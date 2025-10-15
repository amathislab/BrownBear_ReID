import torch
import numpy as np
from project.utils.reranking import re_ranking
import time
from datetime import datetime
import torch
import torch.nn as nn


class MatchCalculator(nn.Module):
    def __init__(self, q_pids, g_pids):
        super(MatchCalculator, self).__init__()
        self.q_pids = q_pids
        self.g_pids = g_pids

    def forward(self, indices):
        # Assuming indices is a tensor that indexes g_pids
        # Adjust this logic based on your specific computation
        g_pids_selected = self.g_pids[indices]
        matches = (g_pids_selected == self.q_pids[:, None]).float()
        return matches
    
    
def euclidean_distance(qf, gf):
    time_start = time.time()
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    # dist_mat.addmm_(1, -2, qf, gf.t())
    dist_mat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    
    t = time.time() - time_start
    # print("euclidean distance time taken", t)
    return dist_mat.cpu().numpy()

def batched_euclidean_distance(qf, gf, batch_size=100):
    time_start = time.time()
    m, n = qf.shape[0], gf.shape[0]
    dist_mat = torch.zeros(m, n, device=qf.device, dtype=qf.dtype)

    for i in range(0, m, batch_size):
        end = min(i + batch_size, m)
        
        # Compute the squared norms for the batch and the full gf
        norms_qf = torch.pow(qf[i:end], 2).sum(dim=1, keepdim=True)
        norms_gf = torch.pow(gf, 2).sum(dim=1, keepdim=True).t()

        # Compute the distance matrix for the batch
        dist_mat_batch = norms_qf + norms_gf
        dist_mat_batch.addmm_(qf[i:end], gf.t(), beta=1, alpha=-2)

        dist_mat[i:end, :] = dist_mat_batch

    t = time.time() - time_start
    # print("batched distance - time taken", t)
    return dist_mat.cpu().numpy()

# def compute_cosine_distance(qf, gf):
#     qf_norm = nn.functional.normalize(qf, p=2, dim=1)
#     gf_norm = nn.functional.normalize(gf, p=2, dim=1)
#     cosine_similarity_matrix = torch.matmul(qf_norm, gf_norm.T)
#     dist_mat = 1 - cosine_similarity_matrix
#     dist_mat = dist_mat.cpu().numpy()
#     return dist_mat


def compute_cosine_distance(qf, gf, batch_size=1000):
    m, n = qf.shape[0], gf.shape[0]
    dist_mat = torch.zeros(m, n, device=qf.device, dtype=qf.dtype)

    for i in range(0, m, batch_size):
        end = min(i + batch_size, m)
        qf_norm = nn.functional.normalize(qf[i:end], p=2, dim=1)
        gf_norm = nn.functional.normalize(gf, p=2, dim=1)
        cosine_similarity_matrix = torch.matmul(qf_norm, gf_norm.T)
        dist_mat_batch = 1 - cosine_similarity_matrix
        dist_mat[i:end, :] = dist_mat_batch

    return dist_mat.cpu().numpy()

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat

def create_mapping(q_data, g_data):
    combined = np.concatenate((q_data, g_data))
    unique = np.unique(combined)
    data_to_int = {val: i for i, val in enumerate(unique)}
    q_data_int = np.array([data_to_int[val] for val in q_data])
    g_data_int = np.array([data_to_int[val] for val in g_data])
    return q_data_int, g_data_int

def datetime2date(date):
    # convert datetime to date
    if isinstance(date, str):
        date = date.split(" ")[0]
    if isinstance(date, datetime):
        date = date.date()
    return date
    
def batch_sort(distmat, batch_size=100):
    num_queries = distmat.shape[0]
    indices = torch.empty_like(distmat, dtype=torch.long)
    for start_idx in range(0, num_queries, batch_size):
        end_idx = min(start_idx + batch_size, num_queries)
        indices[start_idx:end_idx] = torch.argsort(distmat[start_idx:end_idx], dim=1)
    return indices

def compute_matches_in_batches(q_pids, g_pids, indices, batch_size=100):
    num_queries, num_gallery = indices.shape
    # Initialize the matches tensor with zeros
    print("num_queries", num_queries, "num_gallery", num_gallery)
    matches = torch.zeros(num_queries, num_gallery, dtype=torch.float32, device='cuda')
    
    # Process each batch of indices
    for start_idx in range(0, num_gallery, batch_size):
        end_idx = min(start_idx + batch_size, num_gallery)
        batch_indices = indices[:, start_idx:end_idx]
        # Expand q_pids to match the batch_indices shape for broadcasting
        expanded_q_pids = q_pids[:, None].expand(-1, end_idx-start_idx)
        # Compare gallery PIDs with query PIDs and convert to float for matches
        matches[:, start_idx:end_idx] = (g_pids[batch_indices] == expanded_q_pids).float()
    
    return matches


def eval_func_gpu(distmat, 
                  q_pids, 
                  g_pids, 
                  q_dates, 
                  g_dates, 
                  q_paths,
                  g_paths,
                  max_rank=50, 
                  device='cuda', 
                  mAP_for_max_rank=False, 
                  filter_date=False,):
    """
        Modified - evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.

        ### Note: for bear, we don't need ignore any index - they were removed when splitting query and gallery
        ### we don't need to remove the same camera view
        ### but we want to remove the same date
        
    """
    
    # Ensure all inputs are tensors and transferred to the GPU -- turn into int first
    _q_pids = q_pids.copy()
    _g_pids = g_pids.copy()
    q_pids_int, g_pids_int = create_mapping(q_pids, g_pids)

    distmat = torch.tensor(distmat, dtype=torch.float32).cuda().to(device)
    q_pids = torch.tensor(q_pids_int, dtype=torch.int64).cuda().to(device)
    g_pids = torch.tensor(g_pids_int, dtype=torch.int64).cuda().to(device)
    
    ## check if the dates are in the same format, only date, no time
    q_dates = [datetime2date(date) for date in q_dates]
    g_dates = [datetime2date(date) for date in g_dates]
    q_dates_int, g_dates_int = create_mapping(q_dates, g_dates)
    q_dates = torch.tensor(q_dates_int, dtype=torch.int64).cuda().to(device)
    g_dates = torch.tensor(g_dates_int, dtype=torch.int64).cuda().to(device)
    
    if q_paths is not None and g_paths is not None:
        ## convert to int
        q_paths_int, g_paths_int = create_mapping(q_paths, g_paths)
        q_paths = torch.tensor(q_paths_int, dtype=torch.int64).cuda().to(device)
        g_paths = torch.tensor(g_paths_int, dtype=torch.int64).cuda().to(device)
    
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    # indices = torch.argsort(distmat, dim=1)
    indices = batch_sort(distmat, batch_size=1000)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, None]).float()

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
        
    valid_indices = []
    for q_idx in range(num_q):
        
        # get query pid and date
        q_pid = q_pids[q_idx]
        q_date = q_dates[q_idx]

        if filter_date:
            # remove gallery samples that have the same pid and date with query
            order = indices[q_idx]  # select one row
            remove = (g_pids[order] == q_pid) & (g_dates[order] == q_date)
            keep = ~remove
        else:
            keep = torch.ones(num_g, dtype=torch.bool).cuda().to(device) # all True
            ### check if the query and gallery are the exact same, if yes, filter the same index (so not to include the same image in the ranking)
            ## check the path of query and gallery, only keep the different ones
            keep = torch.ones(num_g, dtype=torch.bool).cuda().to(device) # all True
            if q_paths is not None and g_paths is not None:
                remove = q_paths[q_idx] == g_paths[indices[q_idx]]
                keep = ~remove

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        
        valid_indices.append(indices[q_idx][keep].cpu().numpy())
        if mAP_for_max_rank:
            orig_cmc = orig_cmc[:max_rank]
            
        if not torch.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            all_AP.append(np.nan)
            continue

        cmc = orig_cmc.cumsum(dim=0)  ## count the number of correct matches at each rank
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank].cpu().numpy())
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum().item() # number of relevant images
        tmp_cmc = orig_cmc.cumsum(dim=0) 
        y = torch.arange(1, tmp_cmc.size(0) + 1).float().cuda().to(device) # 1 to n
        tmp_cmc = tmp_cmc / y
        tmp_cmc = tmp_cmc * orig_cmc  ## precision at each rank
        AP = tmp_cmc.sum().item() / num_rel ## average precision
        all_AP.append(AP)
    
    # assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    if num_valid_q == 0:
        print("Error: all query identities do not appear in gallery, returning empty lists")
        return [], 0, [], [], []
    else:
        all_cmc = np.array(all_cmc)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.nansum(all_AP) / num_valid_q
        return all_cmc, mAP, all_AP, indices, valid_indices


def eval_func(distmat, q_pids, g_pids, q_dates, g_dates, max_rank=50, filter_date=False):
    """
        Modified - evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.

        ### Note: for bear, we don't need ignore any index - they were removed when splitting query and gallery
        ### we don't need to remove the same camera view
        ### but we want to remove the same date
    """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        
        # get query pid and date
        q_pid = q_pids[q_idx]
        q_date = q_dates[q_idx]

        if filter_date:
            # remove gallery samples that have the same pid and date with query
            order = indices[q_idx]  # select one row
            remove = (g_pids[order] == q_pid) & (g_dates[order] == q_date)
            keep = np.invert(remove)
        else:
            keep = np.ones(num_g, dtype=bool) # all True

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            all_AP.append(np.nan)
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc)
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.nansum(all_AP) / num_valid_q

    return all_cmc, mAP, all_AP, indices


class R1_mAP_eval():
    def __init__(self, logger, max_rank=50, feat_norm=True, reranking=False, device='cuda', mAP_for_max_rank=False, filter_date=True, distance_norm=False):
        super(R1_mAP_eval, self).__init__()
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
        self.logger = logger
        self.device = device
        self.mAP_for_max_rank = mAP_for_max_rank
        self.filter_date = filter_date
        self.distance_norm = distance_norm

    def reset(self):
        self.qf = []
        self.gf = []
        self.q_ids = []
        self.g_ids = []

    def update(self, qfeat, qid, gfeat, gid, qdate, gdate, qpaths=None, gpaths=None):
        self.qf = qfeat
        self.gf = gfeat
        self.q_ids = qid
        self.g_ids = gid
        self.q_dates = qdate
        self.g_dates = gdate
        self.q_paths = qpaths
        self.g_paths = gpaths
        
    def compute_distmat(self):
        if isinstance(self.qf, tuple):
            self.qf = torch.stack(self.qf)
            self.gf = torch.stack(self.gf)
        if self.feat_norm:
            # self.logger.info("The test feature is normalized")
            self.qf = torch.nn.functional.normalize(self.qf, dim=1, p=2)  # along channel
            self.gf = torch.nn.functional.normalize(self.gf, dim=1, p=2)  # along channel
        
        if self.reranking:
            # self.logger.info('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(self.qf, self.gf, k1=50, k2=15, lambda_value=0.3)

        else:
            # self.logger.info('=> Computing DistMat with euclidean_distance')
            # distmat = euclidean_distance(self.qf, self.gf)
            # distmat = cosine_similarity(self.qf, self.gf)
            # distmat = batched_euclidean_distance(self.qf, self.gf, batch_size=1000) ### speed up the process when gallery is large
            distmat = compute_cosine_distance(self.qf, self.gf, batch_size=1000) ### speed up

        return distmat
        

    def compute(self):
        if isinstance(self.qf, tuple):
            self.qf = torch.stack(self.qf)
            self.gf = torch.stack(self.gf)
        if self.feat_norm:
            # self.logger.info("The test feature is normalized")
            self.qf = torch.nn.functional.normalize(self.qf, dim=1, p=2)  # along channel
            self.gf = torch.nn.functional.normalize(self.gf, dim=1, p=2)  # along channel
        
        if self.reranking:
            # self.logger.info('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(self.qf, self.gf, k1=50, k2=15, lambda_value=0.3)

        else:
            # self.logger.info('=> Computing DistMat')
            distmat = compute_cosine_distance(self.qf, self.gf, batch_size=1000) ### speed up
            # distmat = batched_euclidean_distance(self.qf, self.gf, batch_size=1000) ### speed up the process when gallery is large

        cmc, mAP, all_AP, sorted_qg_indices, valid_sorted_qg_indices = eval_func_gpu(distmat, 
                                                                                        self.q_ids, 
                                                                                        self.g_ids, 
                                                                                        self.q_dates, 
                                                                                        self.g_dates, 
                                                                                        self.q_paths,
                                                                                        self.g_paths,
                                                                                        max_rank=self.max_rank, 
                                                                                        device=self.device, 
                                                                                        mAP_for_max_rank=self.mAP_for_max_rank,
                                                                                        filter_date=self.filter_date)
        
        return cmc, mAP, all_AP, valid_sorted_qg_indices, distmat
