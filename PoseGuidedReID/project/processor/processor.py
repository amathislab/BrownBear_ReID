import logging
import os
import time
import torch
import torch.nn as nn
from project.utils.meter import AverageMeter
from project.utils.metrics import R1_mAP_eval
from project.utils.tools import get_feature_result_path, save_results2csv, save_model, get_DDP_model
from project.utils.data_analysis import get_top_k_matched_images
from torch.cuda import amp
import torch.distributed as dist
import numpy as np
from pytorch_metric_learning import losses, miners
from torch.autograd import Variable
from tqdm import tqdm
import wandb
import cv2
import math
from datetime import datetime
import pandas as pd

def train_model(cfg, 
                train_loader,
                model, 
                criterion, 
                criterion_triplet,
                optimizer, 
                scheduler, 
                local_rank=0,
                device="cuda",
                wb_run=None,
                ):

    train_dataset = train_loader.dataset
    batchsize = cfg.SOLVER.IMS_PER_BATCH
    epochs = cfg.SOLVER.MAX_EPOCHS
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    warm_epoch = cfg.SOLVER.WARMUP_EPOCHS
    log_period = cfg.SOLVER.LOG_PERIOD

    logger = logging.getLogger("project.train")
    logger.info('start training')

    model.to(device)
    if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
        model = get_DDP_model(cfg, model, local_rank, logger)
    else:
        ### only with single gpu
        local_rank = 0
        
    ## if resume training
    if cfg.MODEL.RESUME:
        try:
            model_path = os.path.join(cfg.OUTPUT_DIR, 'net_60.pth')
            if not os.path.exists(model_path):
                model_path = os.path.join(cfg.OUTPUT_DIR, 'net_last.pth')
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'module.' in list(checkpoint.keys())[0]: # Check if saved from DDP model
                checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
            model.load_state_dict(checkpoint, strict=False)
            start_epoch = 60  ### TODO -- change to the correct epoch
            print(f"resume from {model_path}, continue training...")
        except:
            start_epoch = 0
            print(f"no resume model found, start from epoch 0")
        
    best_acc = 0.0
    warm_up = 0.1 # We start from the 0.1*lrRate
    warm_iteration = round(len(train_dataset)/batchsize) * warm_epoch # first 5 epoch

    miner = miners.MultiSimilarityMiner()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for epoch in range(1, epochs+1):
        if cfg.MODEL.RESUME:
            epoch += start_epoch

        ### only vis feat for the first epoch (save to wandb)
        model.train()  # Set model to training mode
        loss_meter.reset()
        acc_meter.reset()

        for iter, data in enumerate(train_loader):
            inputs, labels = data[:2]
            now_batch_size,c,h,w = inputs.shape
            if now_batch_size < batchsize:  # skip the last batch
                continue
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            ## if transreid with SIE
            if cfg.MODEL.JPM and cfg.MODEL.TYPE == 'vit' and cfg.DATASETS.NAMES == 'bear':
                cameras = np.array(data[-1])[:, 2].astype(np.int)
                cameras = torch.tensor(cameras).to(device)
                outputs = model(inputs, cam_label=cameras)
            else:
                # others
                outputs = model(inputs)

            logits, ff, pose_conf = outputs

            if isinstance(ff, list):  ### for transreid 
                fnorm = [torch.norm(f, p=2, dim=1, keepdim=True) for f in ff]
                ff = [ff[i].div(f.expand_as(ff[i])) for i, f in enumerate(fnorm)]

                id_loss = [criterion(logit, labels) for logit in logits[1:]]
                id_loss = sum(id_loss) / len(id_loss)
                id_loss = 0.5 * id_loss + 0.5 * criterion(logits[0], labels)
                _, preds = torch.max(logits[0].data, 1)
                
                other_loss =  [criterion_triplet(f, labels, miner(f, labels)) for f in ff[1:]]
                other_loss = sum(other_loss) / len(other_loss)
                other_loss = 0.5 * other_loss + 0.5 * criterion_triplet(ff[0], labels, miner(ff[0], labels))
                loss = cfg.MODEL.ID_LOSS_WEIGHT * id_loss + cfg.MODEL.TRIPLET_LOSS_WEIGHT * other_loss
            
            else:
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                ff = ff.div(fnorm.expand_as(ff))
                loss = criterion(logits, labels)
                _, preds = torch.max(logits.data, 1)
                hard_pairs = miner(ff, labels)
                loss +=  criterion_triplet(ff, labels, hard_pairs)

            log_images = []
            if local_rank == 0 and wb_run is not None:
                # log_images = [wandb.Image(img, caption="Pred:{} Truth:{}".format(preds[i].item(), labels.data[i])) for i, img in enumerate(inputs[:5])]
                log_images = [
                    wandb.Image(inputs[0], caption="Pred:{} Truth:{}".format(preds[0].item(), labels.data[0])),
                    # wandb.Image(inputs[0], caption="Pose Pred:{} Truth:{}".format(pose_preds[0].item(), labels.data[0])),
                    ]

            del inputs  

            # backward + optimize only if in training phase
            if epoch < warm_epoch: 
                warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                loss = loss * warm_up

            loss.backward()
            optimizer.step()

            acc = float(torch.sum(preds == labels.data)) / now_batch_size
            loss_meter.update(loss.item(), now_batch_size)
            acc_meter.update(acc, 1)
            
            if local_rank == 0:
                if iter % log_period == 0 or iter == len(train_dataset)//batchsize - 1:
                    logger.info('[Epoch{}-iter{}/{}]: Loss: {:.4f} Acc: {:.4f}'.format(
                        epoch, iter, len(train_dataset)//batchsize, loss_meter.avg, acc_meter.avg) )

        ### print epoch loss
        if local_rank == 0:        
            logger.info('[Epoch{}]: Loss: {:.4f} Acc: {:.4f}'.format(
                    epoch, loss_meter.avg, acc_meter.avg) )

            if wb_run is not None:
                wb_run.log({
                    "Train samples": log_images,
                    "Acc": 100. * acc_meter.avg,
                    "loss": loss_meter.avg,
                })

        ### save best model by best acc
        if acc_meter.avg > best_acc:
            best_acc = acc_meter.avg
            model_save_path = os.path.join(cfg.OUTPUT_DIR, 'net_best.pth')
            save_model(cfg, model, model_save_path)        

        if epoch % checkpoint_period == 0:
            model_save_path = os.path.join(cfg.OUTPUT_DIR, 'net_{}.pth'.format(epoch))
            save_model(cfg, model, model_save_path)

        ## final checkpoints
        model_path = os.path.join(cfg.OUTPUT_DIR, f'net_last.pth')
        save_model(cfg, model, model_path)
        
        scheduler.step()

    ### release gpu memory
    torch.cuda.empty_cache()

    return model


### TODO: DDP setting

def fliplr(img, device="cuda"):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long().to(device)  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def extract_feature(cfg, model, dataloader, out_dim=768, device="cuda", filter_bad_pose=True, pose_conf_thres=0.8):
    #features = torch.FloatTensor()

    ms = [1]
    ms = [math.sqrt(s_f) for s_f in ms]
    meta_list = []
    count = 0
    features = []
    if cfg.MODEL.TYPE == 'vit':
        out_dim = out_dim * 5
    
    for n_iter, data in enumerate(tqdm(dataloader)):
        img, _, _, meta_info = data
        img_path = meta_info[0][5]
        img = img.to(device)
        model = model.to(device)
        
        n_batch, c, h, w = img.size()
        count += n_batch
        ff = torch.FloatTensor(n_batch, out_dim).zero_().to(device)
        valid_feature = True

        for i in range(2):
            if(i==1):
                img = fliplr(img, device)
            input_img = Variable(img)
            for scale in ms:
                if scale != 1:
                    # bicubic is only available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                            ## if transreid with SIE
                
                
                ### TODO: make outputs of all models consistent (logits, feat, pose_conf)
                
                if cfg.MODEL.JPM and cfg.MODEL.TYPE == 'vit' and cfg.DATASETS.NAMES == 'bear':
                    cameras = np.array(meta_info)[:, 2].astype(np.int)
                    cameras = torch.tensor(cameras).to(device)
                    outputs = model(input_img, cam_label=cameras)
                else:
                    outputs = model(input_img)
                    
                ### 
                if len(outputs) == 4:  ### Transreid
                    feat = outputs[2]
                else:
                    ### swint
                    if isinstance(outputs, list):
                        feat = outputs[1]
                        pose_conf = outputs[2]
                        mean_pose_conf = pose_conf.mean()
                        ### check if nan in feat
                        if torch.isnan(feat).sum()>0:
                            valid_feature = False
                        if mean_pose_conf < pose_conf_thres and filter_bad_pose:
                            print("skip bad pose:", img_path)
                            valid_feature = False
                            break
                    else:
                        feat = outputs
                        
                # print(f"niter: {n_iter}, i: {i}, feat.shape: {feat.shape}, {ff.shape}")
                ff += feat
                
            if not valid_feature:
                break
            
        if valid_feature:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
            
            # ### check if nan in feat
            # if torch.isnan(ff).sum()>0:
            #     print("nan in ff", img_path)
            
            features.append(ff.cpu().numpy())
            meta_list.extend(meta_info)
            
    features = np.concatenate(features, axis=0)
    
    return features, meta_list


def predict(cfg, 
            model, 
            dataloader, 
            out_dim=768, 
            device="cuda", 
            all_classes=None):

    meta_list = []
    count = 0
    if cfg.MODEL.TYPE == 'vit':
        out_dim = out_dim * 5
    
    ## get train_meta_info
        
    for n_iter, data in enumerate(tqdm(dataloader)):
        imgs, labels, _, meta_info = data
        imgs = imgs.to(device)
        model = model.to(device)
        meta_list.extend(meta_info)
        n_batch, c, h, w = imgs.size()
        count += n_batch

        input_imgs = Variable(imgs)
        if cfg.MODEL.JPM and cfg.MODEL.TYPE == 'vit' and cfg.DATASETS.NAMES == 'bear':
            cameras = np.array(meta_info)[:, 2].astype(np.int)
            cameras = torch.tensor(cameras).to(device)
            outputs = model(input_imgs, cam_label=cameras)
        else:
            outputs = model(input_imgs)
            
        if len(outputs) == 4:
            logits = outputs[0]
        else:
            if isinstance(outputs, list):
                logits = outputs[0]
            else:
                logits = outputs
        conf, pred = torch.max(logits.data, 1)
        
        ## convert tensor to numpy
        pred = pred.cpu().numpy()
        conf = conf.cpu().numpy()

        ## match pred id to real 'train' labels (because the data were mapped to different pid)
        train_id_label_pairs = all_classes['train']
        pred_labels = [train_id_label_pairs[pred_id] for pred_id in pred]
        
        ## get real ood labels
        meta_info = np.array(meta_info)
        
        if n_iter == 0:
            predictions = []
            confidences = []
        predictions.extend(pred_labels)
        confidences.extend(conf)
        
    ## normalize confidences to [0, 1]
    confidences = np.array(confidences)
    confidences = (confidences - np.min(confidences)) / (np.max(confidences) - np.min(confidences))
    
    meta_list = np.array(meta_list)
    gt_labels = meta_list[:, 1]
    gt_paths = meta_list[:, 5]
    
    # save prediction jsons
    df_results = pd.DataFrame({'paths':gt_paths, 
                               'labels':gt_labels, 
                               'predictions':predictions, 
                               'confidences':confidences, 
                               })
    
    return predictions, df_results



def save_feature(cfg, feature, meta_list, logger, result_path):

    keys = ['id', 'label', 'camera', 'date', 'unique_ids', 'path', 'keypoints', 'sex']
    meta_dict = dict(zip(keys, zip(*meta_list)))

    ## if feature is tensor, convert to numpy
    if isinstance(feature, torch.Tensor):
        feature = feature.numpy()
    result = {'feature':feature}
    result.update(meta_dict)
    
    np.savez(result_path, **result)
    logger.info(f'output feature saved to : {result_path}')


def do_evaluation(cfg,
                  query_ids,
                  query_feature,
                  gallery_ids,
                  gallery_feature,
                  query_timestamps,
                  gallery_timestamps,
                  gallery_type='train',
                  query_type='iid'):
    
    
    logger = logging.getLogger(f"project.evaluate.{query_type}")
    logger.info(f"-------------- Query: {query_type}, Gallery: {gallery_type} ---------------")
    logger.info("Validation Results ")
    logger.info("-----------------------------")
    logger.info(f"query: {len(query_ids)}, gallery: {len(gallery_ids)}")
    logger.info(f"q_id: {len(set(query_ids))}, g_id: {len(set(gallery_ids))}")
    
    ### metrics from TransReID 
    evaluator = R1_mAP_eval(logger, max_rank=cfg.TEST.MAX_RANK, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING, device=cfg.MODEL.DEVICE_ID[0], mAP_for_max_rank=cfg.TEST.MAP_MAX_RANK, filter_date=cfg.TEST.FILTER_DATE)
    evaluator.reset()
    evaluator.update(query_feature, query_ids, gallery_feature, gallery_ids, query_timestamps, gallery_timestamps)
    cmc, mAP, all_AP, sorted_qg_matrix, distmat = evaluator.compute()

    logger.info("mAP: {:.1%}".format(mAP))
    
    if len(cmc) >= 10:
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    else:
        logger.info("CMC curve, Rank-1:{:.1%}".format(cmc[0]))
    logger.info("-----------------------------")
    
    return cmc, mAP, all_AP, sorted_qg_matrix, distmat


def run_evaluation_pipeline(cfg,
                            query_data,
                            gallery_data,
                            query_type='iid',
                            gallery_type='train',
                            split_type='',  ## '' or 'unqiue' or 'full'
                            top_k=10,
                            filter_date=True):
    
    '''
    Args:
        postprocessing: PostProcessing object
        out_dir: str, output directory
        data_type: str, 'iid' or 'ood'
        random_split_seed: int, random seed for splitting the gallery set or None
        split_type: str, '' or 'unique' or 'full'
        save_results: bool, whether to save the results to csv
        top_k: int, top k matched images from gallery set
    Returns:
        cmc: numpy array, shape: (len(query), len(gallery)), cmc curve
        mAP: float, mean average precision
        all_AP: numpy array, shape: (len(query), len(gallery)), AP for each query
        sorted_qg_indices: numpy array, shape: (len(query), len(gallery)), sorted indices of gallery images for each query image
    '''

    ### get query-gallery set
    query_paths, query_labels, query_feature, query_cameras, query_timestamps, query_is_unique, query_ids = query_data
    gallery_paths, gallery_labels, gallery_feature, gallery_cameras, gallery_timestamps, gallery_is_unique, gallery_ids = gallery_data
    if isinstance(query_feature, list):
        query_feature = torch.stack(query_feature)
    if isinstance(gallery_feature, list):
        gallery_feature = torch.stack(gallery_feature)
    query_ids = np.array(query_ids)
    gallery_ids = np.array(gallery_ids)
    query_labels = np.array(query_labels)
    gallery_labels = np.array(gallery_labels)
    query_timestamps = np.array(query_timestamps)
    gallery_timestamps = np.array(gallery_timestamps)

    if split_type == '':
        split_type = 'known'
    query_type = f'{query_type}-{split_type}'

    cmc, mAP, all_AP, sorted_qg_indices, distmat = do_evaluation(cfg, 
                                                        query_labels,
                                                        query_feature,
                                                        gallery_labels,
                                                        gallery_feature,
                                                        query_timestamps,
                                                        gallery_timestamps,
                                                        query_type=query_type,
                                                        gallery_type=gallery_type)


    query_timestamps = np.array([date.split(" ")[0] for date in query_timestamps])
    gallery_timestamps = np.array([date.split(" ")[0] for date in gallery_timestamps])

    results, result_image = get_top_k_matched_images(cfg,
                                                     all_AP, 
                                                    query_labels, 
                                                    query_paths, 
                                                    query_ids, 
                                                    gallery_labels, 
                                                    gallery_paths, 
                                                    gallery_ids, 
                                                    query_timestamps,
                                                    gallery_timestamps,
                                                    sorted_qg_indices, 
                                                    k=top_k,
                                                    filter_date=filter_date)


    return cmc, mAP, all_AP, sorted_qg_indices, len(query_ids), len(gallery_ids), results, result_image


def do_inference(cfg,
                 model,
                 val_loader,
                 data_type='iid',
                 out_dir=None,
                 local_rank=0,
                 device="cuda"):
    
    logger = logging.getLogger("project.inference")
    logger.info("Enter inferencing")

    if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
        model = get_DDP_model(cfg, model, local_rank, logger)

    model.eval()

    ## save features to disk
    if out_dir is None:
        out_dir = cfg.OUTPUT_DIR
    
    if cfg.MODEL.TYPE == 'swin':
        out_dim = cfg.MODEL.SWIN.EMBED_DIM
        if cfg.DATASETS.NAMES == 'macaque':
            out_dim = 256
    elif cfg.MODEL.TYPE == 'vit':
        out_dim = cfg.MODEL.VIT_EMBED_DIM
    elif cfg.MODEL.TYPE == 'resnet':
        out_dim = cfg.MODEL.RESNET_EMBED_DIM
    elif cfg.MODEL.TYPE == 'swinv2':
        out_dim = cfg.MODEL.SWINV2.EMBED_DIM

    weight = cfg.TEST.WEIGHT  ## net_xxx.pth
    epoch = weight.split('_')[-1].split('.')[0]
    result_path = get_feature_result_path(cfg, out_dir=out_dir, data_type=data_type, epoch=epoch)

    with torch.no_grad():
        logger.info('get model predictions...')
        features, meta_list = extract_feature(cfg, model, val_loader, out_dim=out_dim, device=device)
    logger.info("Inference done!")
    save_feature(cfg, features, meta_list, logger, result_path)

    return result_path



def do_prediction(cfg,
                 model,
                 val_loader,
                 data_type='iid',
                 out_dir=None,
                 local_rank=0,
                 device="cuda",
                 all_classes=None):
    
    logger = logging.getLogger("project.prediction")
    logger.info("Enter predictions")

    if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
        model = get_DDP_model(cfg, model, local_rank, logger)

    model.eval()

    ## save features to disk
    if out_dir is None:
        out_dir = cfg.OUTPUT_DIR
        
    if cfg.MODEL.TYPE == 'swin':
        out_dim = cfg.MODEL.SWIN.EMBED_DIM
        if cfg.DATASETS.NAMES == 'macaque':
            out_dim = 256
    elif cfg.MODEL.TYPE == 'vit':
        out_dim = cfg.MODEL.VIT_EMBED_DIM
    elif cfg.MODEL.TYPE == 'resnet':
        out_dim = cfg.MODEL.RESNET_EMBED_DIM
    elif cfg.MODEL.TYPE == 'swinv2':
        out_dim = cfg.MODEL.SWINV2.EMBED_DIM

    weight = cfg.TEST.WEIGHT  ## net_xxx.pth
    epoch = weight.split('_')[-1].split('.')[0]
    result_path = get_feature_result_path(cfg, out_dir=out_dir, data_type=data_type, epoch=epoch)
    result_path = result_path.replace('.npz', '.csv')

    with torch.no_grad():
        logger.info('get model predictions...')
        predictions, df_results = predict(cfg, model, val_loader, out_dim=out_dim, device=device, all_classes=all_classes)
    logger.info("Prediction done!")
    
    ### save to csv
    save_results2csv(df_results, result_path)
    print("predictions save to :", result_path)

    return df_results
