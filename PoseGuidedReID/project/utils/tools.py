from __future__ import division, print_function, absolute_import
import os
import sys
import json
import time
import errno
import numpy as np
import random
import os.path as osp
import warnings
import PIL
import torch
from PIL import Image
import torch.nn as nn
import pandas as pd
import torch.distributed as dist
import cv2
import matplotlib.pyplot as plt

__all__ = [
    'mkdir_if_missing', 'check_isfile', 'read_json', 'write_json',
    'set_random_seed', 'download_url', 'read_image', 'collect_env_info',
    'listdir_nohidden'
]


def mkdir_if_missing(dirname):
    """Creates dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def check_isfile(fpath):
    """Checks if the given path is a file.

    Args:
        fpath (str): file path.

    Returns:
       bool
    """
    isfile = osp.isfile(fpath)
    if not isfile:
        warnings.warn('No file found at "{}"'.format(fpath))
    return isfile


def read_json(fpath):
    """Reads json file from a path."""
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    """Writes to a json file."""
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def download_url(url, dst):
    """Downloads file from a url to a destination.

    Args:
        url (str): url to download file.
        dst (str): destination path.
    """
    from six.moves import urllib
    print('* url="{}"'.format(url))
    print('* destination="{}"'.format(dst))

    def _reporthook(count, block_size, total_size):
        global start_time
        if count == 0:
            start_time = time.time()
            return
        duration = time.time() - start_time
        progress_size = int(count * block_size)
        speed = 0 if duration == 0 else int(progress_size / (1024*duration))
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(
            '\r...%d%%, %d MB, %d KB/s, %d seconds passed' %
            (percent, progress_size / (1024*1024), speed, duration)
        )
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dst, _reporthook)
    sys.stdout.write('\n')


def read_image(path):
    """Reads image from path using ``PIL.Image``.

    Args:
        path (str): path to an image.

    Returns:
        PIL image
    """
    got_img = False
    if not osp.exists(path):
        raise IOError('"{}" does not exist'.format(path))
    while not got_img:
        try:
            img = Image.open(path).convert('RGB')
            got_img = True
        except IOError:
            print(
                'IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'
                .format(path)
            )
    return img


def collect_env_info():
    """Returns env info as a string.

    Code source: github.com/facebookresearch/maskrcnn-benchmark
    """
    from torch.utils.collect_env import get_pretty_env_info
    env_str = get_pretty_env_info()
    env_str += '\n        Pillow ({})'.format(PIL.__version__)
    return env_str


def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory.

    Args:
         path (str): directory path.
         sort (bool): sort the items.
    """
    items = [f for f in os.listdir(path) if not f.startswith('.')]
    if sort:
        items.sort()
    return items


def get_DDP_model(cfg, model, local_rank, logger=None):

    device = torch.device('cuda', local_rank)
    if dist.get_rank() == 0 and logger is not None:
        logger.info('Using {} GPU(s)'.format(cfg.MODEL.DEVICE_ID))
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)
    model._set_static_graph()

    return model



def save_model(cfg, model, model_path):
    if cfg.MODEL.DIST_TRAIN:
        if dist.get_rank() == 0:
            torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)



def load_model(model, checkpoint_path, logger, remove_fc=False, local_rank=0, is_swin=True):

    if remove_fc:
        model.classifier.classifier = nn.Sequential()

    if os.path.isfile(checkpoint_path):
        if local_rank == 0:
            logger.info("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'module.' in list(checkpoint.keys())[0]: # Check if saved from DDP model
            checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        if remove_fc:
            checkpoint = {k: v for k, v in checkpoint.items() if 'classifier.classifier' not in k}
            if not is_swin:
                checkpoint = {k: v for k, v in checkpoint.items() if 'classifier' not in k}  ## for transreid, resnet
                checkpoint = {k: v for k, v in checkpoint.items() if 'bottleneck' not in k}  ## for transreid, resnet
        for k, v in checkpoint.items():
            if k not in model.state_dict().keys():
                print(k, 'not in model')
        for k, v in model.state_dict().items():
            if k not in checkpoint.keys():
                print(k, 'not in checkpoint')

        model.load_state_dict(checkpoint, strict=False)
    else:
        if local_rank == 0:
            logger.info("=> no checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()

    return model



def get_feature_result_path(cfg, out_dir=None, data_type='iid', epoch=None):
    
    if out_dir is None:
        out_dir = cfg.OUTPUT_DIR
    out_dir = os.path.join(out_dir, 'pred_features', data_type)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    if epoch is None:
        weight = cfg.TEST.WEIGHT  ## net_xxx.pth
        epoch = weight.split('_')[-1].split('.')[0]
        
    if cfg.INPUT.PRE_SCALING is not None:
        epoch = f'{epoch}_size{cfg.INPUT.PRE_SCALING}'
        
    result_path = os.path.join(out_dir, f'pytorch_result_e{epoch}.npz')

    return result_path

def get_prediction_result_path(out_dir, query_type, gallery_type, seed=None, split_single_year=None):

    filename = f'results_Q{query_type}_G{gallery_type}' if split_single_year is None else f'results_Q{query_type}_G{gallery_type}_{split_single_year}'
    if seed is not None:
        filename = f'{filename}_s{seed}'
    
    return os.path.join(out_dir, f'{filename}.csv')

def save_results2csv(results, output_path):
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)

def setup_ddp_training():
    
    # Set default values for single-GPU training
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12355"  # Choose any free port
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"
        
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="nccl")
    print(
        f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
    )

    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    world_size = dist.get_world_size()

    return local_rank, world_size, device


def overlay_keypoints_on_image(image, keypoints):
    radius = 5
    if image.shape[0] > 500:
        radius = 10
    if image.shape[0] > 1000:
        radius = 30

    colors = plt.cm.get_cmap('hsv', len(keypoints))
    colors = [[colors(i)[0]*255, colors(i)[1]*255, colors(i)[2]*255 ] for i in range(colors.N)]
    
    for i, (x, y) in enumerate(keypoints):
        color = colors[i]
        cv2.circle(image, (int(x), int(y)), radius, color, -1)

    return image


def get_full_path(im_path, img_dir = './data', dataset='bear', body_input=False):
    # im_path
    if dataset == 'macaque':
        img_dir= os.getcwd() + '/../../../datasets_other_animals/MacaqueFaces/MacaqueFaces'
        im_path = os.path.join(img_dir, im_path[1:])  
    elif dataset == 'bear':
        if not body_input:
            if '2022/' in im_path:
                im_path = os.path.join(img_dir, im_path)
            else:
                year = im_path[:4]
                im_path = os.path.join(img_dir, year, im_path)
        else:
            im_path = os.path.join(img_dir, im_path)
         
    return im_path


def fuse_all_conv_bn(model):
    from torch.nn.utils import fuse_conv_bn_eval
    stack = []
    for name, module in model.named_children():
        if list(module.named_children()):
            fuse_all_conv_bn(module)
            
        if isinstance(module, nn.BatchNorm2d):
            if not stack:
                continue
            if isinstance(stack[-1][1], nn.Conv2d):
                setattr(model, stack[-1][0], fuse_conv_bn_eval(stack[-1][1], module))
                setattr(model, name, nn.Identity())
        else:
            stack.append((name, module))
    return model




def get_id(img_path, norm_filename=False, iid=False):

    camera_ids = []
    labels = []
    date_ids = []
    unique_ids = []
    paths = []
    for path in img_path:
        label = path.split('/')[-2]
        filename = path.split('/')[-1]
        year = filename[:4]
        if norm_filename:
            camera = int(filename.split('_')[1].split('c')[-1])
            date = int(filename.split('_')[1].split('d')[-1][0])
            if iid:
                # date = int(filename.split('_')[1].split('d')[-1].split('c')[0])
                unique_id = 0
            else:
                # date = int(filename.split('_')[1].split('d')[-1].split('u')[0])
                unique_id = int(filename.split('_')[1].split('u')[-1][0])
        else:
            camera = 0
            date = 0
            unique_id = 0
        labels.append(label)
        camera_ids.append(int(camera))
        date_ids.append(date)
        unique_ids.append(unique_id)
        paths.append(path)

    return paths, camera_ids, labels, date_ids, unique_ids



def get_result_metric_csv_path(out_dir=''):
    return os.path.join(out_dir, 'metrics.csv')



def split_test_to_query_gallery(test_path, test_feature, 
                                norm_filename=False, 
                                iid=False, 
                                seed=10, 
                                month=None,
                                sex=None,
                                ):
    # Get the camera info, label info, date info, and unique ids info
    test_path, test_camera, test_label, test_date, test_unique_ids = get_id(test_path, norm_filename=norm_filename, iid=iid)

    # print(test_date)

    # Combine the info into a list of dictionaries
    image_info = [
        {"path": path.rstrip(), "label": label, "camera": camera, "date": date, "unique_id": unique_id, 'feature': feature}
        for path, label, camera, date, unique_id, feature in zip(test_path, test_label, test_camera, test_date, test_unique_ids, test_feature)
    ]

    # Group images by label
    images_by_label = {}
    for image in image_info:
        # print("dd", image["date"])
        if image["label"] not in images_by_label:
            images_by_label[image["label"]] = []
        images_by_label[image["label"]].append(image)
        
    query_path = []
    gallery_path = []
    query_feature = []
    gallery_feature = []
    query_label = []
    gallery_label = []

    unique_query_path = []
    unique_gallery_path = []
    unique_query_feature = []
    unique_gallery_feature = []
    unique_query_label = []
    unique_gallery_label = []

    full_query_path = []
    full_query_feature = []
    full_query_label = []

    # Randomly pick 10% images from each label for the query set
    for label, images in images_by_label.items():
        random.shuffle(images)

        # Separate images by date
        images_by_date = {}
        for image in images:
            if image["date"] not in images_by_date:
                images_by_date[image["date"]] = []
            images_by_date[image["date"]].append(image)

        # Sort dates by the number of images
        sorted_dates = sorted(images_by_date.keys(), key=lambda x: len(images_by_date[x]), reverse=True)

        # print("sorted_dates", len(sorted_dates), images_by_date.keys())

        if len(sorted_dates) == 1: ## if the bear images are taken from 1 day, ignored
            continue


        # Pick 1 or 2 non-overlapped date id(s) for the query set
        num_dates_for_query = 1 if len(sorted_dates) == 2 else int(len(sorted_dates)/3)
        query_dates = random.sample(sorted_dates, num_dates_for_query)
        gallery_dates = list(set(sorted_dates).difference(query_dates))

        # print("query date", query_dates)
        # print("gallery date", gallery_dates)

        # Add images to the query set and gallery set
        for image in images:
            
            if image["unique_id"] == 1:  ### if the bear is unique in this year, ignored
                ## TODO: split in another set
                if image["date"] in query_dates:
                    unique_query_path.append(image["path"])
                    unique_query_feature.append(image['feature'])
                    unique_query_label.append(image['label'])
                    # print("query", image['label'])
                    full_query_path.append(image["path"])
                    full_query_feature.append(image['feature'])
                    full_query_label.append(image['label'])
                # else:
                #     # unique_gallery_path.append(image["path"])
                #     # unique_gallery_feature.append(image['feature'])
                #     # unique_gallery_label.append(image['label'])
                #     gallery_path.append(image["path"])
                #     gallery_feature.append(image['feature'])
                #     gallery_label.append(image['label'])
            if image["date"] in query_dates:
                query_path.append(image["path"])
                query_feature.append(image['feature'])
                query_label.append(image['label'])

                full_query_path.append(image["path"])
                full_query_feature.append(image['feature'])
                full_query_label.append(image['label'])
                # print("query", image['label'])
            else:
                year = os.path.basename(image["path"])[:4]
                gallery_path.append(image["path"])
                gallery_feature.append(image['feature'])
                gallery_label.append(image['label'])
                # unique_gallery_path.append(image["path"])
                # unique_gallery_feature.append(image['feature'])
                # unique_gallery_label.append(image['label'])

    gallery_feature = torch.stack(gallery_feature)
    query_feature = torch.stack(query_feature)
    # unique_gallery_feature = torch.stack(unique_gallery_feature)
    full_query_feature = torch.stack(full_query_feature)
    if unique_query_feature!= []:
        unique_query_feature = torch.stack(unique_query_feature)


    # img_dir = '/media/mu/bear/other_projects_mu/bear_data_project/experiments/heads_5years/test_on_2021'
    
    # filtered_query_path = set(query_path) - set(unique_query_path)
    # save_splitted_data(filtered_query_path, seed)

    # for query_img in filtered_query_path:
    #     src = query_img.rstrip()
    #     bear_id = os.path.basename(os.path.split(src)[0])
    #     dst_dir = os.path.join(img_dir, f'seed_{opt.seed}', 'filtered_query', bear_id)
    #     if not os.path.isdir(dst_dir):
    #         os.makedirs(dst_dir)
    #     dst = os.path.join(dst_dir, os.path.split(src)[1])
    #     if os.path.isfile(dst):
    #         continue
    #     os.symlink(src, dst)
    #     print(src, dst)

    # for query_img in unique_query_path:
    #     src = query_img.rstrip()
    #     bear_id = os.path.basename(os.path.split(src)[0])
    #     dst_dir = os.path.join(img_dir, f'seed_{opt.seed}', 'unique_query', bear_id)
        
    #     if not os.path.isdir(dst_dir):
    #         os.makedirs(dst_dir)
    #     dst = os.path.join(dst_dir, os.path.split(src)[1])
    #     if os.path.isfile(dst):
    #         continue
    #     os.symlink(src, dst)
    #     print(src, dst)

    # for query_img in query_path:
    #     src = query_img.rstrip()
    #     bear_id = os.path.basename(os.path.split(src)[0])
    #     dst_dir = os.path.join(img_dir, f'seed_{opt.seed}', 'query', bear_id)
        
    #     if not os.path.isdir(dst_dir):
    #         os.makedirs(dst_dir)
    #     dst = os.path.join(dst_dir, os.path.split(src)[1])
    #     if os.path.isfile(dst):
    #         continue
    #     os.symlink(src, dst)
    #     print(src, dst)

    # good = 0
    # for gallery_img in gallery_path:
    #     src = gallery_img.rstrip()
    #     bear_id = os.path.basename(os.path.split(src)[0])
    #     dst_dir = os.path.join(img_dir, f'seed_{opt.seed}', 'gallery', bear_id)
    #     if not os.path.isdir(dst_dir):
    #         os.makedirs(dst_dir)
    #     dst = os.path.join(dst_dir, os.path.split(src)[1])
    #     if os.path.isfile(dst):
    #         continue
    #     else:
    #         os.symlink(src, dst)
    #         good+=1
    # print(len(gallery_path), good)
    # gallery_path = [g.rstrip() for g in gallery_path]
    # print(len(gallery_path), len(np.unique(gallery_path)))

    ### note: unique_gallery_xx are empty, because during evaluation, we want to make sure the query can match
    # the images taken from another day, and make sure there is no consecutive images. 
    # Therefore, we only evaluate the images from other days. 
    return query_path, gallery_path, query_feature, gallery_feature, query_label, gallery_label, \
        unique_query_path, unique_gallery_path, unique_query_feature, unique_gallery_feature, unique_query_label, unique_gallery_label,\
        full_query_path, full_query_feature, full_query_label

