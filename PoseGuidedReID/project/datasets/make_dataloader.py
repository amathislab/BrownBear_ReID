import torch
import torchvision.transforms as T
import cv2

from torch.utils.data import DataLoader

from .bases import ImageDataset, BaseImageDataset, SplittedImageDataset
from timm.data.random_erasing import RandomErasing
from .transforms import CustomCoarseDropout
from .sampler import RandomIdentitySampler
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist

__factory = {
    'bear': SplittedImageDataset,
    'base': BaseImageDataset,
    'macaque': SplittedImageDataset
}

def train_collate_fn(batch):
    imgs, pids, camids, meta_info = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, meta_info

def val_collate_fn(batch):
    imgs, pids, camids, meta_info = zip(*batch)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, meta_info

def get_transforms(cfg, 
                   is_train=True):
    if is_train:
        transforms = T.Compose([
            T.Resize(cfg.INPUT.IMG_SIZE),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.IMG_SIZE),
            # T.RandomRotation(degrees=90),
            # T.ColorJitter(brightness=0.2, contrast=0.15, saturation=0, hue=0),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        ],
        )
    else:
        transforms = T.Compose([
            T.Resize(cfg.INPUT.IMG_SIZE),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        ])
        
    return transforms


def create_dataloader(dataset, batch_size, num_workers, sampler=None, is_train=False):
    """Helper function to create dataloaders with consistent parameters"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        shuffle=is_train if sampler is None else False,
        collate_fn=train_collate_fn if is_train else val_collate_fn,
        pin_memory=True
    )

def create_train_sampler(cfg, train_data):
    """Create appropriate sampler based on configuration"""
    if not cfg.MODEL.DIST_TRAIN:
        return RandomIdentitySampler(train_data, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
    
    mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
    data_sampler = RandomIdentitySampler_DDP(train_data, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
    return torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)

def make_dataloader(cfg):
    """
    Make a dataloader for a dataset.

    Args:
        cfg (dict): The configuration dictionary.

    Returns:
        tuple: A tuple containing  
                the training dataloader, 
                the normal training dataloader, 
                the test iid dataloader, 
                the test ood dataloader, 
                the validation iid dataloader, 
                the pid container, 
                the number of camera classes, 
                the number of training classes, 
                the number of test iid classes, 
                the number of test classes, 
                the number of validation classes.

    """
    
    train_transforms = get_transforms(cfg, is_train=True)
    val_transforms = get_transforms(cfg, is_train=False)

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](cfg,
                                            root=cfg.DATASETS.ROOT_DIR, 
                                            img_dir=cfg.DATASETS.IMG_DIR)


    train_set = ImageDataset(dataset.train, train_transforms, overfitting=cfg.DATASETS.OVERFITTING)
    train_set_normal = ImageDataset(dataset.train, val_transforms, pre_scaling=cfg.INPUT.PRE_SCALING, overfitting=cfg.DATASETS.OVERFITTING)
    test_iid_set = ImageDataset(dataset.test_iid, val_transforms, pre_scaling=cfg.INPUT.PRE_SCALING, overfitting=cfg.DATASETS.OVERFITTING)
    test_ood_set = ImageDataset(dataset.test, val_transforms, pre_scaling=cfg.INPUT.PRE_SCALING, overfitting=cfg.DATASETS.OVERFITTING)
    val_iid_set = ImageDataset(dataset.val, val_transforms, pre_scaling=cfg.INPUT.PRE_SCALING, overfitting=cfg.DATASETS.OVERFITTING)
    
    if 'triplet' in cfg.DATALOADER.SAMPLER:
        sampler = create_train_sampler(cfg, train_set)
        train_loader = create_dataloader(train_set, 
                                         cfg.SOLVER.IMS_PER_BATCH, 
                                         num_workers, 
                                         sampler, 
                                         is_train=True)
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = create_dataloader(train_set, 
                                         cfg.SOLVER.IMS_PER_BATCH, 
                                         num_workers, 
                                         is_train=True)
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))
        
    loaders = {
        'train': train_loader,
        'train_normal': create_dataloader(train_set_normal, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_WORKERS),
        'test_iid': create_dataloader(test_iid_set, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_WORKERS),
        'test_ood': create_dataloader(test_ood_set, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_WORKERS),
        'val_iid': create_dataloader(val_iid_set, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_WORKERS)
    }
    return (
        loaders['train'],
        loaders['train_normal'],
        loaders['test_iid'],
        loaders['test_ood'],
        loaders['val_iid'],
        dataset.pid_container,
        len(dataset.cid_container),
        dataset.n_train_classes,
        dataset.n_test_iid_classes,
        dataset.n_test_classes,
        dataset.n_val_classes
    )


def make_base_dataloader(cfg, 
                         root_dir,
                         img_dir,
                         is_train=True
                         ):
    
    """
    Make a dataloader for a base dataset.

    Args:
        cfg (dict): The configuration dictionary.
        root_dir (str): The root directory of the dataset.
        img_dir (str): The image directory of the dataset.
        is_train (bool, optional): Whether to create a training dataloader. Defaults to True.

    Returns:
        tuple: A tuple containing the dataloader and the number of classes in the dataset.
    """

    transforms = get_transforms(cfg, is_train=is_train)
    batch_size = cfg.SOLVER.IMS_PER_BATCH if is_train else cfg.TEST.IMS_PER_BATCH
    
    dataset = BaseImageDataset(cfg=cfg,
                               root=root_dir, 
                               img_dir=img_dir)
    
    image_set = ImageDataset(dataset.data, transforms, overfitting=cfg.DATASETS.OVERFITTING)
    
    loader = create_dataloader(image_set, batch_size, cfg.DATALOADER.NUM_WORKERS, is_train=is_train)
    return loader, dataset.n_classes, len(dataset.cid_container)

