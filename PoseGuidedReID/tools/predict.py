from project.utils.logger import setup_logger
from project.datasets import make_dataloader, make_base_dataloader, PostProcessing
from project.models import make_model
from project.solver import make_optimizer
from project.solver.scheduler_factory import create_scheduler
from project.losses import make_loss

import random
import torch
import torch.optim as optim
import numpy as np
import os
import argparse
from project.config import cfg
import sys
import glob
from project.utils.tools import load_model, setup_ddp_training, fuse_all_conv_bn
from project.processor.processor import train_model, do_prediction, get_DDP_model
from tools.evaluation import run_evaluation
from tools.evaluation_katmai import run_evaluation as run_evaluation_katmai
from project.utils.wandb_tools import WandbLogger
import shutil
import wandb
wandb.login()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--notes", default="training...", type=str)
    parser.add_argument("--out_dim", default=512, type=int)
    parser.add_argument("--do_prediction", action='store_true', help='do inference on the dataset (train/test_iid/test_ood)')
    parser.add_argument("--do_evaluation", action='store_true', help='evaluate the model on the dataset (ood/iid vs gallery)')
    parser.add_argument("--do_training", action='store_true', help='train the model')
    parser.add_argument("--data_type", default=None, type=str)
    parser.add_argument("--wb_run_id", default=None, type=str)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    local_rank = args.local_rank

    output_dir = cfg.OUTPUT_DIR

    exp_name = output_dir.split('/')[-1]  ### 'test_on_20xx'
    full_data = output_dir.split('/')[-3].split('_')[1]
    project_name = full_data + '_' + output_dir.split('/')[-2]  ### e.g. '5years_swin_triplet_inference_eval'
    
    wb_run=None
    wb_run_id=args.wb_run_id
    # if local_rank == 0:
    #     if args.do_training:
    #         exp_name = exp_name + '_train'
    #     if args.do_prediction:
    #         exp_name = exp_name + '_inf'
    #     if args.do_evaluation:
    #         exp_name = exp_name + '_eval'

    #     if args.model_name != "":
    #         exp_name = exp_name + '_' + args.model_name

    #     wb = WandbLogger(cfg)
    #     wb_config = wb.build_config(cfg, args.config_file, exp_name, project_name, output_dir)
    #     if wb_run_id is None:
    #         wb_run = wb.init_wandb(project_name, exp_name, wb_config, notes=args.notes)
    #         wb_run_id = wb_run.id
    #     else:
    #         wb_run = wb.resume(run_id=wb_run_id, resume='allow')
        
    if_train = args.do_training
    
    logger = setup_logger("project", output_dir, if_train=if_train)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        local_rank, world_size, device = setup_ddp_training()
    else:
        device_ids = cfg.MODEL.DEVICE_ID[0]
        device = torch.device("cuda:{}".format(device_ids))

    root_dir = cfg.DATASETS.TEST_ROOT_DIR if 'train' not in args.data_type else cfg.DATASETS.ROOT_DIR
    img_dir = cfg.DATASETS.TEST_IMG_DIR if 'train' not in args.data_type else cfg.DATASETS.IMG_DIR

    if cfg.DATASETS.NAMES == 'base':
        dataloader, num_classes, camera_num = make_base_dataloader(cfg, 
                                                                   is_train=args.do_training,
                                                                   root_dir=root_dir,
                                                                   img_dir=img_dir
                                                                   )
    else:
        raise ValueError(f"Please use the make_base_dataloader for prediction")

    model_weights_path = cfg.TEST.WEIGHT
    if model_weights_path is None or model_weights_path == '':
        model_weights_path = os.path.join(cfg.OUTPUT_DIR, 'net_best.pth')
        
    data_type = args.data_type
    model_structure = make_model(cfg, num_classes=num_classes, logger=logger, return_feature=True, device=device)
    model = load_model(model_structure, model_weights_path, logger=logger, remove_fc=False, local_rank=local_rank)
    model.eval()
    model = fuse_all_conv_bn(model).to(device)
    predictions = do_prediction(cfg, 
                                model, 
                                dataloader, 
                                data_type=data_type, 
                                out_dir=cfg.OUTPUT_DIR, 
                                local_rank=local_rank, 
                                device=device)