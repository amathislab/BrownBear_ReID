# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
import torch.nn as nn
from pytorch_metric_learning import losses

def make_loss():
    criterion = nn.CrossEntropyLoss()
    criterion_triplet = losses.TripletMarginLoss(margin=0.3)
    return criterion, criterion_triplet


def make_loss_old(cfg, num_classes, feat_dim=2048):
    sampler = cfg.DATALOADER.SAMPLER

    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("Using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)
            print("Using triplet loss with margin: {}".format(cfg.SOLVER.MARGIN))
    else:
        print('Expected METRIC_LOSS_TYPE to be triplet, but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("Label smooth on, numclasses:", num_classes)

    def loss_func(score, feat, target):
        if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                ID_LOSS = xent(score[0], target) if not isinstance(score, list) else 0.5 * xent(score[0], target)
                TRI_LOSS = triplet(feat[0], target)[0] if not isinstance(feat, list) else 0.5 * triplet(feat[0], target)[0]
            else:
                ID_LOSS = F.cross_entropy(score, target) if not isinstance(score, list) else 0.5 * F.cross_entropy(score[0], target)
                TRI_LOSS = triplet(feat, target)[0] if not isinstance(feat, list) else 0.5 * triplet(feat[0], target)[0]

            if isinstance(score, list):
                ID_LOSS += sum([xent(scor, target) for scor in score[1:]]) / len(score[1:])
            if isinstance(feat, list):
                TRI_LOSS += sum([triplet(feats, target)[0] for feats in feat[1:]]) / len(feat[1:])

            return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
        else:
            print('Expected METRIC_LOSS_TYPE to be triplet, but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if sampler == 'softmax':
        return F.cross_entropy

    elif sampler == 'softmax_triplet':
        return loss_func

    else:
        print('Expected sampler to be softmax or softmax_triplet, but got {}'.format(cfg.DATALOADER.SAMPLER))


