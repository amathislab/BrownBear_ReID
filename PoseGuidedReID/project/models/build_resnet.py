import torch
import torch.nn as nn
import copy
from .backbones.basics import *

class Backbone(nn.Module):
    def __init__(self, num_classes, cfg, model, linear_num, return_feature=False):
        super(Backbone, self).__init__()
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.base = model
        self.return_feature = return_feature
        self.in_planes = linear_num

        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.base(x)
        global_feat = self.gap(x)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        feat = self.bottleneck(global_feat)

        if self.return_feature:
            cls_score = self.classifier(feat)
            return [cls_score, global_feat]
        else:
            cls_score = self.classifier(feat)
            return cls_score

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))
