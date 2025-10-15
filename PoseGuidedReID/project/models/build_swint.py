from .backbones.swin_transformer import SwinTransformer
from .backbones.swin_transformer_v2 import SwinTransformerV2
import torch.nn as nn
from .backbones.basics import *
import timm
from .backbones.pose_net import SimpleHRNet
import cv2
import copy

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)

class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, linear=512, return_feature = False):
        super(ClassBlock, self).__init__()
        self.return_feature = return_feature
        add_block = []
        if linear>0:
            add_block += [nn.Linear(input_dim, linear)]
        else:
            linear = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(linear)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(linear, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.return_feature:
            feature = x
            x = self.classifier(x)
            return [x, feature]
        else:
            x = self.classifier(x)
            return x
        

class ft_net_swin(nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2, return_feature=True, linear_num=512, cfg=None, model_ft=None):
        super(ft_net_swin, self).__init__()

        # if model_ft is None:
        #     print('loading timm model -- swin_base_patch4_window7_224, Make sure timm==0.6.13')
        #     model_ft = timm.create_model('swin_base_patch4_window7_224', pretrained=True, drop_path_rate = 0.2, cfg=cfg)
        # else:
        #     print('loading our swin model')

        # avg pooling to global pooling
        #model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.head = nn.Sequential() # save memory
        self.return_feature = return_feature
        
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = ClassBlock(1024, class_num, droprate, linear=linear_num, return_feature=return_feature)

    def forward(self, x):

        global_feat, pose_conf = self.model.forward_features(x) ## b, 7, 7, 1024
        if len(global_feat.shape) == 4:
            # the shape is different in higher version timm
            # b, 7, 7, 1024 -> b, 7*7, 1024
            global_feat = global_feat.view(global_feat.shape[0], global_feat.shape[1]*global_feat.shape[2], global_feat.shape[3])
        # swin is update in latest timm>0.6.0, so I add the following two lines.
        global_feat = self.avgpool( global_feat.permute((0,2,1)) )
        global_feat = global_feat.view(global_feat.size(0), global_feat.size(1))
        x, global_feat = self.classifier(global_feat)
        return [x, global_feat, pose_conf]

class ft_net_swinv2(nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2, return_feature=True, linear_num=512, cfg=None, model_ft=None):
        super(ft_net_swinv2, self).__init__()

        # avg pooling to global pooling
        model_ft.head = nn.Sequential() # save memory
        self.return_feature = return_feature
        
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = ClassBlock(1536, class_num, droprate, linear=linear_num, return_feature=return_feature)

    def forward(self, x):

        global_feat, pose_conf = self.model.forward_features(x) ## b, 64, 1536
        if len(global_feat.shape) == 4:
            # the shape is different in higher version timm
            # b, 8,8, 1536 -> b, 8*8, 1536
            global_feat = global_feat.view(global_feat.shape[0], global_feat.shape[1]*global_feat.shape[2], global_feat.shape[3])
        global_feat = self.avgpool( global_feat.permute((0,2,1)) )
        global_feat = global_feat.view(global_feat.size(0), global_feat.size(1))
        x, global_feat = self.classifier(global_feat)
        return [x, global_feat, pose_conf]
    
    
def get_swin_model(num_classes, cfg, logger, load_weights=True, device='cuda'):

    model_type = cfg.MODEL.TYPE
    img_size = cfg.INPUT.IMG_SIZE

    # accelerate layernorm
    if cfg.FUSED_LAYERNORM:
        try:
            import apex as amp
            layernorm = amp.normalization.FusedLayerNorm
        except:
            layernorm = None
            print("To use FusedLayerNorm, please install apex.")
    else:
        import torch.nn as nn
        layernorm = nn.LayerNorm

    if model_type == 'swin':

        if cfg.MODEL.AGG_POSE_FEATURE:
            pose_model = get_hrnet_model(cfg, max_batch_size=64, device=device)
        else:
            pose_model = None

        model = SwinTransformer(img_size=img_size,
                                patch_size=cfg.MODEL.SWIN.PATCH_SIZE,
                                in_chans=cfg.MODEL.SWIN.IN_CHANS,
                                num_classes=num_classes,
                                embed_dim=cfg.MODEL.SWIN.EMBED_DIM,
                                depths=cfg.MODEL.SWIN.DEPTHS,
                                num_heads=cfg.MODEL.SWIN.NUM_HEADS,
                                window_size=cfg.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=cfg.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=cfg.MODEL.SWIN.QKV_BIAS,
                                qk_scale=cfg.MODEL.SWIN.QK_SCALE,
                                drop_rate=cfg.MODEL.DROP_RATE,
                                drop_path_rate=cfg.MODEL.DROP_PATH_RATE,
                                ape=cfg.MODEL.SWIN.APE,
                                norm_layer=layernorm,
                                patch_norm=cfg.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=cfg.MODEL.USE_CHECKPOINT,
                                fused_window_process=cfg.FUSED_WINDOW_PROCESS,
                                cfg=cfg,
                                pose_model=pose_model)
        if load_weights:
            load_pretrained(cfg, model, logger, device=device)
            # model.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH))

    elif model_type == 'swinv2':
        if cfg.MODEL.AGG_POSE_FEATURE:
            pose_model = get_hrnet_model(cfg, max_batch_size=64, device=device)
        else:
            pose_model = None
        model = SwinTransformerV2(img_size=img_size,
                                  patch_size=cfg.MODEL.SWINV2.PATCH_SIZE,
                                  in_chans=cfg.MODEL.SWINV2.IN_CHANS,
                                  num_classes=num_classes,
                                  embed_dim=cfg.MODEL.SWINV2.EMBED_DIM,
                                  depths=cfg.MODEL.SWINV2.DEPTHS,
                                  num_heads=cfg.MODEL.SWINV2.NUM_HEADS,
                                  window_size=cfg.MODEL.SWINV2.WINDOW_SIZE,
                                  mlp_ratio=cfg.MODEL.SWINV2.MLP_RATIO,
                                  qkv_bias=cfg.MODEL.SWINV2.QKV_BIAS,
                                  drop_rate=cfg.MODEL.DROP_RATE,
                                  drop_path_rate=cfg.MODEL.DROP_PATH_RATE,
                                  ape=cfg.MODEL.SWINV2.APE,
                                  patch_norm=cfg.MODEL.SWINV2.PATCH_NORM,
                                  use_checkpoint=cfg.MODEL.SWINV2.USE_CHECKPOINT,
                                  pretrained_window_sizes=cfg.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES,
                                cfg=cfg,
                                pose_model=pose_model)
        model.to(device)
        print('model.device', 'device', device)
        if load_weights:
            load_pretrained(cfg, model, logger, device=device)
            # model.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH))
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")
    
    return model

def get_hrnet_model(cfg, max_batch_size=96, device='cuda'):

    n_joints = cfg.MODEL.NUM_JOINTS
    # pose config
    n_channel = int(cfg.MODEL.POSE_HRNET[-2:])
    device_ids = cfg.MODEL.DEVICE_ID[0]
    device = torch.device("cuda:{}".format(device_ids))
    # device = torch.device("cuda")
    pose_model = SimpleHRNet(n_channel,
                            n_joints, 
                            cfg.MODEL.POSE_WEIGHT,
                            model_name='HRNet',
                            resolution = (256, 256),  ### fixed -- because the pose model is trained with this resolution
                            max_batch_size=max_batch_size,
                            device=device,
                            )

    return pose_model



def build_swin_reid(num_classes, logger, linear_num=512, cfg=None, load_weights=True, return_feature=True, device='cuda'):

    swin_model = get_swin_model(num_classes, cfg, logger, load_weights=load_weights, device=device)
    model = ft_net_swin(class_num=num_classes, return_feature=return_feature, linear_num=linear_num, cfg=cfg, model_ft=swin_model)

    return model

def build_swinv2_reid(num_classes, logger, linear_num=512, cfg=None, load_weights=True, return_feature=True, device='cuda'):

    swin_model = get_swin_model(num_classes, cfg, logger, load_weights=load_weights, device=device)
    model = ft_net_swinv2(class_num=num_classes, return_feature=return_feature, linear_num=linear_num, cfg=cfg, model_ft=swin_model)

    return model

def load_pretrained(config, model, logger, device='cuda'):
    logger.info(f"==============> Loading weight {config.MODEL.PRETRAIN_PATH} for fine-tuning......")
    checkpoint = torch.load(config.MODEL.PRETRAIN_PATH, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model']

    # check classifier, if not match, then re-init classifier to zero
    head_bias_pretrained = state_dict['head.bias']
    Nc1 = head_bias_pretrained.shape[0]
    Nc2 = model.head.bias.shape[0]
    if (Nc1 != Nc2):
        if Nc1 == 21841 and Nc2 == 1000:
            logger.info("loading ImageNet-22K weight to ImageNet-1K ......")
            map22kto1k_path = f'data/map22kto1k.txt'
            with open(map22kto1k_path) as f:
                map22kto1k = f.readlines()
            map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
            state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
            state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
        else:
            torch.nn.init.constant_(model.head.bias, 0.)
            torch.nn.init.constant_(model.head.weight, 0.)
            del state_dict['head.weight']
            del state_dict['head.bias']
            logger.warning(f"Error in loading classifier head, re-init classifier head to 0")

    msg = model.load_state_dict(state_dict, strict=False)
    logger.warning(msg)
    
    model = model.to(device)

    logger.info(f"=> loaded successfully '{config.MODEL.PRETRAIN_PATH}'")

    del checkpoint
    torch.cuda.empty_cache()

