from .backbones.transreid import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID
from .build_vit import build_transformer
from .build_transreid import build_transreid
from .build_swint import build_swin_reid, build_swinv2_reid
from .build_resnet import Backbone
from .backbones.resnet import ResNet, Bottleneck
from torchvision import models

__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID,
}

__model_factory = [
    'vit', 
    'swin', 
    'swinv2', 
    'resnet',
]


def build_base_model(num_classes, cfg, logger, load_weights=True, camera_num=0, view_num=0):

    model_name = cfg.MODEL.NAME
    model_path = cfg.MODEL.PRETRAIN_PATH
    pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE

    # check if model_name is in our list
    assert cfg.MODEL.TYPE in __model_factory, \
        'expected model name to be one of {}, but got {}'.format(__model_factory, cfg.MODEL.TYPE)

    if cfg.MODEL.TYPE == 'vit':
        print(f'===========building Vision Transformer or TransReID: {model_name} ===========')

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        model = __factory_T_type[model_name](img_size=cfg.INPUT.IMG_SIZE, 
                                            sie_xishu=cfg.MODEL.SIE_COE,
                                            camera=camera_num, 
                                            view=view_num, 
                                            stride_size=cfg.MODEL.STRIDE_SIZE, 
                                            drop_path_rate=cfg.MODEL.DROP_PATH,
                                            drop_rate= cfg.MODEL.DROP_OUT,
                                            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
                                            local_feature=cfg.MODEL.JPM,)
        in_dim = 768
        
        if model_name == 'deit_small_patch16_224_TransReID':
            in_dim = 384
        if pretrain_choice == 'imagenet' and load_weights:
            model.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

    elif cfg.MODEL.TYPE == 'resnet':
        
        in_dim = 2048
        model = models.resnet50(pretrained=True)
        # model = ResNet(last_stride=2,
        #                 block=Bottleneck,
        #                 layers=[3, 4, 6, 3])

        # if pretrain_choice == 'imagenet' and load_weights:
        #     model.load_param(model_path)
        #     print('Loading pretrained ImageNet model......from {}'.format(model_path))

        print(f'===========building ResNet-50 ===========')

    return model, in_dim


def make_model(cfg, 
               num_classes, 
               logger, 
               load_weights=True, 
               return_feature=True, 
               device='cuda', 
               camera_num=0, 
               view_num=0):

    model_name = cfg.MODEL.NAME
    
    if cfg.MODEL.TYPE == 'swin':
        linear_num = cfg.MODEL.SWIN.EMBED_DIM
        if cfg.DATASETS.NAMES == 'macaque':
            linear_num = 256
    elif cfg.MODEL.TYPE == 'vit':
        linear_num = cfg.MODEL.VIT_EMBED_DIM
    elif cfg.MODEL.TYPE == 'resnet':
        linear_num = cfg.MODEL.RESNET_EMBED_DIM
    elif cfg.MODEL.TYPE == 'swinv2':
        linear_num = cfg.MODEL.SWINV2.EMBED_DIM
        

    if 'swin' in cfg.MODEL.TYPE :

        if cfg.MODEL.AGG_POSE_FEATURE:
            print(f'===========building *Pose* Swin Transformer: {model_name} ===========')
        else:
            print(f'===========building Swin Transformer: {model_name} ===========')
        
        if 'swinv2' in cfg.MODEL.TYPE :
            model = build_swinv2_reid(num_classes, logger, linear_num=linear_num, cfg=cfg, load_weights=load_weights, return_feature=return_feature, device=device)
        else:
            model = build_swin_reid(num_classes, logger, linear_num=linear_num, cfg=cfg, load_weights=load_weights, return_feature=return_feature, device=device)

    else:
        base, in_dim = build_base_model(num_classes, cfg, logger, load_weights=True, camera_num=camera_num, view_num=view_num)

        if cfg.MODEL.JPM and cfg.MODEL.TYPE == 'vit':
            print('===========building TransReID with JPM ===========')
            model = build_transreid(num_classes, cfg, base, rearrange=cfg.MODEL.RE_ARRANGE, in_dim=in_dim)
            
        elif cfg.MODEL.TYPE == 'resnet':
            model = Backbone(num_classes, cfg, base, return_feature=return_feature, linear_num=linear_num)

        

    return model
