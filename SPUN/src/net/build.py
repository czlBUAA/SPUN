from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch

from .krn2019 import KeypointRegressionNet

from .urso import BGnet
from .EfficientNetV2 import EfficientNetV2S
from src.utils.utils import num_total_parameters, num_trainable_parameters
from torchvision.models import resnet50, efficientnet_v2_s,  efficientnet_v2_m
logger = logging.getLogger(__name__)
def load_pretrained(model):
    ckpt_path  = '/media/computer/study/CZL/Bingham-Guass/models_ckp/dragon/sdn/resnet50/0.001/0.001/True/checkpoint_resnet50_149.tar'
    ckpt = torch.load(ckpt_path, map_location='cpu')

    # 根据 .tar 文件结构提取 state_dict
    # 一般是 {'state_dict': ..., 'epoch': ..., ...}
    state_dict = ckpt.get('state_dict', ckpt)

    # 有的保存形式前缀会是 "module." 或 "model."
    # 我们统一移除前缀，确保 key 是 'fc.weight' 之类的
    processed_state_dict = {}
    for k, v in state_dict.items():
        new_key = k
        if k.startswith("module."):
            new_key = k[len("module."):]
        elif k.startswith("model."):
            new_key = k[len("model."):]
        processed_state_dict[new_key] = v
    
        # ---------- 2. 提取 fc 层的权重 ----------
    print("fc 相关的 key：")
    for key in processed_state_dict:
        if 'fc' in key:
            print(key)
    fc_weight = processed_state_dict['resnet.fc.weight']  # (2048, 36)
    fc_bias = processed_state_dict.get('resnet.fc.bias', None)  # (36,)

    
    fc_weight_expanded = torch.cat([fc_weight[-1:].clone(),fc_weight], dim=0) #(37,2048)
    if fc_bias is not None:
        fc_bias_expanded = torch.cat([fc_bias[-1:].clone(), fc_bias], dim=0)

    # ---------- 4. 将扩展后的权重赋到新模型 fc 的 3 个段 ----------
    with torch.no_grad():
        model.resnet.fc.weight[0:37, :] = fc_weight_expanded
        model.resnet.fc.weight[37:74, :] = fc_weight_expanded
        model.resnet.fc.weight[74:111, :] = fc_weight_expanded

        if fc_bias is not None:
            model.resnet.fc.bias[0:37] = fc_bias_expanded
            model.resnet.fc.bias[37:74] = fc_bias_expanded
            model.resnet.fc.bias[74:111] = fc_bias_expanded

def get_model(cfg):

    if cfg.model_name == 'krn':
        if cfg.loss_name=='sdn':
            out_dim = 36
        elif cfg.loss_name=='S3R3loss':
            out_dim = 28
        model = KeypointRegressionNet(out_dim)
        logger.info('KRN created')
    elif cfg.model_name == 'resnet50':
        if cfg.loss_name=='sdn':
            out_dim = 36
        elif cfg.loss_name=='S3R3loss':
            out_dim = 28
        elif cfg.loss_name=='mutimodal':
            out_dim = 36*cfg.mix_count
        elif cfg.loss_name=='PVSPE':
            out_dim = 19
        elif cfg.loss_name=='mutimodal_attitude':
            out_dim = 20*cfg.mix_count
        model = BGnet(resnet50(pretrained=True), out_dim)
        # if cfg.loss_name=='mutimodal':
        #    load_pretrained(model)
        logger.info('URSO created')
    elif cfg.model_name == 'efficientNetV2-S':
        if cfg.loss_name=='sdn':
            out_dim = 36
        elif cfg.loss_name=='S3R3loss':
            out_dim = 28
        elif cfg.loss_name=='mutimodal':
            out_dim = 36*cfg.mix_count
        elif cfg.loss_name=='PVSPE':
            out_dim = 19
        model = EfficientNetV2S(efficientnet_v2_s(pretrained=True), out_dim)
    elif cfg.model_name == 'efficientNetV2-M':
        if cfg.loss_name=='sdn':
            out_dim = 36
        elif cfg.loss_name=='S3R3loss':
            out_dim = 28
        elif cfg.loss_name=='mutimodal':
            out_dim = 36*cfg.mix_count
        elif cfg.loss_name=='PVSPE':
            out_dim = 19
        model = EfficientNetV2S(efficientnet_v2_m(pretrained=True), out_dim)
    logger.info('   - Number of total parameters:     {:,}'.format(num_total_parameters(model)))
    logger.info('   - Number of trainable parameters: {:,}'.format(num_trainable_parameters(model)))

    return model
def get_optimizer(cfg, model):
    param = filter(lambda p:p.requires_grad, model.parameters())

    if cfg.optimizer == 'sgd':
        optimizer = torch.optim.SGD(param,
                        lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(param,
                        lr=cfg.lr, alpha=cfg.momentum, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'adam':
        optimizer = torch.optim.Adam(param,
                        lr=cfg.lr, betas=(cfg.momentum,0.999), weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param,
                        lr=cfg.lr, betas=(cfg.momentum,0.999), weight_decay=cfg.weight_decay)

    logger.info('Optimizer created: {}'.format(cfg.optimizer))

    return optimizer