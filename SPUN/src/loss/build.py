from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch

from .mutimodal import  mutimodal
from .bd_mixture_loss import BGMixtureLoss
from .bd_loss import BGLloss
from .S3R3_loss import S3R3loss
from .S3R3_mixture_loss import S3R3MixtureLoss
from .Separation_distribution import PVSPE
from .bingham_mixture_loss import BinghamMixtureLoss
logger = logging.getLogger(__name__)

def get_loss(cfg):

    if cfg.loss_name=='sdn':
        loss_function = BGLloss()
        logger.info('sdn loss created')
        print('sdn loss created')
    elif cfg.loss_name=='mutimodal':
        loss_function =  BGMixtureLoss()
        logger.info('mutimodal loss created')
        print('mutimodal loss created')
    elif cfg.loss_name=='Nonprobability':
        #loss_function = BinghamMixtureLoss(4)
        logger.info('Nonprobability loss created')
    elif cfg.loss_name=='PVSPE':
        #loss_function = BinghamMixtureLoss(4)
        loss_function =  PVSPE()
        logger.info('PVSPE loss created')
        print('PVSPE loss created')
    elif cfg.loss_name=='S3R3loss':
        loss_function = S3R3loss()
        logger.info('S3R3loss loss created')
    elif cfg.loss_name=='mutimodal_attitude':
        loss_function = BinghamMixtureLoss()
        logger.info('BinghamMixtureLoss loss created')


    return loss_function