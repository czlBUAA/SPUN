import time
import torch
from config import cfg
from utils import AverageMeter  
import random
class Trainer(object):
    def __init__(self, device, floating_point_type="float"):
        self._device = device
        self._floating_point_type = floating_point_type

    @staticmethod
    def adjust_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 2 
    def train_epoch(self, train_loader, model, loss_function,optimizer, epoch, writer_train ,writer_val, val_loader, styleAugmentor=None):
        losses = AverageMeter()
        model.train()
        if self._floating_point_type == "double":
            model = model.double()
        if hasattr(model, 'is_sequential'):
            is_sequential = True
        else:
            is_sequential = False
        timings_start = time.time()
        
        # Current learning rate
        for pg in optimizer.param_groups:
            lr = pg['lr']
        
        for i , data in enumerate(train_loader):
            if i % 20 == 0:
                if i > 0 and i % 200 == 0:
                    print("Elapsed time: {}".format(
                        str(time.time()-timings_start)))
                    timings_start = time.time()

                if is_sequential:
                    model.reset_state(batch=data['image'].shape[0],
                                      device=self._device)
                self.validate(self._device, val_loader, model,
                              loss_function, writer_val, i, epoch,
                              len(train_loader), 0.1)

                # switch to train mode
                model.train()
            if self._floating_point_type == "double":
                target_varr = data["quater"].double().to(self._device)
                target_vart = data["pos"].double().to(self._device)
                input_var = data["image"].double().to(self._device)
            else:
                target_varr = data["quater"].float().to(self._device)
                target_vart = data["pos"].float().to(self._device)
                input_var = data["image"].float().to(self._device)
            if torch.sum(torch.isnan(target_varr)) > 0:
                continue
            if torch.sum(torch.isnan(target_vart)) > 0:
                continue
            if torch.sum(torch.isnan(input_var)) > 0:
                print("Input contains NaN")
                continue
            # compute output
            if is_sequential:
                model.reset_state(batch=data['image'].shape[0],
                                  device=self._device)
                model.to(self._device)
            # Randomize texture?
            if styleAugmentor is not None and random.random() < cfg.texture_ratio:
                input_var = styleAugmentor(input_var)
            # imshow(images[0].cpu())
            output = model(input_var)
            if cfg.loss_name=='mutimodal':
                
                loss, log_likelihood = loss_function(target_varr, target_vart, output, epoch)
            elif cfg.loss_name=='mutimodal_attitude':
                loss, log_likelihood = loss_function(target_varr, output, epoch)
            else:
                loss, log_likelihood = loss_function(target_varr, target_vart, output)
            optimizer.zero_grad()
            if cfg.data_style == 'dragon' or cfg.data_style == 'speed':
                l2_norm = sum(param.pow(2.0).sum() for param in model.parameters())
                loss = loss + cfg.L2reg * l2_norm
            loss.backward()
            optimizer.step()
            if self._floating_point_type == "double":
                loss = loss.double() / data["image"].shape[0]
            else:
                loss = loss.float() / data["image"].shape[0]
            losses.update(loss.item(), data["image"].size(0))
            writer_train.add_scalar('data/loss', loss,
                                    i + len(train_loader) * epoch)
            writer_train.add_scalar('data/log_likelihood', log_likelihood,
                                    i + len(train_loader) * epoch)
            cur_iter = epoch * len(train_loader) + i
            # if cfg.loss_name=='mutimodal':
            #     stats = loss_function.statistics(target_varr, target_vart, output, epoch)
            # else:
            #     stats = loss_function.statistics(target_varr, target_vart, output)
            #Trainer.report_stats(writer_train, stats, cur_iter)
            # writer_train.add_scalar('data/rmse_trans', stats["rmse_trans"],
            #                         i + len(train_loader) * epoch)
            # writer_train.add_scalar('data/me_trans', stats["me_trans"],
            #                         i + len(train_loader) * epoch)
            # writer_train.add_scalar('data/rmse_ang', stats["rmse_ang"],
            #                         i + len(train_loader) * epoch)
            # writer_train.add_scalar('data/me_ang', stats["me_ang"],
            #                         i + len(train_loader) * epoch)

            print("Epoch: [{0}][{1}/{2}]\t Loss {loss.last_val:.4f} "
                  "({loss.avg:.4f})\t".format(
                    epoch, i, len(train_loader), loss=losses))
            
    def validate(self, device, val_loader, model,  loss_function, writer,
                 index=None, cur_epoch=None, epoch_length=None, eval_fraction=1):
        model.eval()

        losses = AverageMeter()
        log_likelihoods = AverageMeter()
        maads = AverageMeter()
        averaged_stats = AverageMeter()
        val_load_iter = iter(val_loader)
        
        for i in range(int(len(val_loader) * eval_fraction)):
            data = val_load_iter.next()

            if self._floating_point_type == "double":
                target_varr = data["quater"].double().to(self._device)
                target_vart = data["pos"].double().to(self._device)
                input_var = data["image"].double().to(self._device)
            else:
                target_varr = data["quater"].float().to(self._device)
                target_vart = data["pos"].float().to(self._device)
                input_var = data["image"].float().to(self._device)
            if torch.sum(torch.isnan(target_varr)) > 0:
                continue
            if torch.sum(torch.isnan(target_vart)) > 0:
                continue
            output = model(input_var)
            if cfg.loss_name=='mutimodal':
                loss, log_likelihood = loss_function(target_varr, target_vart, output, cur_epoch)
            elif cfg.loss_name=='mutimodal_attitude':
                loss, log_likelihood = loss_function(target_varr, output, cur_epoch)
            else:
                  loss, log_likelihood = loss_function(target_varr, target_vart, output)
            if self._floating_point_type == "double":
                loss = loss.double() / data["image"].shape[0]
            else:
                loss = loss.float() / data["image"].shape[0]
             # measure accuracy and record loss
            losses.update(loss.item(), data["image"].size(0))
            log_likelihoods.update(log_likelihood.item(), data["image"].size(0))
            
             # TODO: Unify reporting to the style below.
            # if cfg.loss_name=='mutimodal':
            #     stats = loss_function.statistics(target_varr, target_vart, output, cur_epoch)
            # else:
            #     stats = loss_function.statistics(target_varr, target_vart, output)
            # averaged_stats.update(stats, data["image"].size(0))
            
        if index is not None:
            cur_iter = cur_epoch * epoch_length + index
            writer.add_scalar('data/loss', losses.avg, cur_iter)
            writer.add_scalar('data/log_likelihood', log_likelihoods.avg,
                              cur_iter)
            # writer.add_scalar('data/rmse_trans', losses.avg, cur_iter)
            # writer.add_scalar('data/me_trans', log_likelihoods.avg,
            #                   cur_iter)
            #Trainer.report_stats(writer, averaged_stats.avg, cur_iter)

            print('Test:[{0}][{1}/{2}]\tLoss {loss.last_val:.4f} '
                  '({loss.avg:.4f})\t'.format(
                    cur_epoch, index, epoch_length, loss=losses))
    @staticmethod
    def report_stats(writer, stats, cur_iter):
        for key in stats:
            writer.add_scalar(
                'data/' + key, stats[key], cur_iter)
