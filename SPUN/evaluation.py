import torch
from utils import AverageMeter, eaad_bingham, eatd_Guassian
import numpy as np
import csv
import time
import sys
import se3lib
def report_progress(epoch_iter, epoch_size, time, is_train=True, **kwargs):
    # Construct message
    blength = 30
    percent = float(epoch_iter / epoch_size)
    arrow   = '█' * int(round(percent * blength))
    spaces  = ' ' * (blength - len(arrow))
    msg = "\rTraining " if is_train else "\rTesting "

    msg += "{iter:04d}/{esize:04d} [{prog}{percent:03d}%] [{time_v:.0f} ({time_a:.0f}) ms] ".format(
        iter=epoch_iter, esize=epoch_size, time_v=time.last_val, time_a=time.avg,
        prog=arrow+spaces, percent=round(percent*100))

    #Add losses to report
    # for key, item in kwargs.items():
    #     if item is not None:
    #         msg += '{}: {:.4f} ({:.4f}) '.format(key, item.last_val, item.avg)

    # Report loss
    sys.stdout.write(msg)
    sys.stdout.flush()

    # To next line if at the end of the epoch
    if epoch_iter == epoch_size:
        sys.stdout.write('\n')
        sys.stdout.flush()

def run_evaluation(model, dataset, loss_function, device, floating_point_type="float"):
    model.eval()
    losses = AverageMeter()
    log_likelihoods = AverageMeter()
    averaged_stats = AverageMeter()
    eaads = AverageMeter()
    eatds = AverageMeter()
    min_eaads = AverageMeter()
    min_eatds = AverageMeter()
    min_maads = AverageMeter()
    min_matds = AverageMeter() 
    rmse_transs = AverageMeter()
    me_transs = AverageMeter()
    rmse_angs =AverageMeter()
    me_angs =AverageMeter()
    test_time_meter = AverageMeter()
    val_load_iter = iter(dataset)
    eval_fraction = 0.05
    errors = []
    pyr_errors = []
    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters: ", total_params)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable parameters: ", trainable_params)
    print(len(dataset))
    for i , data in enumerate(dataset):
    #for i in range(int(len(dataset)*eval_fraction)):
        #data = val_load_iter.next()
        start = time.time()
        if floating_point_type == "double":
            target_varr = data["quater"].double().to(device)
            target_vart = data["pos"].double().to(device)
            input_var = data["image"].double().to(device)
        else:
            target_varr = data["quater"].float().to(device)
            target_vart = data["pos"].float().to(device)
            input_var = data["image"].float().to(device)

        if torch.sum(torch.isnan(target_varr)) > 0:
            continue
            # compute output
        with torch.no_grad():
            output = model(input_var)
        
        
        if loss_function.__class__.__name__ == "Nonprobability":
            print("soon avaliable: maads, min_maads")
        else:
            stats = loss_function.statistics(target_varr,  target_vart, output)
            # solve maad/matd
            test_time = (time.time()-start)*1000 /  data["image"].shape[0]
            test_time_meter.update(test_time, data["image"].size(0))
            rmse_angs.update(stats["rmse_ang"], data["image"].size(0))
            rmse_transs.update(stats["rmse_trans"], data["image"].size(0))
            me_angs.update(stats["me_ang"], data["image"].size(0))
            error = np.array([stats["me_ang"], stats["me_trans"], float(target_vart[0,2])])
            errors.append(error)
            q_est = np.array([stats["est_q"][0,1], stats["est_q"][0,2], stats["est_q"][0,3],stats["est_q"][0,0]])
            q_gt = np.array([float(target_varr[0,1]),float(target_varr[0,2]), float(target_varr[0,3]),float(target_varr[0,0])])
            error_q = se3lib.quat_mult(q_est, se3lib.quat_inv(q_gt))
            error_q = np.array([error_q[0,0],error_q[0,1],error_q[0,2], error_q[0,3]])
            v, theta = se3lib.quat2angleaxis(error_q)
            p,y,r = se3lib.quat2euler(error_q)
            pyr_error = np.array([p,y,r])
            pyr_errors.append(np.abs(pyr_error))
            me_transs.update(stats["me_trans"], data["image"].size(0))
            if loss_function.__class__.__name__ == "BinghamMixtureLoss":
                min_maads.update(stats["mmaad"], data["image"].size(0))
            else:
                min_maads.update(stats["rmse_ang"], data["image"].size(0))
            # solve eaad 
            eaad, min_eaad = bingham_z_to_eaad(
                stats, loss_function
            ) 
            eatd, min_eatd = dual_cov_to_eatd(stats, loss_function)
            eaads.update(eaad, data["image"].size(0))
            min_eaads.update(min_eaad, data["image"].size(0))
            eatds.update(eatd, data["image"].size(0))
            min_eatds.update(min_eatd, data["image"].size(0))
        if loss_function.__class__.__name__ == "Nonprobability":
            # norm over the last dimension (ie. orientations)
            print('soon available')
        if loss_function.__class__.__name__ == "BGMixtureLoss":
            loss, log_likelihood = loss_function(target_varr,  target_vart, output)
        elif loss_function.__class__.__name__ == "BGLloss":
            loss, log_likelihood = loss_function(target_varr,  target_vart, output)
        elif loss_function.__class__.__name__ == "PVSPE":
             loss, log_likelihood = loss_function(target_varr,  target_vart, output)
        elif loss_function.__class__.__name__ == "S3R3loss":
             loss, log_likelihood = loss_function(target_varr,  target_vart, output)

        if floating_point_type == "double":
            loss = loss.double() / data["image"].shape[0]
        else:
            loss = loss.float() / data["image"].shape[0]
        # measure accuracy and record loss
        losses.update(loss.item(), data["image"].size(0))
        log_likelihoods.update(log_likelihood.item(), data["image"].size(0))
        report_progress(epoch_iter=i+1, epoch_size=len(dataset),
                        time=test_time_meter, is_train=False, eT=me_transs, eR=me_angs, rmse_ang = rmse_angs, eaads = eaads, rmse_transs = rmse_transs, eatds =eatds, loss =losses
                        )
        # print("iter_count: [{0}/{1}]\t Loss {loss.last_val:.4f} "
        #           "({loss.avg:.4f})\t".format(
        #             i, len(dataset), loss=losses))
        # print("iter_count: [{0}/{1}]\t Rmse_trans {Rmse_transs.last_val:.4f} "
        #           "({Rmse_transs.avg:.4f})\t".format(
        #             i, len(dataset), Rmse_transs=rmse_transs))
        # print("iter_count: [{0}/{1}]\t Me_trans {Me_transs.last_val:.4f} "
        #           "({Me_transs.avg:.4f})\t".format(
        #             i, len(dataset), Me_transs=me_transs))
        # print("iter_count: [{0}/{1}]\t EATD {EATD.last_val:.4f} "
        #           "({EATD.avg:.4f})\t".format(
        #             i, len(dataset), EATD=eatds))
            
            
    filename = "results_mutimodal_speedplus.csv"

    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(errors)
    filename = "PYR.csv"

    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(pyr_errors)

    print("CSV文件保存成功。")

    if "Bingham" or "Separation_distribution" in loss_function.__class__.__name__:
        print("Loss: {}, Log Likelihood: {}, rmse_ang: {}, rmse_trans: {}, EAAD: {}, EATD: {}, me_ang: {}, me_trans: {}".format(
                losses.avg, log_likelihoods.avg, rmse_angs.avg, rmse_transs.avg, eaads.avg, eatds.avg, me_angs.avg, me_transs.avg))
    else:
        print("Loss: {}, Log Likelhood: {}, MAAD: {}".format(losses.avg, log_likelihoods.avg))

    
def bingham_z_to_eaad(stats, loss_function):
    eaads = []

   
    z_0, z_1, z_2 = stats["z_0"], stats["z_1"], stats["z_2"]
    #bingham_z = np.array([z_0, z_1, z_2, 0])
    bingham_z = np.array([0, z_2, z_1, z_0]) ##z一定要降序排列，为了确保M是单位矩阵。也就是说M的第一列是（1，0，0，0），这是由于我们的四元数是scalar first
    eaad = eaad_bingham(bingham_z)
    eaads.append(eaad)
    return sum(eaads)/len(eaads), min(eaads)


def dual_cov_to_eatd(stats, loss_function):
    eatds = []
    
  
    cov = stats["cov"]
    eatd = eatd_Guassian(cov)
    eatds.append(eatd)

    return sum(eatds)/len(eatds), min(eatds)
    
    
