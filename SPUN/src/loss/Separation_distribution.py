import numpy as np
import torch
import symmartix
import QCQP as qcqp
from QCQP import QuadQuatFastSover
import math
import utils
import dill
import os
import caculatation_nc
from config import cfg

class PVSPE(object):
    def __init__(self):
       print("initial")
    def __call__(self, target_qr,  target_qt, output):
        if target_qr.is_cuda:
            device = target_qr.get_device()
        B, mu, cov = self._output_to_B_G(output)
        #print('qrqt', target_qr.dim(), target_qr.shape[1])
        log_likelihood = torch.sum(self._log_BG_loss(target_qr, target_qt, B, mu, cov))
        loss = -log_likelihood
        return loss, log_likelihood / target_qr.shape[0]
    def statistics(self, target_qr,  target_qt, output):
        """ Reports some additional loss statistics.

        Arguments:
            target (torch.Tensor): Ground-truth shaped as loss input.
            output (torch.Tensor): Network output.
            epoch (int): Current epoch. Currently unused.

        Returns:
            stats (dict): Bingham parameters and angular deviation.
        """


        if target_qr.is_cuda:
            device = target_qr.get_device()
        B, mu, cov = self._output_to_B_G(output)
        cov = -0.5*torch.linalg.inv(cov)
        cov = torch.mean(cov, dim=0)
        
        val, M = torch.linalg.eigh(B) #升序排列的
        bd_z = torch.mean(val-val[:, 3].unsqueeze(1), 0)  #broadcast mechanism #这里还是升序的
        rmse_ang, me_ang= rmse_me_ang(target_qr, M)
        rmse_trans, me_trans = rmse_me_trans(target_qr, target_qt, mu)
        stats = {
            "z_0": float(bd_z[0]),
            "z_1": float(bd_z[1]),
            "z_2":  float(bd_z[2]),
            "cov": cov.detach().cpu().numpy(),
            "rmse_trans": float(rmse_trans),
            "me_trans": float(me_trans),
            "rmse_ang": float(rmse_ang),
            "me_ang": float(me_ang),
            "est_q": M[:, :, 3].detach().cpu().numpy(),
            "gt_q":  target_qr.detach().cpu().numpy(),
        }
        return stats
    @staticmethod
    def _log_BG_loss(target_qr, target_qt, B, mu, cov):
        #print('qrqt1', target_qr.dim(), target_qr.shape[1])
        assert target_qr.dim() == 2 and target_qr.shape[1] == 4, \
            "Wrong dimensionality of target tensor."
        assert target_qt.dim() == 2 and target_qt.shape[1] == 3, \
            "Wrong dimensionality of target tensor."
        assert B.dim() == 3 and B.shape[1:3] == (4, 4), \
            "Wrong dimensionality of location parameter matrix M."

        if target_qr.is_cuda:
            device = target_qr.get_device()
        else:
            device = "cpu"
        
        Z = torch.zeros((B.shape[0], 4))
        B_ = torch.zeros((B.shape[0], 4, 4))
        for i in range(B.shape[0]):
            vals, Mi = torch.linalg.eigh(B[i])
            sorted_vals, _ = torch.sort(vals, descending=True)
            z_padded = vals - sorted_vals[0]
            z_as_matrices = torch.diag_embed(z_padded)
            Z[i, :] = sorted_vals-sorted_vals[0]
            B_[i,:,:] = torch.mm(torch.mm(Mi, z_as_matrices), Mi.T)
        Z = Z.to(device)
        B_ = B_.to(device)
        norm_const = BG_RBF.apply(Z)
        expanded_qt = utils.expand_qt(target_qt)
        N = expanded_qt.shape[0]
        likelihoods = (torch.bmm(torch.bmm(
                target_qr.unsqueeze(1),
                B_),
                target_qr.unsqueeze(2))).squeeze()-0.5* (torch.bmm(torch.bmm(
                (target_qt-mu).unsqueeze(1),
                torch.linalg.inv(cov)),
                (target_qt-mu).unsqueeze(2))).squeeze() - torch.log(norm_const)- torch.log(2 * math.pi* torch.sqrt(torch.linalg.det(cov)))
        return likelihoods
    def _output_to_B_G(self, output):
        #print('out', output)
        #\output = output/output.norm(dim=1).view(-1,1)
        #output = -torch.exp(output)
        if  output.is_cuda:
            device =  output.get_device()
        else:
            device = "cpu"
        b = output[:,0:10]
        mu = output[:,10:13]
        cov = output[:,13:19]
        cov = symmartix.convert_Avec_to_Avec_psd(cov)
        cov = symmartix.convert_Avec_to_A(cov)
        B = symmartix.convert_Avec_to_A(b)
        B = B.to(device)
        mu = mu.to(device)
        cov = cov.to(device)
        return B, mu, cov
class BG_RBF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Z):
        norm_const = np.zeros(Z.shape[0])
        ctx.save_for_backward(Z)
        v = Z.detach().cpu().numpy()
        for i in range(Z.shape[0]):
            matrix = v[i]
            caculatation_nc.t0 = - np.max(matrix)-2
            norm_const[i] = caculatation_nc.nc(matrix)
        tensor_type = Z.type()
        if Z.is_cuda:
            device = Z.get_device()
            result = torch.tensor(norm_const, device=device).type(tensor_type)
        else:
            result = torch.tensor(norm_const).type(tensor_type)

        return result
    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.needs_input_grad[0]:
            return None

        Z = ctx.saved_tensors[0]
        grad_Z = torch.zeros(Z.shape[0], 4)
        v = Z.detach().cpu().numpy()
        for i in range(Z.shape[0]):
            matrix = v[i]
            caculatation_nc.t0 = - np.max(matrix)-2
            grad_n = torch.tensor([
                caculatation_nc.nc_der(matrix, 1),
                caculatation_nc.nc_der(matrix, 2),
                caculatation_nc.nc_der(matrix, 3),
                caculatation_nc.nc_der(matrix, 4)
            ], device=Z.device)
            grad_Z[i] = grad_output[i] * grad_n
            #print("grad", grad_n, matrix, caculatation_nc.t0)
        
        tensor_type = grad_output.type()
        if grad_output.is_cuda:
            device = grad_output.get_device()
            result = torch.tensor(grad_Z, device=device).type(tensor_type)
        else:
            result = torch.tensor(grad_Z).type(tensor_type)
        
        return result, None
def angular_loss_mse(target, predicted):
    """ Returns the angle between two quaternions.
        Note that for a quaternion q, -q = q so the
        angle of rotation must be less than 180 degrees.

        Inputs:
          target = target quaternion
          predicted = predicted quaternion
    """
    quat_ang = torch.clamp(torch.abs(torch.dot(target, predicted)), min=0,
                           max=1)
    acos_val = torch.acos(quat_ang)
    diff_ang = acos_val * 2
    return diff_ang**2
def angular_loss_me(target, predicted):
    """ Returns the angle between two quaternions.
        Note that for a quaternion q, -q = q so the
        angle of rotation must be less than 180 degrees.

        Inputs:
          target = target quaternion
          predicted = predicted quaternion
    """
    quat_ang = torch.clamp(torch.abs(torch.dot(target, predicted)), min=0,
                           max=1)
    acos_val = torch.acos(quat_ang)
    diff_ang = acos_val * 2
    return diff_ang
def quat2euler(q):
    """ Convert left-handed quaternion to euler angles (X,Y,Z) (valid)"""

    sqx = q[0] * q[0]
    sqy = q[1] * q[1]
    sqz = q[2] * q[2]
    test = q[0]*q[2] + q[1]*q[3]
    if test > 0.499: # singularity at north pole
        pitch = 2 * np.arctan2(q[0], q[3])
        yaw = - np.pi / 2
        roll = 0
    elif test < -0.499: # singularity at south pole
        pitch = -2 * np.arctan2(q[0], q[3])
        yaw = np.pi / 2
        roll = 0
    else:
        pitch = np.arctan2(2*(q[1]*q[2] - q[0]*q[3]), 1-2*sqx-2*sqy)
        yaw = np.arcsin(-2*(q[0]*q[2]+q[1]*q[3]))
        roll = np.arctan2(2*(q[0]*q[1] - q[2]*q[3]), 1-2*sqy-2*sqz)

    # Keeps pitch between [-180, 180] under singularities
    if pitch > np.pi:
        pitch = 2*np.pi - pitch
    if pitch < -np.pi:
        pitch = 2*np.pi + pitch

    return pitch*180/np.pi, yaw*180/np.pi, roll*180/np.pi

def rmse_me_ang(target, M):
    """ Computes mean absolute angular deviation between a pair of quaternions

    Parameters:
        predicted (torch.Tensor): Output from network of shape (N, 16) if
            orthogonalization is "gram_schmidt" and (N, 4) if it is
            "quaternion_matrix".
       target (torch.Tensor): Ground truth of shape N x 4
       orthogonalization (str): Orthogonalization method to use. Can be
            "gram_schmidt" for usage of the classical gram-schmidt method.
            "modified_gram_schmidt" for a more robust variant, or
            "quaternion_matrix" for usage of a orthogonal matrix representation
            of an output quaternion.
    """
    angular_rmse = 0
    angular_me = 0
    
    batch_size = target.shape[0]
    for i in range(batch_size):
        angular_rmse += angular_loss_mse(
            target[i], M[i, :, 3])
    for i in range(batch_size):
        angular_me += angular_loss_me(
            target[i], M[i, :, 3])

    return torch.sqrt(angular_rmse / target.shape[0]), angular_me / target.shape[0]
def rmse_me_trans(target_qr, target_qt, mu):
    """ Computes mean absolute angular deviation between a pair of quaternions

    Parameters:
        predicted (torch.Tensor): Output from network of shape (N, 16) if
            orthogonalization is "gram_schmidt" and (N, 4) if it is
            "quaternion_matrix".
       target (torch.Tensor): Ground truth of shape N x 4
       orthogonalization (str): Orthogonalization method to use. Can be
            "gram_schmidt" for usage of the classical gram-schmidt method.
            "modified_gram_schmidt" for a more robust variant, or
            "quaternion_matrix" for usage of a orthogonal matrix representation
            of an output quaternion.
    """
    dual_dev = 0
    trans_dev = 0
    
    batch_size = target_qr.shape[0]
    for i in range(batch_size):
       
        dual_dev += torch.norm(target_qt[i]-mu[i].squeeze())**2
        trans_dev += torch.norm(target_qt[i]-mu[i].squeeze())

    return torch.sqrt(dual_dev / target_qr.shape[0]), trans_dev/target_qr.shape[0]
    
    
    
    
    


