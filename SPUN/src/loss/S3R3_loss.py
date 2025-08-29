import torch.linalg
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

class S3R3loss(object):
    def __init__(self):
       print("initial")
    def __call__(self, target_qr,  target_qt, output):
        if target_qr.is_cuda:
            device = target_qr.get_device()
        B, mu, P, sigmac = self._output_to_parameter(output)
        #print('qrqt', target_qr.dim(), target_qr.shape[1])
        log_likelihood = torch.sum(self._log_BG_loss(target_qr, target_qt, B, mu, P, sigmac))
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
        B, mu, P, sigmac = self._output_to_parameter(output)
        sigmac = torch.mean(sigmac, dim=0)
        
        val, M = torch.linalg.eigh(B) #升序排列的
        bd_z = torch.mean(val-val[:, 3].unsqueeze(1), 0)  #broadcast mechanism #这里还是升序的
        rmse_ang, me_ang = rmse_me_ang(target_qr, M)
        B_ = torch.zeros((B.shape[0], 4, 4))
        for i in range(B.shape[0]):
            vals, Mi = torch.linalg.eigh(B[i])# eigh分解的特征值默认升序的
            z_padded = vals - vals[3] #broadcast mechanism
            z_as_matrices = torch.diag_embed(z_padded) 
            B_[i,:,:] = torch.mm(torch.mm(Mi, z_as_matrices), Mi.T)
        mu_c = encode_mu_c(B_, target_qr, mu, P)
        rmse_trans, me_trans = rmse_me_trans(target_qt, mu_c)
        stats = {
            "z_0": float(bd_z[0]),
            "z_1": float(bd_z[1]),
            "z_2":  float(bd_z[2]),
            "cov": sigmac.detach().cpu().numpy(),
            "rmse_trans": float(rmse_trans),
            "me_trans": float(me_trans),
            "rmse_ang": float(rmse_ang),
            "me_ang": float(me_ang)
            
        }
        return stats
    @staticmethod
    def _log_BG_loss(target_qr, target_qt, B, mu, P, sigmac):
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
            vals, Mi = torch.linalg.eigh(B[i])# eigh分解的特征值默认升序的
            z_padded = vals - vals[3] #broadcast mechanism
            Z[i, :] = z_padded
            z_as_matrices = torch.diag_embed(z_padded) 
            B_[i,:,:] = torch.mm(torch.mm(Mi, z_as_matrices), Mi.T)
        Z = Z.to(device)
        B_ = B_.to(device)
        norm_const = BG_RBF.apply(Z)
        mu_c = encode_mu_c(B_, target_qr, mu, P)
        try:
            part1 = torch.bmm(torch.bmm(target_qr.unsqueeze(1), B_), target_qr.unsqueeze(2)).squeeze()
            if torch.isnan(part1).any():
                print("NaN found in part1!")
                # print("target_qr:", target_qr)
                # print("B:", B)
        except Exception as e:
            print("Error occurred:", e)
        likelihoods = (torch.bmm(torch.bmm(
                target_qr.unsqueeze(1),
                B_),
                target_qr.unsqueeze(2))).squeeze()-0.5* (torch.bmm(torch.bmm(
                (target_qt-mu_c).unsqueeze(1),
                torch.linalg.inv(sigmac)),
                (target_qt-mu_c).unsqueeze(2))).squeeze() - torch.log(norm_const)- torch.log(((2 * math.pi)**(3/2))* torch.sqrt(torch.linalg.det(sigmac)))
        return likelihoods
    def _output_to_parameter(self, output):
        #print('out', output)
        #\output = output/output.norm(dim=1).view(-1,1)
        #output = -torch.exp(output)
        if  output.is_cuda:
            device =  output.get_device()
        else:
            device = "cpu"
        b = output[:,0:10]
        mu = output[:,10:13]
        P = output[:,13:22]
        sigma_c = output[:,22:28]
        sigma_c = symmartix.convert_Avec_to_Avec_psd(sigma_c)
        B = symmartix.convert_Avec_to_A(b)
        sigma_c = symmartix.convert_Avec_to_A(sigma_c)
        #print('A_3',A_3)
        P = P.reshape(P.shape[0],3,3)
        if B.dim()<3:
            B = B.unsqueeze(dim=0)
        if mu.dim()<2:
            mu = mu.unsqueeze(dim=0)
        if P.dim()<3:
            P = P.unsqueeze(dim=0)
        if sigma_c.dim()<3:
            sigma_c = sigma_c.unsqueeze(dim=0)
        B = B.to(device)
        mu = mu.to(device)
        P = P.to(device)
        sigma_c = sigma_c.to(device)
        return B, mu, P, sigma_c
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
            #print("grad_n", grad_n, matrix)
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
def rmse_me_trans(target_qt, mu_c):
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
    trans_rmse = 0
    trans_me = 0
    batch_size = target_qt.shape[0]
    for i in range(batch_size):
        trans_rmse += torch.norm(target_qt[i]-mu_c[i])**2
        trans_me += torch.norm(target_qt[i]-mu_c[i])

    return torch.sqrt(trans_rmse / target_qt.shape[0]), trans_me/target_qt.shape[0]
    
def encode_mu_c(B_, target_qr, mu, P):
    if target_qr.is_cuda:
        device = target_qr.get_device()
    else:
        device = "cpu"
    B_ = B_.to(device)
    Z, M = torch.linalg.eigh(B_)
    M = M[:, :, [3, 2, 1, 0]] 
    Z = Z[:, [3, 2, 1, 0]] #变为降序排列。默认0是第一个
    M = torch.transpose(M,2,1)
    p = torch.bmm(M, target_qr.unsqueeze(2)).squeeze()
    if p.dim() == 1 :
        p = p.unsqueeze(0)
    vq = torch.zeros((M.shape[0], 3)).to(device)
    for i in range(M.shape[0]):
        vq[i,0] = (Z[i, 2]-Z[i,3]) * p[i,2] *p[i,3]-Z[i,1]*p[i,0] *p[i,1]
        vq[i,1] = (Z[i, 3]-Z[i,1]) * p[i,1] *p[i,3]-Z[i,2]*p[i,0] *p[i,2]
        vq[i,2] = (Z[i, 1]-Z[i,2]) * p[i,1] *p[i,2]-Z[i,3]*p[i,0] *p[i,3]
    
    mu_c = mu + torch.bmm(P, vq.unsqueeze(2)).squeeze() #这里多乘了一个0.5
    return mu_c
    
    
    
    
    
     
    


