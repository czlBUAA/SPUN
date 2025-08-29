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

class BGLloss(object):
    def __init__(self):
       print("initial")
    def __call__(self, target_qr,  target_qt, output):
        if target_qr.is_cuda:
            device = target_qr.get_device()
        B, G, cov = self._output_to_B_G(output)
        #print('qrqt', target_qr.dim(), target_qr.shape[1])
        log_likelihood = torch.sum(self._log_BG_loss(target_qr, target_qt, B, G, cov))
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
        B, G, cov = self._output_to_B_G(output)
        cov = -0.5*torch.linalg.inv(cov)
        cov = torch.mean(cov, dim=0)
        
        val, M = torch.linalg.eigh(B) #升序排列的
        bd_z = torch.mean(val-val[:, 3].unsqueeze(1), 0)  #broadcast mechanism #这里还是升序的
        rmse_ang, me_ang= rmse_me_ang(target_qr, M)
        rmse_trans, me_trans = rmse_me_trans(target_qr, target_qt, G, M)
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
    def _log_BG_loss(target_qr, target_qt, B, G, cov):
        #print('qrqt1', target_qr.dim(), target_qr.shape[1])
        assert target_qr.dim() == 2 and target_qr.shape[1] == 4, \
            "Wrong dimensionality of target tensor."
        assert target_qt.dim() == 2 and target_qt.shape[1] == 3, \
            "Wrong dimensionality of target tensor."
        
        assert B.dim() == 3 and B.shape[1:3] == (4, 4), \
            "Wrong dimensionality of location parameter matrix M."

        assert G.dim() == 3 and G.shape[1] == 4, \
            "Wrong dimensionality of location parameter matrix Z."

        assert G.shape[0] == B.shape[0] and G.shape[0] == target_qr.shape[0], \
            "Number of samples does not agree with number of parameters."

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
        expanded_qt = utils.expand_qt(target_qt)
        #print("czl123", Z, cov)
        N = expanded_qt.shape[0]
        target_qd = torch.empty((N, 4), device=device, dtype=expanded_qt.dtype)
        for i in range(N):
            target_qd[i] = 0.5*utils.Qmultiply(expanded_qt[i], target_qr[i]) #qd=0.5qtqr
        #print('czl', target_qr.size(), B.size(), G.size(), cov.size(), target_qd.size(), norm_const.size())
        try:
            part1 = torch.bmm(torch.bmm(target_qr.unsqueeze(1), B_), target_qr.unsqueeze(2)).squeeze()
            if torch.isnan(part1).any():
                print("NaN found in part1!")
                # print("target_qr:", target_qr)
                # print("B:", B)

          
            temp = target_qd - torch.bmm(G, target_qr.unsqueeze(2)).squeeze()
            part2 = torch.bmm(torch.bmm(temp.unsqueeze(1), cov), temp.unsqueeze(2)).squeeze()
            if torch.isnan(part2).any():
                print("NaN found in part2!")
                # print("target_qd:", target_qd)
                # print("G:", G)
                # print("target_qr:", target_qr)
                # print("cov:", cov)

           
            log_norm_const = torch.log(norm_const)
            if torch.isnan(log_norm_const).any():
                print("NaN found in log_norm_const!")
                #print("Z:", Z)

            
            if torch.isnan(torch.linalg.inv(cov)).any():
                print("NaN found in likelihoods!")
                # print("part1:", part1)
                # print("part2:", part2)
                # print("log_norm_const:", log_norm_const)
                # print("cov determinant:", torch.linalg.det(-0.5 * torch.linalg.inv(cov)))

        except Exception as e:
            print("Error occurred:", e)
        likelihoods = (torch.bmm(torch.bmm(
                target_qr.unsqueeze(1),
                B_),
                target_qr.unsqueeze(2))).squeeze()+ (torch.bmm(torch.bmm(
                (target_qd-torch.bmm(G,target_qr.unsqueeze(2)).squeeze()).unsqueeze(1),
                cov),
                (target_qd-torch.bmm(G,target_qr.unsqueeze(2)).squeeze()).unsqueeze(2))).squeeze() - torch.log(norm_const)- torch.log(2 * math.pi* torch.sqrt(torch.linalg.det( -0.5 *torch.linalg.inv(cov))))
        return likelihoods
    def _output_to_B_G(self, output):
        #print('out', output)
        #\output = output/output.norm(dim=1).view(-1,1)
        #output = -torch.exp(output)
        if  output.is_cuda:
            device =  output.get_device()
        else:
            device = "cpu"
        A1 = output[:,0:10]
        A2 = output[:,10:26]
        A3 = output[:,26:36]
        A3 = symmartix.convert_Avec_to_Avec_psd(A3)
        A_1 = symmartix.convert_Avec_to_A(A1)
        A_3= -symmartix.convert_Avec_to_A(A3)
        #print('A_3',A_3)
        A_2 = A2.reshape(A2.shape[0],4,4)
        if A_1.dim()<3:
            A_1 = A_1.unsqueeze(dim=0)
        if A_3.dim()<3:
            A_3 = A_3.unsqueeze(dim=0)
        B_mat = symmartix.matrix_bingham(A_1,A_2,A_3)
        G_mat = symmartix.matrix_gaussian(A_2,A_3)
        #B_vec = symmartix.convert_A_to_Avec(B_mat)
        #B_vec = -symmartix.convert_Avec_to_Avec_psd(B_vec)
        #B_vec = symmartix.normalize_Avec(B_vec)
        #B = symmartix.convert_Avec_to_A(B_vec)
        if B_mat.dim()<3:
            B_mat = B_mat.unsqueeze(dim=0)
        #q_r_t,nu = qcqp.solve_waha_fast(B_mat)
        #G_mat = qcqp.generate_dual_part(G_mat)
        G_cov = A_3
        #print('BGCOV', B_mat.size(), G_mat.size(), G_cov.size())
        B_mat = B_mat.to(device)
        G_mat = G_mat.to(device)
        G_cov = G_cov.to(device)
        return B_mat, G_mat, G_cov
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
def rmse_me_trans(target_qr, target_qt, G, M):
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
    qd = torch.bmm(G, target_qr.unsqueeze(2)).squeeze() #prediction
    qdd = torch.bmm(G, M[:,:,3].unsqueeze(2)).squeeze() # for compution translation
    expanded_qt = utils.expand_qt(target_qt)
    N = expanded_qt.shape[0]
    target_qd = torch.empty((N, 4), device=target_qr.device, dtype=expanded_qt.dtype)
    for i in range(N):
        target_qd[i] = 0.5*utils.Qmultiply(expanded_qt[i], target_qr[i]) 
    if qdd.dim() <2:
        qdd = qdd.unsqueeze(0)
    for i in range(batch_size):
       
        t = 2*torch.mm(torch.t(utils.q2R(M[i, :, 3])[:, 1:]), qdd[i].unsqueeze(1))
        dual_dev += torch.norm(target_qd[i]-qd[i])**2
        trans_dev += torch.norm(target_qt[i]-t.squeeze())

    return torch.sqrt(dual_dev / target_qr.shape[0]), trans_dev/target_qr.shape[0]
    
    


