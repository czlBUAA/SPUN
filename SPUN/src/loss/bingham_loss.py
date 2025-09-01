"""Implementation of the Bingham loss function"""
from __future__ import print_function

import dill
import os

import bingham_distribution as ms
import numpy as np
import torch
from scipy.interpolate import Rbf
import caculatation_nc
import utils

from .maad import maad_bingham
from .gram_schmidt import gram_schmidt, gram_schmidt_batched
from utils import generate_coordinates, vec_to_bingham_z_many


def batched_logprob(target, mu, sigma):
    """ Mean of log probability of targets given mu and sigmas of a Gaussian
    distribution """
    target = target.reshape(mu.shape)
    dist = torch.distributions.normal.Normal(mu, sigma)
    return torch.mean(dist.log_prob(target))


def batched_norm(target, output):
    """ Mean of norm error between target and output matrices """
    target = target.reshape(output.shape)
    diff = target - output
    loss = torch.mean(torch.norm(diff, dim=-1))
    return loss


class BinghamLoss(object):
    """
    Calculates the bingham log likelihood loss on a batch of target-output
    values.

    Arguments:
        lookup_table_file (str): Path to the location of the lookup table.
        interpolation_kernel (str): The kernel to use for rbf interpolaition.
            Can be "multiquadric" (default) or "gaussian".
        orthogonalization (str): Orthogonalization method to use. Can be
            "gram_schmidt" for usage of the classical gram-schmidt method.
            "modified_gram_schmidt" for a more robust variant, or
            "quaternion_matrix" for usage of a orthogonal matrix representation
            of an output quaternion.

    Inputs:
        target (torch.Tensor): Target values at which the likelihood is
            evaluated of shape (N, 4)
        output (torch.Tensor): Output values from which M and Z are extracted of
            shape (N, 19) if orthogonalization is "gram_schmidt" and shape (N,7)
            if it is "quaternion_matrix"
    Result:
        loss: The loss of the current batch.
        log_likelihood: Average log likelihood.

    """

    def __init__(self):
       print("initial")

    def __call__(self, target, output):
        if target.is_cuda:
            device = target.get_device()
        M, Z = self._output_to_m_z(output)
        log_likelihood = torch.sum(
                self._log_bingham_loss(
                    target, M, Z.squeeze(0),
                    self.rbf))

        loss = -log_likelihood
        return loss, log_likelihood / target.shape[0]

    def statistics(self, target, output, epoch):
        """ Reports some additional loss statistics.

        Arguments:
            target (torch.Tensor): Ground-truth shaped as loss input.
            output (torch.Tensor): Network output.
            epoch (int): Current epoch. Currently unused.

        Returns:
            stats (dict): Bingham parameters and angular deviation.
        """

        bd_z = torch.mean(vec_to_bingham_z_many(output[:, :3]).squeeze(0), 0)
        cur_maad = maad_bingham(target, output[:, 3:], self.orthogonalization)

        stats = {
            "z_0": float(bd_z[0]),
            "z_1": float(bd_z[1]),
            "z_2": float(bd_z[2]),
            "maad": float(cur_maad)
        }
        return stats

    @staticmethod
    def _log_bingham_loss(target, M, Z, rbf=None):
        r"""Log Bingham likelihood loss.

        The Bingham distribution is parametrized as

        f(x) = N(Z) * exp(x^T MZM^Tx)

        with x being defined on the hypershere, i.e. ||x||=1.

        Note: This has been developed using CPU-only storage of Tensors and may
         require adaptation when used with GPU.

        Parameters:
            target: Target values at which the likelihood is evaluated of shape
                (N, 4).
            M: Bingham distribution location and axes parameter of shape
                (N,4,4). M is expected to be an orthonormal matrix.
            Z: Tensor representing the Z parameter matrix of shape (N, 3).
                The parameters are expected to be negative and given in an
                ascending order.
            rbf: RBF object
        Returns:
            log likelihood: log value of the pdf for each of the target samples.
        """

        assert target.dim() == 2 and target.shape[1] == 4, \
            "Wrong dimensionality of target tensor."

        assert M.dim() == 3 and M.shape[1:3] == (4, 4), \
            "Wrong dimensionality of location parameter matrix M."

        assert Z.dim() == 2 and Z.shape[1] == 3, \
            "Wrong dimensionality of location parameter matrix Z."

        assert Z.shape[0] == M.shape[0] and Z.shape[0] == target.shape[0], \
            "Number of samples does not agree with number of parameters."

        if target.is_cuda:
            device = target.get_device()
        else:
            device = "cpu"

        # Adds missing 0 to vectors Z and turns them into diagonal matrices.
        z_padded = torch.cat(
            (Z, torch.zeros((Z.shape[0], 1), device=device, dtype=M.dtype)),
            dim=1) #Ascending
        z_as_matrices = torch.diag_embed(z_padded)

        norm_const = BinghamInterpolationRBF.apply(Z)
        likelihoods = (torch.bmm(torch.bmm(torch.bmm(torch.bmm(
                target.unsqueeze(1),
                M),
                z_as_matrices),
                M.transpose(1, 2)),
                target.unsqueeze(2))
                          ).squeeze() - norm_const
        return likelihoods

    def _output_to_m_z(self, output):
        """ Creates orthogonal matrix from output.

        This method does not support vectorization yet.

        Parameters:
            output (torch.Tensor): Output values from which M is extracted,
                shape (19,) for gram-schmidt orthogonalization and (7,) for
                quaternion_matrix orthogonalization.
        """
        bd_z = utils.vec_to_bingham_z_many(output[:, :3])
        bd_m = vec_to_bingham_m(output[:, 3:], "gram_schmidt")

        return bd_m, bd_z

class BinghamInterpolationRBF(torch.autograd.Function):
    r"""Computes the Bingham interpolation constant and its derivatives.

    Input:
        Z: Tensor representing the Z parameters of shape (N, 3).

    Returns:
        norm_const: Von Mises normalization constants evaluated for each set of kappas 
        in matrix.

    """

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

def vec_to_bingham_m(output, orthogonalization):
    """ Creates orthogonal matrix from output.

    This operates on an entire batch.

    Parameters:
        output (torch.Tensor): Output values from which M is extracted,
            shape (batch_size, 16) for gram-schmidt orthogonalization
            and (batch_size, 4) for quaternion_matrix orthogonalization.
        orthogonalization (str): orthogonalization (str): Orthogonalization
            method to use. Can be "gram_schmidt" for usage of the classical
            gram-schmidt method. "modified_gram_schmidt" for a more robust
            variant, or "quaternion_matrix" for usage of a orthogonal matrix
            representation of an output quaternion. The latter is not supported
            yet.
    """
    batch_size = output.shape[0]
    if orthogonalization == "gram_schmidt":
        reshaped_output = output.reshape(batch_size, 4, 4)
        bd_m = gram_schmidt_batched(reshaped_output)
    elif orthogonalization == "modified_gram_schmidt":
        reshaped_output = output.reshape(batch_size, 4, 4)
        bd_m = gram_schmidt_batched(reshaped_output, modified=True)
    elif orthogonalization == "quaternion_matrix":
       #TODO batched version
        bd_m = torch.zeros(output.shape[0], 4, 4).to(device=output.device, dtype=output.dtype)
        for i in range(output.shape[0]):
            bd_m[i] = quaternion_matrix(output)
    else:
        raise ValueError("Invalid orthogonalization type.")
    return bd_m



