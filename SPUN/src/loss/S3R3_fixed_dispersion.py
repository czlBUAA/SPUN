import torch
import utils
import symmartix
class S3R3FixedDispersionLoss(object):
    """
    Class for calculating bingham loss assuming a fixed Z.

    Parameters:
        bd_z (list): Values of parameter matrix Z of size 3 (the bingham is four
            dimensional but the last parameter is assumed to be 0). All must be
            negative and in ascending order.
        orthogonalization (str): Orthogonalization method to use. Can be
            "gram_schmidt" for usage of the classical gram-schmidt method.
            "modified_gram_schmidt" for a more robust variant, or
            "quaternion_matrix" for usage of a orthogonal matrix representation
            of an output quaternion.
    """
    def __init__(self, bd_z, bd_cov, orthogonalization="gram_schmidt"):

        self.name = "bingham_fixed_z"
        self.bd_z = bd_z
        self.bd_cov = bd_cov
        self.orthogonalization = orthogonalization

    def __call__(self, target_qr,  target_qt, output):
        """
        Calculates the bingham fixed dispersion log likelihood loss
        on a batch of target-output values.

        Inputs:
           target: Target values at which the likelihood is evaluated
                of shape (N, 4)
           output: Output values from which M is computed, shape
                (N, 16) if orthogonalization is "gram_schmidt" and (N, 4) if it
                is "quaternion_matrix".
        Result:
           loss: The loss of the current batch.
           log_likelihood: Average log likelihood.

        """
        if type(self.bd_z) != torch.Tensor:
            bd_z = torch.tensor([
                [self.bd_z[0], 0, 0, 0],
                [0, self.bd_z[1], 0, 0],
                [0, 0, self.bd_z[2], 0],
                [0, 0, 0, 0]
            ], device=output.device, dtype=output.dtype)
        if type(self.bd_cov) != torch.Tensor:
            bd_cov = torch.tensor([
                [self.bd_cov[0], 0, 0, 0],
                [0, self.bd_cov[1], 0, 0],
                [0, 0, self.bd_cov[2], 0],
                [0, 0, 0, 0]
            ], device=output.device, dtype=output.dtype)

        log_likelihood = 0.0
        B, G, cov = self._output_to_B_G(output)
        _, M = torch.linalg.eigh(B)

        for i in range(output.shape[0]):
            log_likelihood \
                += self._bingham_loss_fixed_dispersion_single_sample(
                    target_qr[i], target_qt[i], M[i], bd_z, G[i], bd_cov)
        loss = -log_likelihood
        return loss, log_likelihood / output.shape[0]
    def statistics(self, target, output, epoch):
        """ Reports some additional loss statistics.

        Arguments:
            target (torch.Tensor): Ground-truth shaped as loss input.
            output (torch.Tensor): Network output.
            epoch (int): Current epoch. Currently unused.

        Returns:
            stats (dict): Bingham parameters and angular deviation.
        """
        stats = {
            "maad": float(maad_quaternion(
                            target, output, self.orthogonalization))
        }

        return stats

    @staticmethod
    def _bingham_loss_fixed_dispersion_single_sample(target_qr, target_qt, bd_m, bd_z, G, ga_cov):
        """
        Calculates the bingham likelihood loss on
        a single sample.

        Parameters:
           target: Target value at which the likelihood is
                evaluated
           bd_m: Bingham distribution location and axes parameter of shape
           (1, 4, 4)
           bd_z: Z parameter matrix of shape (1, 4, 4)
        """
        target_qr = target_qr.reshape(1, 4)
        target_qt = target_qt.reshape(1, 3)
        target_qt = torch.cat((target_qt, torch.zeros(1,  device=target_qt.get_device(), dtype=target_qt.dtype)), dim=1)
        target_qd = utils.Qmultiply(target_qt[0], target_qr[0])
        target_qd = target_qd.reshape(1, 4)
        
        loss = torch.mm(torch.mm(torch.mm(torch.mm(
            target_qr, bd_m), bd_z), torch.t(bd_m)), torch.t(target_qr))+ \
        torch.mm(torch.mm((target_qd-torch.t(torch.mm(G, torch.t(target_qr)))), ga_cov), torch.t(target_qd-torch.t(torch.mm(G, torch.t(target_qr)))))
        return loss

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
