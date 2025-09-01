"""Implementation of the Bingham Mixture Loss"""
import torch
from .S3R3_fixed_dispersion import S3R3FixedDispersionLoss
from .S3R3_loss import S3R3loss
from utils import vec_to_bingham_z_many


class S3R3MixtureLoss(object):
    """ Bingham Mixture Loss

    Computes the log likelihood bingham mixture loss on a batch. Can be
    configured such that for a predefined number of epochs

    Arguments:
        lookup_table_file (str): Path to the location of the lookup table.
        mixture_component_count (int): Number of Bingham mixture components.
        interpolation_kernel (str): The kernel to use for rbf interpolaition
            (can be "multiquadric" or "gaussian").
        fixed_dispersion_stage (int): Number of epochs in which the network is
            trained using a fixed dispersion parameter z.
        fixed_param_z (list): The fixed dispersion parameter Z used for all
            mixture components during the fixed dispersion stage.

    Inputs:
        target (torch.Tensor): Target values at which the likelihood is
            evaluated of shape (N, 4)
        output (torch.Tensor): Output values from which M and Z are extracted of
            shape (N, MIXTURE_COMPONENT_COUNT * 20). The first of the 20 values
            per mixture component is for computing the weight of that component.
            The remaining 19 are passed on to the BinghamLoss class.
    """
    def __init__(self, mixture_component_count=2,
                 fixed_dispersion_stage=25,
                 fixed_param_z=[-1, -1, -1, 0], fixed_param_cov=[1,1,1,1,1,1]):

        self._num_components = mixture_component_count
        self._fixed_dispersion_stage = fixed_dispersion_stage
        self._softmax = torch.nn.Softmax(dim=1)
        self._S3R3_loss = S3R3loss()

    def __call__(self, target_qr,  target_qt, output):
        batch_size = output.shape[0]
        weights = self._softmax(output[:, 0:-1:29])

        log_likelihood = torch.tensor(0., device=output.device, dtype=output.dtype)
        for i in range(batch_size):
            current_likelihood = torch.tensor(
                0., device=output.device, dtype=output.dtype)
            for j in range(self._num_components):
                bd_log_likelihood = self._S3R3_loss(
                        target_qr[i].unsqueeze(0), target_qt[i].unsqueeze(0), 
                        output[i, (j*29+1):((j+1)*29)].unsqueeze(0))[1]
                
                current_likelihood += weights[i, j] * \
                    torch.exp(bd_log_likelihood).squeeze()

            log_likelihood += torch.log(current_likelihood)

        loss = -log_likelihood
        log_likelihood /= batch_size

        return loss, log_likelihood

    def statistics(self, target, output, epoch):
        """ Reports some additional loss statistics.

        Arguments:
            target (torch.Tensor): Ground-truth shaped as loss input.
            output (torch.Tensor): NN output shaped as loss output parameter.
            epoch (int): Current epoch. Currently unused.

        Returns:
            stats (dict): Bingham parameters and angular deviation.
        """
        batch_size = output.shape[0]
        weights = self._softmax(output[:, 0:-1:20])

        stats = {
            "maad": float(1),
            "mmaad": float(2)
        }

        return stats
def angular_loss_single_sample(target, predicted):
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
