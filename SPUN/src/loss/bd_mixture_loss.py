"""Implementation of the Bingham Mixture Loss"""
import torch
from .bd_fixed_dispersion import BGFixedDispersionLoss
from .bd_loss import BGLloss
from utils import vec_to_bingham_z_many
import symmartix
import utils
import torch.nn.functional as F
from config import cfg
class BGMixtureLoss(object):
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
    def __init__(self, mixture_component_count=cfg.mix_count,
                 fixed_dispersion_stage=-1, fixed_param_z=[-1, -1, -1, 0],
        fixed_param_cov=[-1,-1,-1,-1] ):

        self._num_components = mixture_component_count
        self._fixed_dispersion_stage = fixed_dispersion_stage
        self._softmax = torch.nn.Softmax(dim=1)
        self._bingham_loss = BGLloss()
        self._fixed_dispersion_loss = BGFixedDispersionLoss(
            fixed_param_z, fixed_param_cov, orthogonalization="gram_schmidt")
    # def __call__(self, target_qr, target_qt, output, epoch=160):
    #基于log-sum-exp策略
    #     """
    #     Args:
    #         target_qr: [B,4] 目标四元数
    #         target_qt: [B,3] 目标平移
    #         output:    [B, K*37 + ?] 网络预测，0:-1:37 对应 weight logits
    #     Returns:
    #         loss:       标量，等于 -∑_i log ∑_j w_ij e^{ℓ_ij}
    #         avg_ll:     平均对数似然
    #     """
    #     B = output.shape[0]
    #     K = self._num_components

    #     # 1) 先算出每个样本每个分量的“权重” (softmax over logits)
    #     weights = self._softmax(output[:, 0:-1:37])  # [B, K]

    #     total_log_likelihood = torch.tensor(
    #         0., device=output.device, dtype=output.dtype
    #     )
    #     eps = 1e-12  # 防止 log(0)

    #     # 2) 双重循环
    #     for i in range(B):
    #         # 2.1 取出第 i 个样本的 K 个 log weight
    #         log_w_i = torch.log(weights[i] + eps)     # [K]

    #         # 2.2 把每个分量的 (log_w + ℓ) 存到列表里
    #         component_terms = []
    #         for j in range(K):
    #             # 取出 params_j
    #             params_j = output[i, (j*37+1):(j+1)*37].unsqueeze(0)
    #             # 计算 log-likelihood ℓ_ij
                
    #             ll_j = self._bingham_loss(
    #                     target_qr[i].unsqueeze(0),
    #                     target_qt[i].unsqueeze(0),
    #                     params_j
    #                 )[1]

    #             # 把 NaN / -inf 换成一个大负数
    #             #ll_j = torch.nan_to_num(ll_j, nan=-1e6, neginf=-1e6).squeeze()
    #             # 累加 log_w + ll_j
    #             component_terms.append(log_w_i[j] + ll_j)

    #         # 2.3 手写 log-sum-exp
    #         #    m = max_j (log_w_i[j] + ll_j)
    #         term_stack = torch.stack(component_terms)     # [K]
            
    #         m = torch.max(term_stack)
    #         sum_exp = torch.exp(term_stack - m).sum()
    #         logsumexp = m + torch.log(sum_exp)

    #         # 累到总对数似然
    #         total_log_likelihood += logsumexp

    #     # 3) 最终 loss 和 平均 log-likelihood
    #     loss   = - total_log_likelihood
    #     avg_ll = total_log_likelihood / B

    #     return loss, avg_ll
    def __call__(self, target_qr, target_qt, output, epoch=160):
        """
        Args:
            target_qr: [B,4] 目标四元数
            target_qt: [B,3] 目标平移
            output:    [B, M*37 + ...] 预测张量，
                       output[:, 0:-1:37] 用于权重（此处忽略，WTA 不用 softmax）
        Returns:
            loss:   标量 Tensor = ∑_i L_i
            avg_ll: 平均对数似然 = ∑_i ∑_j π_{ij} * ℓ_{ij} / B
        """
        #基于RTWA策略
        B = output.shape[0]
        total_loss = torch.tensor(
            0., device=output.device, dtype=output.dtype
        )
        total_loglik = torch.tensor(
            0., device=output.device, dtype=output.dtype
        )

        for i in range(B):
            # 1) 计算每个分量的 log-likelihood ℓ_{ij}
            ll_list = []
            for j in range(self._num_components):
                params_j = output[i, (j*36):(j+1)*36].unsqueeze(0)
                if epoch < self._fixed_dispersion_stage:
                    ll_j = self._fixed_dispersion_loss(
                        target_qr[i].unsqueeze(0),
                        target_qt[i].unsqueeze(0),
                        params_j
                    )[1]
                else:
                    ll_j = self._bingham_loss(
                        target_qr[i].unsqueeze(0),
                        target_qt[i].unsqueeze(0),
                        params_j
                    )[1]
                ll_list.append(ll_j.squeeze())

            ll = torch.stack(ll_list)  # ℓ shape: (M,)

            # 2) 找到最佳分量 k
            k = torch.argmax(ll)

            # 3) 计算 RWTA 权重 π_j
            #    最优分量：1 - sigma；其余分量：sigma / (M - 1)
            pi = torch.full((self._num_components,),
                            fill_value=0.05 / (self._num_components - 1),
                            device=output.device,
                            dtype=output.dtype)
            pi[k] = 1.0 - 0.05    # 确保 sum_j pi_j = 1

            # 4) 样本 i 的损失 L_i = -∑_j π_j * ℓ_{ij}
            loss_i = -(pi * ll).sum()
            total_loss += loss_i

            # 用于 avg_ll = ∑_j π_j * ℓ_{ij} 的累加
            total_loglik += (pi * ll).sum()

        loss = total_loss
        avg_ll = total_loglik / B
        return loss, avg_ll
    # def __call__(self, target_qr,  target_qt, output, epoch=160):
    #基于muti-step training策略
    #     batch_size = output.shape[0]
    #     weights = self._softmax(output[:, 0:-1:37])
    #     #weights = output[:, 0:-1:37]
    #     log_likelihood = torch.tensor(0., device=output.device, dtype=output.dtype)
       
    #     # fixed_param_z[0] = fixed_param_z[0]-  epoch/1.5
    #     # fixed_param_z[1] = fixed_param_z[1]- epoch/1.5
    #     # fixed_param_z[2] = fixed_param_z[2]-  epoch/1.5
    #     # fixed_param_cov[0] = fixed_param_cov[0] - epoch/1.5
    #     # fixed_param_cov[1] = fixed_param_cov[1] - epoch/1.5
    #     # fixed_param_cov[2] = fixed_param_cov[2] - epoch/1.5
    #     # fixed_param_cov[3] = fixed_param_cov[3] - epoch/1.5
        
    #     for i in range(batch_size):
    #         current_likelihood = torch.tensor(
    #             0., device=output.device, dtype=output.dtype)
    #         for j in range(self._num_components):
                
    #             if epoch < self._fixed_dispersion_stage:
    #                 bd_log_likelihood = self._fixed_dispersion_loss(
    #                     target_qr[i].unsqueeze(0), target_qt[i].unsqueeze(0), 
    #                     output[i, (j*37+1):((j+1)*37)].unsqueeze(0))[1]
    #             else:
    #                 bd_log_likelihood = self._bingham_loss(
    #                     target_qr[i].unsqueeze(0), target_qt[i].unsqueeze(0), 
    #                     output[i, (j*37+1):((j+1)*37)].unsqueeze(0))[1]
    #             #weights[i,j]=1
    #             current_likelihood += weights[i, j] * \
    #                 torch.exp(bd_log_likelihood).squeeze()
                
    #             #print("test", current_likelihood)
            
    #         log_likelihood += torch.log(current_likelihood)

    #     loss = -log_likelihood
    #     log_likelihood /= batch_size

    #     return loss, log_likelihood

    # def statistics(self, target_varr, target_vart, output, epoch=None):
    #     """ Reports some additional loss statistics.

    #     Arguments:
    #         target (torch.Tensor): Ground-truth shaped as loss input.
    #         output (torch.Tensor): NN output shaped as loss output parameter.
    #         epoch (int): Current epoch. Currently unused.

    #     Returns:
    #         stats (dict): Bingham parameters and angular deviation.
    #     """
    #     batch_size = output.shape[0]
    #     weights = self._softmax(output[:, 0:-1:37])
    #     max_indices = torch.argmax(weights, dim=1)
    #     param_m = torch.zeros((batch_size, 4, 4),
    #                           device=output.device, dtype=output.dtype)
    #     param_G = torch.zeros((batch_size, 4, 4),
    #                           device=output.device, dtype=output.dtype)
    #     param_cov = torch.zeros((batch_size, 4, 4),
    #                           device=output.device, dtype=output.dtype)
    #     vall = torch.zeros((batch_size, 4),
    #                           device=output.device, dtype=output.dtype)
    #     covv =  torch.zeros((batch_size, 4, 4),
    #                           device=output.device, dtype=output.dtype)
        
    #     for j in range(batch_size):
    #         B, G, cov = self._output_to_B_G(
    #             output[j, (max_indices[j]*37+1):((max_indices[j]+1)*37)]
    #         )
    #         val, M = torch.linalg.eigh(B) #升序排列的
    #         vall[j,: ] = val-val[:, 3]
    #         if M.dim()<3:
    #             M = M.unsqueeze(0)
    #         if G.dim()<3:
    #             G = G.unsqueeze(0)
    #         if cov.dim()<3:
    #             cov = cov.unsqueeze(0)
    #         param_m[j, :, :] = M
    #         param_G[j, :, :] = G
    #         covv[j, :, :] =cov
    #     covv = -0.5*torch.linalg.inv(covv)
    #     covv = torch.mean(covv, dim=0)
    #     bd_z = torch.mean(vall, 0) 
    #     rmse_ang, me_ang= rmse_me_ang(target_varr, param_m)
    #     rmse_trans, me_trans = rmse_me_trans(target_varr, target_vart, param_G, param_m)
    #     #print("czl", bd_z)
    #     stats = {
    #         "z_0": float(bd_z[0]),
    #         "z_1": float(bd_z[1]),
    #         "z_2":  float(bd_z[2]),
    #         "cov": covv.detach().cpu().numpy(),
    #         "rmse_trans": float(rmse_trans),
    #         "me_trans": float(me_trans),
    #         "rmse_ang": float(rmse_ang),
    #         "me_ang": float(me_ang),
    #         "est_q": M[:, :, 3].detach().cpu().numpy()
    #     }

    #     return stats
    # def statistics(self, target_qr,  target_qt, output, epoch=None):
    #     """ Reports some additional loss statistics.

    #     Arguments:
    #         target (torch.Tensor): Ground-truth shaped as loss input.
    #         output (torch.Tensor): NN output shaped as loss output parameter.
    #         epoch (int): Current epoch. Currently unused.

    #     Returns:
    #         stats (dict): Bingham parameters and angular deviation.
    #     """
    #     batch_size = output.shape[0]
    #     weights = self._softmax(output[:, 0:-1:37])
        
    #     rmse_trans = torch.zeros(
    #         batch_size, device=output.device, dtype=output.dtype)
    #     me_trans = torch.zeros(
    #         batch_size, device=output.device, dtype=output.dtype)
    #     rmse_ang = torch.zeros(
    #         batch_size, device=output.device, dtype=output.dtype)
    #     me_ang = torch.zeros(
    #         batch_size, device=output.device, dtype=output.dtype)
    #     mode_stats = dict()
       
    #     param_m = torch.zeros((batch_size, self._num_components, 4, 4),
    #                           device=output.device, dtype=output.dtype)
    #     param_G = torch.zeros((batch_size, self._num_components, 4, 4),
    #                           device=output.device, dtype=output.dtype)
    #     for j in range(self._num_components):
    #         B, G, cov = self._output_to_B_G(
    #             output[:, (j*37+1):((j+1)*37)]
    #          )
    #         val, M = torch.linalg.eigh(B)
    #         print("M", j, M[:,3])
    #         param_m[:, j, :, :] = M
    #         param_G[:, j, :, :] = G
        

    #     # Setting mmaad to 10 such that the minimum succeeds in the first run.
    #     for i in range(batch_size):
    #         for j in range(self._num_components):
    #             cur_rmse_ang, cur_me_ang = rmse_me_ang(
    #                 target_qr[i], param_m[i, j, :, 3])
    #             cur_rmse_trans, cur_me_trans = rmse_me_trans(
    #                 target_qr[i], target_qt[i], param_G[i, j, :, :], param_m[i, j, :, 3])

    #             me_ang[i] += cur_me_ang * weights[i, j]
    #             me_trans[i] += cur_me_trans * weights[i, j]
           

    #     me_ang = torch.mean(me_ang)
    #     me_trans = torch.mean(me_trans)
    #     stats = {
    #         "z_0": float(-1),
    #         "z_1": float(-1),
    #         "z_2":  float(-1),
    #         "cov": cov.detach().cpu().numpy(),
    #         "rmse_trans": float(1.231),
    #         "me_trans": float(me_trans),
    #         "rmse_ang": float(1.231),
    #         "me_ang": float(me_ang),
    #         "est_q": cov.detach().cpu().numpy(),
    #         "gt_q":  cov.detach().cpu().numpy(),
    #     }
    #     stats.update(mode_stats)

    #     return stats
    
    def statistics(self, target_qr, target_qt, output, epoch=None):

        B = output.shape[0]
        M = self._num_components
        sigma = 0.05  # RWTA 松弛系数

        me_ang = torch.zeros(B, device=output.device, dtype=output.dtype)
        me_trans = torch.zeros(B, device=output.device, dtype=output.dtype)
        rmse_ang = torch.zeros(B, device=output.device, dtype=output.dtype)
        rmse_trans = torch.zeros(B, device=output.device, dtype=output.dtype)

        # 先解出每个分量的 B 和 G 矩阵
        param_m = torch.zeros((B, M, 4, 4), device=output.device, dtype=output.dtype)
        param_G = torch.zeros((B, M, 4, 4), device=output.device, dtype=output.dtype)
        for j in range(M):
            B_mat, G_mat, cov = self._output_to_B_G(
                output[:, (j*36):((j+1)*36)]
            )
            _, M_eig = torch.linalg.eigh(B_mat)
            # M_eig[..., 3] 是主特征向量
            param_m[:, j, :, :] = M_eig
            param_G[:, j, :, :] = G_mat

        # 对每个样本计算 RWTA 权重并加权误差
        for i in range(B):
            # 1) 计算每个分量的 log-likelihood ll_j
            ll_list = []
            for j in range(M):
                params_j = output[i, (j*36):((j+1)*36)].unsqueeze(0)
                if epoch is not None and epoch < self._fixed_dispersion_stage:
                    ll_j = self._fixed_dispersion_loss(
                        target_qr[i].unsqueeze(0),
                        target_qt[i].unsqueeze(0),
                        params_j
                    )[1]
                else:
                    ll_j = self._bingham_loss(
                        target_qr[i].unsqueeze(0),
                        target_qt[i].unsqueeze(0),
                        params_j
                    )[1]
                ll_list.append(ll_j.squeeze())

            ll = torch.stack(ll_list)  # (M,)

            # 2) 找到最佳分量 k
            k = torch.argmax(ll)

            # 3) 构造 RWTA 权重 π_j
            pi = torch.full((M,),
                            fill_value=sigma / (M - 1),
                            device=output.device,
                            dtype=output.dtype)
            pi[k] = 1.0 - sigma

            # 4) 计算加权误差
            # for j in range(M):
            #     # 角度误差和位置误差
            #     cur_rmse_ang, cur_me_ang = rmse_me_ang(
            #         target_qr[i],
            #         param_m[i, j, :, 3]
            #     )
            #     cur_rmse_trans, cur_me_trans = rmse_me_trans(
            #         target_qr[i],
            #         target_qt[i],
            #         param_G[i, j, :, :],
            #         param_m[i, j, :, 3]
            #     )
            #     me_ang[i]   += cur_me_ang   * pi[j]
            #     me_trans[i] += cur_me_trans * pi[j]
            #5) 计算权值最大的分布对应的误差
            cur_rmse_ang, cur_me_ang = rmse_me_ang(
                    target_qr[i],
                    param_m[i, k, :, 3]
                )
            cur_rmse_trans, cur_me_trans = rmse_me_trans(
                    target_qr[i],
                    target_qt[i],
                    param_G[i, k, :, :],
                    param_m[i, k, :, 3]
                )
            me_ang[i]   = cur_me_ang
            me_trans[i] = cur_me_trans
            rmse_ang[i]   = cur_rmse_ang
            rmse_trans[i] = cur_rmse_trans
        me_ang =torch.mean(me_ang)
        me_trans =torch.mean(me_trans)
        rmse_ang =torch.sqrt(torch.mean(rmse_ang))
        rmse_trans =torch.sqrt(torch.mean(rmse_trans))

        # 对整个 batch 取平均
        stats = {
             "z_0": float(-1),
             "z_1": float(-1),
            "z_2":  float(-1),
            "cov": cov.detach().cpu().numpy(),
           "rmse_trans": float(rmse_trans),
            "me_trans": float(me_trans),
             "rmse_ang": float(rmse_ang ),
            "me_ang": float(me_ang),
            "est_q": param_m[:, k, :, 3].detach().cpu().numpy(),
            "gt_q":  cov.detach().cpu().numpy(),
        }
        return stats

    
    def _output_to_B_G(self, output):
        #print('out', output)
        #\output = output/output.norm(dim=1).view(-1,1)
        #output = -torch.exp(output)
        if  output.is_cuda:
            device =  output.get_device()
        else:
            device = "cpu"
        if output.dim()<2:
           output = output.unsqueeze(0)
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
def gram_schmidt(input_mat, reverse=False, modified=False):
    """ Carries out the Gram-Schmidt orthogonalization of a matrix.

    Arguments:
        input_mat (torch.Tensor): A quadratic matrix that will be turned into an
            orthogonal matrix.
        reverse (bool): Starts gram Schmidt method beginning from the last
            column if set to True.
        modified (bool): Uses modified Gram-Schmidt as described.
    """
    mat_size = input_mat.shape[0]
    Q = torch.zeros(mat_size, mat_size,
                    device=input_mat.device, dtype=input_mat.dtype)

    if modified:
        if reverse:
            outer_iterator = range(mat_size - 1, -1, -1)
            def inner_iterator(k): return range(k, -1, -1)
        else:
            outer_iterator = range(mat_size)
            def inner_iterator(k): return range(k+1, mat_size)

        # This implementation mostly follows the description from
        # https://www.math.uci.edu/~ttrogdon/105A/html/Lecture23.html
        # The more complex form is due to pytorch not allowing for inplace
        # operations of variables needed for gradient computation.
        v = input_mat
        for j in outer_iterator:
            Q[:, j] = v[:, j] / torch.norm(v[:, j])

            v_old = v
            v = torch.zeros(mat_size, mat_size,
                            device=input_mat.device, dtype=input_mat.dtype)

            for i in inner_iterator(j):
                v[:, i] = v_old[:, i] \
                          - (torch.dot(Q[:, j].clone(), v_old[:, i])
                             * Q[:, j].clone())

    elif not modified:
        if reverse:
            outer_iterator = range(mat_size - 1, -1, -1)
            def inner_iterator(k): return range(mat_size - 1, k, -1)
        else:
            outer_iterator = range(mat_size)
            def inner_iterator(k): return range(k)

        for j in outer_iterator:
            v = input_mat[:, j]
            for i in inner_iterator(j):
                p = torch.dot(Q[:, i].clone(), v) * Q[:, i].clone()
                v = v - p

            Q[:, j] = v / torch.norm(v)

    return Q


def gram_schmidt_batched(input_mat, reverse=False, modified=False):
    """ Carries out the Gram-Schmidt orthogonalization of a matrix on an
        entire batch.

    Arguments:
        input_mat (torch.Tensor): A tensor containing quadratic matrices each of
            which will be orthogonalized of shape (batch_size, m, m).
        reverse (bool): Starts gram Schmidt method beginning from the last
            column if set to True.
        modified (bool): Uses modified Gram-Schmidt as described.

    Returns:
        Q (torch.Tensor): A batch of orthogonal matrices of same shape as
            input_mat.
    """
    batch_size = input_mat.shape[0]
    mat_size = input_mat.shape[1]
    Q = torch.zeros(batch_size, mat_size, mat_size,
                    device=input_mat.device, dtype=input_mat.dtype)

    if modified:
    #TODO implement batched version
        for i in range(input_mat.shape[0]):
            q = gram_schmidt(input_mat[i], reverse, modified)
            Q[i] = q 
    elif not modified:
        if reverse:
            raise NotImplementedError
        else:
            outer_iterator = range(mat_size)
            def inner_iterator(k): return range(k)

        for j in outer_iterator:
            v = input_mat[:, :, j].view(batch_size, mat_size, 1)
            for i in inner_iterator(j):
                q_squeezed = Q[:, :, i].view(batch_size, 1, mat_size).clone()
                dot_products = torch.bmm(q_squeezed, v)
                p = dot_products.repeat((1, mat_size, 1)) \
                    * Q[:, :, i].unsqueeze(2).clone()
                v = v - p

            Q[:, :, j] = v.squeeze() / torch.norm(v, dim=1).repeat(1, mat_size)

    return Q
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
    angular_me = 0
    

    angular_me += angular_loss_me(
            target, M)
    

    return angular_me**2 , angular_me 
def rmse_me_trans(target_qr, target_qt, G, M):
    """ Computes mean absolute angular deviation between a pair of quaternions

    Parameters:
        predicted (torch.Tensor): Output from network of shape (N, 16) if
            orthogonalization is "gram_schmidt" and (N, 4) if it is
            "quaternion_matrix".
       target (torch.Tensor): Ground truth of shape N x 4
       orthogonalization (str): Orthogonalization method to use . Can be
            "gram_schmidt" for usage of the classical gram-schmidt method.
            "modified_gram_schmidt" for a more robust variant, or
            "quaternion_matrix" for usage of a orthogonal matrix representation
            of an output quaternion.
    """
    trans_dev = 0
    
    qdd = torch.mm(G, M.unsqueeze(1)).squeeze() # for compution translation

    t = 2*torch.mm(torch.t(utils.q2R(M)[:, 1:]), qdd.unsqueeze(1))
    trans_dev += torch.norm(target_qt-t.squeeze())

    return trans_dev**2, trans_dev