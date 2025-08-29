import torch
import numpy as np
import symmartix
import utils


def generate_dual_part(G,q_r):
    if q_r.dim()<3:
        quat = q_r.unsqueeze(dim=2)
    #for i in range(quat.shape[0]):
    #    quat[i] = quat[i]/torch.norm(quat[i])
    #dual part
    dual = torch.bmm(G,quat)
    #dual = dual.squeeze(-1)
    #quat = quat.squeeze(-1)
    #for i in range(quat.shape[0]):
    #    dual[i] = dual[i]-torch.dot(quat[i],dual[i])*quat[i]
    #q_s = torch.cat([quat,dual],dim=1)
    #return q_s.squeeze(dim=2)
    return dual.squeeze(-1)

def solve_waha_fast(B, compute_gap=False):
    '''
    Use a fast eigenvalue solution to the dual of the 'generalized Wahba' problem to solve the primal.
    :param B: quadratic cost matrix
    :param redundant_constraints: boolean indicating whether to use redundant constraints
    :return: Optimal q, optimal dual var. nu, time to solve, duality gap
    '''
    nus,qs = torch.linalg.eigh(B) #nus-->eigenvalues, qs-->eigenvectors
    nu_min, nu_argmin = torch.min(nus,1)
    q_opt = qs[torch.arange(B.shape[0]),:,nu_argmin]
    q_opt = q_opt*(torch.sign(q_opt[:,3]).unsqueeze(1))
    nu_opt = -1*nu_min.unsqueeze(1)
    if compute_gap:
        p = torch.einsum('bn,bnm,bm->b',q_opt,B,q_opt).unsqueeze(1)
        gap = p+nu_opt
        return q_opt,nu_opt,gap
    return q_opt,nu_opt


class QuadQuatFastSover(torch.autograd.Function):
    '''
    Differentiable QCQP solver
    Input: Batch x 10 tensor which encodes symmetric 4x4 matrices, B
    Output: q that minimizes QCQP
    '''
    @staticmethod
    def forward(ctx,B_vec):
        B = symmartix.convert_Avec_to_A(B_vec)
        if B.dim()<3:
            B = B.unsqueeze(dim=0)
        q,nu = solve_waha_fast(B)
        ctx.save_for_backward(B,q,nu)
        return q
    
    @staticmethod
    def backward(ctx,grad_output):
        B,q,nu = ctx.saved_tensors
        grad_qcqp = compute_grad_fast(B,nu,q)
        outgrad = torch.einsum('bkq,bk->bq',grad_qcqp,grad_output)
        return outgrad


def compute_grad_fast(A, nu, q):
    """
    Input: A_vec: (B,4,4) tensor (parametrices B symmetric 4x4 matrices)
           nu: (B,) tensor (optimal lagrange multipliers)
           q: (B,4) tensor (optimal unit quaternions)
    
    Output: grad: (B, 4, 10) tensor (gradient)
           
    Applies the implicit function theorem to compute gradients of qT*A*q s.t |q| = 1, assuming A is symmetric 
    """

    assert(A.dim() > 2 and nu.dim() > 0 and q.dim() > 1)
    
    M = A.new_zeros((A.shape[0], 5, 5))
    I = A.new_zeros((A.shape[0], 4, 4))

    I[:,0,0] = I[:,1,1] = I[:,2,2] = I[:,3,3] = 1.

    #M[:, :4, :4] = A + I*nu.view(-1,1,1)# if need a negative character on here
    M[:, :4, :4] = A - I*nu.view(-1,1,1)# if need a negative character on here
    M[:, 4,:4] = q
    M[:, :4,4] = q

    b = A.new_zeros((A.shape[0], 5, 10))

    #symmetric matrix indices
    idx = torch.triu_indices(4,4)

    i = torch.arange(10)
    I_ij = A.new_zeros((10, 4, 4))

    I_ij[i, idx[0], idx[1]] = 1.
    I_ij[i, idx[1], idx[0]] = 1.
    
    I_ij = I_ij.expand(A.shape[0], 10, 4, 4)

    b[:, :4, :] = torch.einsum('bkij,bi->bjk',I_ij, q) 

    #This solves all gradients simultaneously!
    #print('M shape: ',M.shape)
    #try:
    X, _ = torch.solve(b, M)
    grad = -1*X[:,:4,:]
    #except Exception as e:
    #    grad = torch.ones(I_ij.shape[0],4,10).to(I_ij.device)
        #grad = grad*0.0001
        #print("fuck!!!")
        #utils.send_email(str(e))
    #print('b shape', b.shape)
    #print('X shape: ',X.shape)
    return grad


if __name__ == '__main__':
    B = torch.rand((2,4,4))
    q,nu = solve_waha_fast(B)
    print(q)
    print(q.norm(dim=1))
    print(nu)
