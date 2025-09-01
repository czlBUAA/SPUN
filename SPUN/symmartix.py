import numpy as np
import torch



#Net output: 36 elements
# 10 for A1
# 16 for A2
# 10 for A3

#generate symmetric A
#A_vec: 10
#A_matrix: 4x4
def convert_Avec_to_A(A_vec):
    '''
    Convert BXM tersor to BXNXN symmetric matrix
    '''
    if A_vec.dim()<2:
        A_vec = A_vec.unsqueeze(dim=0)
    
    if A_vec.shape[1]==10:
        A_dim=4
    elif A_vec.shape[1] == 6:
        A_dim = 3
    elif A_vec.shape[1]==55:
        A_dim=10
    idx = torch.triu_indices(A_dim,A_dim)
    A = A_vec.new_zeros((A_vec.shape[0],A_dim,A_dim))
    A[:,idx[0],idx[1]]=A_vec
    A[:,idx[1],idx[0]]=A_vec
    return A.squeeze()

#A_mat: 4x4
#A_vec: 10
def convert_A_to_Avec(A):
    if A.dim()<3:
        A = A.unsqueeze(dim=0)
    idx = torch.triu_indices(A.shape[1],A.shape[1])
    A_vec = A[:,idx[0],idx[1]]
    return A_vec.squeeze()


#normalize to generate a unit Frobenius norm
def normalize_Avec(A_vec):
    A = convert_Avec_to_A(A_vec)
    if A.dim()<3:
        A = A.unsqueeze(dim=0)
    A = A/A.norm(dim=[1,2],keepdim=True)
    return convert_A_to_Avec(A).squeeze()


#convert to positive semidefinite vector
def convert_Avec_to_Avec_psd(A_vec):
    """ Convert BxM tensor (encodes symmetric NxN matrices) to BxM tensor  
    (encodes symmetric and positive semi-definite 4x4 matrices)"""

    if A_vec.dim() < 2:
        A_vec = A_vec.unsqueeze(dim=0)
    
    if A_vec.shape[1] == 10:
        A_dim = 4
    elif A_vec.shape[1] == 6:
        A_dim = 3
    elif A_vec.shape[1] == 55:
        A_dim = 10
    else:
        raise ValueError("Arbitrary A_vec not yet implemented")

    idx = torch.tril_indices(A_dim,A_dim)
    L = A_vec.new_zeros((A_vec.shape[0],A_dim,A_dim))   
    L[:, idx[0], idx[1]] = A_vec
    A = L.bmm(L.transpose(1,2))
    A_vec_psd = convert_A_to_Avec(A)
    return A_vec_psd


#A_vec to rotation quaternion
def A_vec_to_quat(A_vec):
    A = convert_Avec_to_A(A_vec)
    _, evs = torch.symeig(A, eigenvectors=True)
    return evs[:,0].squeeze()


#for bingham matrix
def matrix_bingham(A1,A2,A3):
    A2_trans = A2.transpose(1,2)
    #print(A2_trans.shape)
    #print(A3.shape)
    A3_inv = torch.inverse(A3)
    #print(A3_inv.shape)
    M1 = torch.bmm(A2_trans,A3_inv)
    #M1 = torch.bmm(A2.transpose(1,2),torch.inverse(A3))
    M2 = torch.bmm(M1,A2)
    B = A1-M2
    return B


#for gaussian matrix
def matrix_gaussian(A2,A3):
    G = -torch.bmm(torch.inverse(A3),A2)
    #G = torch.bmm(torch.inverse(A3),A2)
    return G


if __name__ == '__main__':
    output = torch.rand(3,36)
    A1 = output[:,0:10]
    A2 = output[:,10:26]
    A3 = output[:,26:36]
    A_1 = convert_Avec_to_A(A1)
    A_3 = convert_Avec_to_A(A3)
    A_2 = A2.reshape(A2.shape[0],4,4)
    #print(normalize_Avec(B_vec))
    #print(A_vec_to_quat(B_vec))
    
    B_m = matrix_bingham(A_1,A_2,A_3)
    B_vec = convert_A_to_Avec(B_m)
    print(B_vec)
    quat = A_vec_to_quat(B_vec) #generate quaternion
    print(quat[0,:].norm())
    print(quat[1,:].norm())
    print(quat[2,:].norm())

