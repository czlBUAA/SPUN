#逐一排查
#验证torch.symeig()能否自动反向求导，如果可以验证Magnus文章公式计算的结果和自动求导结果是否一致  ---可以自动反向求导，且特征值对vec(特征矩阵)的导数与理论结果一样
#在姿态偏差是5°，位置偏差是0.5m的情况下算一下我的loss的绝对值有多大。   ---还未验证   去matlab验证
#B矩阵有必要强制正定和F范数的归一化吗？
#在平移分支时，生成伪旋转qr，为啥要对网络输出先归一化呢？
#我的qr, qd有没有编码正确。
#验证我的混合分布归一化常数是否被正确计算
#四元数的标量是在第几维（x,y,z,w)?
#输入的图像是否裁剪？？和ursonet一样 使用def mold_inputs(self, images)函数
import torch
import numpy as np
import caculatation_nc
import symmartix
from dataset import SPEdataset
import torchvision.transforms as transforms
import QCQP
import utils
import math
import bingham_distribution as ms
from torchvision.models import resnet50
import net
# 设置随机种子以便复现
# torch.manual_seed(0)


# def map_to_symmetric_matrix(a):
#     # 使用 torch.tensor 创建对称矩阵
#     A = a.new_zeros(2,2)
#     A[0,0]=a[0]
#     A[0,1]=a[1]
#     A[1,0]=a[1]
#     A[1,1]=a[2]
#     return A

# # 定义损失函数
# def loss_fn(a):
#     A = map_to_symmetric_matrix(a)  # 在损失函数内计算 A
#     # 进行特征分解
#     eigenvalues, eigenvectors = torch.linalg.eigh(A)
#     # 获取最小特征值对应的特征向量
#     min_eigenvector = eigenvectors[:, 0]
#     # 返回特征向量的均值作为损失
#     return min_eigenvector.mean()
# def loss_fn1(a):
#     A = map_to_symmetric_matrix(a)  # 在损失函数内计算 A
#     # 进行特征分解
#     eigenvalues, eigenvectors = torch.linalg.eigh(A)
#     # 返回特征值的平均作为损失
#     return eigenvalues[0], eigenvectors[:, 0]
# a = torch.randn(3, requires_grad=True)
# print(a)
# # 计算损失
# loss, eigenvectors= loss_fn1(a)

# # 反向传播
# loss.backward()

# # 输出损失值和关于 a 的梯度
# print("Gradient of a:\n", a.grad)
# print(eigenvectors)
def skew(X):
    return torch.tensor([[0, -X[2], X[1]],
                         [X[2], 0, -X[0]],
                         [-X[1], X[0], 0]], dtype=X.dtype)
def Qmultiply(X, Y):
    skew_X = skew(X[:3])
    term1 = torch.matmul(skew_X, Y[:3]) + X[3] * Y[:3] + Y[3] * X[:3]
    term2 = X[3] * Y[3] - torch.dot(X[:3], Y[:3])
    term2 = torch.tensor(term2.item(), dtype=X.dtype).unsqueeze(0)
    return torch.cat((term1, torch.tensor(term2, dtype=X.dtype)), dim=0)
root = "./UE4/SPE"
transform = transforms.Compose([
        transforms.CenterCrop((960, 960)),
        transforms.ToTensor()
        ])
train_dataset = SPEdataset(root, "train", transform)
val_dataset = SPEdataset(root, "val", transform)
validationloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=2, shuffle=True)
trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=True)
A_vec = torch.randn((2,36), requires_grad=True)
A1 = A_vec[:,0:10]
A2 = A_vec[:,10:26]
A3 = A_vec[:,26:36]
A3 = symmartix.convert_Avec_to_Avec_psd(A3)
A_1 = symmartix.convert_Avec_to_A(A1)
A_3 = -symmartix.convert_Avec_to_A(A3)#negative definite
A_2 = A2.reshape(A2.shape[0],4,4)
B_mat = symmartix.matrix_bingham(A_1,A_2,A_3)
G_mat = symmartix.matrix_gaussian(A_2,A_3)
G_cov = A_3
q_r,nu = QCQP.solve_waha_fast(B_mat)
q_d = QCQP.generate_dual_part(G_mat,q_r)
q = torch.tensor([0.0454372, 0.0416356, 0.0454372, 0.9970644])
q_r1 = torch.zeros((2,4))
q_r1[0] =  Qmultiply(q_r[0], q)
q_r1[1] = Qmultiply(q_r[1], q)
q_d1 = torch.zeros((2,4))
q_d1 = q_d+ 0.5
Z = torch.zeros((B_mat.shape[0], 4))
for i in range(B_mat.shape[0]):
        vals, _ = torch.linalg.eigh(B_mat[i])
        sorted_vals, _ = torch.sort(vals, descending=True)
        Z[i, :] = sorted_vals
nc_options = {"epsrel": 1e-3, "epsabs": 1e-7}
print(Z)
Z = np.array([-300, -200, -200, 0]) #Z对角元素的相反数等于Maximum likelihood estimation of the Fisher–Bingham distribution文章里的theta。theta要满足一些条件，参加文章的Notation1
#Z = np.flip(Z, axis=0)
caculatation_nc.t0 = - np.max(Z)-2
n1 = caculatation_nc.nc(Z)
n2 = ms.BinghamDistribution.normalization_constant(Z,"numerical", nc_options)  
n11 = caculatation_nc.nc_der(Z, 1)
n12 = caculatation_nc.nc_der(Z, 2)
n13 = caculatation_nc.nc_der(Z, 3)
n14 = caculatation_nc.nc_der(Z, 4)
nder = ms.BinghamDistribution.normalization_constant_deriv(Z, "default")
print("nc is", n1, n2)
print("der is", n11,n12,n13,n14, nder)
# likelihoods = (torch.bmm(torch.bmm(
#                 q_r1.unsqueeze(1),
#                 B_mat),
#                 q_r1.unsqueeze(2))).squeeze()+ (torch.bmm(torch.bmm(
#                 (q_d1-q_d).unsqueeze(1),
#                 G_cov),
#                 (q_d1-q_d).unsqueeze(2))).squeeze() + torch.log(norm_const)- torch.log(2 * math.pi * torch.sqrt( torch.linalg.det(torch.linalg.inv(-0.5 * G_cov))))
# print('似然概率是', likelihoods)
# eigenvalues, eigenvectors = torch.linalg.eigh(B_mat)
# print("origanial", B_mat)
# print(eigenvectors)
# B_vec = symmartix.convert_A_to_Avec(B_mat)
# #B_vec = symmartix.convert_Avec_to_Avec_psd(B_vec) #半正定  ---特征值都是非负
# B_vec = symmartix.normalize_Avec(B_vec)    # 强制归一化  --为啥呢？
# B = symmartix.convert_Avec_to_A(B_vec)
# eigenvalues, eigenvectors = torch.linalg.eigh(B)
# print("psd/nor", B)
# print(eigenvectors)
# A = torch.tensor([[1, 2, 3, 4],
#               [2, 5, 6, 7],
#               [3, 6, 9, 12],
#               [4, 7, 12, 11]], dtype=torch.float)
# rank = torch.matrix_rank(A)
# eigenvalues, eigenvectors = torch.linalg.eigh(B_mat)
# eigenvectors1, eigenvalues1  = QCQP.solve_waha_fast(B_mat)
# print(eigenvectors[:,:,0],eigenvectors1, rank.item())
# print(eigenvalues)



