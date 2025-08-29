import qcqp
from qcqp import QuadQuatFastSover
import torch
import numpy as np
import torch.nn as nn
import symmatrix
import torchvision
import torch.nn.functional as F

def conv_box(in_channel,out_channel,kernel_size=3,stride=2,padding=1,batchnorm=True):
    if batchnorm:
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channel,out_channel,kernel_size=kernel_size,stride=stride,padding=padding),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.ReLU()
        )
    else:
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, out_channel,kernel_size=kernel_size,stride=stride,padding=padding),
            torch.nn.ReLU()
        )

def deconv_box(in_channels, out_channels,kernel_size=3,stride=2,padding=1,batchnorm=True):
    if batchnorm:
        return torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ConvTranspose2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        )
    else:
        return torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        )


class CNN_module(nn.Module): #继承了nn.Module可以直接使用_call_方法
    def __init__(self,dim_in, dim_out,normalize_output=True,batchnorm=True):
        super(CNN_module,self).__init__()
        self.normalize_output = normalize_output
        self.cnn = nn.Sequential(
            #conv_box(dim_in,64,kernel_size=3,stride=2,padding=1,batchnorm=batchnorm),
            conv_box(dim_in,64,kernel_size=5,stride=2,padding=1,batchnorm=batchnorm),
            # conv_box(32,64,kernel_size=3,stride=2,padding=1,batchnorm=batchnorm),
            conv_box(64,128,kernel_size=5,stride=2,padding=1,batchnorm=batchnorm),
            conv_box(128,256,kernel_size=5,stride=2,padding=1,batchnorm=batchnorm),
            conv_box(256,512,kernel_size=5,stride=2,padding=1,batchnorm=batchnorm),
            conv_box(512,1024,kernel_size=3,stride=2,padding=1,batchnorm=batchnorm),
            conv_box(1024,1024,kernel_size=3,stride=2,padding=1,batchnorm=batchnorm),
            conv_box(1024,2048,kernel_size=3,stride=2,padding=1,batchnorm=batchnorm),
            conv_box(2048,2048,kernel_size=3,stride=2,padding=1,batchnorm=batchnorm),
            #conv_box(2048,2048,kernel_size=3,stride=2,padding=1,batchnorm=batchnorm)
            #new added
            #conv_box(1024,1024,kernel_size=3,stride=2,padding=1,batchnorm=batchnorm)
        )
        self.fc = nn.Sequential(
            #nn.Linear(61440,4096),
            nn.Linear(20480,4096),
            #nn.Linear(12288,4096),
            #nn.BatchNorm1d(4096),
            #nn.ReLU(), #new added
            #nn.Dropout(0.3),
            nn.Linear(4096,1024),
            #nn.BatchNorm1d(1024),
            #nn.Dropout(0.5),
            #nn.ReLU(),
            #new added
            #nn.Linear(4096,1024),
            nn.Linear(1024,512),
            #nn.ReLU(),
            nn.Linear(512,dim_out)
        )
        '''
        self.coeff = nn.Sequential(
            nn.Linear(6144,1024),
            nn.BatchNorm1d(1024),
            nn.PReLU(),
            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Linear(512,self.num_coeff),
            nn.BatchNorm1d(self.num_coeff)
        )
        '''
        #self.fc1 = nn.Linear(4096,512)
        #self.relu = nn.ReLU()
        #self.fc2 = nn.Linear(512,dim_out)

    def forward(self,x):
        out = self.cnn(x)
        out = out.view(out.shape[0],-1)
        #TODO: show the shape of output vector
        #print(out.shape)
        #out = out.view(-1,16*1024*3*10)
        out = self.fc(out)
        #coeff = self.coeff(x)
        #coeff = F.relu(coeff)
        ''''
        if self.normalize_output:
            out = out/out.norm(dim=1).view(-1,1)
        '''
        return out

#for kitti dataset
class DualQuatNet(nn.Module):
    def __init__(self,enforce_psd = True,rot_net=True, trans_net = True, unit_frob_norm=True, dim_in=2, batchnorm=True):
        super(DualQuatNet,self).__init__()
        #self.net,self.coeff = CNN_module(dim_in=dim_in, dim_out=36,normalize_output=True,batchnorm=batchnorm)
        self.net = CNN_module(dim_in=dim_in, dim_out=36,normalize_output=True,batchnorm=batchnorm)
        self.enforce_psd = enforce_psd
        self.unit_forb_norm = unit_frob_norm
        #self.num_coeff = num_coeff
        self.rot_net = rot_net
        self.trans_net = trans_net
        self.qcqp_solver = QuadQuatFastSover.apply
        '''
        self.coeff = nn.Sequential(
            #nn.Linear(6144,4096),
            nn.Linear(20480,4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,self.num_coeff),
            nn.BatchNorm1d(self.num_coeff)
        )
        '''

    def output_A(self,x):
        A_vec = self.net(x)
        A1 = A_vec[:,0:10]
        A2 = A_vec[:,10:26]
        A3 = A_vec[:,26:36]
        A3 = symmatrix.convert_Avec_to_Avec_psd(A3)
        A_1 = symmatrix.convert_Avec_to_A(A1)
        A_3= -symmatrix.convert_Avec_to_A(A3)
        #print('A_3',A_3)
        A_2 = A2.reshape(A2.shape[0],4,4)
        B_mat = symmatrix.matrix_bingham(A_1,A_2,A_3)
        G_mat = symmatrix.matrix_gaussian(A_2,A_3)
        B_vec = symmatrix.convert_A_to_Avec(B_mat)
        if self.enforce_psd:
            B_vec = symmatrix.convert_Avec_to_Avec_psd(B_vec)
        if self.unit_forb_norm:
            B_vec = symmatrix.normalize_Avec(B_vec)
        return B_vec,G_mat
    #it works fine
    
    def forward(self,x):
        A_vec = self.net(x)
        if self.rot_net:
            A1 = A_vec[:,0:10]
            A2 = A_vec[:,10:26]
            A3 = A_vec[:,26:36]
            A3 = symmatrix.convert_Avec_to_Avec_psd(A3)
            A_1 = symmatrix.convert_Avec_to_A(A1)
            #A_2 = symmatrix.convert_Avec_to_A(A2)
            A_3 = -symmatrix.convert_Avec_to_A(A3)#negative definite
            A_2 = A2.reshape(A2.shape[0],4,4)
            if A_1.dim()<3:
                A_1 = A_1.unsqueeze(dim=0)
            if A_3.dim()<3:
                A_3 = A_3.unsqueeze(dim=0)
            B_mat = symmatrix.matrix_bingham(A_1,A_2,A_3)
            #print(B_mat)
            B_vec = symmatrix.convert_A_to_Avec(B_mat)
            if self.enforce_psd:
                B_vec = symmatrix.convert_Avec_to_Avec_psd(B_vec)
            if self.unit_forb_norm:
                B_vec = symmatrix.normalize_Avec(B_vec)
            q_r = self.qcqp_solver(B_vec)
        if self.trans_net:
            #A_vec = A_vec/A_vec.norm(dim=1).view(-1,1)#normalize output
            A1_t = A_vec[:,0:10]
            A2_t = A_vec[:,10:26]
            A3_t = A_vec[:,26:36]
            A3_t = symmatrix.convert_Avec_to_Avec_psd(A3_t)
            A_1_t = symmatrix.convert_Avec_to_A(A1_t)
            A_2_t = A2_t.reshape(A2_t.shape[0],4,4)
            #A_2_t = symmatrix.convert_Avec_to_A(A2_t)
            A_3_t = -symmatrix.convert_Avec_to_A(A3_t)#negative definite
            if A_1_t.dim()<3:
                A_1_t= A_1_t.unsqueeze(dim=0)
            if A_3_t.dim()<3:
                A_3_t = A_3_t.unsqueeze(dim=0)
            B_mat_t = symmatrix.matrix_bingham(A_1_t,A_2_t,A_3_t)
            G_mat_t = symmatrix.matrix_gaussian(A_2_t,A_3_t)
            #print(G_mat_t)
            B_vec_t = symmatrix.convert_A_to_Avec(B_mat_t)
            if self.enforce_psd:
                B_vec_t = symmatrix.convert_Avec_to_Avec_psd(B_vec_t)
            if self.unit_forb_norm:
                B_vec_t = symmatrix.normalize_Avec(B_vec_t)
            q_r_t = self.qcqp_solver(B_vec_t)
            q_s= qcqp.generate_dual_part(G_mat_t,q_r_t)

        return q_r,q_s#,B_mat, G_mat_t


class DualQuatEncoder(nn.Module):
    def __init__(self,dim_in,dim_latent,dim_transition,batchnorm=False):
        super(DualQuatEncoder,self).__init__()
        self.cnn = nn.Sequential(
            conv_box(dim_in,64,kernel_size=3,stride=2,padding=1,batchnorm=batchnorm),
            conv_box(64,128,kernel_size=3,stride=2,padding=1,batchnorm=batchnorm),
            conv_box(128,256,kernel_size=3,stride=2,padding=1,batchnorm=batchnorm),
            conv_box(256,512,kernel_size=3,stride=2,padding=1,batchnorm=batchnorm),
            conv_box(512,1024,kernel_size=3,stride=2,padding=1,batchnorm=batchnorm),
            conv_box(1024,1024,kernel_size=3,stride=2,padding=1,batchnorm=batchnorm),
            conv_box(1024,1024,kernel_size=3,stride=2,padding=1,batchnorm=batchnorm)
        )
        self.fc_encoder=nn.Sequential(
            nn.Linear(4096,dim_transition),
            nn.ReLU(),
            nn.Linear(dim_transition,dim_latent)
        )
        self.fc_decoder=nn.Sequential(
            nn.ReLU(),
            nn.Linear(dim_latent,dim_transition),
            nn.ReLU(),
            nn.Linear(dim_transition,4096)
        )
        self.cnn_decoder = nn.Sequential(
            deconv_box(1024,1024,kernel_size=3,stride=2,padding=1,batchnorm=batchnorm),
            deconv_box(1024,1024,kernel_size=3,stride=2,padding=0,batchnorm=batchnorm),
            deconv_box(1024,512,kernel_size=2,stride=2,padding=0,batchnorm=batchnorm),
            deconv_box(512,256,kernel_size=2,stride=2,padding=0,batchnorm=batchnorm),
            deconv_box(256,128,kernel_size=2,stride=2,padding=0,batchnorm=batchnorm),
            deconv_box(128,64,kernel_size=2,stride=2,padding=0,batchnorm=batchnorm),
            deconv_box(64,dim_in,kernel_size=2,stride=2,padding=0,batchnorm=batchnorm)
        )
    
    def encode(self,x):
        code_out = self.cnn(x)
        #print("cnn_out:", code_out.shape)
        code_out = code_out.view(code_out.shape[0],-1)
        #print(code_out.shape)
        code_out = self.fc_encoder(code_out)
        #print("fc_out:", code_out.shape)
        return code_out
    
    def decode(self,x):
        decode_out = self.fc_decoder(x)
        #print("before decode_out: ", decode_out.shape)
        decode_out = decode_out.view(-1,1024,2,2)
        #print("decode_out: ", decode_out.shape)
        decode_out = self.cnn_decoder(decode_out)
        #print("cnn_decoder:", decode_out.shape)
        return decode_out
    
    def forward(self,x):
        out = self.encode(x)
        out = self.decode(out)
        return out

class ResNet(torch.nn.Module):
    def __init__(self,dim_out,normalize_output = False):
        super(ResNet,self).__init__()
        self.cnn = torchvision.models.resnet50(pretrained=True)
        self.num_ftrs = self.cnn.fc.in_features
        self.normalize_output = normalize_output
    
    def forward(self,x):
        y = self.cnn(x)
        if self.normalize_output:
            y = y/y.norm(dim=1).view(-1,1)
        return y
    
    def freeze_layers(self):
        for param in self.cnn.parameters():
            param.requires_grad=False
        for param in self.cnn.fc.parameters():
            param.requires_grad=True


class DualQuatResNet(torch.nn.Module):
    def __init__(self,num_coeff=2,enforce_psd=True,unit_forb_norm=True,rot_net = True, trans_net= True):
        super(DualQuatResNet,self).__init__()
        self.net = ResNet(dim_out=36,normalize_output=False)
        self.enforce_psd = enforce_psd
        self.unit_forb_norm = unit_forb_norm
        self.qcqp_solver = QuadQuatFastSover.apply
        self.rot_net = rot_net
        self.trans_net = trans_net
        self.num_coeff = num_coeff
        self.coeff = nn.Sequential(
            #nn.Linear(6144,4096),
            nn.Linear(20480,4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,self.num_coeff),
            nn.BatchNorm1d(self.num_coeff)
        )
    
        
    def forward(self,x):
        A_vec = self.net(x)
        if self.rot_net:
            A1 = A_vec[:,0:10]
            A2 = A_vec[:,10:26]
            A3 = A_vec[:,26:36]
            A3 = symmatrix.convert_Avec_to_Avec_psd(A3)
            A_1 = symmatrix.convert_Avec_to_A(A1)
            #A_2 = symmatrix.convert_Avec_to_A(A2)
            A_3 = symmatrix.convert_Avec_to_A(A3)#negative definite
            A_2 = A2.reshape(A2.shape[0],4,4)
            if A_1.dim()<3:
                A_1 = A_1.unsqueeze(dim=0)
            if A_3.dim()<3:
                A_3 = A_3.unsqueeze(dim=0)
            B_mat = symmatrix.matrix_bingham(A_1,A_2,A_3)
            B_vec = symmatrix.convert_A_to_Avec(B_mat)
            if self.enforce_psd:
                B_vec = symmatrix.convert_Avec_to_Avec_psd(B_vec)
            if self.unit_forb_norm:
                B_vec = symmatrix.normalize_Avec(B_vec)
            q_r = self.qcqp_solver(B_vec)
        if self.trans_net:
            A_vec = A_vec/A_vec.norm(dim=1).view(-1,1)#normalize output
            A1_t = A_vec[:,0:10]
            A2_t = A_vec[:,10:26]
            A3_t = A_vec[:,26:36]
            A3_t = symmatrix.convert_Avec_to_Avec_psd(A3_t)
            A_1_t = symmatrix.convert_Avec_to_A(A1_t)
            A_2_t = A2_t.reshape(A2_t.shape[0],4,4)
            #A_2_t = symmatrix.convert_Avec_to_A(A2_t)
            A_3_t = symmatrix.convert_Avec_to_A(A3_t)#negative definite
            if A_1_t.dim()<3:
                A_1_t= A_1_t.unsqueeze(dim=0)
            if A_3_t.dim()<3:
                A_3_t = A_3_t.unsqueeze(dim=0)
            B_mat_t = symmatrix.matrix_bingham(A_1_t,A_2_t,A_3_t)
            G_mat_t = symmatrix.matrix_gaussian(A_2_t,A_3_t)
            B_vec_t = symmatrix.convert_A_to_Avec(B_mat_t)
            if self.enforce_psd:
                B_vec_t = symmatrix.convert_Avec_to_Avec_psd(B_vec_t)
            if self.unit_forb_norm:
                B_vec_t = symmatrix.normalize_Avec(B_vec_t)
            q_r_t = self.qcqp_solver(B_vec_t)
            q_s= qcqp.generate_dual_part(G_mat_t,q_r_t)

        coeff = self.net.cnn(x)
        coeff = coeff.view(coeff.shape[0],-1)
        loss_coeff = self.coeff(coeff)
        #loss_coeff = F.relu(loss_coeff)
        loss_coeff = F.softmax(loss_coeff)
        #print(loss_coeff)
        return q_r,q_s#,loss_coeff[:,0],loss_coeff[:,1]