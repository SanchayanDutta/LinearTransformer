###########################################
# This file contains the following:
# 1. Linear Transformer Model
# 2. Function for clipping gradient
# 3. Function for generating random data
#
# The notation for linear attention follows
# the paper at https://arxiv.org/pdf/2306.00297.pdf
###########################################


import torch
from torch import nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definition of a single linear attention unit for linear-regression data
# P is the value matrix
# Q is the product of key,query matrices
# the dimensions of the input are
# B: batch-size of prompts
# N: context length (excluding query)
# d: covariate dimension
# P,Q are d x d matrices
# Z is a B x (N+1) + (d+1) matrix
# Output is also B x (N+1) + (d+1)

# For linear attention, activation = None
# For standard attention, activation(x) = torch.nn.functional.softmax(x, dim = 2)
# For ReLU attention, activation(x) = torch.nn.relu(x)
'''
def attention(P,Q,Z, activation = None):
    B= Z.shape[0]
    N = Z.shape[1]-1
    d = Z.shape[2]-1
    P_full =  torch.cat([P,torch.zeros(1,d).to(device)],dim=0)
    P_full =  torch.cat([P_full,torch.zeros(d+1,1).to(device)],dim=1)
    P_full[d,d] = 1
    Q_full = torch.cat([Q, torch.zeros(1,d).to(device)],dim=0)
    Q_full = torch.cat([Q_full, torch.zeros(d+1,1).to(device)],dim=1)
    A = torch.eye(N+1).to(device)
    A[N,N] = 0
    Attn = torch.einsum('BNi, ij, BMj -> BNM', (Z,Q_full,Z))
    if activation is not None:
        Attn = activation(Attn)
    Attn =torch.nn.functional.softmax(Attn, dim = 2)
    key = torch.einsum('ij, BNj -> BNi', (P_full,Z))
    Output = torch.einsum('BNM,ML, BLi -> BNi', (Attn,A,key))
    return Output /N
'''
def softmax_activation(mask, Attn):
    #Attn = torch.einsum('BNi, BMi -> BNM', (QZ,KZ))
    Attn = Attn.masked_fill(mask, float('-inf'))
    return torch.nn.functional.softmax(Attn, dim = 1)

def relu_activation(mask, Attn):
    #Attn = torch.einsum('BNi, BMi -> BNM', (QZ,KZ))
    Attn = Attn.masked_fill(mask, float('0'))
    return torch.nn.functional.relu(Attn)

def exp_activation(mask, Attn):
    #Attn = torch.einsum('BNi, BMi -> BNM', (QZ,KZ))
    #normQZ = QZ.norm(p=2,dim=2)**2
    #normKZ = KZ.norm(p=2,dim=2)**2
    #Attn = Attn - 0.5 * normQZ[:,:,None]
    #Attn = Attn - 0.5 * normKZ[:,None,:]
    Attn = torch.exp(Attn)
    Attn = Attn.masked_fill(mask, float('0'))
    return Attn

def linear_activation(mask, Attn):
    #Attn = torch.einsum('BNi, BMi -> BNM', (QZ,KZ))
    Attn = Attn.masked_fill(mask, float('0'))
    return Attn

def attention(P,Q,K,Z,mask, activation = 'exp'):
    B= Z.shape[0]
    N = Z.shape[1]-1
    d = Z.shape[2]-1
    QK = torch.einsum('ji,jk->ik', (Q,K))
    Attn = torch.einsum('BNi, ij, BMj -> BNM', (Z,QK,Z))
    #QZ = torch.einsum('ij, BNj -> BNi', (Q,Z))
    #KZ = torch.einsum('ij, BNj -> BNi', (K,Z))
    PZ = torch.einsum('ij, BNj -> BNi', (P,Z))
    
    if activation == 'exp':
        Attn = exp_activation(mask, Attn)
    elif activation == 'linear':
        Attn = linear_activation(mask, Attn)
    elif activation == 'relu':
        Attn = relu_activation(mask, Attn)
    elif activation == 'softmax':
        Attn = softmax_activation(mask, Attn)
    else:
        assert(False)
    Output = torch.einsum('BNi, BNM -> BMi', (PZ,Attn))
    return Output /N



# The Linear Transformer module
# n_layer denotes the number of layers
# n_head denotes the number of heads. In most of our experiments, n_head = 1
# d denotes the dimension of covariates
# var denotes the variance of initialization. It needs to be sufficiently small, but exact value is not important
# allparam: contains all the parameters, has dimension n_layer x n_head x 2 x d x d
# For example
# - P matrix at layer i, head j is allparam[i,j,0,:,:]
# - Q matrix at layer i, head j is allparam[i,j,1,:,:]
class Transformer_F(nn.Module):
    def __init__(self, n_layer, n_head, d, var, N=20):
        super(Transformer_F, self).__init__()
        self.register_parameter('allparam', torch.nn.Parameter(torch.zeros(n_layer, n_head, 3, d+1, d+1)))
        with torch.no_grad():
            self.allparam.normal_(0,var)
        self.n_layer = n_layer
        self.n_head = n_head
        self.register_buffer('mask',torch.zeros([1,N+1,N+1], dtype=torch.bool))
        self.mask[:,-1, :] = True
        self.register_buffer('param_mask',torch.zeros([1,1,3,d+1,d+1], dtype=torch.bool))
        self.param_mask[0,0,0,d,:d]=True
        self.param_mask[0,0,0,:d,d]=True
        self.param_mask[0,0,1,:d+1,d]=True
        self.param_mask[0,0,1,d,:d+1]=True
        self.param_mask[0,0,2,:d+1,d]=True
        self.param_mask[0,0,2,d,:d+1]=True
        

    def forward(self, Z, activation):
        for i in range(self.n_layer):
            Zi = Z
            residues = 0
            # the forwarad map of each layer is given by F(Z) = Z + attention(Z)
            for j in range(self.n_head):
                Pij = self.allparam[i,j,0,:,:]
                Qij = self.allparam[i,j,1,:,:]
                Kij = self.allparam[i,j,2,:,:]
                residues = residues + attention(Pij,Qij,Kij, Zi, self.mask, activation)
            Z = Zi + residues
            
        if Z.norm() > 1e10:
            Z = 1e10* Z/Z.norm()
        return Z
    
    def zero_row_col(self):
        with torch.no_grad():
            self.allparam.data = self.allparam.data.masked_fill(self.param_mask, 0)
    #enforces top-left-dxd-block sparsity on p
    def zero_block_P(self):
        d = self.allparam.shape[4]
        for i in range(self.n_layer):
            for j in range(self.n_head):
                with torch.no_grad():
                    self.allparam[i,j,0,:d-1,:d-1].zero_()

# evaluate the loss of model, given data (Z,y)
def in_context_loss(model, Z, y, activation):
    N = Z.shape[1]-1
    d = Z.shape[2]-1
    output = model(Z, activation)
    diff = output[:,N,d]+y
    loss = ((diff)**2).mean() 
    return loss

def euclidean_kernel(X):
    kernel_mat = torch.einsum('BNi,BMi->BNM', (X,X))
    return kernel_mat

def relu_kernel(X):
    kernel_mat = torch.einsum('BNi,BMi->BNM', (X,X))
    return torch.nn.functional.relu(kernel_mat)

def exp_kernel(X,sigma=1):
    kernel_mat = torch.einsum('BNi,BMi->BNM', (X,X))
    return torch.exp(1/(2*sigma**2)*kernel_mat)

def combination_kernel(X,sigma=1):
    kernel_mat0 = torch.einsum('BNi,BMi->BNM', (X[:,:,0:2],X[:,:,0:2]))
    kernel_mat1 = torch.einsum('BNi,BMi->BNM', (X[:,:,2:-1],X[:,:,2:-1]))
    return torch.exp(1/(2*sigma**2)*kernel_mat0) + kernel_mat1

def generate_data_inplace(Z, kernel, U=None, D=None):
    B = Z.shape[0]
    N = Z.shape[1]-1
    d = Z.shape[2]-1
    X = Z[:,:,0:-1]
    X.normal_(0, 1).cuda()
    
    X.div_(X.norm(p=2,dim=2)[:,:,None])
    
    #W= torch.FloatTensor(B, d).normal_(0,1).cuda()
    #Z[:,:,-1] = torch.einsum('bi,bni->bn', (W, Z[:,:,0:-1])) #y update
    # new sampling scheme for y
    if kernel=='euclidean':
        kernel_matrices = euclidean_kernel(X) + torch.eye(N+1,N+1).unsqueeze(0).cuda() * 1e-8 #regularization for cholesky
    elif kernel=='exp':
        kernel_matrices = exp_kernel(X)
    elif kernel=='relu':
        kernel_matrices = relu_kernel(X)
    elif kernel=='comb':
        kernel_matrices = combination_kernel(X)
    else:
        assert False
    L, Q = torch.linalg.eigh(kernel_matrices)
    #if torch.min(L)<0:
    #    print(min(L))
    #    assert False
    Z[:,:,-1].normal_(0,1)
    Z[:,:,-1] = torch.einsum('BNM,BM,BM-> BN', (Q, L.abs()**0.5, Z[:,:,-1]))
    
    #cholesky_matrices = torch.linalg.cholesky(kernel_matrices, upper=False)
    #Z[:,:,-1].normal_(0,1)
    #Z[:,:,-1] = torch.einsum('BM,BNM-> BN', (Z[:,:,-1], cholesky_matrices))
    
    #print((torch.einsum('BLN, BMN-> BLM', (cholesky_matrices,cholesky_matrices))-kernel_matrices).norm())
    
    y_test = Z[:,-1,-1].detach().clone()
    Z[:,-1,-1].zero_()
    if U is not None:
        U = U.to(device)
        D = D.to(device)
        Z[:,:,0:-1] = torch.einsum('ij, jk, BNk -> BNi', (U,D,X))
    return Z.to(device),y_test.to(device)

def bayes_prediction(Z, kernel, U, D):
    B = Z.shape[0]
    N = Z.shape[1]-1
    d = Z.shape[2]-1
    #X = torch.einsum('ij, kj, BNk -> BNi', (torch.diag(torch.diag(D)**(-1)), U, Z[:,:,0:-1]))
    X = Z[:,:,0:-1]
    y = Z[:,0:-1,-1]
    
    if kernel=='euclidean':
        kernel_matrices = euclidean_kernel(X) + torch.eye(N+1,N+1).unsqueeze(0).cuda() * 1e-4 #regularization for cholesky
    elif kernel=='exp':
        kernel_matrices = exp_kernel(X) 
    elif kernel=='relu':
        kernel_matrices = relu_kernel(X)
    elif kernel=='comb':
        kernel_matrices = combination_kernel(X)
    else:
        assert False
        
    
    L, Q = torch.linalg.eigh(kernel_matrices)
    kernel_matrices = torch.einsum('BNM,BM,BOM-> BNO', (Q, L.abs(), Q))
    v = kernel_matrices[:,0:-1,-1]
    #Kinv = torch.einsum('BNM,BM,BOM-> BNO', (Q, L.abs()**(-1), Q))
    
    #Kinv = torch.linalg.pinv(kernel_matrices[:,0:-1,0:-1])
    #bayes_pred = torch.einsum('Bi,Bij,Bj->B',(v,Kinv,y))
    
    tt = torch.linalg.solve(kernel_matrices[:,0:-1,0:-1],y)
    bayes_pred = torch.einsum('Bi,Bi->B',(v,tt))
    
    return bayes_pred

def bayes_loss(Z, y, activation, U, D):
    N = Z.shape[1]-1
    d = Z.shape[2]-1
    output = bayes_prediction(Z, activation, U, D)
    diff = output-y
    loss = ((diff)**2).mean() 
    
    #print(diff)
    return loss

class Transformer_LN_MLP(nn.Module):
    def __init__(self, n_layer, n_head, d, var, N=20):
        super(Transformer_LN_MLP, self).__init__()
        self.register_parameter('allparam', torch.nn.Parameter(torch.zeros(n_layer, n_head, 3, d+1, d+1)))
        with torch.no_grad():
            self.allparam.normal_(0,var)
        self.n_layer = n_layer
        self.n_head = n_head
        self.register_buffer('mask',torch.zeros([1,N+1,N+1], dtype=torch.bool))
        self.mask[:,-1, :] = True
        self.register_buffer('param_mask',torch.zeros([1,1,3,d+1,d+1], dtype=torch.bool))
        self.param_mask[0,0,0,d,:d]=True
        self.param_mask[0,0,0,:d,d]=True
        self.param_mask[0,0,1,:d+1,d]=True
        self.param_mask[0,0,1,d,:d+1]=True
        self.param_mask[0,0,2,:d+1,d]=True
        self.param_mask[0,0,2,d,:d+1]=True
        assert n_head==1
        self.register_parameter('ln_weights', torch.nn.Parameter(torch.ones(n_layer, n_head, 2, d+1)))
        self.register_parameter('mlp_weights', torch.nn.Parameter(torch.zeros(n_layer, n_head, 2, d+1,d+1)))
        
        
    def forward(self, Z, activation):
        for i in range(self.n_layer):
            Zi = torch.nn.functional.layer_norm(Z, self.ln_weights[i,0,0].shape, self.ln_weights[i,0,0], None, 1e-5)
            residues = 0
            # the forwarad map of each layer is given by F(Z) = Z + attention(Z)
            j=0
            Pij = self.allparam[i,j,0,:,:]
            Qij = self.allparam[i,j,1,:,:]
            Kij = self.allparam[i,j,2,:,:]
            Ai = attention(Pij,Qij,Kij, Zi, self.mask, activation)
            
            Mi = torch.nn.functional.layer_norm(Z, self.ln_weights[i,0,1].shape, self.ln_weights[i,0,1], None, 1e-5)
            Mi = torch.nn.functional.linear(Mi, self.mlp_weights[i,0,0],None)
            Mi = torch.nn.functional.gelu(Mi)
            Mi = torch.nn.functional.linear(Mi, self.mlp_weights[i,0,1],None)
            residues = residues + Ai + Mi
            Z = Zi + residues
            
        if Z.norm() > 1e10:
            Z = 1e10* Z/Z.norm()
        return Z
    
    def zero_row_col(self):
        with torch.no_grad():
            self.allparam.data = self.allparam.data.masked_fill(self.param_mask, 0)
            
class Transformer_C(nn.Module):
    def __init__(self, n_layer, n_head, d, var, N=20):
        super(Transformer_C, self).__init__()
        self.register_parameter('allparam', torch.nn.Parameter(torch.zeros(n_layer, n_head, 3, d+1, d+1)))
        with torch.no_grad():
            self.allparam.normal_(0,var)
        self.n_layer = n_layer
        self.n_head = n_head
        self.register_buffer('mask',torch.zeros([1,N+1,N+1], dtype=torch.bool))
        self.mask[:,-1, :] = True
        self.register_buffer('param_mask',torch.zeros([1,1,3,d+1,d+1], dtype=torch.bool))
        self.param_mask[0,0,0,d,:d]=True
        self.param_mask[0,0,0,:d,d]=True
        self.param_mask[0,0,1,:d+1,d]=True
        self.param_mask[0,0,1,d,:d+1]=True
        self.param_mask[0,0,2,:d+1,d]=True
        self.param_mask[0,0,2,d,:d+1]=True
        assert n_head == 2
        

    def forward(self, Z, activation):
        #ignore activation
        for i in range(self.n_layer):
            Zi = Z
            residues = 0
            # the forwarad map of each layer is given by F(Z) = Z + attention(Z)
            j=0
            Pij = self.allparam[i,j,0,:,:]
            Qij = self.allparam[i,j,1,:,:]
            Kij = self.allparam[i,j,2,:,:]
            residues = residues + attention(Pij,Qij,Kij, Zi, self.mask, 'exp')
            j=1
            Pij = self.allparam[i,j,0,:,:]
            Qij = self.allparam[i,j,1,:,:]
            Kij = self.allparam[i,j,2,:,:]
            residues = residues + attention(Pij,Qij,Kij, Zi, self.mask, 'linear')
            Z = Zi + residues
            
        if Z.norm() > 1e10:
            Z = 1e10* Z/Z.norm()
        return Z
    
    def zero_row_col(self):
        with torch.no_grad():
            self.allparam.data = self.allparam.data.masked_fill(self.param_mask, 0)