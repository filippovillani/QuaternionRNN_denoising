import numpy as np
import torch
from scipy.stats import chi
import torch.nn as nn

def check_input(input):
    if input.dim() not in {2,3}:
        raise RuntimeError(
            'Quaternion linear accepts only input of dimension 2 or 3.'
            'input.dim = '+ str(input.dim())
            )

    nb_hidden = input.shape[-1]

    if nb_hidden % 4 != 0:
        raise RuntimeError(
            "Quaternion Tensors must be divisible by 4."
            " input.shape[-1] = " + str(nb_hidden)
        )

#%% GETTERS
# This getters will extract the dimension 2 which contains the four channels of the mic

def get_r(input):
    check_input(input)
    nb_hidden = input.size()[-1]
    if input.dim() == 2:
        return input.narrow(1,0,nb_hidden//4)
    elif input.dim() == 3:
        return input.narrow(2,0,nb_hidden//4)


def get_i(input):
    check_input(input)
    nb_hidden = input.size()[-1]
    if input.dim() == 2:
        return input.narrow(1,nb_hidden//4,nb_hidden//4)
    if input.dim() == 3:
        return input.narrow(2,nb_hidden//4,nb_hidden//4)

def get_j(input):
    check_input(input)
    nb_hidden = input.size()[-1]
    if input.dim() == 2:
        return input.narrow(1,nb_hidden//2,nb_hidden//4)
    if input.dim() == 3:
        return input.narrow(2,nb_hidden//2, nb_hidden//4)

def get_k(input):
    check_input(input)
    nb_hidden = input.size()[-1]
    if input.dim() == 2:
        return input.narrow(1,nb_hidden-nb_hidden//4, nb_hidden//4)
    if input.dim() == 3:
        return input.narrow(2,nb_hidden-nb_hidden//4,nb_hidden//4)
  


#%% Weigth initialization

def unitary_init(in_features,out_features,init_criterion):
    kernel_shape = (in_features,out_features)
    number_of_weights = np.prod(kernel_shape)
    v_r = np.random.uniform(-1.0,1.0,number_of_weights)
    v_i = np.random.uniform(-1.0,1.0,number_of_weights)
    v_j = np.random.uniform(-1.0,1.0,number_of_weights)
    v_k = np.random.uniform(-1.0,1.0,number_of_weights)
    
    for i in range(number_of_weights):
        norm = np.sqrt(v_r[i]**2+v_i[i]**2+v_j[i]**2+v_k[i]**2)+1e-4
        v_r[i]/=norm
        v_i[i]/=norm
        v_j[i]/=norm
        v_k[i]/=norm
    v_r = v_r.reshape(kernel_shape)
    v_i = v_i.reshape(kernel_shape)
    v_j = v_j.reshape(kernel_shape)
    v_k = v_k.reshape(kernel_shape)       
    return (v_r,v_i,v_j,v_k)

#%% This initialization is described in Quaternion Recurrent NNs pag.6

def quaternion_init(in_features,out_features,criterion='glorot'):
    num_neurons_in = in_features
    num_neurons_out = out_features
    kernel_shape = (in_features,out_features)

    
    if criterion == 'glorot':
        sigma = 1./np.sqrt(2*(num_neurons_in + num_neurons_out))
    elif criterion == 'he':
        sigma = 1./np.sqrt(2*num_neurons_in)
    else:
        raise ValueError('Invalid criterion: '+ criterion)
    
    # Generating random and purely imaginary quaternions
# Weigths follow a Chi-distribution with 4 degrees of freedom and mean value = 0

#Step 4
    theta = np.random.uniform(low=-np.pi,high=np.pi)
    
# Step 5    
    phi = chi.rvs(4,loc=0,scale=sigma,size=kernel_shape)

# Step 6
    number_of_weights = np.prod(kernel_shape)
    v_i = np.random.uniform(0.0,1.0,number_of_weights)
    v_j = np.random.uniform(0.0,1.0,number_of_weights)
    v_k = np.random.uniform(0.0,1.0,number_of_weights)

# Steps 7-8
    for i in range(number_of_weights):
        norm = np.sqrt(v_i[i]**2+v_j[i]**2+v_k[i]**2)+1e-4
        v_i[i]/=norm
        v_j[i]/=norm
        v_k[i]/=norm
    v_i = v_i.reshape(kernel_shape)
    v_j = v_j.reshape(kernel_shape)
    v_k = v_k.reshape(kernel_shape)
    
# Steps 9-10-11-12
    w_r = phi * np.cos(theta)
    w_i = phi * v_i * np.sin(theta)
    w_j = phi * v_j * np.sin(theta)
    w_k = phi * v_k * np.sin(theta)
    return (w_r,w_i,w_j,w_k)

#%% 
def affect_init(w_r,w_i,w_j,w_k,init_func,init_criterion='glorot'):
    if w_r.size() != w_i.size() or w_r.size() != w_j.size() or w_r.size() != w_k.size():
        raise ValueError('Real and imaginary weigths should have the same size. \
                         Found: r:'+str(w_r.size())
                         +'i:'+str(w_i.size())
                         +'j:'+str(w_j.size())
                         +'k:'+str(w_k.size()))
    elif w_r.dim()!=2:
        raise Exception('affect_init accepts only matrices. \
                        Found dimension ='+str(w_r.dim()))
    
    r, i, j, k  = init_func(w_r.size(0),w_r.size(1),init_criterion)
    r, i, j, k  = torch.from_numpy(r),torch.from_numpy(i),torch.from_numpy(j),torch.from_numpy(k)

    w_r.data = r.type_as(w_r.data)
    w_i.data = i.type_as(w_i.data)
    w_j.data = j.type_as(w_j.data)
    w_k.data = k.type_as(w_k.data)
                
#%% 
class QuaternionLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,input,w_r,w_i,w_j,w_k,bias=None):
        # save_for_backward must be used when saving input or output of the
        # forward phase to be used in the backward
        ctx.save_for_backward(input,w_r,w_i,w_j,w_k,bias)
        check_input(input)
        
        # Computes W*Inputs + bias, where * is the hamilton product.
       
        # Building the product as described in 'QNN for distant speech recognition'
        # Eq. (6)
        cat_kernels_4_r = torch.cat([w_r, -w_i, -w_j, -w_k], dim=0)
        cat_kernels_4_i = torch.cat([w_i,  w_r, -w_k, w_j], dim=0)
        cat_kernels_4_j = torch.cat([w_j,  w_k, w_r, -w_i], dim=0)
        cat_kernels_4_k = torch.cat([w_k,  -w_j, w_i, w_r], dim=0) 
        cat_kernels_4_quaternion = torch.cat([cat_kernels_4_r, cat_kernels_4_i, cat_kernels_4_j, cat_kernels_4_k], dim=1)
        # print('input.dim',input.dim())
        if input.dim() == 2:
            if bias is not None:
                # print('bias',bias.shape)
                # print('input',input.shape)
                # print('cat_ker',cat_kernels_4_quaternion.shape)
                # print('mm',torch.mm(input,cat_kernels_4_quaternion).shape)
                return torch.addmm(bias,input,cat_kernels_4_quaternion)
            else:
                return torch.mm(input,cat_kernels_4_quaternion) #mm doesn't broadcast
        else:
            output = torch.matmul(input,cat_kernels_4_quaternion) #this broadcasts
            if bias is not None:
                return output+bias
            else:
                return output
    
    # Backward is given as many tensors as there were outputs, with each of them
    # representing gradient w.r.t. that output.
    # It should return as many tensors as there were inputs, with each of them
    # representing gradient w.r.t its corresponding input
    
    # grad_output will be the loss function
    @staticmethod
    def backward(ctx,grad_output):
        # Unpacking save_tensors and initialize all gradients w.r.t. inputs to None
        input,w_r,w_i,w_j,w_k,bias = ctx.saved_tensors
        grad_input = grad_w_r = grad_w_i = grad_w_j = grad_w_k = grad_bias = None
        
        # Building quaternion weights matrix
        input_r = torch.cat([w_r, -w_i, -w_j, -w_k], dim=0)
        input_i = torch.cat([w_i,  w_r, -w_k, w_j], dim=0)
        input_j = torch.cat([w_j,  w_k, w_r, -w_i], dim=0)
        input_k = torch.cat([w_k,  -w_j, w_i, w_r], dim=0)
        cat_kernels_4_quaternion_T = torch.cat([input_r, input_i, input_j, input_k], dim=1).permute(1,0).requires_grad_(False)
        
        # Building quaternion input matrix
        r = get_r(input)
        i = get_i(input)
        j = get_j(input)
        k = get_k(input)
        input_r = torch.cat([r, -i, -j, -k], dim=0)
        input_i = torch.cat([i,  r, -k, j], dim=0)
        input_j = torch.cat([j,  k, r, -i], dim=0)
        input_k = torch.cat([k,  -j, i, r], dim=0)
        input_mat = torch.cat([input_r, input_i, input_j, input_k], dim=1).requires_grad_(False)
       
        # Building quaternion output gradient matrix 
        r = get_r(grad_output)
        i = get_i(grad_output)
        j = get_j(grad_output)
        k = get_k(grad_output)
        input_r = torch.cat([r, i, j, k], dim=1)
        input_i = torch.cat([-i,  r, k, -j], dim=1)
        input_j = torch.cat([-j,  -k, r, i], dim=1)
        input_k = torch.cat([-k,  j, -i, r], dim=1)
        grad_mat = torch.cat([input_r, input_i, input_j, input_k], dim=0)
        # print('grad_mat.shape',grad_mat.shape)
        
        if ctx.needs_input_grad[0]:
            grad_input  = torch.mm(grad_output,cat_kernels_4_quaternion_T)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_mat.permute(1,0).mm(input_mat).permute(1,0)
            unit_size_x = w_r.size(0)
            unit_size_y = w_r.size(1)
            grad_w_r = grad_weight.narrow(0,0,unit_size_x).narrow(1,0,unit_size_y)
            grad_w_i = grad_weight.narrow(0,0,unit_size_x).narrow(1,unit_size_y,unit_size_y)
            grad_w_j = grad_weight.narrow(0,0,unit_size_x).narrow(1,unit_size_y*2,unit_size_y)
            grad_w_k = grad_weight.narrow(0,0,unit_size_x).narrow(1,unit_size_y*3,unit_size_y)
        if ctx.needs_input_grad[5]:
            grad_bias   = grad_output.sum(0).squeeze(0)

        return grad_input, grad_w_r, grad_w_i, grad_w_j, grad_w_k, grad_bias
    
#%%
class QuaternionLinear(nn.Module):
    def __init__(self,input_features,output_features,bias=False,
                 init_criterion='glorot', weight_init='quaternion'):
        super(QuaternionLinear,self).__init__()
        self.input_features = input_features//4 
        self.output_features = output_features//4
        
        self.w_r = nn.Parameter(torch.empty(self.input_features,self.output_features))
        self.w_i = nn.Parameter(torch.empty(self.input_features,self.output_features))    
        self.w_j = nn.Parameter(torch.empty(self.input_features,self.output_features))
        self.w_k = nn.Parameter(torch.empty(self.input_features,self.output_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_features))
        else:
            self.register_parameter('bias',None)
        
        # weight initialization
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.reset_parameters()
    
    def reset_parameters(self):
        w_init = {'quaternion':quaternion_init,
                  'unitary': unitary_init}[self.weight_init]
        if self.bias is not None:
            self.bias.data.fill_(0)
        affect_init(self.w_r,self.w_i,self.w_j,self.w_k,w_init,self.init_criterion)
        
    def forward(self,input):
        if input.dim() == 3:
            T, N, C = input.size()
            input = input.reshape(T * N, C)
            output = QuaternionLinearFunction.apply(input, self.w_r, self.w_i, self.w_j, self.w_k, self.bias)
            output = output.reshape(T, N, output.size(1))
        elif input.dim() == 2:
            output = QuaternionLinearFunction.apply(input, self.w_r, self.w_i, self.w_j, self.w_k, self.bias)
        return output
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.input_features) \
            + ', out_features=' + str(self.output_features) \
            + ', bias=' + str(self.bias is not None) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init=' + str(self.weight_init) \

        
        
        
        
        
            
        