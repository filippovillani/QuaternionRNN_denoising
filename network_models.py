import torch
import torch.nn as nn
from quaternion_ops import QuaternionLinear
import numpy as np

# This function is used for Bidirectional (Q)LSTM
def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

#%%
# Input shape: [time_step,batch,channel]
class QLSTM(nn.Module):
    def __init__(self, feat_size, hidden_size, output_size, bidirectional,bias,CUDA):
        super(QLSTM, self).__init__()
        
        # Reading options:
        self.act        = nn.Tanh()
        self.act_gate   = nn.Sigmoid()
        self.input_size  = feat_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.CUDA       = CUDA
        
        # Gates initialization
        self.wfx  = QuaternionLinear(self.input_size, self.hidden_size) # Forget
        self.ufh  = QuaternionLinear(self.hidden_size, self.hidden_size, bias=bias) # Forget
        
        self.wix  = QuaternionLinear(self.input_size, self.hidden_size) # Input
        self.uih  = QuaternionLinear(self.hidden_size, self.hidden_size, bias=bias) # Input  
        
        self.wox  = QuaternionLinear(self.input_size, self.hidden_size) # Output
        self.uoh  = QuaternionLinear(self.hidden_size, self.hidden_size, bias=bias) # Output
        
        self.wcx  = QuaternionLinear(self.input_size, self.hidden_size) # Cell 
        self.uch  = QuaternionLinear(self.hidden_size, self.hidden_size, bias=bias) # Cell 

        # Output layer initialization

        
        if self.bidirectional:
            self.fco  = nn.Linear(self.hidden_size*2, self.output_size)
        else:
            self.fco  = nn.Linear(self.hidden_size, self.output_size)
               
    
    def forward(self, x):
        if self.bidirectional:
            h_init = torch.zeros(2*x.shape[1],self.hidden_size)
            x = torch.cat([x,flip(x,0)],1)
        else:
            h_init = torch.zeros(x.shape[1],self.hidden_size)  
                     
        if self.CUDA:
            x = x.to('cuda')
            h_init = h_init.to('cuda')
                 
        # Feed-forward affine transformation
        wfx_out=self.wfx(x)
        wix_out=self.wix(x)
        wox_out=self.wox(x)
        wcx_out=self.wcx(x)
          
        # Processing time steps
        hiddens = []
        
        c=h_init
        h=h_init
        
        for k in range(x.shape[0]):
            ft=self.act_gate(wfx_out[k]+self.ufh(h))
            it=self.act_gate(wix_out[k]+self.uih(h))
            ot=self.act_gate(wox_out[k]+self.uoh(h))
                  
            at = wcx_out[k]+self.uch(h)       
            c = it*self.act(at)+ft*c
            h = ot*self.act(c)            
            hiddens.append(h)  
        h = torch.stack(hiddens)
        if self.bidirectional:
            h_f = h[:,0:int(x.shape[1]/2)] # forward
            h_b = flip(h[:,int(x.shape[1]/2):x.shape[1]].contiguous(),0) # backward
            h = torch.cat([h_f,h_b],2) 
            
        output = self.fco(h)

     
        return output
               
                          
#%% Multilayer QLSTM
class MultiLayerQLSTM(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,nb_layers,bidirectional,use_cuda):
        super(MultiLayerQLSTM,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.nb_layers = nb_layers
        self.use_cuda = use_cuda
        
        self.wfx  = nn.ModuleList([]) # Forget
        self.ufh  = nn.ModuleList([]) # Forget

        self.wix  = nn.ModuleList([]) # Input
        self.uih  = nn.ModuleList([]) # Input

        self.wox  = nn.ModuleList([]) # Output
        self.uoh  = nn.ModuleList([]) # Output

        self.wcx  = nn.ModuleList([]) # Cell state
        self.uch  = nn.ModuleList([])  # Cell state

        self.act  = nn.ModuleList([]) # Activations
        
        self.drop = nn.Dropout(p=0.2)     
        
        if self.bidirectional:
            self.fco  = nn.Linear(self.hidden_size*2, self.output_size)
        else:
            self.fco  = nn.Linear(self.hidden_size, self.output_size)
        curr_input = self.input_size
        
        for i in range(self.nb_layers):
            self.act.append(nn.Tanh())
            self.wfx.append(QuaternionLinear(curr_input, self.hidden_size,bias=False))
            self.wix.append(QuaternionLinear(curr_input, self.hidden_size,bias=False))
            self.wox.append(QuaternionLinear(curr_input, self.hidden_size,bias=False))
            self.wcx.append(QuaternionLinear(curr_input, self.hidden_size,bias=False))

            # Recurrent connections
            self.ufh.append(QuaternionLinear(self.hidden_size, self.hidden_size,bias=False))
            self.uih.append(QuaternionLinear(self.hidden_size, self.hidden_size,bias=False))
            self.uoh.append(QuaternionLinear(self.hidden_size, self.hidden_size,bias=False))
            self.uch.append(QuaternionLinear(self.hidden_size, self.hidden_size,bias=False))
                       
            if self.bidirectional:
                curr_input = 2*hidden_size
            else:
                curr_input = hidden_size
        
    def forward(self,x):
        for i in range(self.nb_layers):
            if self.bidirectional:
                h_init = torch.zeros(2*x.shape[1],self.hidden_size)
                x = torch.cat([x,flip(x,0)],1)
            else:
                h_init = torch.zeros(x.shape[1],self.hidden_size)
            
            if self.use_cuda:
                h_init = h_init.to('cuda')
                x = x.to('cuda')
            
            # Feed-forward affine transformations (all steps in parallel)
            wfx_out=self.wfx[i](x)
            wix_out=self.wix[i](x)
            wox_out=self.wox[i](x)
            wcx_out=self.wcx[i](x)
            
            # Processing time-steps
            hiddens = []
            ct = h_init
            ht = h_init
            
            # Equations (8) in QNNs for multi-channel distant speech recognition
            for k in range(x.shape[0]): 
                ft = torch.sigmoid(wfx_out[k] + self.ufh[i](ht))
                it = torch.sigmoid(wix_out[k] + self.uih[i](ht))
                ot = torch.sigmoid(wox_out[k] + self.uoh[i](ht))
                if self.nb_layers == 1:
                    ct = it*self.act[i](wcx_out[k] + self.uch[i](ht))+ft*ct
                else:                   
                    ct = self.drop(it*self.act[i](wcx_out[k] + self.uch[i](ht)))+ft*ct
                ht = ot*self.act[i](ct)
                
                hiddens.append(ht)
            
            h = torch.stack(hiddens)
            
            if self.bidirectional:
                h_f = h[:,0:int(x.shape[1]/2)] # forward
                h_b = flip(h[:,int(x.shape[1]/2):x.shape[1]].contiguous(),0) # backward
                h = torch.cat([h_f,h_b],2)
            
            x = h
            output = self.fco(h)

        return output
             
#%% LSTM    
class LSTM(nn.Module):
    def __init__(self, feat_size, hidden_size, output_size, bidirectional, CUDA):
        super(LSTM, self).__init__()
        self.act        = nn.Tanh()
        self.act_gate   = nn.Sigmoid()
        self.input_size  = feat_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.CUDA       = CUDA
        
        # Gates initialization
        self.wfx  = nn.Linear(self.input_size, self.hidden_size) # Forget
        self.ufh  = nn.Linear(self.hidden_size, self.hidden_size, bias=False) # Forget
        
        self.wix  = nn.Linear(self.input_size, self.hidden_size) # Input
        self.uih  = nn.Linear(self.hidden_size, self.hidden_size, bias=False) # Input  
        
        self.wox  = nn.Linear(self.input_size, self.hidden_size) # Output
        self.uoh  = nn.Linear(self.hidden_size, self.hidden_size, bias=False) # Output
        
        self.wcx  = nn.Linear(self.input_size, self.hidden_size) # Cell 
        self.uch  = nn.Linear(self.hidden_size, self.hidden_size, bias=False) # Cell 

        # Output layer initialization        
        if self.bidirectional:
            self.fco  = nn.Linear(self.hidden_size*2, self.output_size)
        else:
            self.fco  = nn.Linear(self.hidden_size, self.output_size)
               
    
    def forward(self, x):
        if self.bidirectional:
            h_init = torch.zeros(2*x.shape[1],self.hidden_size)
            x = torch.cat([x,flip(x,0)],1)
        else:
            h_init = torch.zeros(x.shape[1],self.hidden_size)  
                     
        if self.CUDA:
            x = x.to('cuda')
            h_init = h_init.to('cuda')
                 
        # Feed-forward affine transformation
        wfx_out=self.wfx(x)
        wix_out=self.wix(x)
        wox_out=self.wox(x)
        wcx_out=self.wcx(x)
          
        # Processing time steps
        hiddens = []
        
        c=h_init
        h=h_init
        
        for k in range(x.shape[0]):
            ft=self.act_gate(wfx_out[k]+self.ufh(h))
            it=self.act_gate(wix_out[k]+self.uih(h))
            ot=self.act_gate(wox_out[k]+self.uoh(h))
                  
            at = wcx_out[k]+self.uch(h)       
            c = it*self.act(at)+ft*c
            h = ot*self.act(c)            
            hiddens.append(h)  
        h = torch.stack(hiddens)
        if self.bidirectional:
            h_f = h[:,0:int(x.shape[1]/2)] # forward
            h_b = flip(h[:,int(x.shape[1]/2):x.shape[1]].contiguous(),0) # backward
            h = torch.cat([h_f,h_b],2) 
            
        output = self.fco(h)

     
        return output


#%%
class MultiLayerLSTM(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,nb_layers,bidirectional,use_cuda):
        super(MultiLayerLSTM,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.nb_layers = nb_layers
        self.use_cuda = use_cuda
        
        self.wfx  = nn.ModuleList([]) # Forget
        self.ufh  = nn.ModuleList([]) # Forget

        self.wix  = nn.ModuleList([]) # Input
        self.uih  = nn.ModuleList([]) # Input

        self.wox  = nn.ModuleList([]) # Output
        self.uoh  = nn.ModuleList([]) # Output

        self.wcx  = nn.ModuleList([]) # Cell state
        self.uch  = nn.ModuleList([])  # Cell state

        self.act  = nn.ModuleList([]) # Activations
       
        self.drop = nn.Dropout(p=0.2)
        if self.bidirectional:
            self.fco  = nn.Linear(self.hidden_size*2, self.output_size)
        else:
            self.fco  = nn.Linear(self.hidden_size, self.output_size)
       
        curr_input = self.input_size
        
        for i in range(self.nb_layers):
            self.act.append(nn.Tanh())
            self.wfx.append(nn.Linear(curr_input, self.hidden_size,bias=False))
            self.wix.append(nn.Linear(curr_input, self.hidden_size,bias=False))
            self.wox.append(nn.Linear(curr_input, self.hidden_size,bias=False))
            self.wcx.append(nn.Linear(curr_input, self.hidden_size,bias=False))

            # Recurrent connections
            self.ufh.append(nn.Linear(self.hidden_size, self.hidden_size,bias=False))
            self.uih.append(nn.Linear(self.hidden_size, self.hidden_size,bias=False))
            self.uoh.append(nn.Linear(self.hidden_size, self.hidden_size,bias=False))
            self.uch.append(nn.Linear(self.hidden_size, self.hidden_size,bias=False))
            
            if self.bidirectional:
                curr_input = 2*hidden_size
            else:
                curr_input = hidden_size
        
    def forward(self,x):
        for i in range(self.nb_layers):
            if self.bidirectional:
                h_init = torch.zeros(2*x.shape[1],self.hidden_size)
                x = torch.cat([x,flip(x,0)],1)
            else:
                h_init = torch.zeros(x.shape[1],self.hidden_size)
            
            if self.use_cuda:
                h_init = h_init.to('cuda')
                x = x.to('cuda')
            
            # Feed-forward affine transformations (all steps in parallel)
            wfx_out=self.wfx[i](x)
            wix_out=self.wix[i](x)
            wox_out=self.wox[i](x)
            wcx_out=self.wcx[i](x)
            
            # Initializing hidden state and cell state
            hiddens = []
            ct = h_init
            ht = h_init
            
            # Processing time-steps
            # Equations (8) in QNNs for multi-channel distant speech recognition
            for k in range(x.shape[0]): 
                ft = torch.sigmoid(wfx_out[k] + self.ufh[i](ht))
                it = torch.sigmoid(wix_out[k] + self.uih[i](ht))
                ot = torch.sigmoid(wox_out[k] + self.uoh[i](ht))
                ct = self.drop(it*self.act[i](wcx_out[k] + self.uch[i](ht)))+ft*ct
                ht = ot*self.act[i](ct)
                
                hiddens.append(ht)
            
            h = torch.stack(hiddens)
            
            if self.bidirectional:
                h_f = h[:,0:int(x.shape[1]/2)] # forward
                h_b = flip(h[:,int(x.shape[1]/2):x.shape[1]].contiguous(),0) # backward
                h = torch.cat([h_f,h_b],2)
            
            x = h
            output = self.fco(h)

        return output

#%% Pytorch LSTM implementation
class TorchLSTM(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,nb_layers,bias,bidirectional):
        super(TorchLSTM,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.nb_layers = nb_layers
        self.bidirectional = bidirectional
        self.bias = bias
        
        self.lstm = nn.LSTM(self.input_size,self.hidden_size,self.nb_layers,
                            bias=self.bias,dropout=0.2,bidirectional=self.bidirectional)
        
        if self.bidirectional:
            self.fco = nn.Linear(self.hidden_size*2,self.output_size)
        else:
            self.fco = nn.Linear(self.hidden_size,self.output_size)
            
    def forward(self,x):
        out = self.lstm(x)
        # lstm layers return a tuple containing [output, hidden, cell]
        # I'm interested only in the output (i.e. last hidden)
        out = self.fco(out[0])
        return out
    

#%% Bi-LSTM
