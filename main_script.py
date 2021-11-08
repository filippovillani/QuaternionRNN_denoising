# sine wave denoiser with four channels (FINAL VERSION)
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from time import time
import os
from utility_functions import load_model, save_model, create_sine_dataset, noise, evaluate
import torch
import torch.nn as nn
import torch.utils.data as utils
from network_models import LSTM, QLSTM, TorchLSTM
from sklearn.model_selection import train_test_split

np.random.seed(0)
multi_channel = True # set to false if training a real-valued network
model_name = 'ProvaBi-QLSTM_bias'
###### HYPERPARAMETERS #######
hidden_size = 80
lr = 1e-3
batch_size = 64
epochs = 150
patience = 20
bidirectional = True 
bias = True 
##############################
# Changes checkpoint_dir everytime you start a new experiment
checkpoint_main_dir = r'C:\Users\User\Desktop\SineWaveDenoiser\RESULTS'
checkpoint_dir = os.path.join(checkpoint_main_dir,f'{model_name}_b{batch_size}h{hidden_size}')
results_path = checkpoint_dir
training_output_path = os.path.join(results_path,'training_output')
##############################

if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
    
if not os.path.exists(training_output_path):
    os.mkdir(training_output_path)
#%% 
if torch.cuda.is_available():
    use_cuda = True
    device = 'cuda'
else:
    use_cuda = False
    device = 'cpu'

input_size = 4 # because we have sine waves over 4 channels, i.e. 4 features
output_size = 1 # I want to estimate a 1D sine wave
# nb_layers = 1


# model = TorchLSTM(input_size,hidden_size,output_size,nb_layers,bias,bidirectional)
# model = QLSTM(input_size,hidden_size,output_size,bidirectional,bias,use_cuda)
model = LSTM(input_size,hidden_size,output_size,bidirectional,use_cuda)

model_params = sum([np.prod(p.size()) for p in model.parameters()])
print(f'Total {model_name} parameters: {model_params}')

model = model.to(device)
criterion = nn.MSELoss()
optimizer = Adam(params=model.parameters(),lr=lr)

#%% Building the data


data_pred, data_target = create_sine_dataset(multi_channel=multi_channel)

train_pred,test_pred, train_target,test_target = train_test_split(data_pred,data_target,test_size=0.2)
train_pred, val_pred, train_target,val_target = train_test_split(train_pred,train_target,test_size=0.35)

# Converting each array into tensor
if multi_channel:
    test_pred = torch.Tensor(test_pred).permute(0,2,1)
    test_target = torch.Tensor(test_target).permute(0,2,1)
    train_pred = torch.Tensor(train_pred).permute(0,2,1)
    train_target = torch.Tensor(train_target).permute(0,2,1)
    val_pred = torch.Tensor(val_pred).permute(0,2,1)
    val_target = torch.Tensor(val_target).permute(0,2,1)
else:
    test_pred = torch.Tensor(test_pred.reshape((test_pred.shape[0],-1,1)))
    test_target = torch.Tensor(test_target.reshape((test_target.shape[0],-1,1)))
    train_pred = torch.Tensor(train_pred.reshape((train_pred.shape[0],-1,1)))
    train_target = torch.Tensor(train_target.reshape((train_target.shape[0],-1,1)))
    val_pred = torch.Tensor(val_pred.reshape((val_pred.shape[0],-1,1)))
    val_target = torch.Tensor(val_target.reshape((val_target.shape[0],-1,1)))   

# Dataloader
train_loader = utils.DataLoader(utils.TensorDataset(train_pred,train_target),shuffle=True,batch_size=batch_size)
val_loader = utils.DataLoader(utils.TensorDataset(val_pred,val_target),batch_size=batch_size)
test_loader = utils.DataLoader(utils.TensorDataset(test_pred,test_target),batch_size=batch_size)
'''
The dataset has this shape: [batch_size,feature_size(mics),seq_length]
where n_samples is the number of instances and sample_size is the temporal index
We will feed [seq_length,batch_size,feature_size(mics)]
'''


# State dictionary for training

state = {"step" : 0,
         "worse_epochs" : 0,
         "epochs" : 0,
         "best_loss" : np.Inf,
         "best_checkpoint": None}

#%% TRAIN MODEL
print('_________________________')
print('     Training start')
print('_________________________')
start_train = time()
train_loss_hist = []
val_loss_hist = []
epoch = 1

while state["worse_epochs"] < patience and epoch<=epochs:
    if state['worse_epochs'] == 3 and lr > 1e-5:
        lr /= 2
        for g in optimizer.param_groups:
            g['lr'] = lr
            print('lr = ',g['lr'])
        
    start_epoch = time()
    print(f'\n§ Train epoch: {epoch}\n')
    model.train()
    train_loss = 0
    
    for example_num, (x,target) in enumerate(train_loader):
        
        if multi_channel:
            x, target = x.permute(1,0,2), target.permute(1,0,2)           
        
        optimizer.zero_grad()
        x = x.to(device)
        target = target.to(device)
        outputs = model(x)
        
        if example_num == 0 and epoch%20==0:
            img_path = os.path.join(training_output_path,f'e0{epoch}.png')
            if multi_channel:
                plt.figure()
                plt.plot(outputs[:,0].cpu().detach().numpy(),label='output')
                plt.plot(target[:,0].cpu().detach().numpy(),label='target')
                plt.legend()
                plt.title(f'epoch {epoch}')
                plt.savefig(img_path)
            else:
                plt.figure()
                plt.plot(outputs[0,:,0].cpu().detach().numpy(),label='output')
                plt.plot(target[0,:,0].cpu().detach().numpy(),label='target')
                plt.legend()
                plt.title(f'epoch {epoch}')
                plt.savefig(img_path)
        loss = criterion(outputs,target)
        loss.backward()
        
        
        train_loss += ((1./(example_num+1))*(loss-train_loss)).item()
        optimizer.step()
        state['step'] += 1

    # Pass validation data
    val_loss = evaluate(model, criterion, val_loader)
    print(f'\nTraining loss: {train_loss}')
    print(f'Validation loss: {val_loss}\n')
    
    # Early stopping
    checkpoint_path = os.path.join(checkpoint_dir,'checkpoint')
    
    if val_loss >= state['best_loss']:
        state['worse_epochs']+=1
    else:
        print(f'\nMODEL IMPROVED ON VALIDATION SET ON EPOCH {epoch}\n')
        state['worse_epochs'] = 0
        state['best_loss'] = val_loss
        state['best_checkpoint'] = checkpoint_path
        print('Saving Model...')
        #save_model(model,optimizer,state,checkpoint_path)
    
    state['epochs']+=1
    train_loss_hist.append(train_loss)
    val_loss_hist.append(val_loss)
    print(f'Epoch time: {int(((time()-start_epoch))//60)} min {int((((time()-start_epoch))%60)*60/100)} s')
    print('_____________________________')
    epoch += 1
print(f'Train finished in {(time()-start_train)/60} min\n')
#%% Load best model and compute loss for all sets
print('_________________________')
print('       Test start')
print('_________________________')
start3 = time()
# Load best model based on validation loss
state = load_model(model,None,checkpoint_path,cuda=use_cuda)
# compute loss
train_loss = evaluate(model,criterion,train_loader)
val_loss = evaluate(model,criterion,val_loader)
test_loss = evaluate(model,criterion,test_loader)

# Print and save results
results = {'train_loss': train_loss,
           'val_loss': val_loss,
           'test_loss': test_loss,
           'train_loss_hist': train_loss_hist,
           'val_loss_hist': val_loss_hist}

           
print(f'Test finished in {(time()-start3)/60} min')
print('RESULTS')
for i in results:
    if 'hist' not in i:
        print(i, results[i])
out_path = os.path.join(results_path,'results_dict')
#np.save(out_path,results)
# To load them back and retrieve the dictionary you have to use
# load = np.load(...,allow_pickle=True).item()

#%% plotting loss history
n_epochs = len(train_loss_hist)
plt.figure()
plt.plot(train_loss_hist, label='Train Loss')
plt.plot(val_loss_hist, label='Validation Loss')
plt.title(f'Loss history, {model_name} with batch={batch_size} and hidden={hidden_size}')
plt.legend()
plt.grid()
plt.savefig(os.path.join(results_path,'LOSS_HISTORY.png'))

#%% Plotting target, predictor and predicted example

loaded = torch.load(checkpoint_path)
model.load_state_dict(loaded['model_state_dict'])

sample_num = 12
plt.figure()
if multi_channel:
    example = test_pred[sample_num].reshape(-1,1,4).to(device)
    predicted = model(example).to(device)
    plt.plot(predicted[:,0,0].cpu().detach().numpy(),label='Predicted')
    plt.plot(test_target[sample_num,:,0].cpu().detach().numpy(),label='Target')
    plt.plot(test_pred[sample_num,:,0].cpu().detach().numpy(),label='Predictor, ch0',linewidth=0.5,linestyle='dashed')
    plt.title(f'Prediction example, {model_name} with batch={batch_size} and hidden={hidden_size}')
    plt.legend()
    plt.xlabel('n')
    plt.ylabel('amplitude')
    plt.grid()
    plt.savefig(os.path.join(results_path,'prediction_example.png'))
else:
    example = test_pred[sample_num].reshape(-1,1,1).to(device)
    predicted = model(example).to(device)
    plt.plot(predicted[:,0,0].cpu().detach().numpy(),label='Predicted')
    plt.plot(test_target[sample_num,:,0].cpu().detach().numpy(),label='Target')
    plt.plot(test_pred[sample_num,:,0].cpu().detach().numpy(),label='Predictor, ch0',linewidth=0.5,linestyle='dashed')
    plt.title(f'Prediction example, {model_name} with batch={batch_size} and hidden={hidden_size}')
    plt.legend()
    plt.grid()
    plt.xlabel('n')
    plt.ylabel('amplitude')
    plt.savefig(os.path.join(results_path,'prediction_example.png'))



#%% SNR computation
loaded = torch.load(checkpoint_path)
model.load_state_dict(loaded['model_state_dict'])
model = model.to(device)

test_pred = test_pred.permute(1,0,2)
test_target = test_target.permute(1,0,2)
test_pred = test_pred.to(device)

SNR_in = []
SNR_out = []
nAttempts = 20
for k in range(nAttempts):
    predicted = model(test_pred)
    predicted = predicted.cpu().detach().numpy()
    error = test_target - predicted
    signal_power = np.mean((test_target.numpy())**2,axis=0).squeeze()
    error_power = np.mean((error.numpy())**2,axis=0).squeeze()
    SNR_single_sample = signal_power/error_power
    SNR = np.mean(SNR_single_sample)
    SNR_out.append(10*np.log10(SNR))
    
    noise_vector = np.zeros(100)
    noise_vector = noise(noise_vector)
    noise_power = np.mean(noise_vector**2)
    SNR = np.mean(signal_power/noise_power)
    SNR_in.append(10*np.log10(SNR))

SNR_in_mean = np.mean(SNR_in)
SNR_out_mean = np.mean(SNR_out)

print('SNR_in',SNR_in_mean,'dB')
print('SNR_out',SNR_out_mean,'dB')
    
#%%
plt.figure()
plt.plot(error[:,0,0],label='error')
plt.plot(predicted[:,0,0],label='predicted')
plt.plot(test_pred[:,0,0].cpu().detach(),label='predictor ch0')
plt.plot(test_target[:,0,0],label='target')
plt.legend()
plt.grid()

