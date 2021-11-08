# L3DAS21 utility_functions
import os
import torch
import numpy as np
import random


def evaluate(model,criterion,dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for example_num, (x,target) in enumerate(dataloader):
            x, target = x.permute(1,0,2), target.permute(1,0,2)
            x = x.to(device)
            target = target.to(device)
            outputs = model(x)
            loss = criterion(outputs,target)
            test_loss += ((1./(example_num+1))*(loss-test_loss)).item()
    return test_loss

def save_model(model, optimizer, state, path):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # save state dict of wrapped module
    if len(os.path.dirname(path)) > 0 and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'state': state,  # state of training loop (was 'step')
    }, path)


def load_model(model, optimizer, path, cuda):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # load state dict of wrapped module
    if cuda:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location='cpu')
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except:
        # work-around for loading checkpoints where DataParallel was saved instead of inner module
        from collections import OrderedDict
        model_state_dict_fixed = OrderedDict()
        prefix = 'module.'
        for k, v in checkpoint['model_state_dict'].items():
            if k.startswith(prefix):
                k = k[len(prefix):]
            model_state_dict_fixed[k] = v
        model.load_state_dict(model_state_dict_fixed)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'state' in checkpoint:
        state = checkpoint['state']
    else:
        # older checkpoints only store step, rest of state won't be there
        state = {'step': checkpoint['step']}
    return state


#%% Dataset creation
def sine(X,fs = 60.):
    return np.sin(2*np.pi*X/fs)    

def noise(Y, noise_range=(-.35,.35)):
    noise = np.random.normal(scale=0.2,size=Y.shape)
    return Y+noise


def sample(sample_size,random_offset):
    X = np.arange(sample_size)
    out = sine(X+random_offset)
    inp = noise(out)
    return inp, out


def create_sine_dataset(n_samples=3000,n_channels=4,sample_size=100,multi_channel=True):
    if multi_channel:
        data_pred = np.zeros((n_samples,n_channels,sample_size))
        data_target = np.zeros((n_samples,1,sample_size))
        
        for i in range(n_samples):
            random_offset = random.randint(0,sample_size)
            for j in range(n_channels):           
                sample_inp, sample_out = sample(sample_size,random_offset)
                data_pred[i,j,:] = sample_inp
                data_target[i,:,:] = sample_out          
        return data_pred, data_target
    else:
        data_target = np.zeros((n_samples, sample_size))
        data_pred = np.zeros((n_samples, sample_size))
        
        for i in range(n_samples):
            random_offset = random.randint(0,sample_size)
            sample_inp, sample_out = sample(sample_size,random_offset)
            data_pred[i, :] = sample_inp
            data_target[i, :] = sample_out
        return data_pred, data_target