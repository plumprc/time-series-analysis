import numpy as np
import torch
from torch.utils import data as datautil
import torch.optim as optim
from cpc import CPC
import os
from timeit import default_timer as timer
import soundfile as sf  
import argparse

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:3" if use_cuda else "cpu")
global_timer = timer() # global timer
log_interval = 200
best_acc = 0
best_loss = np.inf
best_epoch = -1 
epochs = 10

class RawDataset(datautil.Dataset):
    
    def __init__(self, directory, audio_window):

        self.audio_window = audio_window 
        self.audiolist = []
        self.idx2file = {}
        self.files = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(directory):
            for file in f:
                if '.flac' in file:
                    self.files.append(os.path.join(r, file))

        for idx, filepath in enumerate(self.files):
            self.idx2file[idx] = filepath

    def __len__(self):
        """Denotes the total number of utterances """
        return len(self.files)

    def __getitem__(self, index):
        filepath = self.idx2file[index]
        audiodata, samplerate = sf.read(filepath)
        utt_len = audiodata.shape[0] 
        # get the index to read part of the utterance into memory 
        index = np.random.randint(utt_len - self.audio_window + 1) 
        return audiodata[index:index+self.audio_window]


class ScheduledOptim(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = 128 
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0 
        self.delta = 1

    def state_dict(self):
        self.optimizer.state_dict()

    def step(self):
        """Step by the inner optimizer"""
        self.optimizer.step()

    def zero_grad(self):
        """Zero out the gradients by the inner optimizer"""
        self.optimizer.zero_grad()

    def increase_delta(self):
        self.delta *= 2

    def update_learning_rate(self):
        """Learning rate scheduling per step"""

        self.n_current_steps += self.delta
        new_lr = np.power(self.d_model, -0.5) * np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr


def save_model(model,name):
    torch.save(model.state_dict(),name)
    
def load_model(model,name):
    model.load_state_dict(torch.load(name))

def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.float().unsqueeze(1).to(device) # add channel dimension
        optimizer.zero_grad()
        hidden = model.init_hidden(len(data), device, use_gpu=True)
        acc, loss, hidden = model(data, hidden)

        loss.backward()
        optimizer.step()
        lr = optimizer.update_learning_rate()
        if batch_idx % log_interval == 0:
            print('Train Epoch:{}[{}/{}({:.0f}%)]\tlr:{:.5f}\tAcc:{:.4f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), lr, acc, loss.item()))
            
def validation(model, device, data_loader):
    print("Starting Validation")
    model.eval()
    total_loss = 0
    total_acc  = 0 

    with torch.no_grad():
        for data in data_loader:
            data = data.float().unsqueeze(1).to(device) # add channel dimension
            hidden = model.init_hidden(len(data), device, use_gpu=True)
            acc, loss, hidden = model(data, hidden)
            total_loss += len(data) * loss 
            total_acc  += len(data) * acc

    total_loss /= len(data_loader.dataset) # average loss
    total_acc  /= len(data_loader.dataset) # average acc

    print('===> Validation set: Average loss: {:.4f}\tAccuracy: {:.4f}\n'.format(
                total_loss, total_acc))

    return total_acc, total_loss

params = {'num_workers': 0,
          'pin_memory': False} if use_cuda else {}

training_set = RawDataset(directory = "../data/LibriSpeech/train-clean-100", 
                          audio_window = 20480)

validation_set = RawDataset(directory = "../data/LibriSpeech/dev-clean", 
                          audio_window = 20480)

train_loader = datautil.DataLoader(training_set, batch_size=32, 
                                   shuffle=True, **params) # set shuffle to True

validation_loader = datautil.DataLoader(validation_set, batch_size=32, 
                                        shuffle=False, **params) # set shuffle to False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=2, help='predict k timestamps in future (defaults to 2)')
    args = parser.parse_args()
    model = CPC(K=args.k, seq_len=20480).to(device)
    # load_model(model,"checkpoints/CPC_K2.pth")
    optimizer = ScheduledOptim(
    optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),
        n_warmup_steps = 50)
    
    for epoch in range(1, epochs + 1):
        epoch_timer = timer()
        
        train(log_interval, model, device, train_loader, optimizer, epoch)
        val_acc, val_loss = validation(model, device, validation_loader)

        if val_acc > best_acc:
            print("new best val_acc", val_acc)
            save_model(model,"checkpoints/CPC_k_" + str(args.k) + ".pth")
            best_acc = max(val_acc, best_acc)
            best_epoch = epoch + 1
        elif epoch - best_epoch > 2:
            optimizer.increase_delta()
            best_epoch = epoch + 1
        
        end_epoch_timer = timer()
        print("#### End epoch {}/{}, elapsed time: {}".format(epoch,
                                                    epochs, 
                                                   end_epoch_timer - epoch_timer))
    
    end_global_timer = timer()
    print("Total elapsed time: %s" % (end_global_timer - global_timer))
    save_model(model,"checkpoints/CPC_K" + str(args.k) + "_fin.pth")
