import torch
import torch.nn as nn
import numpy as np

class CPC(nn.Module):
    def __init__(self, K, seq_len):
        """
        K: this is K in the paper, the K timesteps in the future relative to
           the last context timestep  are used as targets to teach the model 
           to predict K steps into the future
        """
        super(CPC, self).__init__()

        self.seq_len = seq_len
        self.K = K
        self.c_size = 256
        self.z_size = 512
        self.dwn_fac = 160
        # the downsampling factor of this CNN is 160
        # so the output sequence length is 20480/160 = 128
        self.encoder = nn.Sequential( 
            nn.Conv1d(1, self.z_size, kernel_size=10, stride=5, padding=3, bias=False),
            nn.BatchNorm1d(self.z_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.z_size, self.z_size, kernel_size=8, stride=4, padding=2, bias=False),
            nn.BatchNorm1d(self.z_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.z_size, self.z_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(self.z_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.z_size, self.z_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(self.z_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.z_size, self.z_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(self.z_size),
            nn.ReLU(inplace=True)
        )
        
        self.gru = nn.GRU(self.z_size, self.c_size, num_layers = 1, 
                          bidirectional = False, batch_first = True)
        
        # These are all trained
        self.Wk = nn.ModuleList([nn.Linear(self.c_size,self.z_size) for i in range(self.K)])
        self.softmax = nn.Softmax(dim=1)
        self.lsoftmax = nn.LogSoftmax(dim=1)

    def init_hidden(self, batch_size, device, use_gpu=True):
        if use_gpu: return torch.zeros(1, batch_size, self.c_size).to(device)
        else: return torch.zeros(1, batch_size, self.c_size)

    def forward(self, x, hidden):
        """
        x: torch.float32 shape (batch_size,channels=1,seq_len)
        hidden: torch.float32 
                shape (num_layers*num_directions=1,batch_size,hidden_size=256) 
        """
        batch_size = x.size()[0] 
        
        # input x is shape (batch_size, channels, seq_len), e.g. 8*1*20480
        z = self.encoder(x) 
        # encoded sequence z is shape (batch_size, z_size, seq_len), e.g. 8*512*128
        
        z = z.transpose(1,2) # reshape->(batch_size,seq_len,z_size) for GRU, e.g. 8*128*512
        
        # pick timestep to be the last in the context, time_C, later ones are targets 
        highest = self.seq_len//self.dwn_fac - self.K # 128 -12 = 116
        time_C = torch.randint(highest, size=(1,)).long() # some number between 0 and 116

        # encode_samples (K, batch_size, z_size)ie 12,8,512, 
        z_t_k = z[:, time_C + 1:time_C + self.K + 1, :].clone().cpu().float()
        z_t_k = z_t_k.transpose(1,0)
        
        z_0_T = z[:,:time_C + 1,:] # e.g. size 8*100*512
        output, hidden = self.gru(z_0_T, hidden) # output size e.g. 8*100*256
        c_t = output[:,time_C,:].view(batch_size, self.c_size) # c_t e.g. size 8*256
        
        # For the future K timesteps, predict their z_t+k, 
        W_c_k = torch.empty((self.K, batch_size, self.z_size)).float() # e.g. size 12*8*512
        for k in np.arange(0, self.K):
            linear = self.Wk[k] # c_t is size 256, Wk is a 512x256 matrix 
            W_c_k[k] = linear(c_t) # Wk*c_t e.g. size 8*512
            
        nce = 0 # average over timestep and batch
        for k in np.arange(0, self.K):
            
            # (batch_size, z_size)x(z_size, batch_size) = (batch_size, batch_size)
            zWc = torch.mm(z_t_k[k], torch.transpose(W_c_k[k],0,1)) 
            # print(zWc)
            # total has shape (batch_size, batch_size) e.g. size 8*8
            
            logsof_zWc = self.lsoftmax(zWc)
            #print(logsof_zWc)
            nce += torch.sum(torch.diag(logsof_zWc)) # nce is a tensor
            
        nce /= -1.*batch_size*self.K
        
        argmax = torch.argmax(self.softmax(zWc), dim=0)# softmax not required if argmax 
        correct = torch.sum( torch.eq( argmax, torch.arange(0, batch_size) ) ) 
        accuracy = 1.*correct.item()/batch_size

        return accuracy, nce, hidden

    def predict(self, x, hidden):

        # input sequence is N*C*L, e.g. 8*1*20480
        
        z = self.encoder(x) # encoded sequence is N*C*L, e.g. 8*512*128
        
        z = z.transpose(1,2) # reshape to N*L*C for GRU, e.g. 8*128*512
        output, hidden = self.gru(z, hidden) # output size e.g. 8*128*256

        return output, hidden # return every frame
        #return output[:,-1,:], hidden # only return the last frame per utt
