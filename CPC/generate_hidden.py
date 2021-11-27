import os
import soundfile as sf  
import numpy as np
import torch
from torch.utils import data as datautil
from cpc import CPC
import argparse

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:2" if use_cuda else "cpu")

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


def load_model(model, name, device):
    model.load_state_dict(torch.load(name, map_location=device))

validation_set = RawDataset(directory="../data/LibriSpeech/dev-clean", 
                          audio_window=20480)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=2, help='predict k timestamps in future (defaults to 2)')
    parser.add_argument('--checkpoints', type=str, default='', help='path of weights (no default, plz check by yourself!)')
    args = parser.parse_args()
    model = CPC(K=args.k, seq_len=20480).to(device)
    load_model(model, "checkpoints/" + args.checkpoints, device)
    model.eval()

    params = {}
    loader = datautil.DataLoader(validation_set, batch_size=1024, shuffle=False, **params)
    for batch_idx, utterance in enumerate(loader):
        break
    
    utterance = utterance.float().float().unsqueeze(1).to(device)
    with torch.no_grad():
        hidden = model.init_hidden(len(utterance), device, use_gpu=use_cuda)
        acc, loss, hidden = model(utterance, hidden)
        print(acc, loss, hidden.shape)
    
    hidden_np = hidden.squeeze(0)
    hidden_np = hidden_np.numpy()
    np.save('hidden_1024_k' + str(args.k) + '.npy', hidden_np)
