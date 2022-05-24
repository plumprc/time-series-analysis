import torch
from models.model import Informer
from data.data_loader import Dataset_ETT_hour
from torch.utils.data import DataLoader

if __name__ == '__main__':
    seq_len, label_len, pred_len = 96, 48, 24

    data_set = Dataset_ETT_hour(
        root_path='data/ETT',
        data_path='ETTh1.csv',
        size=[seq_len, label_len, pred_len],
        features='M', # multivariate predict multivariate
        freq='h' # ['month','day','weekday','hour'], data-agnostic
    )
    '''
        seq_x and seq_y are overlapping!
        len(seq_x) = seq_len + label_len
        len(seq_y) = label_len + pred_len
    '''

    data_loader = DataLoader(
        data_set,
        batch_size=32,
        shuffle=True
    )

    batch_x, batch_y, batch_x_mark, batch_y_mark = iter(data_loader).next()
    pred_seq = torch.zeros([batch_y.shape[0], pred_len, batch_y.shape[-1]]).float()
    pred_seq = torch.cat([batch_y[:,:label_len,:], pred_seq], dim=1).float()
    true = batch_y[:,-pred_len:,0:]

    model = Informer(7, 7, 7, seq_len, label_len, pred_len)
    pred = model(batch_x.float(), batch_x_mark.float(), pred_seq.float(), batch_y_mark.float())

    shift_mean = torch.nn.AvgPool1d(kernel_size=17, stride=1, padding=0)
    pred_shift_mean = shift_mean(batch_x.transpose(1, 2)).transpose(1, 2)
    true_shift_mean = shift_mean(batch_y.transpose(1, 2)).transpose(1, 2)

    print(batch_x.shape, true_shift_mean.shape)

    batch_x = torch.cat((batch_x, true_shift_mean), dim=1)
    print(batch_x.shape)
