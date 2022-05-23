import os
import numpy as np
from ts2vec import TS2Vec
import datautils
from tasks.classification import eval_classification
from utils import init_dl_program
import argparse

def UCR_cls(data, device, aug):
    train_data, train_labels, test_data, test_labels = datautils.load_UCR(data)
    model = TS2Vec(
        input_dims=train_data.shape[-1],
        device=device,
        output_dims=320,
        batch_size=32,
        aug=aug
    )
    loss_log = model.fit(
        train_data,
        verbose=True
    )
    
    _, acc = eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='svm')
    print(data, acc)

    # return model.encode(
    #     test_data,
    #     casual=True,
    #     sliding_length=1,
    #     sliding_padding=50
    # )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--aug', type=str, default='crop', help='augmentation')
    parser.add_argument('--mode', type=str, default='all', help='test mode')
    args = parser.parse_args()

    device = init_dl_program('cpu', seed=42, max_threads=4)
    if args.mode == 'all':
        data_name = os.listdir('datasets/UCR')
        data_name = list(filter(lambda x:x!='Missing_value_and_variable_length_datasets_adjusted', data_name))
        data_name.sort()

        for data in data_name:
            UCR_cls(data, device, args.aug)
    
    if args.mode == 'uni':
        UCR_cls('GestureMidAirD3', device, args.aug)
