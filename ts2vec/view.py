import os
from ts2vec import TS2Vec
import datautils
from tasks.classification import eval_classification
from utils import init_dl_program
import argparse

device = init_dl_program('cuda:2', seed=42, max_threads=4)
data_name = ['ACSF1', 'Adiac', 'AllGestureWiimoteX', 'AllGestureWiimoteY', 'AllGestureWiimoteZ']

def UCR_cls(data, aug):
    train_data, train_labels, test_data, test_labels = datautils.load_UCR(data)
    model = TS2Vec(
        input_dims=train_data.shape[-1],
        device=device,
        output_dims=320,
        aug=aug
    )
    loss_log = model.fit(
        train_data,
        verbose=False
    )
    _, acc = eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='svm')
    print(acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--aug', type=str, default='crop', help='augmentation')
    args = parser.parse_args()
    for data in data_name:
        UCR_cls(data, args.aug)
