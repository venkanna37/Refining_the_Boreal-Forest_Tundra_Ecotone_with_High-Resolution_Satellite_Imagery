import sys
import argparse

from botocore import args

sys.path.append("..")
from pipelines import training

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general parameters
    parser.add_argument("--keyword", type=str, default="test")
    parser.add_argument("--server", type=str, default="lumi")

    # data parameters
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patch_size", type=int, default=256) # this is train patch size
    parser.add_argument("--stretch_setting", type=str, default=1)
    parser.add_argument("--data_fold", type=str, default=1)

    # model and training parameters
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--boundary_weight", type=float, default=2)  # weight to boundary pixels
    parser.add_argument("--model_size", type=str, default='small')
    parser.add_argument("--loss_function", type=str, default='tversky')
    parser.add_argument("--alpha", type=float, help='tversky loss param', default=0.7)
    parser.add_argument("--geom_aug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_wb", action=argparse.BooleanOptionalAction, default=False)


    params = vars(parser.parse_args())
    params['data_dir'] = f'../data/geofolds/fold{params["data_fold"]}/'
    if params['server'] == 'lumi':
        params['runs_dir'] = '/scratch/project_465002698/venky/projects/arctic/runs'
    if params['server'] == 'local':
        params['runs_dir'] = 'runs'

    train = training.Training(**params)
    train.train()