import os
import sys
import glob
import argparse

sys.path.append("..")
from tools import predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Parameters
    parser.add_argument("--keyword", type=str, default='test_run',
                        help='Keyword that used for saving checkpoint while training')

    # prediction parameters
    parser.add_argument("--in_dir", type=str, default='',
                        help='Folder where all input images are available')
    parser.add_argument("--out_dir", type=str, default='',
                        help='Directory where all predictions will be saved within a new directory with keyword name')
    parser.add_argument("--patch_size", type=int, default=2048,
                        help='Subset of large image to predict')
    parser.add_argument("--ensemble_mode", type=str, default='majority',
                        help='Option to combine results from multiple models if ensemble=True',
                        choices=['majority', 'intersection', 'atleast1', 'atleast2', 'atleast3', 'atleast4'])
    parser.add_argument("--ensemble",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wt_file", type=str, default='best_f1',
                        help='Select the weights file name based on metric',
                        choices=['best_precision', 'best_recall', 'best_f1', 'latest'])
    parser.add_argument("--pred_setting", type=int, default=1,
                        help='1: select all models, 2: First five models 3: last five models')
    # 1 :- all models
    # 2 :- first five models
    # 3 :- last five models

    parser.add_argument("--data_dir", type=str, default='./datasets',
                        help='Directory to the datasets')
    parser.add_argument('--checkpoints_dir', type=str, default='./runs',
                        help='output directory to save check points and logs')

    parser.add_argument("--chunk_size", type=int, default=1000,
                        help='Devides total number of images into N number of sets (chunks) based on this size')
    parser.add_argument("--chunk_id", type=int, default=0,
                        help='chunk_id is the id to select the set/chunk from list of all sets/chunks')

    params = vars(parser.parse_args())

    files = sorted(glob.glob(os.path.join(params['in_dir'], '*.tif')))
    assert len(files) != 0, f'There are no file in {params["in_dir"]}'
    print('Total number of images are', len(files))

    # split all images into chunk with the size of 320
    chunks = [files[i:i + params['chunk_size']] for i in range(0, len(files), params['chunk_size'])]
    params['images'] = chunks[params['chunk_id']]
    print(f'Total number of images in chunk id {params["chunk_id"]}:', len(params['images'] ))
    print(f'Total number of chunks {len(chunks)}')

    # list of trained models on
    if params['ensemble']:
        if params['pred_setting'] == 1:
            start_idx, end_idx = 1, 11
        elif params['pred_setting'] == 2:
            start_idx, end_idx = 1, 6
        elif params['pred_setting'] == 3:
            start_idx, end_idx = 6, 11
        else:
            raise ValueError

        pretrained_model = []
        for i in range(start_idx, end_idx):
            path = os.path.join(params['checkpoints_dir'], f'{params["keyword"]}_split{i}', f'{params["wt_file"]}.pth')
            pretrained_model.append(path)

    else:
        pretrained_model = [os.path.join(params['checkpoints_dir'],
                                        f'{params["keyword"]}',
                                        f'{params["wt_file"]}.pth')]

    # Create output folder insize output directory
    params['out_dir'] = os.path.join(params['out_dir'], params['keyword'])
    if not os.path.isdir(params['out_dir']):
        os.makedirs(params['out_dir'])

    params['pretrained_model'] = pretrained_model
    print(f'Total pretrained models for ensembling are: {len(pretrained_model)}')

    # this is where status of the predictions save
    params['csv_path'] = os.path.join(params['out_dir'], f"{params['keyword']}_split{params['chunk_id']}.csv")


    predict.ArcticPredict(**params).run()