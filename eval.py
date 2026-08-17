import os
import sys
import argparse

sys.path.append("..")
from tools import evaluation


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # general paramters
    parser.add_argument("--keyword", type=str, default='test_run',
                        help='Keyword that used for saving checkpoint while training')

    # prediction parameters
    parser.add_argument("--ensemble_mode", type=str, default='majority',
                        help='Option to combine results from multiple models if ensemble=True',
                        choices=['majority', 'intersection', 'atleast1', 'atleast2', 'atleast3', 'atleast4'])
    parser.add_argument("--ensemble",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--set_name", type=str, default='test',
                        help='Set name on which we want to evaluate')
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

    params = vars(parser.parse_args())
    params['image_dir'] = os.path.join(params['data_dir'], params['set_name'], 'images')
    params['label_dir'] = os.path.join(params['data_dir'], params['set_name'], 'labels')

    # select models based on pred_setting
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
    params['pretrained_model'] = pretrained_model

    evaluation.ArcticEvaluation(**params).run()