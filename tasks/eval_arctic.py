import os
import sys
import argparse

sys.path.append("..")
from pipelines import evaluation


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Parameters
    parser.add_argument("--keyword", type=str, default='bw15')
    parser.add_argument("--server", type=str, default='lumi')
    parser.add_argument("--ensemble_mode", type=str, default='majority')
    parser.add_argument("--set_name", type=str, default='test')
    parser.add_argument("--wt_file", type=str, default='best_f1')
    parser.add_argument("--model_name", type=str, default='unet_elu')
    parser.add_argument("--pred_setting", type=int, default=1)
    # 1 :- all models
    # 2 :- first five models
    # 3 :- last five models

    parser.add_argument("--data_dir", type=str,
                        default='../data/')
    parser.add_argument("--ensemble",
                        action=argparse.BooleanOptionalAction, default=True)
    params = vars(parser.parse_args())

    params['image_dir'] = os.path.join(params['data_dir'], params['set_name'], 'images')
    params['label_dir'] = os.path.join(params['data_dir'], params['set_name'], 'labels')
    params['results_csv'] = '../documents/results.csv'  #fixme not using this for now

    if params['server'] == 'lumi':
        run_dir = "/scratch/project_465002698/venky/projects/arctic/runs"
    elif params['server'] == 'local':
        run_dir = '../runs/'
    else:
        raise FileNotFoundError

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
            path = os.path.join(run_dir, f'{params["keyword"]}_split{i}', f'{params["wt_file"]}.pth')
            pretrained_model.append(path)

    # # list of trained models on
    # if params['ensemble']:
    #     pretrained_model = []
    #     for i in range(1, 6):
    #         pretrained_model.append(os.path.join(run_dir,
    #                                              f'{params["keyword"]}_fold{i}',
    #                                              f'{params["wt_file"]}.pth'))
    #     if params['keyword2'] is not None:
    #         for i in range(1, 6):
    #             path = os.path.join(run_dir, f'{params["keyword2"]}_fold{i}', f'{params["wt_file"]}.pth')
    #             pretrained_model.append(path)
    else:
        pretrained_model = [os.path.join(run_dir,
                                        f'{params["keyword"]}',
                                        f'{params["wt_file"]}.pth')]
    params['pretrained_model'] = pretrained_model

    evaluation.ArcticEvaluation(**params).run()