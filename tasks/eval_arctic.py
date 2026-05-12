import os
import sys
import argparse

sys.path.append("..")
from pipelines import evaluation


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Parameters
    parser.add_argument("--keyword", type=str, default='ls2')
    parser.add_argument("--server", type=str, default='local')
    parser.add_argument("--set_name", type=str, default='train')
    parser.add_argument("--wt_file", type=str, default='best_f1')
    parser.add_argument("--data_dir", type=str, default='../data/geofolds/fold1')
    parser.add_argument("--ensemble", action=argparse.BooleanOptionalAction, default=False)
    params = vars(parser.parse_args())

    params['image_dir'] = os.path.join(params['data_dir'], params['set_name'], 'images')
    params['label_dir'] = os.path.join(params['data_dir'], params['set_name'], 'labels')
    params['results_csv'] = '../documents/results.csv'

    if params['server'] == 'lumi':
        run_dir = "/scratch/project_465002698/venky/projects/arctic/runs"
    elif params['server'] == 'local':
        run_dir = '../runs/'
    else:
        raise FileNotFoundError

    # list of trained models on
    if params['ensemble']:
        pretrained_model = []
        for i in range(1, 6):
            pretrained_model.append(os.path.join(run_dir,
                                                 f'{params["keyword"]}_fold{i}',
                                                 f'{params["wt_file"]}.pth'))
    else:
        pretrained_model = os.path.join(run_dir,
                                        f'{params["keyword"]}_fold1',
                                        f'{params["wt_file"]}.pth')
    params['pretrained_model'] = pretrained_model

    evaluation.ArcticEvaluation(**params).run()