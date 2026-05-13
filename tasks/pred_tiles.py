import os
import sys
import argparse

sys.path.append("..")
from pipelines import predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Parameters
    parser.add_argument("--keyword", type=str, default='ls2')
    parser.add_argument("--server", type=str, default='lumi')
    parser.add_argument("--set_name", type=str, default='mosaics')
    parser.add_argument("--wt_file", type=str, default='best_f1')
    parser.add_argument("--ensemble",
                        action=argparse.BooleanOptionalAction, default=False)
    params = vars(parser.parse_args())

    # if params['set_name'] in ['train', 'val', 'test']:
    #     params['image_dir'] = os.path.join(params['data_dir'], params['set_name'], 'images')
    # elif params['set_name'] == 'mosaics':

    if params['server'] == 'lumi':
        run_dir = "/scratch/project_465002698/venky/projects/arctic/runs"
        params['image_dir'] = '/scratch/project_465002698/venky/projects/arctic/mosaics/'
        base_dir = '/scratch/project_465002698/venky/projects/arctic/predictions/'
        params['out_dir'] = os.path.join(base_dir, params['keyword'])


    elif params['server'] == 'local':
        run_dir = '../runs/'
        base_dir = '/home/venky/Documents/projects/data/arctic/venky_predictions/'
        params['out_dir'] = os.path.join(base_dir, params['keyword'])
        params['image_dir'] = '../data/geofolds/test/images/'
    else:
        raise FileNotFoundError

    if not os.path.isdir(params['out_dir']):
        os.makedirs(params['out_dir'])

    # list of trained models on
    if params['ensemble']:
        pretrained_model = []
        for i in range(1, 6):
            pretrained_model.append(os.path.join(run_dir,
                                                 f'{params["keyword"]}_fold{i}',
                                                 f'{params["wt_file"]}.pth'))
    else:
        pretrained_model = [os.path.join(run_dir,
                                        f'{params["keyword"]}',
                                        f'{params["wt_file"]}.pth')]
    params['pretrained_model'] = pretrained_model

    predict.ArcticPredict(**params).run()