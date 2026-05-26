import os
import sys
import argparse

sys.path.append("..")
from pipelines import predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Parameters
    parser.add_argument("--keyword", type=str, default='ls2')
    parser.add_argument("--keyword2", type=str, default=None)
    parser.add_argument("--server", type=str, default='lumi')
    parser.add_argument("--ensemble_mode", type=str, default='majority')
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--wt_file", type=str, default='best_f1')
    parser.add_argument("--ensemble",
                        action=argparse.BooleanOptionalAction, default=False)
    params = vars(parser.parse_args())

    if params['server'] == 'lumi':
        run_dir = "/scratch/project_465002698/venky/projects/arctic/runs"
        # params['image_dir'] = '/scratch/project_465002698/venky/projects/arctic/mosaics/'  # Corrected mosaics
        params['image_dir'] = '/scratch/project_465002698/venky/projects/arctic/image/'    # Raw mosaics
        base_dir = '/scratch/project_465002698/venky/projects/arctic/predictions/'
        params['out_dir'] = os.path.join(base_dir, f"both_{params['keyword']}_{params['ensemble_mode']}")

    elif params['server'] == 'local':
        run_dir = '../runs/'
        base_dir = '/home/venky/Documents/projects/data/arctic/venky_predictions/'
        params['image_dir'] = '../data/test/images/'
        params['out_dir'] = os.path.join(base_dir, f"test_{params['keyword']}_{params['ensemble_mode']}")
    else:
        raise FileNotFoundError

    if not os.path.isdir(params['out_dir']):
        os.makedirs(params['out_dir'])

    # list of trained models on
    if params['ensemble']:
        pretrained_model = []
        for i in range(1, 6):
            path = os.path.join(run_dir, f'{params["keyword"]}_fold{i}', f'{params["wt_file"]}.pth')
            pretrained_model.append(path)
        if params['keyword2'] is not None:
            for i in range(1, 6):
                path = os.path.join(run_dir, f'{params["keyword2"]}_fold{i}', f'{params["wt_file"]}.pth')
                pretrained_model.append(path)

    else:
        pretrained_model = [os.path.join(run_dir,
                                        f'{params["keyword"]}',
                                        f'{params["wt_file"]}.pth')]
    params['pretrained_model'] = pretrained_model
    print(f'Total pretrained models for ensembling are: {len(pretrained_model)}')

    predict.ArcticPredict(**params).run()