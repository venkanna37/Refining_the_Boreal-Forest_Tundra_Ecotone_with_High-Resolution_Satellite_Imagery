import os
import sys
import glob
import argparse

sys.path.append("..")
from pipelines import predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Parameters
    parser.add_argument("--keyword", type=str, default='bw5')
    parser.add_argument("--server", type=str, default='lumi')
    parser.add_argument("--ensemble_mode", type=str, default='majority')
    parser.add_argument("--csv_path", type=str, default='predictions.csv')
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--pred_setting", type=int, default=1)
    # 1 :- all models
    # 2 :- first five models
    # 3 :- last five models
    parser.add_argument("--chunk_id", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=105)
    parser.add_argument("--wt_file", type=str, default='best_f1')
    parser.add_argument("--ensemble",
                        action=argparse.BooleanOptionalAction, default=True)
    params = vars(parser.parse_args())

    if params['server'] == 'lumi':
        run_dir = "/scratch/project_465002698/venky/projects/arctic/runs"
        # params['image_dir'] = '/scratch/project_465002698/venky/projects/arctic/mosaics/'  # Corrected mosaics
        # params['image_dir'] = '/scratch/project_465002698/venky/projects/arctic/image/'    # Raw mosaics
        # params['images'] = sorted(glob(os.path.join(kwargs['image_dir'], '*.tif')))

        # params['image_dir'] = '/scratch/project_465002698/venky/projects/arctic/rclone_download'    # All mosaics
        params['image_dir'] = '/scratch/project_465002698/venky/projects/arctic/all_mosaics'    # Single mosaics
        files = [
            f for f in glob.glob(f"{params['image_dir']}/**/*mosaic.tif", recursive=True)
            if os.path.dirname(f) != params['image_dir'] and 'browse' not in f
        ]
        print('Total number of images are', len(files))
        # split all images into chunk with the size of 320
        chunks = [files[i:i + params['chunk_size']] for i in range(0, len(files), params['chunk_size'])]
        params['images'] = chunks[params['chunk_id']]
        print(f'Total number of images in chunk id {params["chunk_id"]}', len(params['images'] ))
        print(f'Total number of chunks {len(chunks)}')

        base_dir = '/scratch/project_465002698/venky/projects/arctic/predictions/'
        # params['out_dir'] = os.path.join(base_dir, f"final_{params['keyword']}_{params['ensemble_mode']}")


    elif params['server'] == 'local':
        run_dir = '../runs/'
        base_dir = '/home/venky/Documents/projects/data/arctic/venky_predictions/'
        params['image_dir'] = '../data/test/images/'
        # params['out_dir'] = os.path.join(base_dir, f"test_{params['keyword']}_{params['ensemble_mode']}")
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
        params['out_dir'] = os.path.join(base_dir,  f"{params['keyword']}_{params['pred_setting']}_{params['ensemble_mode']}")
        if not os.path.isdir(params['out_dir']):
            os.makedirs(params['out_dir'])

    # # list of trained models on
    # if params['ensemble']:
    #     pretrained_model = []
    #     for i in range(1, 6):
    #         path = os.path.join(run_dir, f'{params["keyword"]}_fold{i}', f'{params["wt_file"]}.pth')
    #         pretrained_model.append(path)
    #     if params['keyword2'] is not None:
    #         for i in range(1, 6):
    #             path = os.path.join(run_dir, f'{params["keyword2"]}_fold{i}', f'{params["wt_file"]}.pth')
    #             pretrained_model.append(path)

    else:
        pretrained_model = [os.path.join(run_dir,
                                        f'{params["keyword"]}',
                                        f'{params["wt_file"]}.pth')]
    params['pretrained_model'] = pretrained_model
    print(f'Total pretrained models for ensembling are: {len(pretrained_model)}')

    predict.ArcticPredict(**params).run()