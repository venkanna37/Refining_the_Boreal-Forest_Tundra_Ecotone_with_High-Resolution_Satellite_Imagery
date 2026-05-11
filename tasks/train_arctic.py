import sys
import argparse

sys.path.append("..")
from pipelines import training

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patch_size", type=int, default=256) # this is train patch size
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--stretch_setting", type=str, default=1)
    parser.add_argument("--runs_dir", type=str,
                        default='/scratch/project_465002698/venky/projects/arctic/runs')

    # model parameters
    parser.add_argument("--weight", type=float, default=2)  # weight to boundary pixels
    parser.add_argument("--load", type=str, help="load model weights")
    parser.add_argument("--model_size", type=str, default='small')
    parser.add_argument("--loss_function", type=str, default='tversky')
    parser.add_argument("--alpha", type=float, help='Parameter for tversky loss',
                        default=0.7)
    parser.add_argument("--geom_aug", action=argparse.BooleanOptionalAction,
                        help="Apply geometry augmentation or not", default=False)

    params = vars(parser.parse_args())
    train = training.Training(**params)
    train.train()