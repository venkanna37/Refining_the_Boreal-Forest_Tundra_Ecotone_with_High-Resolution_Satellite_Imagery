import os
import sys
import argparse

sys.path.append("..")
from tools import training


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general parameters
    parser.add_argument("--keyword", type=str, default="test_run",
                        help='Keyword for naming checkpoint and run in weights and biases')
    
    # data parameters
    parser.add_argument("--batch_size", type=int, default=64,
                        help='Batch size for training')
    parser.add_argument("--patch_size", type=int, default=256,
                        help='Patch size while training')
    parser.add_argument("--data_dir", type=str, default='./datasets',
                        help='Directory to the datasets')

    # model and training parameters
    parser.add_argument("--epochs", type=int, default=20000,
                        help='Number of epochs')
    parser.add_argument("--num_workers", type=int, default=4,
                        help='Number of workers')
    parser.add_argument("--learning_rate", type=float, default=0.000009,
                        help='Learning rate')
    parser.add_argument("--boundary_weight", type=float, default=15,
                        help='Weight for boundary pixels of trees')
    parser.add_argument("--target_weight", type=float, default=1,
                        help='Weight for target pixels of trees')
    parser.add_argument("--model_size", type=str, default='small',
                        help='Size of the model, there are two different size of model')
    parser.add_argument("--model_name", type=str, default='unet_elu',
                        help='Model with different activation and batch normalization',
                        choices=['unet_relu_bn', 'unet_elu_bn', 'unet_elu'])
    # unet_relu_bn: U-Net with ReLU activation and batch normalization
    # unet_elu_bn: U-Net with ELU activation and batch normalization
    # unet_elu: U-Net with ELU activation and no batch normalization
    parser.add_argument("--loss_function", type=str, default='tversky',
                        help='Loss function',
                        choices=['tversky', 'ce'])
    parser.add_argument("--alpha", type=float, default=0.7,
                        help='tversky loss param')
    parser.add_argument('--checkpoints_dir', type=str, default='./runs',
                        help='output directory to save check points and logs')

    # visualisation params
    parser.add_argument("--use_wb", action=argparse.BooleanOptionalAction, default=False,
                        help='Use weights and biases for visualisation')
    params = vars(parser.parse_args())
    train = training.Training(**params)
    train.train()