import os
import argparse
import torch
import torch.nn as nn
from data.dataset import get_pointnet_dataloader, get_ocnn_dataloader
from models.ocnn import OCNNClassifier
from models.pointnet2 import get_model
from train import train, evaluate


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--model', type=str, default='ocnn', help='Model: ocnn or pointnet2')
    parser.add_argument('--data_dir', type=str, default='data/ModelNet40.ply.normalize', help='Data directory')
    parser.add_argument('--log_dir', type=str, default='runs', help='Tensorboard log directory')

    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model
    if args.model == 'ocnn':
        model = OCNNClassifier()
    elif args.model == 'pointnet2':
        model = get_model(num_class=40, normal_channel=False)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model.to(DEVICE)

    # dataloader - 根据 model 选择对应的 dataloader 和数据目录
    if args.model == 'ocnn':
        data_dir = 'data/ModelNet40.ply.normalize'
        train_loader = get_ocnn_dataloader(data_dir, split='train')
        test_loader = get_ocnn_dataloader(data_dir, split='test')
    else:  # pointnet2
        data_dir = 'data/ModelNet40'
        train_loader = get_pointnet_dataloader(
            data_dir, batch_size=32, num_points=1024, split='train', num_workers=4)
        test_loader = get_pointnet_dataloader(
            data_dir, batch_size=32, num_points=1024, split='test', num_workers=4)

    train(model, train_loader, test_loader, DEVICE, args)
    evaluate(model, test_loader, DEVICE, args)


if __name__ == "__main__":
    main()