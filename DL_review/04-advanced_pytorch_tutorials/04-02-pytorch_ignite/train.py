import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from model import ImageClassifier
from trainer import Trainer
from data_loader import get_loaders

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)

    p.add_argument('--train_ratio', type=float, default=.8)

    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=20)
    p.add_argument('--verbose', type=int, default=2)

    config = p.parse_args()

    return config


def main(config):
    # Set device based on user defined configuration.
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    train_loader, valid_loader, test_loader = get_loaders(config)

    print("Train:", len(train_loader.dataset))
    print("Valid:", len(valid_loader.dataset))
    print("Test:", len(test_loader.dataset))
    
    # model.py
    model = ImageClassifier(28**2, 10).to(device)
    optimizer = optim.Adam(model.parameters())
    crit = nn.NLLLoss()

    trainer = Trainer(config)
    trainer.train(model, crit, optimizer, train_loader, valid_loader)

if __name__ == '__main__':
    config = define_argparser()
    main(config)
