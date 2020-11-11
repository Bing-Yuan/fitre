""" Shared snippets for experiments, regardless of SGD or KFAC. """

import argparse
import logging
import random
import time
from math import floor
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

from nn_models import *
import resnet1 as rsn1
import resnet2 as rsn2


class ExpArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        """ Override constructor to add more arguments or modify existing defaults.
        :param log_dir: if not None, the directory for log file to dump, the log file name is determined
        """
        super().__init__(*args, **kwargs)

        self.add_argument('--benchmark', type=str, choices=['cifar10', 'cifar100'], default='cifar10',
                          help='the benchmark to run')
        self.add_argument('--opt', type=str, choices=['sgd', 'kfac'],
                          help='the optimizer to use')

        self.add_argument('--model', type=str, default='QAlexNetS',
                          choices=['QAlexNetS', 'QAlexNetSb', 'VGG16', 'VGG16b', 'ResNet18', 'ResNet18b'],
                          help='neural network model')
        self.add_argument('--init', type=str, default='def', choices=['def', 'km', 'xavier', '0', '1'],
                          help='init network model')

        self.add_argument('--batch-size', type=int, default=1000,
                          help='input batch size for training (default: 1000)')
        self.add_argument('--test-batch-size', type=int, default=200,
                          help='input batch size for testing (default: 200)')
        self.add_argument('--epochs', type=int, default=10,
                          help='number of epochs to train (default: 10)')
        self.add_argument('--da', type=int, default=0)
        self.add_argument('--decay-epoch', type=int, nargs='+', default=[99, 199, 299],
                          help='learning rate decay epoch')

        self.add_argument('--seed', type=int, default=1, metavar='S',
                          help='random seed (default: 1)')
        self.add_argument('--no-cuda', action='store_true', default=False,
                          help='disables CUDA training')
        self.add_argument('--log-dir', type=str, default='logs',
                          help='directory to save logs')
        self.add_argument('--resume', type=int, default=0)

        group = self.add_mutually_exclusive_group()
        group.add_argument("--quiet", action="store_true", default=False,
                           help='show warning level logs (default: info)')
        group.add_argument("--debug", action="store_true", default=False,
                           help='show debug level logs (default: info)')
        return

    def parse_args(self, args=None, namespace=None):
        res = super().parse_args(args, namespace)
        self.random_seed(res.seed)
        self.setup_logger(res)

        # extra processing
        res.cuda = (not res.no_cuda) and torch.cuda.is_available()
        return res

    @staticmethod
    def random_seed(seed):
        """ Set random seed for all. """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        return

    @staticmethod
    def setup_logger(args: argparse.Namespace):
        logger = logging.getLogger()
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')

        if args.quiet:
            # default to be warning level
            pass
        elif args.debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        args.stamp = f'{args.opt}-{args.benchmark}-{timestamp}'
        logger.handlers = []  # reset, otherwise it may duplicate many times when calling setup_logger() multiple times

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = Path(log_dir, f'{args.stamp}.log')
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return
    pass


def get_net(args: argparse.Namespace) -> nn.Module:
    num_classes = {
        'cifar10': 10,
        'cifar100': 100
    }[args.benchmark]

    model_list = {
        'QAlexNetS': lambda: QAlexNetS(num_classes=num_classes),
        'QAlexNetSb': lambda: QAlexNetSb(num_classes=num_classes),
        'VGG16': lambda: VGG('VGG16', num_class=num_classes),
        'VGG16b': lambda: VGGb('VGG16', num_class=num_classes),
        'ResNet18': lambda: rsn2.resnet(depth=20, num_classes=num_classes),
        'ResNet18b': lambda: rsn1.resnet(depth=20, num_classes=num_classes),
    }
    model = model_list[args.model]()

    if args.init == "km":
        model.apply(init_params)
    elif args.init == "xavier":
        model.apply(normal_init)
    elif args.init == "0":
        model.apply(zeros_init)
    elif args.init == '1':
        model.apply(ones_init)
    return model


def get_data_loader(args: argparse.Namespace, train: bool, for_training: bool) -> DataLoader:
    """ Note: old experiment had data on '/scratch2/skylasa/solvers/cnn-data-augmented'. """
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    if args.benchmark == 'cifar10':
        if args.da == 0:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])
        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        if train and for_training:
            ds = CIFAR10('./data', train=True, download=True, transform=transform_train)
            return DataLoader(ds, batch_size=args.batch_size, shuffle=True, **kwargs)
        if train and (not for_training):
            ds = CIFAR10('./data', train=True, download=True, transform=transform_test)
            return DataLoader(ds, batch_size=args.batch_size, shuffle=False, **kwargs)
        if not train:
            ds = CIFAR10('./data', train=False, download=True, transform=transform_test)
            return DataLoader(ds, batch_size=args.batch_size, shuffle=False, **kwargs)

    if args.benchmark == 'cifar100':
        if args.da == 0:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ])
        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        if train and for_training:
            ds = CIFAR100('./data', train=True, download=True, transform=transform_train)
            return DataLoader(ds, batch_size=args.batch_size, shuffle=True, **kwargs)
        if train and (not for_training):
            ds = CIFAR100('./data', train=True, download=True, transform=transform_test)
            return DataLoader(ds, batch_size=args.batch_size, shuffle=False, **kwargs)
        if not train:
            ds = CIFAR100('./data', train=False, download=True, transform=transform_test)
            return DataLoader(ds, batch_size=args.batch_size, shuffle=False, **kwargs)

    raise NotImplementedError(f'Benchmark {args.benchmark} unsupported yet.')


def eval_test(model, test_loader, loss_f, device):
    model.eval()
    test_loss = 0.
    correct = 0
    total = len(test_loader.dataset)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            test_loss += loss_f(output, target).item() * len(data)  # sum up batch loss
            correct += (pred == target).sum().item()

    test_loss /= total
    acc = 1.0 * correct / total
    model.train()
    return test_loss, acc
