#!/bin/bash

# KFAC-TR
python cifar10_str_batch_kfac.py --batch-size 200 --epochs 200 --model ResNet18 --seed 1 --init def --da 1 --damp=0.01 --check-grad

# KFAC-TR
python cifar10_str_batch_kfac.py --batch-size 200 --epochs 10 --model QAlexNetS --seed 1 --init xavier --damp=0.01

# SGD
python cifar10_sgd.py --batch-size 200 --epochs 200 --model ResNet18b --seed 1 --init def --da 1 --lr 0.01 --decay-epoch 49 99 149
