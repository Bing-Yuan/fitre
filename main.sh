#!/bin/bash

# KFAC-TR
python3 kfac_exp.py --batch-size 200 --epochs 200 --model QAlexNetS --seed 1 --init def --da 1 --damp=0.01 --check-grad --debug

# KFAC-TR
python3 kfac_exp.py --batch-size 200 --epochs 10 --model QAlexNetS --seed 1 --init xavier --damp=0.01 --debug

# SGD
python3 sgd_exp.py --batch-size 200 --epochs 200 --model QAlexNetS --seed 1 --init def --da 1 --lr 0.01 --decay-epoch 49 99 149 --debug
