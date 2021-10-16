#!/bin/bash

python3 arch/label2obj/train.py --name meronym_128 --continue_train --dataset_mode meronym --dataroot ./datasets/ --gpu_ids 0 --batchSize 1 --tf_log  --niter 10 --niter_decay 10 --no_instance \
 --checkpoints_dir weights --load_size 128 --crop_size 128 --not_sort --no_flip --which_epoch 10
