"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
from collections import OrderedDict
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.pix2pix_trainer import Pix2PixTrainer
from models.pix2pix_model import Pix2PixModel
# from SBGAN.modules.fid_score import calculate_fid_given_acts, get_activations

# parse options
opt = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)

# create trainer for our model
trainer = Pix2PixTrainer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

model = Pix2PixModel(opt)
model.eval()
# create tool for visualization
visualizer = Visualizer(opt)

# for epoch in iter_counter.training_epochs():
epoch = 1
iter_counter.record_epoch_start(epoch)
for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
    # iter_counter.record_one_iteration()
    
    generated = model(data_i, mode='inference')

    img_path = data_i['path']
    for b in range(generated.shape[0]):
        print('process image... %s' % img_path[b])
        visuals = OrderedDict([('input_label', data_i['label'][b]),
                               ('synthesized_image', generated[b])])
        # visualizer.save_images(webpage, visuals, img_path[b:b + 1])
        visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far, data_i['path'])
