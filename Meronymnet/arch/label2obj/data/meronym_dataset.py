"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset
import os

class MeronymDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        if is_train:
            parser.set_defaults(load_size=286)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=25)
        # parser.set_defaults(semantic_nc=24)
        parser.set_defaults(contain_dontcare_label=False)
        parser.set_defaults(cache_filelist_read=False)
        parser.set_defaults(cache_filelist_write=False)
        parser.set_defaults(no_instance=True)
        return parser

    def get_paths(self, opt):
        root = opt.dataroot
        phase = '1' if opt.phase == 'test' else ''

        label_dir = os.path.join(root, 'visw/%s'%phase)
        label_paths_all = make_dataset(label_dir, recursive=True)
        label_paths = [p for p in label_paths_all if p.endswith('.png')]
        # print(label_paths)

        # instance_paths = []

        image_dir = os.path.join(root, 'visw/%s'%phase)
        image_paths_all = make_dataset(image_dir, recursive=True)
        image_paths = [im for im in image_paths_all if im.endswith('.png')]

        instance_paths = []  # don't use instance map for ade20k

        for i in label_paths:
            if 'aeroplane' in i:
                instance_paths.append(0)
            if 'bicycle' in i:
                instance_paths.append(1)
            if 'bird' in i:
                instance_paths.append(2)
            if 'cat' in i:
                instance_paths.append(3)
            if 'cow' in i:
                instance_paths.append(4)
            if 'dog' in i:
                instance_paths.append(5)
            if 'horse' in i:
                instance_paths.append(6)
            if 'motorbike' in i:
                instance_paths.append(7)
            if 'person' in i:
                instance_paths.append(8)
            if 'sheep' in i:
                instance_paths.append(9)

        return label_paths, image_paths, instance_paths

    # In ADE20k, 'unknown' label is of value 0.
    # Change the 'unknown' label to the last label to match other datasets.
    # we ignored this for the indoor portion!
    def postprocess(self, input_dict):
        label = input_dict['label']
