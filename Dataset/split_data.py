import numpy as np
import pickle

objects = ['cow', 'dog', 'person', 'horse',  'sheep', 'aeroplane', 'bird', 'bicycle', 'cat', 'motorbike', 'car']

for object_name in objects:
    with open(object_name +  '_part_separated_labels', 'rb') as f:
        label = pickle.load(f)
    with open(object_name +  '_part_separated_bbx', 'rb') as f:
        box = pickle.load(f)
    with open(object_name +  '_part_separated_masks', 'rb') as f:
        mask = pickle.load(f)
    with open(object_name +  '_images', 'rb') as f:
        o_images = pickle.load(f)

    size = len(label)
    train_split = int((75/100)*size)
    validation_split = int((10/100)*size)
    test_split = int((15/100)*size)

    #train
    with open(object_name+'_train_label', 'wb') as f:
        pickle.dump(label[0:train_split], f)
    with open(object_name+'_train_bbx', 'wb') as f:
        pickle.dump(box[0:train_split], f)
    with open(object_name+'_train_masks', 'wb') as f:
        pickle.dump(mask[0:train_split], f)
    with open(object_name+'_train_images', 'wb') as f:
        pickle.dump(o_images[0:train_split], f)

    #vaidation
    with open(object_name+'_validation_label', 'wb') as f:
        pickle.dump(label[train_split:train_split+validation_split], f)
    with open(object_name+'_validation_bbx', 'wb') as f:
        pickle.dump(box[train_split:train_split+validation_split], f)
    with open(object_name+'_validation_masks', 'wb') as f:
        pickle.dump(mask[train_split:train_split+validation_split], f)
    with open(object_name+'_validation_images', 'wb') as f:
        pickle.dump(o_images[train_split:train_split+validation_split], f)

    #test
    with open(object_name+'_test_label', 'wb') as f:
        pickle.dump(label[train_split+validation_split::], f)
    with open(object_name+'_test_bbx', 'wb') as f:
        pickle.dump(box[train_split+validation_split::], f)
    with open(object_name+'_test_masks', 'wb') as f:
        pickle.dump(mask[train_split+validation_split::], f)
    with open(object_name+'_test_images', 'wb') as f:
        pickle.dump(o_images[train_split+validation_split::], f)

