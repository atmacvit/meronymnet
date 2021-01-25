import sys
import os
sys.path.append(os.path.abspath('../../'))
print(sys.path)
from arch.layoutvae import bboxVAE
from preprocess import *
import argparse




parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, help='path to dataset as numpy arrays')
parser.add_argument('--model_path', type=str, default='.', help='model save directory')
parser.add_argument('--resume_train', type=str, default='s', help='s for train from scratch and l for resume from last saved ckpt')


args = parser.parse_args()
if args.data_path is None:
  parser.error('path to data missing')

data_path = args.data_path
model_path = args.model_path
resume_train = args.resume_train

checkpoint_filepath = model_path + '/' + 'layoutvae_pascalvoc.h5'

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_filepath, monitor='loss', verbose=1,
    save_best_only=True, mode='auto', period=1)

X_train, masks, rgb_masks, rgb_imgs, class_v = make_data(object_names, data_path, 'train')

X_train_val, masks_val, rgb_masks_val, rgb_imgs_val, class_v_val = make_data(object_names, data_path, 'valid')

label_count = X_train[:, :, 0]
bbox_gt = X_train[:, :, 1:]

label_count_val = X_train_val[:, :, 0]
bbox_gt_val = X_train_val[:, :, 1:]

tf.config.experimental_run_functions_eagerly(True)
model = BBoxVAE(is_class_condition=True)

x1 = tf.random.normal((32, 24))
x2 = tf.random.normal((32, 24, 4))
x3 = tf.random.normal((32, 10))
model([x1, x2, x3])

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt)
if resume_train=='l':
  model.load_weights(checkpoint_filepath)
  
model([label_count, bbox_gt, class_v])

model.fit(x=[label_count, bbox_gt, class_v],validation_data=([label_count_val,bbox_gt_val,class_v_val]),shuffle=True,batch_size=32,epochs=150, callbacks=[model_checkpoint_callback])



