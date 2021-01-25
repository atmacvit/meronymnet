import sys
from preprocess import *
import os
import numpy as np
import argparse
sys.path.append(os.path.abspath('../../'))
from arch.layoutvae import bboxVAE

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model_path', type=str, help='model load path')
parser.add_argument('--data_path', type=str, help='path to dataset as numpy arrays')

args = parser.parse_args()

if args.model_path is None:
  parser.error('Specify model path')
if args.data_path is None:
  parser.error('Specify data path')


model_path = args.model_path
data_path = args.data_path

tf.config.experimental_run_functions_eagerly(True)
model = BBoxVAE(is_class_condition=True)
x1 = tf.random.normal((32, 24))
x2 = tf.random.normal((32, 24, 4))
x3 = tf.random.normal((32, 10))
model([x1, x2, x3])
model.load_weights(model_path)
test_samples = 10

X_train_test, masks_test, rgb_masks_test, rgb_imgs_test, class_v_test = make_data(object_names, data_path, 'test')
label_count = X_train_test[:, :, 0]
bbox_gt = X_train_test[:, :, 1:]

class_input = class_v_test


model([label_count, bbox_gt, class_input])
out = model.predict([label_count, class_input], batch_size=10)
label_count = np.expand_dims(label_count, axis=2)
class_vec_sketch = []
label_vec_sketch = []
bbx_gen_sketch = []
bbx_gen_sketch_l2im = []

for bbx, pos, cls in zip(out, label_count, class_input):
  
    class_vec_sketch.append(cls)
    label_vec_sketch.append(pos)
    bbxp = bbx * pos * 550.0
    bbxl = centre_object(bbxp, (canvas_size,canvas_size))
    bbxl = bbxl/550.0
    bbx_l = np.array(bbxl)
    bbx_gen_sketch.append(((bbx_l)*pos))

    minx = 1000.0
    miny = 1000.0
    maxx = -1000
    maxy = -1000
    bbx_f = []
    
    for b in bbxp:
        if np.sum(np.array(b))!=0.0:
            if b[0] < minx:
                minx = b[0]
            if b[1] < miny:
                miny = b[1]
                
    for b in bbxp:
        if np.sum(np.array(b))!=0.0:
            x1, y1, x2, y2 = b[0]-minx, b[1]-miny, b[2]-minx, b[3]-miny 
            bbx_f.append([x1, y1, x2, y2])
            if x2 > maxx:
                maxx = x2
            if y2 > maxy:
                maxy = y2
        else:
            bbx_f.append([0, 0, 0, 0])

    for i in range(len(bbx_f)):
        bbx_f[i][0] = bbx_f[i][0]/maxx
        bbx_f[i][1] = bbx_f[i][1]/maxy
        bbx_f[i][2] = bbx_f[i][2]/maxx
        bbx_f[i][3] = bbx_f[i][3]/maxy

    bbx_f = np.array(bbx_f)
    bbx_gen_sketch_l2im.append(((bbx_f)*pos))

class_vec_sketch = np.asarray(class_vec_sketch)
label_vec_sketch = np.asarray(label_vec_sketch)
bbx_gen_sketch = np.asarray(bbx_gen_sketch)
bbx_gen_sketch_l2im = np.asarray(bbx_gen_sketch_l2im)

with open('class_vec_layoutvae', 'wb') as fp:
  pickle.dump(class_vec_sketch, fp)

with open('label_vec_layoutvae', 'wb') as fp:
  pickle.dump(label_vec_sketch, fp)

with open('bbx_gen_layoutvae', 'wb') as fp:
  pickle.dump(bbx_gen_sketch, fp)

with open('bbx_gen_layoutvae_l2im', 'wb') as fp:
  pickle.dump(bbx_gen_sketch_l2im, fp)

  