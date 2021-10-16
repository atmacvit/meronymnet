import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import *
import pickle
from IPython.display import clear_output
import math
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.keras import backend as K
import sys
import cv2

def shuffle_latent(a, b, c):
     p = np.random.permutation(len(a))
     return a[p], b[p], c[p]

def bounder(img):
    result = np.where(img<0.45)
    listOfCoordinates = list(zip(result[0], result[1]))
    for cord in listOfCoordinates:
        img[cord] = 0
    result1 = np.where(img>=0.45)
    listOfCoordinates1 = list(zip(result1[0], result1[1]))
    for cord in listOfCoordinates1:
        img[cord] = 1
    return img

def add_images(canvas,img, ii):
    result = np.where(img!=0)
    listOfCoordinates = list(zip(result[0], result[1]))
    for cord in listOfCoordinates:
        canvas[cord] = ii
    return canvas

label_to_color ={0: (1,1,1),
 1: (1, 0, 0),
 2: (0.737, 0.561, 0.561),
 3: (0.255, 0.412, 0.882),
 4: (0.545, 0.271, 0.0745),
 5: (0.98, 0.502, 0.447),
 6: (0.98, 0.643, 0.376),
 7: (0.18, 0.545, 0.341),
 8: (0.502, 0, 0.502),
 9: (0.627, 0.322, 0.176),
 10:(0.753, 0.753, 0.753),
 11:(0.529, 0.808, 0.922),
 12:(0.416, 0.353, 0.804),
 13:(0.439, 0.502, 0.565),
 14:(0.784, 0.302, 0.565),
 15:(0.867, 0.627, 0.867),
 16:(0, 1, 0.498),
 17:(0.275, 0.51, 0.706),
 18:(0.824, 0.706, 0.549),
 19:(0, 0.502, 0.502),
 20:(0.847, 0.749, 0.847),
 21:(1, 0.388, 0.278),
 22:(0.251, 0.878, 0.816),
 23:(0.933, 0.51, 0.933),
 24:(0.961, 0.871, 0.702)}

def arrangement(a, b, object_name):
    if object_name=='cow' or object_name=='sheep':
        p = [20,22,21,23,10,11,19,18,13,12,14,16,15,17,8,9,0,7,3,4,5,6,1,2]
    elif object_name=='bird':
        p = [ 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,  4, 12,  8,  9, 10, 11,5,  6,  7,  0,  1,  2,  3 ]
    elif object_name=='person':
        p = [ 10,11,19,22,18,21,20,23,13,16,12,15,14,17,9,0,7,8,5,6,1,2,3,4 ]
    elif object_name=='cat':
        p = [ 17,18,19,20,21,22,23,6, 7, 16, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5]
    elif object_name=='dog':
        p = [ 18,19,20,21,22,23,6, 7, 16, 8, 9, 10, 11, 12, 13, 14, 15, 0, 17,1, 2, 3, 4, 5]
    elif object_name=='horse':
        p = [ 22,21,23,10,11,18,13,12,14,16,15,17,8,9,0,7,3,4,5,6,19,20,1,2]
    elif object_name=='aeroplane':
        p = [ 0,20,22,21,23,10,11,19,18,13,12,14,16,15,17,8,9,7,3,4,5,6,1,2]
    elif object_name=='motorbike':
        p = [ 15,20,22,21,23,10,11,19,18,13,12,14,16,17,8,9,0,7,3,4,5,6,1,2 ]
    elif object_name=='bicycle':
        p = [22,21,23,10,11,18,13,12,14,16,15,17,8,9,0,7,3,4,5,6,19,20,1,2 ]
    else:
      print("error")
    return a[p], b[p]

def label_2_image(img):
  rgb_img = np.zeros((img.shape[0],img.shape[1], 3)) 
  for key in label_to_color.keys():
      rgb_img[img == key] = label_to_color[key]
  return rgb_img


batch_size = 1
max_num_node = 24

true_maps = tf.placeholder(tf.float32, [batch_size, max_num_node, 64, 64, 1])
true_masks = tf.placeholder(tf.float32, [batch_size, max_num_node, 64, 64, 1])
true_edges = tf.placeholder(tf.float32, [batch_size, max_num_node, 64, 64, 1])

true_bbxs = tf.placeholder(tf.float32, [batch_size, max_num_node, 4])
cond_bbxs = tf.placeholder(tf.float32, [batch_size, max_num_node, 4])

true_lbls = tf.placeholder(tf.float32, [batch_size, max_num_node, 1])
cond_lbls = tf.placeholder(tf.float32, [batch_size, max_num_node, 1])

true_classes = tf.placeholder(tf.float32, [batch_size, 10])
cond_classes = tf.placeholder(tf.float32, [batch_size, 10])

latent_input = tf.placeholder(tf.float32, [batch_size, 128])

kl_weight = tf.placeholder(tf.float32)
sketch_weight = tf.placeholder(tf.float32 , [1])

def sampling(z_mean, z_log_var):
    epsilon = tf.random_normal(tf.shape(z_log_var), name="epsilon")
    return z_mean + epsilon * tf.exp(z_log_var)

def VAE(maps_, bbxs_, cond_, cond_bbx, cond_class):
    rnn_bbxs = Bidirectional(GRU(4, return_sequences=True))(bbxs_)
    concatenated_bbx_lbl = rnn_bbxs
    dense_cond = Dense(64, activation='tanh')(cond_)
    enc = TimeDistributed(Conv2D(8, kernel_size=3))(maps_)
    enc = TimeDistributed(tf.layers.BatchNormalization(trainable = False))(enc)
    enc = TimeDistributed(Activation('relu'))(enc)


    enc = TimeDistributed(Conv2D(16, kernel_size=3))(enc)
    enc = TimeDistributed(tf.layers.BatchNormalization(trainable = False))(enc)
    enc = TimeDistributed(Activation('relu'))(enc)


    enc = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(enc)

    enc = TimeDistributed(Conv2D(32, kernel_size=3, activation='relu'))(enc)
    enc = TimeDistributed(tf.layers.BatchNormalization(trainable = False))(enc)
    enc = TimeDistributed(Activation('relu'))(enc)
    enc = TimeDistributed(Flatten())(enc)
    TDD = TimeDistributed(Dense(64, activation='relu', name = 'encoded_bitmaps'))
    dense_enc_maps = TDD(enc)

    BGRU = Bidirectional(GRU(32, return_sequences=True))
    rnn_maps = BGRU(dense_enc_maps)

    D = Dense(64, activation='tanh')
    attention = D(concatenated_bbx_lbl)
    sent_representation = Multiply()([rnn_maps, attention])
    sent_representation = Multiply()([sent_representation, dense_cond])
    images_with_attention = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(128,))(sent_representation)

    z_mean = Dense(64, activation='tanh')(images_with_attention)
    z_log_var = Dense(64, activation='tanh')(images_with_attention)

    z_latent = sampling(z_mean, z_log_var)
    cond_bbx = Lambda(lambda xin: K.sum(xin, axis=-1), output_shape=(4,))(cond_bbx)
    cond_cat = cond_bbx

    cond_fully_cat = Dense(64, activation='relu')(cond_cat)
    cond_class_ = Dense(64, activation='relu')(cond_class)
    conditioned_z = concatenate([cond_fully_cat, z_latent], axis=-1)
    conditioned_z = concatenate([conditioned_z, cond_class_], axis=-1)
    decoded = RepeatVector(max_num_node)(conditioned_z)
    decoded = Bidirectional(GRU(32, return_sequences=True))(decoded)
    dec_dense = TimeDistributed(Dense(25088, activation='relu',  name = 'encoding'))(decoded)
    dec_conv = TimeDistributed(Reshape((28, 28, 32)))(dec_dense)

    dec = TimeDistributed(Conv2DTranspose(32, kernel_size=3, padding='same'))(dec_conv)
    dec = TimeDistributed(tf.layers.BatchNormalization(trainable = False))(dec)
    dec = TimeDistributed(Activation('relu'))(dec)

    dec = TimeDistributed(Conv2DTranspose(16, kernel_size=3))(dec)
    dec = TimeDistributed(tf.layers.BatchNormalization(trainable = False))(dec)
    dec = TimeDistributed(Activation('relu'))(dec)

    dec = TimeDistributed(UpSampling2D(size=(2, 2)))(dec)

    dec = TimeDistributed(Conv2DTranspose(8, kernel_size=3))(dec)
    dec = TimeDistributed(tf.layers.BatchNormalization(trainable = False))(dec)
    dec = TimeDistributed(Activation('relu'))(dec)


    decoder_bitmaps = TimeDistributed(Conv2DTranspose(1, kernel_size=3, activation='sigmoid', name = 'decoded_mask'))(dec)
    return  decoder_bitmaps, z_mean, z_log_var, z_latent

pred_masks, z_mean, z_logvar, z_latent = VAE( true_masks, true_bbxs, true_classes, cond_bbxs, cond_classes )

def frange_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio)

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L

kl_loss = tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(tf.exp(z_logvar)) - 2*(z_logvar) - 1, axis=1))
mask_loss = tf.reduce_mean(tf.keras.backend.binary_crossentropy(true_masks,pred_masks))

reconstuction_loss = sketch_weight*(mask_loss)+ (kl_weight*kl_loss)

lr = 0.001
train_op = tf.train.AdamOptimizer(lr).minimize(reconstuction_loss)

nb_train = masks.shape[0]
saver = tf.train.Saver()
nb_epochs = 200

X_train, class_v, masks, X_train_val, class_v_val, masks_val = make_data()

X_train, class_v, masks = shuffle_latent(X_train, class_v, masks)

labels = X_train[:,:,:1] 
bounding_boxes = X_train[:,:,1:]
masks = masks
classes = class_v

labels_val = X_train_val[:,:,:1] 
bounding_boxes_val = X_train_val[:,:,1:]
masks_val = masks_val
classes_val = class_v_val

object_names = ['cow','sheep','bird','person','cat','dog','horse','aeroplane','motorbike','bicycle']#,'sheep','bird','person','cat','dog','horse','cow']

class_dic = {0:'cow',1:'sheep',2:'bird',3:'person',4:'cat',5:'dog',6:'horse',7:'aeroplane',8:'motorbike',9:'bicycle'}

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  saver.restore(sess, "../models/save_bitmaps_colab_new.ckpt")
  start = 0
  for ind_ in range(len(bb)//batch_size):

    start_idx = ind_*batch_size
    end_idx = start_idx + batch_size

    b_in = np.asarray(bb[start_idx:end_idx])
    c_in = np.asarray(cc[start_idx:end_idx])
    for r in range(0,20):
        np.random.seed(r)
        z = np.random.normal(0,1,[batch_size, 64])
        mx= sess.run([pred_masks],feed_dict= {z_latent:z,cond_bbxs: b_in/canvas_size, cond_classes:c_in})
        images= []
        for j in range(batch_size):
            canvas = np.zeros((canvas_size, canvas_size), dtype= 'float32')
            bb_in, mmx = arrangement(b_in[j], mx[0][j], class_dic[np.argmax(c_in)])
            try:
                for i in range(24):
                    x_min, y_min, x_max, y_max = bb_in[i]
                    if x_max-x_min > 0 and y_max-y_min>0:
                        x, y = canvas[ int(y_min):int(y_max), int(x_min):int(x_max) ].shape
                        canvas[ int(y_min):int(y_max), int(x_min):int(x_max) ] = add_images(canvas[ int(y_min):int(y_max), int(x_min):int(x_max)  ],cv2.resize(bounder(np.squeeze(mmx[i]))*(i+1), (y,x)), i+1)
            except:
                print('no problem')
            images.append(canvas)
            sza = 10
            plt.figure(num=None, figsize=(sza, sza))
            plt.axis('off')
            plt.imshow(label_2_image(canvas))
            plt.savefig('../maskResults/'+class_dic[np.argmax(c_in)]+'/'+class_dic[np.argmax(c_in)]+'_'+str(ind_)+'.png')
            plt.show()
