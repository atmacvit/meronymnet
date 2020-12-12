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


train_loss = []
val_loss = []
klw = frange_cycle_linear(nb_epochs)
config_proto = tf.ConfigProto()
off = rewriter_config_pb2.RewriterConfig.OFF
config_proto.graph_options.rewrite_options.arithmetic_optimization = off
config=config_proto

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #saver.restore(sess, "./maskvae.ckpt")
    klw = 0.0
    bbxw = 64
    icoef = 0
    for i in range(1, nb_epochs+1):
        bounding_boxes, labels,masks,classes = shuffle_latent(bounding_boxes, labels,masks,classes)
        klw1 = np.asarray([klw], dtype= 'float32')

        bbxw1 = np.asarray([bbxw], dtype= 'float32')

        nb_batches = nb_train // batch_size
        
        for batch_idx in range(nb_batches):
            
            start_idx = batch_idx*batch_size
            end_idx = batch_idx*batch_size + batch_size
            
            b_in = bounding_boxes[start_idx:end_idx]
            mx_in = masks[start_idx:end_idx]
            cls_in = classes[start_idx:end_idx]
            
            sess.run(train_op, feed_dict = {true_masks: mx_in,
                true_bbxs: b_in,true_classes: cls_in, kl_weight: klw,
                                            sketch_weight:bbxw1,
                                           cond_bbxs:b_in,cond_classes:cls_in
                                           })
        
        kl_loss_, mask_loss_= sess.run([kl_loss, mask_loss],feed_dict= {true_masks: mx_in,
                true_bbxs: b_in, true_classes: cls_in, kl_weight: klw,
                                            sketch_weight:bbxw1,
                                           cond_bbxs:b_in ,cond_classes:cls_in
                                           })

        train_loss.append(mask_loss_)
        clear_output()

        bounding_boxes_val, labels_val,masks_val,classes_val = shuffle_latent(bounding_boxes_val, labels_val,masks_val,classes_val)
        
        start_idx = 0
        end_idx = start_idx + batch_size
        
        b_in = bounding_boxes_val[start_idx:end_idx]
        l_in = labels_val[start_idx:end_idx]
        mx_in = masks_val[start_idx:end_idx]
        cls_in = classes_val[start_idx:end_idx]

        kl_loss_val, mask_loss_val= sess.run([kl_loss, mask_loss],feed_dict= {true_masks: mx_in,
        true_bbxs: b_in, true_classes: cls_in, kl_weight: klw,
                                    sketch_weight:bbxw1,
                                    cond_bbxs:b_in ,cond_classes:cls_in
                                    })
        val_loss.append(mask_loss_val)
        plt.plot(np.asarray(train_loss))
        plt.plot(np.asarray(val_loss))
        plt.show()

        print('epoch:',i,' ',icoef,'kl_weight', klw, 'kl:', kl_loss_,  'train:', mask_loss_, 'val:',mask_loss_val)
        if i % 10 == 0:
            if kl_loss_ > 10.0 and abs(mask_loss_ - mask_loss_val) < 0.1:
              if klw<0.5:
                klw = klw + 0.01
            save_path = saver.save(sess,  "./maskvae.ckpt")
