#%tensorflow_version 1.x
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import *
import pickle
import math
from tensorflow.keras import backend as K
import sys
import cv2

batch_size = 1
latent_dim = 64
label_size = 1
bbx_size = 4
class_size = 10

true_node = tf.placeholder(tf.float32, [batch_size, max_num_node, label_size + bbx_size])

true_class = tf.placeholder(tf.float32 , [batch_size, max_num_node,1])

true_classpred = tf.placeholder(tf.float32 , [batch_size, max_num_node,1])

true_edge = tf.placeholder(tf.float32 , [batch_size, max_num_node, max_num_node])

class_vec = tf.placeholder(tf.float32 , [batch_size, class_size])
class_vecpred = tf.placeholder(tf.float32 , [batch_size, class_size])

kl_weight = tf.placeholder(tf.float32)

dim_vec = tf.placeholder(tf.float32 , [batch_size, 2])
keep_prob = tf.placeholder(tf.float32)

def init_weights(shape):
    init = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(init)

def init_bias(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='VALID')

def max_pool_2b2(x):
    return tf.nn.max_pool(x , ksize = [1,2,2,1] , strides= [1,2,2,1],padding='SAME')

def conv_layer(input_x , shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x,W) + b)

def calc_num_wts():    
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("Total number of trainable parameters:", total_parameters)

def sampling(z_mean, z_log_var):
    epsilon = tf.random_normal(tf.shape(z_log_var), name="epsilon")
    return z_mean + epsilon * tf.exp(z_log_var)

def GCLayer(E, X, out_dims):
    W = tf.Variable(np.random.normal(0, 0.2, size=[X.shape[-1], out_dims]), dtype=tf.float32, name='W')
    W = tf.tile(tf.expand_dims(W, axis=0), [tf.shape(X)[0], 1, 1])
    
    T = tf.convert_to_tensor(tf.keras.backend.sum(E, axis=-1))
    T = tf.convert_to_tensor(tf.linalg.diag(T))
    T = tf.matrix_inverse(T)

    EX = tf.matmul(T, E)
    EX1 = tf.matmul(EX, X)
    EXW = tf.matmul(EX1, W)
    X_out = tf.nn.relu(EXW)

    return X_out

def encoder(E, X_DATA, latent_dim, class_info):
    X1 = GCLayer(E, X_DATA, 32)
    X2 = GCLayer(E, X1, 16)
    
    BOXES =X_DATA[:,:, label_size::]
    BOXES =tf.nn.relu(tf.layers.dense(BOXES, 16))
    
    LABELS =X_DATA[:,:, :label_size]
    LABELS =tf.nn.relu(tf.layers.dense(LABELS, 16))

    MIX = tf.keras.layers.Add()([BOXES, LABELS])

    MIX_FLAT = tf.reshape(MIX, [-1, MIX.shape[1]*MIX.shape[2]])
    MIX_DENSE = tf.nn.relu(tf.layers.dense(MIX_FLAT, 128))

    
    X2_f = tf.reshape(X2, [-1, X2.shape[1]*X2.shape[2]])
    X2_f = tf.keras.layers.concatenate([class_info, X2_f], axis = -1)
    X3 = tf.nn.relu(tf.layers.dense(X2_f, 128))
    X4 = tf.keras.layers.Add()([MIX_DENSE, X3])
    X5 = tf.nn.relu(tf.layers.dense(X4, 128))
    X5 = tf.nn.relu(tf.layers.dense(X5, 128))
        
    z_mean = tf.nn.relu(tf.layers.dense(X5, latent_dim))
    z_logvar = tf.nn.relu(tf.layers.dense(X5, latent_dim))
    
    return z_mean, z_logvar

def decoder(z_latent, num_nodes):
    x1 = tf.nn.relu(tf.layers.dense(z_latent, 128))
    x3d = tf.nn.relu(tf.layers.dense(x1, 128))
    x3 = tf.nn.relu(tf.layers.dense(x3d, 128))
    
    x_bbx = tf.nn.sigmoid(tf.layers.dense(x3, num_nodes*(bbx_size)))
    x_bbx = tf.reshape(x_bbx, [-1, num_nodes, bbx_size])
    
    x_lbl = tf.nn.sigmoid(tf.layers.dense(x3, num_nodes*(label_size)))
    x_lbl = tf.reshape(x_lbl, [-1, num_nodes, label_size])
    
    x_edge = tf.nn.sigmoid(tf.layers.dense(x3, num_nodes*num_nodes))
    x_edge = tf.reshape(x_edge, [-1, num_nodes, num_nodes])
    
    class_ = tf.nn.softmax(tf.layers.dense(x3, class_size))
    
    return x_bbx, x_lbl, x_edge, class_

def conditioning(condition, out_dims):
    W = tf.Variable(np.random.normal(0, 0.2, size=[condition.shape[-1], out_dims]), dtype=tf.float32)
    B = tf.Variable(tf.constant(0.1, shape=[out_dims]))
    
    W = tf.tile(tf.expand_dims(W, axis=0), [tf.shape(condition)[0], 1, 1])
    
    weighted_condition = tf.matmul(condition, W) + B
    return weighted_condition

def condition_z(condition):
    weighted_condition = conditioning(condition, 32)
    reshaped_condition = tf.reshape(weighted_condition, [-1, weighted_condition.shape[1]*weighted_condition.shape[2]])
    return tf.nn.relu(tf.layers.dense(reshaped_condition, 64))

def AutoEncoder(E, X, latent_dimm, condition, class_condition):
    z_mean, z_logvar = encoder(E, X, latent_dimm, class_condition)
    z_latent = sampling(z_mean, z_logvar)
    condition_ = tf.reshape(condition, (-1, condition.shape[1]*condition.shape[2]))
    conditioned_z = tf.keras.layers.concatenate([condition_, z_latent], axis = -1)
    conditioned_z = tf.keras.layers.concatenate([class_condition, conditioned_z], axis = -1)
    node_box_r, node_cls_r, E_recons, class_ = decoder(conditioned_z, max_num_node)    
    return node_box_r, node_cls_r, E_recons, z_latent, z_mean, z_logvar,conditioned_z,class_

node_box_r, node_cls_r, edge_r, z_latent, z_mean, z_logvar, conditioned_z, class_= AutoEncoder(true_edge, true_node, latent_dim,  true_class, class_vec)

node_cls_t = true_node[:, :, :label_size]
node_box_t = true_node[:, :, label_size:]

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

def graph_loss(A_true, A_pred):    
    diag_elem = tf.zeros(A_pred.shape[0:-1])
    diag_elem = tf.cast(diag_elem, tf.float32)
    
    true_nodes = tf.linalg.diag_part(A_true)
    pred_nodes = tf.linalg.diag_part(A_pred)
    
    true_edges = tf.matrix_set_diag(A_true, diag_elem, name='true_edges')
    pred_edges = tf.matrix_set_diag(A_pred, diag_elem, name='pred_edges')

    node_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=true_nodes, logits=pred_nodes))
    edge_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=true_edges, logits=pred_edges))
    
    k = A_true.shape[1]
    k = tf.cast(k, dtype=tf.float32)
    
    total_loss = (node_loss / k) + (edge_loss / (k*(k-1)))
    
    return total_loss

def upper_triangle(mat):
    t = tf.matrix_band_part(mat, 0, -1)
    diag_elem = tf.zeros(t.shape[0:-1])
    diag_elem = tf.cast(diag_elem, tf.float32)
    ut_mat = tf.matrix_set_diag(t, diag_elem, name='ut_mat')
    return ut_mat

def graph_loss_ut(A_true, A_pred):
    A_true_ut = upper_triangle(A_true)
    A_pred_ut = upper_triangle(A_pred)
    
    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=A_true_ut, logits=A_pred_ut))
    k = A_true.shape[1]
    k = tf.cast(k, dtype=tf.float32)
    nb_edges = k*k/2 - k
    loss = loss / (nb_edges)
    return loss

def smooth_l1_loss(y_true, y_pred):
  diff = K.abs(y_true - y_pred)
  less_than_one = K.cast(K.less(diff, 0.01), "float32")
  loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.005)
  return loss

def box_loss(tru_box, gen_box):
    gen_box = ((gen_box)*(tru_box != 0))
    tru_box = ((tru_box)*(tru_box != 0))
    sum_r = tf.dtypes.cast(tf.reduce_sum(tf.keras.losses.MSE(tru_box, gen_box)), tf.float32)
    num_r = tf.dtypes.cast(tf.math.count_nonzero(tf.reduce_sum(tf.keras.losses.MSE(tru_box, gen_box), axis=-1)), tf.float32)
    return (sum_r/(num_r+1))

def area(boxlist):
    x_min, y_min, x_max, y_max = tf.split(
        value=boxlist, num_or_size_splits=4, axis=-1)
    return (y_max - y_min+ 1e-10) * (x_max - x_min + 1e-10)

def aspect_ratio(boxlist):
    x_min, y_min, x_max, y_max = tf.split(
        value=boxlist, num_or_size_splits=4, axis=-1)
    return (y_max - y_min + 1e-10 ) / (x_max - x_min + 1e-10)

def iou(target,  output):
    output = ((output)*(target != 0))
    target = ((target)*(target != 0))
    x1g, y1g, x2g, y2g = tf.split(value=target, num_or_size_splits=4, axis=-1)
    x1, y1, x2, y2 = tf.split(value=output, num_or_size_splits=4, axis=-1)
    xA = tf.maximum(x1g, x1)
    yA = tf.maximum(y1g, y1)
    xB = tf.minimum(x2g, x2)
    yB = tf.minimum(y2g, y2)
    interArea = tf.maximum(0.0, (xB - xA + 1)) * tf.maximum(0.0, yB - yA + 1)
    boxAArea = (x2g - x1g +1) * (y2g - y1g +1)
    boxBArea = (x2 - x1 +1) * (y2 - y1 +1)
    iouk = interArea / (boxAArea + boxBArea - interArea)
    return iouk

def pair_loss(target,  output):
    output = ((output)*(target != 0))
    target = ((target)*(target != 0))
    output_unstacked = tf.unstack(output,num=None,axis=-2)
    target_unstacked = tf.unstack(target,num=None,axis=-2)
    pairwise_iou_output = []
    pairwise_iou_target = []
    for ii in range(len(target_unstacked)):
        jj = ii
        while jj<(len(target_unstacked)):
            pairwise_iou_output.append((tf.keras.losses.MSE(output_unstacked[ii] , output_unstacked[jj])))
            pairwise_iou_target.append((tf.keras.losses.MSE(target_unstacked[ii] , target_unstacked[jj])))
            jj = jj + 1
    pairwise_iou_output = tf.convert_to_tensor(pairwise_iou_output,dtype=tf.float32)
    pairwise_iou_target = tf.convert_to_tensor(pairwise_iou_target,dtype=tf.float32)
    all_loss_sum = tf.reduce_sum(tf.keras.losses.MSE(pairwise_iou_target, pairwise_iou_output))
    total_non_zero = tf.dtypes.cast(tf.math.count_nonzero(tf.reduce_sum(tf.keras.losses.MSE(pairwise_iou_target, pairwise_iou_output),  axis = -1)),  dtype=tf.float32)
    return all_loss_sum/(total_non_zero+1)

def compute_ciou(target,  output):

    output = ((output)*(target != 0))
    target = ((target)*(target != 0))

    x1g, y1g, x2g, y2g = tf.split(value=target, num_or_size_splits=4, axis=-1)
    x1, y1, x2, y2 = tf.split(value=output, num_or_size_splits=4, axis=-1)
    
    w_pred = x2 - x1
    h_pred = y2 - y1
    w_gt = x2g - x1g
    h_gt = y2g - y1g

    x_center = (x2 + x1) / 2
    y_center = (y2 + y1) / 2
    x_center_g = (x1g + x2g) / 2
    y_center_g = (y1g + y2g) / 2

    xc1 = tf.minimum(x1, x1g)
    yc1 = tf.minimum(y1, y1g)
    xc2 = tf.maximum(x2, x2g)
    yc2 = tf.maximum(y2, y2g)
    
    xA = tf.maximum(x1g, x1)
    yA = tf.maximum(y1g, y1)
    xB = tf.minimum(x2g, x2)
    yB = tf.minimum(y2g, y2)
    interArea = tf.maximum(0.0, (xB - xA + 1)) * tf.maximum(0.0, yB - yA + 1)
    boxAArea = (x2g - x1g +1) * (y2g - y1g +1)
    boxBArea = (x2 - x1 +1) * (y2 - y1 +1)
    iouk = interArea / (boxAArea + boxBArea - interArea)
    ciouk = -tf.log(iouk)
    return tf.reduce_mean(ciouk)

kl_loss = tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(tf.exp(z_logvar)) - 2*(z_logvar) - 1, axis=1))
adj_loss = tf.reduce_mean(tf.keras.backend.binary_crossentropy(true_edge, edge_r))
bbox_loss = (compute_ciou(node_box_t, node_box_r)) + box_loss(node_box_t, node_box_r) + pair_loss(node_box_t, node_box_r)
label_loss = tf.reduce_mean(tf.keras.backend.binary_crossentropy(node_cls_t,node_cls_r))
class_loss = tf.reduce_mean(tf.keras.backend.categorical_crossentropy(class_vec, class_))

lr = 0.0001
reconstuction_loss = (bbox_loss + cls_loss + adj_loss + class_vvv)*24*5 + kl_weight*kl_loss
train_op = tf.train.AdamOptimizer(lr).minimize(reconstuction_loss)
saver = tf.train.Saver()

nb_epochs = 500

X_train, class_v, masks, X_train_val, class_v_val, masks_val = make_data()

train_loss = []
val_loss = []
log_likelihood_batch = []
klw = frange_cycle_linear(nb_epochs)
best_loss = -1
cur_step = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #saver.restore(sess, "./boxvae.ckpt")
    icoef = 0
    for i in range(1, nb_epochs+1):
        X_train, adj_train, class_v = shuffle_latent(X_train, adj_train, class_v)
        nb_batches = nb_train // batch_size
        for batch_idx in range(nb_batches):
            start_idx = batch_idx*batch_size
            end_idx = batch_idx*batch_size + batch_size
            
            E_in = adj_train[start_idx:end_idx]
            X_in = X_train[start_idx:end_idx]
            clvec = class_v[start_idx:end_idx]            
            sess.run(train_op, feed_dict = {
                                        true_node: X_in,
                                        true_edge: E_in,
                                        true_class: X_in[:, :, :label_size],
                                        class_vec:clvec,kl_weight: klw[icoef]})
        
        loss_total, loss_lbl, loss_bb, loss_adj, loss_kl, cls_loss = sess.run([reconstuction_loss, 
                                                            label_loss, bbox_loss, adj_loss,kl_loss, class_loss], 
                                                            feed_dict= {
                                                                true_node: X_in,
                                                                true_edge: E_in, 
                                                                true_class: X_in[:, :, :label_size],
                                                                class_vec:clvec,kl_weight: klw[icoef]})

        X_train_val, adj_train_val, class_v_val = shuffle_latent(X_train_val, adj_train_val, class_v_val)
        
        start_idx = 0
        end_idx = start_idx + batch_size
            
        E_in = adj_train_val[start_idx:end_idx]
        X_in = X_train_val[start_idx:end_idx]
        clvec = class_v_val[start_idx:end_idx]            

        loss_total_val, loss_lbl_val, loss_bb_val, loss_adj_val, loss_kl_val, cls_val = sess.run([reconstuction_loss, 
                                                            label_loss, bbox_loss, adj_loss, kl_loss,class_loss], 
                                                            feed_dict= {
                                                                true_node: X_in,
                                                                true_edge: E_in, 
                                                                true_class: X_in[:, :, :label_size],
                                                                class_vec:clvec,kl_weight: klw[icoef]})        

        train_loss.append([loss_bb])
        val_loss.append([loss_bb_val])

        plt.plot(np.asarray(train_loss))
        plt.plot(np.asarray(val_loss))
        plt.show()

        if loss_kl>0.5 and abs(loss_bb - loss_bb_val) < 0.2:
            icoef = icoef + 1

        print('for train')
        print('epoch:',i,'kl_weight', klw[icoef], 'class_loss:',loss_lbl, 'bbx_loss:',loss_bb,'adj_loss:',loss_adj,'kl_latent:',loss_kl, 'classeses: ', cls_loss)

        print('for val')
        print('epoch:',i,'kl_weight', klw[icoef],'class_loss:',loss_cls_val, 'bbx_loss:',loss_bb_val,'adj_loss:',loss_adj_val,'kl_latent:',loss_kl_val, 'classeses: ', cls_val)

        if i % 100 == 0:
            save_path = saver.save(sess,  "./boxvae.ckpt")
