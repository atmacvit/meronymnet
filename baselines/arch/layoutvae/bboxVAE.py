import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, LSTM, Input, Layer
from tensorflow.keras import Model
import numpy as np


def ConditioningInputModel(n_labels, is_class_condition=False):
    label_set_count = Input(shape=(n_labels,))
    curr_label = Input(shape=(n_labels,))
    previous_bbox_encoding = Input(shape=(128,))
    if is_class_condition==True:
      class_input = Input(shape=(10, ))

    i1 = Dense(128, activation='relu')(label_set_count)
    i1 = Dense(128, activation='relu')(i1)

    i2 = Dense(128, activation='relu')(curr_label)
    i2 = Dense(128, activation='relu')(i2)

    i3 = Dense(128, activation='relu')(previous_bbox_encoding)
    
    if is_class_condition==True:
      i4 = Dense(128, activation='relu')(class_input)
      i4 = Dense(128, activation='relu')(i4)
    
    if is_class_condition==True:
      output = Concatenate()([i1, i2, i3, i4])
    else:
      output = Concatenate()([i1, i2, i3])
    
    output = Dense(128)(output)
    
    if is_class_condition==True:
      input_list = [label_set_count, curr_label, previous_bbox_encoding, class_input]
    else:
      input_list = [label_set_count, curr_label, previous_bbox_encoding]

    return Model(inputs=input_list, outputs=output)


def Encoder():
    gt_bbox = Input(shape=(4,))
    conditioning_input = Input(shape=(128,))

    i1 = Dense(128, activation='relu')(gt_bbox)
    i1 = Dense(128)(i1)

    intermediate = Concatenate()([i1, conditioning_input])
    intermediate = Dense(32, activation='relu')(intermediate)
    z_mean = Dense(32)(intermediate)
    z_log_var = Dense(32)(intermediate)

    return Model(inputs=[gt_bbox, conditioning_input], outputs=[z_mean, z_log_var])


def Decoder():
    conditioning_input = Input(shape=(128,))
    latent_dim = Input(shape=(32,))
    output = Concatenate()([conditioning_input, latent_dim])
    output = Dense(128, activation='relu')(output)
    output = Dense(64, activation='relu')(output)
    output = Dense(4,activation='sigmoid')(output)
    return Model(inputs=[conditioning_input, latent_dim], outputs=output)

class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def kl_divergence(p_mean, p_log_var, q_mean, q_log_var):
    kl_div = q_log_var - p_log_var + (tf.exp(p_log_var) + tf.square(p_mean - q_mean))/(tf.exp(q_log_var)) - 1
    kl_div *= 0.5
    return kl_div

def Prior():
  conditioning_input = Input(shape=(128,))
  output = Dense(32, activation='relu')(conditioning_input)
  z_mean = Dense(32)(output)
  z_log_var = Dense(32)(output)
  return Model(inputs=conditioning_input,outputs=[z_mean,z_log_var])


class BBoxVAE(tf.keras.Model):
    def __init__(self, is_class_condition=False):
        super(BBoxVAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.prior = Prior()
        self.condition_model = ConditioningInputModel(24, is_class_condition)
        self.rnn_layer = LSTM(128, return_state=True)
        self.vae_sampling = Sampling()
        self.class_cond = is_class_condition
    def call(self, input, training=True):
        
        label_set = input[0]  # label set with counts i.e 1 Multi-label encoding (batch_size, 24)
        bbox_input = input[1]
        if self.class_cond==True:
          class_input = input[2]

        print("Training value = " + str(training))
        # ground-truth bounding boxes (batch_size, 24, 4)
        num_labels = label_set.shape[1]  # time-steps for LSTM
        batch_size = label_set.shape[0]
        #print(batch_size)
        prev_bounding_boxes_encoding = tf.zeros((batch_size, 128))  # zeros initially
        state_h = None
        state_c = None
        bbox_outputs = []
        kl_losses = []
        for i in range(num_labels):  # num_labels = number of timesteps
            one_hot = tf.Variable(tf.zeros((batch_size, 24)))
            ones = tf.Variable(tf.ones((batch_size,)))
            one_hot[:, i].assign(ones)
            curr_label = one_hot
            if self.class_cond==False:
              conditioning_info = self.condition_model([label_set, curr_label, prev_bounding_boxes_encoding])
            else:
              conditioning_info = self.condition_model([label_set, curr_label, prev_bounding_boxes_encoding, class_input])


            z_mean_c, z_log_var_c = self.prior(conditioning_info)
            ground_truth_bbox = bbox_input[:, i, :]
            z_mean, z_log_var = self.encoder([ground_truth_bbox, conditioning_info])
            
            kl_loss =  kl_divergence(z_mean, z_log_var, z_mean_c, z_log_var_c) #1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)#kl_divergence(z_mean_e, z_log_var_e, z_mean_c, z_log_var_c)
            #kl_loss *= -0.5
            kl_losses.append(kl_loss)
            if training==True:
              z = self.vae_sampling([z_mean, z_log_var])
            else:
              #z = tf.random.normal((batch_size, 32))
              z = self.vae_sampling([z_mean_c, z_log_var_c])#sample from prior

            current_step_bbox = self.decoder([conditioning_info, z])
            lstm_input = tf.concat([curr_label, current_step_bbox], axis=1)
            lstm_input = tf.expand_dims(lstm_input, axis=1)  # (batch_size, 1, features) for single timestep execution
            if i == 0:
                prev_bounding_boxes_encoding, state_h, state_c = self.rnn_layer(lstm_input)
            else:
                prev_bounding_boxes_encoding, state_h, state_c = self.rnn_layer(lstm_input,
                                                                                initial_state=[state_h, state_c])

            prev_bounding_boxes_encoding = tf.squeeze(prev_bounding_boxes_encoding)
            bbox_outputs.append(current_step_bbox)

        bbox_outputs = tf.stack(bbox_outputs, axis=1)
        mse = tf.keras.losses.MeanSquaredError()
        bbox_loss = mse(bbox_input, bbox_outputs)
        print("Bounding Box loss = " + str(bbox_loss))

        kl_losses = tf.stack(kl_losses, axis=1)
        kl_loss_final = tf.reduce_mean(kl_losses)
        print("Mean KL Divergence = " + str(kl_loss_final))
        if training==True:
          self.add_loss(bbox_loss + 0.005 * kl_loss_final)
        else:
          self.add_loss(bbox_loss)





        return bbox_outputs


if __name__ == "__main__":
    model = BBoxVAE()
    x1 = tf.random.normal((32, 24))
    x2 = tf.random.normal((32, 24, 4))
    model([x1, x2])



