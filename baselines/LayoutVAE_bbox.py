import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, LSTM, Input, Layer
from tensorflow.keras import Model
def ConditioningInputModel(n_labels):
    label_set_count = Input(shape=(n_labels, ))
    curr_label = Input(shape=(n_labels, ))
    previous_bbox_encoding = Input(shape=(128, ))

    i1 = Dense(128, activation='relu')(label_set_count)
    i1 = Dense(128, activation='relu')(i1)

    i2 = Dense(128, activation='relu')(curr_label)
    i2 = Dense(128, activation='relu')(i2)

    i3 = Dense(128, activation='relu')(previous_bbox_encoding)

    output = Concatenate()([i1, i2, i3])
    output = Dense(128)(output)
    return Model(inputs=[label_set_count, curr_label, previous_bbox_encoding], outputs=output)

def Encoder():
    gt_bbox = Input(shape=(4, ))
    conditioning_input = Input(shape=(128, ))

    i1 = Dense(128, activation='relu')(gt_bbox)
    i1 = Dense(128)(i1)

    intermediate = Concatenate()([i1, conditioning_input])
    intermediate = Dense(32, activation='relu')(intermediate)
    mean_ = Dense(32)(intermediate)
    sigma_ = Dense(32)(intermediate)

    return Model(inputs=[gt_bbox, conditioning_input], outputs=[mean_, sigma_])

def Decoder():
    conditioning_input = Input(shape=(128,))
    latent_dim = Input(shape=(32, ))
    output = Concatenate()([conditioning_input, latent_dim])
    output = Dense(128, activation='relu')(output)
    output = Dense(64, activation='relu')(output)
    output = Dense(4)(output)
    return Model(inputs=[conditioning_input, latent_dim], outputs=output)

def Prior():
    conditioning_input = Input(shape=(128,))
    intermediate = Dense(32, activation='relu')(conditioning_input)
    z_mean = Dense(32)(intermediate)
    z_log_var = Dense(32)(intermediate)
    return Model(inputs=[conditioning_input], outputs=[z_mean, z_log_var])

class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class BBoxVAE(tf.keras.Model):
    def __init__(self):
        super(BBoxVAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.prior = Prior()
        self.condition_model = ConditioningInputModel(24)
        self.rnn_layer = LSTM(128, return_sequences=True)
        self.vae_sampling = Sampling()
    def call(self, inputs):
        label_set = inputs[0] #label set with counts i.e 1 Multi-label encoding (batch_size, 24)
        bbox_input = inputs[1] # ground-truth bounding boxes (batch_size, 24, 4)

        num_labels = label_set.shape[1] # time-steps for LSTM
        batch_size = label_set.shape[0]
        prev_bounding_boxes_encoding = tf.zeros((batch_size, 128)) # zeros initially
        bbox_outputs = []
        kl_losses = []
        for i in range(num_labels): #num_labels = number of timesteps
            ground_truth_bbox = bbox_input[:, i, :]
            one_hot = tf.Variable(tf.zeros((batch_size, 24)))
            one_hot[:,i].assign(1)
            curr_label = one_hot
            conditioning_info = self.condition_model([label_set, curr_label, prev_bounding_boxes_encoding])
            z_mean, z_log_var = self.encoder([ground_truth_bbox, conditioning_info])
            z = self.vae_sampling([z_mean, z_log_var])
            current_step_bbox = self.decoder([conditioning_info, z])
            lstm_input = tf.concat([curr_label, current_step_bbox], axis=1)
            lstm_input = tf.expand_dims(lstm_input, axis=1) # (batch_size, 1, features) for single timestep execution

            prev_bounding_boxes_encoding = self.rnn_layer(lstm_input, initial_state=[prev_bounding_boxes_encoding, prev_bounding_boxes_encoding])
            prev_bounding_boxes_encoding = tf.squeeze(prev_bounding_boxes_encoding)

            z_mean, z_log_var = self.prior(conditioning_info)
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)

            bbox_outputs.append(current_step_bbox)
            kl_losses.append(kl_loss)

        bbox_outputs = tf.stack(bbox_outputs, axis=1)
        kl_losses = tf.stack(kl_losses, axis=1)
        mse = tf.keras.losses.MeanSquaredError()
        bbox_loss = mse(bbox_input, bbox_outputs)
        kl_loss_final = tf.reduce_mean(kl_losses)
        kl_loss_final *= -0.5
        self.add_loss(bbox_loss + kl_loss_final)

        return bbox_outputs


if __name__ == "__main__":

    model = BBoxVAE()
    x1 = tf.random.normal((32, 24))
    x2 = tf.random.normal((32, 24, 4))
    model([x1, x2])
















