import argparse
import os
import numpy as np
import tensorflow
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Concatenate
# from tensorflow.keras.layers.merge import _Merge
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from functools import partial
from preprocess import *
from model_seq2seq import *
import random

config = {
    'input_shape': (64, 64, 3),
    'point_batch_size': 200,
    'en_n_layers': 2,
    'ef_dim': 32,
    'de_n_layers': 2,
    'df_dim': 32,
    'z_dim': 128,
    'partae_file': 'Models/partae/all_class/model_partae_rgb_ssim_all.h5',
    'part_en_n_layers': 5,
    'part_de_n_layers': 5,
    'target_input_prob': 0.5
}

checkpoint_filepath = 'models/seq2seq/latest/all_data_woreorder_full_loss_200epochs.h5'
model_seq2seq = PartSeq2Seq(config)
r_data = {

    'vox2d': tf.random.normal((32, 24, 64, 64, 3)),  # (B, T, vox_dim, vox_dim, 1),
    'sign': tf.random.normal((32, 24, 1)),
    'affine_input': tf.random.normal((32, 24, 4)),
    'affine_target': tf.random.normal((32, 24, 4)),
    'cond': tf.random.normal((32, 24, 24)),
    'bce_mask': tf.random.normal((32, 24))
}

output_seq, stop_signs = model_seq2seq((
    r_data['vox2d'], r_data['affine_input'], r_data['cond'], r_data['affine_target'],
    r_data['sign'], r_data['bce_mask']))
model_seq2seq.load_weights(checkpoint_filepath)


n_dim = 50
z_dim = 2048
h_dim = 200

def make_generator(n_dim, h_dim, z_dim):
    model = Sequential()
    model.add(Dense(h_dim, input_shape=(n_dim,)))
    model.add(LeakyReLU())
    model.add(Dense(h_dim))
    model.add(LeakyReLU())
    model.add(Dense(h_dim))
    model.add(LeakyReLU())
    model.add(Dense(z_dim, activation='tanh'))
    return model
def make_discriminator(h_dim, z_dim):
    x = Input(shape=(z_dim,))
    inter = Dense(h_dim)(x)
    inter = LeakyReLU()(inter)
    inter = Dense(h_dim)(inter)
    inter = LeakyReLU()(inter)
    output1 = Dense(h_dim)(inter)
    output1 = LeakyReLU()(output1)
    output1 = Dense(1)(output1)
    output2 = Dense(h_dim)(inter)
    output2 = LeakyReLU()(output2)
    output2 = Dense(10, activation='softmax')(output2)

    output3 = Dense(h_dim)(inter)
    output3 = LeakyReLU()(output3)
    output3 = Dense(24, activation='sigmoid')(output3)
    
    return Model(inputs=x, outputs=[output1, output2, output3])

generator = make_generator(n_dim + 10 + 24, h_dim, z_dim)
discriminator = make_discriminator(h_dim, z_dim)

for layer in discriminator.layers:
    layer.trainable = False
discriminator.trainable = False
noise_input = Input(shape=(n_dim,))
class_input = Input(shape=(10,))
part_label_input = Input(shape=(24, ))
generator_input = Concatenate()([noise_input, class_input, part_label_input])
generator_output = generator(generator_input)
# generator_layers = Concatenate()([generator_output, class_input])
discriminator_layers_for_generator, class_pred, part_label_pred = discriminator(generator_output)
generator_model = Model(inputs=[noise_input, class_input, part_label_input],
                        outputs=[discriminator_layers_for_generator, class_pred, part_label_pred])

def visualise_gen_results(decoded_masks, output_box, num_samples, save_dir=None):
    samples = decoded_masks.shape[0]
    for i in range(samples):
        # decoded_masks[i] = decoded_masks[i] #* np.tile(mask[i], (1, 1, 3))
        print(decoded_masks[i].shape)
        print(output_box[i].shape)
        canvas_output = make_rgb_mask(decoded_masks[i], output_box[i].numpy())
        #plt.imshow(canvas_output)
        #plt.savefig(str(save_dir) + '.jpg')
        cv2.imwrite(str(save_dir) + '.jpg', (canvas_output*255.0).astype('uint8'))


def generate_images(num_samples, class_name, part_label_input, generator_model, output_dir, epoch):
    """Feeds random seeds into the generator and tiles and saves the output to a PNG
    file."""
    class_input = np.zeros((1, 10))
    class_input[0][class_dic[class_name]] = 1
    gen_input = tf.concat([np.random.rand(num_samples, n_dim), class_input, part_label_input], axis=1)
    generated_encoding = generator_model.predict(tf.convert_to_tensor(gen_input))
    # generated_encoding = tf.expand_dims(tf.convert_to_tensor(generated_encoding), axis=1)
    generated_encoding = tf.convert_to_tensor(generated_encoding)
    z_dim_half = z_dim // 2
    # print(generated_encoding.shape)
    h1 = generated_encoding[:, :z_dim_half]
    h2 = generated_encoding[:, z_dim_half:]
    decoder_hidden = [h1, h2]
    decoder_input = tf.identity(tf.tile(tf.stop_gradient(model_seq2seq.decoder.init_input), ((num_samples, 1, 1))))
    # print(decoder_input.shape)
    decoder_outputs = []
    stop_signs = []
    curr_max = 0
    stop_idx = 0
    for i in range(24):
        cache, decoder_output, stop_sign = model_seq2seq.decoder(decoder_input, decoder_hidden)
        # print("Decoder output  " + str(decoder_output.shape))
        # print("Stop Sign " + str(stop_sign.shape))
        stop_val = tf.sigmoid(stop_sign[0, 0])
        if stop_val > curr_max:
            curr_max = stop_val
            stop_idx = i
        decoder_outputs.append(decoder_output)
        stop_signs.append(stop_sign)
        decoder_input = tf.expand_dims(decoder_output, axis=1)
        decoder_hidden = cache
    if len(decoder_outputs) > 0:
        decoder_outputs = tf.stack(decoder_outputs, axis=1)
        stop_signs = tf.stack(stop_signs, axis=1)

    # print("Outputs = " + str(decoder_outputs.shape))
    # print("Signs = " + str(stop_signs.shape))

    box_prediction = decoder_outputs[:, :, -4:]
    decoded_masks = model_seq2seq.part_autoencoder.reconstruct(np.reshape(decoder_outputs[:, :, :-4], (-1, 128)))
    # print("Masks Before = " + str(decoded_masks.shape))
    decoded_masks = np.reshape(decoded_masks,
                               (-1, 24, decoded_masks.shape[1], decoded_masks.shape[2], decoded_masks.shape[3]))
    # print("Masks = " + str(decoded_masks.shape))
    # print("Box = " + str(box_prediction.shape))
    visualise_gen_results(decoded_masks, box_prediction, num_samples, save_dir=output_dir + str(class_name) + '_' + str(epoch))

train_classes = ['dog', 'sheep', 'horse', 'person', 'motorbike', 'cow', 'bicycle', 'cat', 'bird', 'aeroplane']
generator_model.load_weights('models/generator_finalall_99.h5')
X_train, masks, rgb_masks, rgb_images, classes = make_data(train_classes, 'test')
data = data_prep(X_train, masks, rgb_masks, classes, train_ratio=0.0, mode='test', model='seq2seq')
print("Samples = " + str(data['classes'].shape[0]))
for i in range(data['classes'].shape[0]):
  cls_idx = np.where(data['classes'][i]==1)[0][0]
  print(data['classes'][i])
  cls_name = object_names[cls_idx]
  print(cls_name)
  if not os.path.isdir('pq-net-generations-new/' + cls_name):
    os.mkdir('pq-net-generations-new/' + cls_name)
  part_list = data['bce_mask'][i]
  part_list = np.expand_dims(part_list, axis=0)
  print(part_list.shape)
  generate_images(1, cls_name, part_list, generator, 'pq-net-generations-new/' + cls_name + '/', i)
