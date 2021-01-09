"""An implementation of the improved WGAN described in https://arxiv.org/abs/1704.00028
The improved WGAN has a term in the loss function which penalizes the network if its
gradient norm moves away from 1. This is included because the Earth Mover (EM) distance
used in WGANs is only easy to calculate for 1-Lipschitz functions (i.e. functions where
the gradient norm has a constant upper bound of 1).
The original WGAN paper enforced this by clipping weights to very small values
[-0.01, 0.01]. However, this drastically reduced network capacity. Penalizing the
gradient norm is more natural, but this requires second-order gradients. These are not
supported for some tensorflow ops (particularly MaxPool and AveragePool) in the current
release (1.0.x), but they are supported in the current nightly builds
(1.1.0-rc1 and higher).
To avoid this, this model uses strided convolutions instead of Average/Maxpooling for
downsampling. If you wish to use pooling operations in your discriminator, please ensure
you update Tensorflow to 1.1.0-rc1 or higher. I haven't tested this with Theano at all.
The model saves images using pillow. If you don't have pillow, either install it or
remove the calls to generate_images.
"""
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

try:
    from PIL import Image
except ImportError:
    print('This script depends on pillow! '
          'Please install it (e.g. with pip install pillow)')
    exit()

BATCH_SIZE = 64
# The training ratio is the number of discriminator updates
# per generator update. The paper uses 5.
TRAINING_RATIO = 5
GRADIENT_PENALTY_WEIGHT = 10  # As per the paper
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
model_seq2seq.trainable = False
model_seq2seq.part_encoder.trainable = False
model_seq2seq.encoder.trainable = False

n_dim = 50
z_dim = 2048
h_dim = 200


def wasserstein_loss(y_true, y_pred):
    """Calculates the Wasserstein loss for a sample batch.
    The Wasserstein loss function is very simple to calculate. In a standard GAN, the
    discriminator has a sigmoid output, representing the probability that samples are
    real or generated. In Wasserstein GANs, however, the output is linear with no
    activation function! Instead of being constrained to [0, 1], the discriminator wants
    to make the distance between its output for real and generated samples as
    large as possible.
    The most natural way to achieve this is to label generated samples -1 and real
    samples 1, instead of the 0 and 1 used in normal GANs, so that multiplying the
    outputs by the labels will give you the loss immediately.
    Note that the nature of this loss means that it can be (and frequently will be)
    less than 0."""
    return K.mean(y_true * y_pred)


def _compute_gradients(tensor, var_list):
    grads = tf.gradients(tensor, var_list)
    return [grad if grad is not None else tf.zeros_like(var)
            for var, grad in zip(var_list, grads)]


def gradient_penalty_loss(y_true, y_pred, averaged_samples,
                          gradient_penalty_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.
    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the
    loss function that penalizes the network if the gradient norm moves away from 1.
    However, it is impossible to evaluate this function at all points in the input
    space. The compromise used in the paper is to choose random points on the lines
    between real and generated samples, and check the gradients at these points. Note
    that it is the gradient w.r.t. the input averaged samples, not the weights of the
    discriminator, that we're penalizing!
    In order to evaluate the gradients, we must first run samples through the generator
    and evaluate the loss. Then we get the gradients of the discriminator w.r.t. the
    input averaged samples. The l2 norm and penalty can then be calculated for this
    gradient.
    Note that this loss function requires the original averaged samples as input, but
    Keras only supports passing y_true and y_pred to loss functions. To get around this,
    we make a partial() of the function with the averaged_samples argument, and use that
    for model training."""
    # first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors
    gradients = _compute_gradients(y_pred, [averaged_samples])[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)


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

    # """Creates a generator model that takes a 100-dimensional noise vector as a "seed",
    # and outputs images of size 28x28x1."""


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

    """Creates a discriminator model that takes an image as input and outputs a single
    value, representing whether the input is real or generated. Unlike normal GANs, the
    output is not sigmoid and does not represent a probability! Instead, the output
    should be as large and negative as possible for generated inputs and as large and
    positive as possible for real inputs.
    Note that the improved WGAN paper suggests that BatchNormalization should not be
    used in the discriminator."""


def tile_images(image_stack):
    """Given a stacked tensor of images, reshapes them into a horizontal tiling for
    display."""
    assert len(image_stack.shape) == 3
    image_list = [image_stack[i, :, :] for i in range(image_stack.shape[0])]
    tiled_images = np.concatenate(image_list, axis=1)
    return tiled_images


class RandomWeightedAverage(tensorflow.keras.layers.Layer):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def call(self, inputs, **kwargs):
        alpha = tensorflow.random.uniform((self.batch_size, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]


def visualise_gen_results(decoded_masks, output_box, num_samples, save_dir=None):
    samples = decoded_masks.shape[0]
    for i in range(samples):
        # decoded_masks[i] = decoded_masks[i] #* np.tile(mask[i], (1, 1, 3))
        print(decoded_masks[i].shape)
        print(output_box[i].shape)
        canvas_output = make_rgb_mask(decoded_masks[i], output_box[i].numpy())
        plt.imshow(canvas_output)
        plt.savefig(str(save_dir) + '.jpg')


def generate_images(num_samples, class_name, part_list, generator_model, output_dir, epoch):
    """Feeds random seeds into the generator and tiles and saves the output to a PNG
    file."""
    part_label_input = part_list
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


# tensorflow.compat.v1.disable_eager_execution()

parser = argparse.ArgumentParser(description="Improved Wasserstein GAN "
                                             "implementation for Keras.")
parser.add_argument("--output_dir", "-o", required=True,
                    help="Directory to output generated files to")
args = parser.parse_args()


# Now we initialize the generator and discriminator.
generator = make_generator(n_dim + 10 + 24, h_dim, z_dim)
discriminator = make_discriminator(h_dim, z_dim)

# The generator_model is used when we want to train the generator layers.
# As such, we ensure that the discriminator layers are not trainable.
# Note that once we compile this model, updating .trainable will have no effect within
# it. As such, it won't cause problems if we later set discriminator.trainable = True
# for the discriminator_model, as long as we compile the generator_model first.
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
# We use the Adam paramaters from Gulrajani et al.
generator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                        loss=[wasserstein_loss, 'categorical_crossentropy', 'mse'], loss_weights=[1.0, 5.0, 1.0])

#generator_model.load_weights('models/generator_sheep.h5')
# Now that the generator_model is compiled, we can make the discriminator
# layers trainable.
for layer in discriminator.layers:
    layer.trainable = True
for layer in generator.layers:
    layer.trainable = False
discriminator.trainable = True
generator.trainable = False

# The discriminator_model is more complex. It takes both real image samples and random
# noise seeds as input. The noise seed is run through the generator model to get
# generated images. Both real and generated images are then run through the
# discriminator. Although we could concatenate the real and generated images into a
# single tensor, we don't (see model compilation for why).
real_samples_latent = Input(shape=(z_dim,))
real_samples_class = Input(shape=(10,))
real_samples_part_labels = Input(shape=(24, ))
generator_noise_for_discriminator = Input(shape=(n_dim,))
generator_classes_for_discriminator = Input(shape=(10,))
generator_part_labels_for_discriminator = Input(shape=(24,))
generator_input_for_discriminator = Concatenate()([generator_noise_for_discriminator, generator_classes_for_discriminator, generator_part_labels_for_discriminator])
generated_output_for_discriminator = generator(generator_input_for_discriminator)
# generated_samples_for_discriminator = Concatenate()(
#     [generated_output_for_discriminator, generator_classes_for_discriminator])
discriminator_output_from_generator, disc_class_from_gen, disc_part_label_from_gen = discriminator(generated_output_for_discriminator)
#real_samples = Concatenate()([real_samples_latent, real_samples_class])
discriminator_output_from_real_samples, disc_class_from_real, disc_part_label_from_real = discriminator(real_samples_latent)

# We also need to generate weighted-averages of real and generated samples,
# to use for the gradient norm penalty.
averaged_samples = RandomWeightedAverage(BATCH_SIZE)([real_samples_latent,
                                                      generated_output_for_discriminator])
# We then run these samples through the discriminator as well. Note that we never
# really use the discriminator output for these samples - we're only running them to
# get the gradient norm for the gradient penalty loss.

averaged_samples_out, avg_class, avg_label = discriminator(averaged_samples)

# The gradient penalty loss function requires the input averaged samples to get
# gradients. However, Keras loss functions can only have two arguments, y_true and
# y_pred. We get around this by making a partial() of the function with the averaged
# samples here.

partial_gp_loss = partial(gradient_penalty_loss,
                          averaged_samples=averaged_samples,
                          gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
# Functions need names or Keras will throw an error
partial_gp_loss.__name__ = 'gradient_penalty'

# Keras requires that inputs and outputs have the same number of samples. This is why
# we didn't concatenate the real samples and generated samples before passing them to
# the discriminator: If we had, it would create an output with 2 * BATCH_SIZE samples,
# while the output of the "averaged" samples for gradient penalty
# would have only BATCH_SIZE samples.

# If we don't concatenate the real and generated samples, however, we get three
# outputs: One of the generated samples, one of the real samples, and one of the
# averaged samples, all of size BATCH_SIZE. This works neatly!
discriminator_model = Model(inputs=[real_samples_latent,
                                    generator_noise_for_discriminator,generator_classes_for_discriminator, generator_part_labels_for_discriminator],
                            outputs=[discriminator_output_from_real_samples, disc_class_from_real, disc_part_label_from_real,
                                     discriminator_output_from_generator, disc_class_from_gen, disc_part_label_from_gen,
                                     averaged_samples_out])

# We use the Adam paramaters from Gulrajani et al. We use the Wasserstein loss for both
# the real and generated samples, and the gradient penalty loss for the averaged samples
discriminator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                            loss=[wasserstein_loss,'categorical_crossentropy','mse',
                                  wasserstein_loss,'categorical_crossentropy','mse',
                                  partial_gp_loss], experimental_run_tf_function=False, loss_weights=[1.0, 5.0, 1.0, 1.0, 5.0, 1.0, 1.0])
#discriminator_model.load_weights('models/discriminator_sheep.h5')
# We make three label vectors for training. positive_y is the label vector for real
# samples, with value 1. negative_y is the label vector for generated samples, with
# value -1. The dummy_y vector is passed to the gradient_penalty loss function and
# is not used.
positive_y = np.ones((BATCH_SIZE, 1), dtype=np.float32)
negative_y = -positive_y
dummy_y = np.zeros((BATCH_SIZE, 1), dtype=np.float32)

positive_y = tf.convert_to_tensor(positive_y)
negative_y = tf.convert_to_tensor(negative_y)
dummy_y = tf.convert_to_tensor(dummy_y)

train_classes = ['dog', 'sheep', 'horse', 'person', 'motorbike', 'cow', 'bicycle', 'cat', 'bird', 'aeroplane']#)['cow', 'sheep', 'bird', 'person', 'cat', 'dog', 'horse']#, 'aeroplane', 'motorbike', 'bicycle']
class_idx = []
for cls in train_classes:
  class_idx.append(class_dic[cls])

X_train, masks, rgb_masks, rgb_images, classes = make_data(train_classes, 'train')
num_s = int(0.9 * X_train.shape[0])
X_train, masks, rgb_masks, rgb_images, classes = X_train[0:num_s], masks[0:num_s], rgb_masks[0:num_s], rgb_images[0:num_s], classes[0:num_s]
data = data_prep(X_train, masks, rgb_masks, classes, train_ratio=1.0, mode='train', model='seq2seq')
indices = np.random.permutation(data['vox2d'].shape[0])
sample_size = (data['vox2d'].shape[0] // (BATCH_SIZE * TRAINING_RATIO)) * (BATCH_SIZE * TRAINING_RATIO)

for key in data.keys():
    data[key] = data[key][indices]
    data[key] = data[key][:sample_size]
sample_size = (data['vox2d'].shape[0] // (BATCH_SIZE * TRAINING_RATIO)) * (BATCH_SIZE * TRAINING_RATIO)
iteration = 0
model_path = ''
for i in train_classes:
    model_path = 'all' 

for epoch in range(100):
    print("Epoch: ", epoch)
    print("Number of batches: ", int(data['vox2d'].shape[0] // BATCH_SIZE))
    discriminator_loss = []
    generator_loss = []
    minibatches_size = BATCH_SIZE * TRAINING_RATIO

    for i in range(int(data['vox2d'].shape[0] // (BATCH_SIZE * TRAINING_RATIO))):
        discriminator_minibatches = {}
        for key in data.keys():
            discriminator_minibatches[key] = data[key][i * minibatches_size:(i + 1) * minibatches_size]
            print(discriminator_minibatches[key].shape)
        for j in range(TRAINING_RATIO):
            row_vox2d = discriminator_minibatches['vox2d'][j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
            affine_input = discriminator_minibatches['affine_input'][j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
            cond = discriminator_minibatches['cond'][j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
            batch_size, max_n_parts, vox_dim = row_vox2d.shape[0], row_vox2d.shape[1], row_vox2d.shape[2]
            batch_vox2d = tf.reshape(row_vox2d, (-1, vox_dim, vox_dim, 3))
            part_geo_features = model_seq2seq.part_encoder(batch_vox2d)
            part_geo_features = tf.reshape(part_geo_features, (batch_size, max_n_parts, -1))
            part_feature_seq = tf.concat([part_geo_features, affine_input, cond], axis=2)
            hidden1, hidden2 = model_seq2seq.encoder(part_feature_seq)

            real_encoding = tf.concat([hidden1[:, -1, :], hidden2[:, -1, :]], axis=1)
            noise = np.random.rand(BATCH_SIZE, n_dim).astype(np.float32)
            noise = tf.convert_to_tensor(noise)
            # zeros = np.zeros((BATCH_SIZE, 10))
            # zeros2 = np.zeros((BATCH_SIZE, 24))
            # random.shuffle(class_idx)
            # rand_cls = random.choices(class_idx, k=BATCH_SIZE)
            # for i in range(BATCH_SIZE):
            #   zeros[i, rand_cls[i]]=1
            #   sampling = random.choices(range(24), k=random.choice(range(25)))
            #   zeros2[i][sampling]=1
            # cls_gen = zeros
            # part_label_gen = zeros2
            real_class_input = tf.convert_to_tensor(
                discriminator_minibatches['classes'][j * BATCH_SIZE:(j + 1) * BATCH_SIZE])
            real_part_label = tf.convert_to_tensor(
                discriminator_minibatches['bce_mask'][j * BATCH_SIZE:(j + 1) * BATCH_SIZE])

            #print("Check class = " + str(real_class_input[0]))
            #final_input = tf.concat([noise, cls_gen], axis=1)
            disc_loss = discriminator_model.train_on_batch(
                [real_encoding, noise, real_class_input, real_part_label],
                [positive_y, real_class_input, real_part_label, negative_y, real_class_input, real_part_label,dummy_y])
            discriminator_loss.append(disc_loss)
            print("Discriminator loss = ", disc_loss)
        

        cls_gen = tf.convert_to_tensor(
        data['classes'][i * BATCH_SIZE:(i + 1) * BATCH_SIZE])
        part_label_gen = tf.convert_to_tensor(
        data['bce_mask'][i * BATCH_SIZE:(i + 1) * BATCH_SIZE])
        noise_gen = np.random.rand(BATCH_SIZE, n_dim)
        gen_loss = generator_model.train_on_batch([noise_gen, cls_gen, part_label_gen],[positive_y, cls_gen, part_label_gen])
        generator_loss.append(gen_loss)
        print("Generator loss = ", gen_loss)
        
    #generate_images(1, 'person', generator, args.output_dir, epoch)
    #generate_images(1, 'horse', generator, args.output_dir, epoch)
    discriminator_model.save_weights('models/discriminator_final' + model_path + '_' + str(epoch) + '.h5')
    generator_model.save_weights('models/generator_final' + model_path + '_' + str(epoch) + '.h5')
    # Still needs some code to display losses from the generator and discriminator,
    # progress bars, etc.
    # generate_images(1, generator, args.output_dir, epoch)



# generate_images(1, generator, args.output_dir, 4)
# generate_images(1, generator, args.output_dir, 5)
