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
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
#from tensorflow.keras.layers.merge import _Merge
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from functools import partial
from preprocess import *
from model_seq2seq import *



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
    'partae_file': 'models/partae/latest/model_partae_rgb_ssim_dog.h5',
    'part_en_n_layers': 5,
    'part_de_n_layers': 5,
    'target_input_prob': 0.5
}

checkpoint_filepath = 'models/seq2seq/latest/all_data_woreorder_100epochs.h5'
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

n_dim = 10
z_dim = 2048
h_dim = 100

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
    model.add(Dense(h_dim, input_shape=(n_dim, )))
    model.add(LeakyReLU())
    model.add(Dense(h_dim))
    model.add(LeakyReLU())
    model.add(Dense(h_dim))
    model.add(LeakyReLU())
    model.add(Dense(z_dim))
    return model

    # """Creates a generator model that takes a 100-dimensional noise vector as a "seed",
    # and outputs images of size 28x28x1."""
    # model = Sequential()
    # model.add(Dense(1024, input_dim=100))
    # model.add(LeakyReLU())
    # model.add(Dense(128 * 7 * 7))
    # model.add(BatchNormalization())
    # model.add(LeakyReLU())
    # if K.image_data_format() == 'channels_first':
    #     model.add(Reshape((128, 7, 7), input_shape=(128 * 7 * 7,)))
    #     bn_axis = 1
    # else:
    #     model.add(Reshape((7, 7, 128), input_shape=(128 * 7 * 7,)))
    #     bn_axis = -1
    # model.add(Conv2DTranspose(128, (5, 5), strides=2, padding='same'))
    # model.add(BatchNormalization(axis=bn_axis))
    # model.add(LeakyReLU())
    # model.add(Conv2D(64, (5, 5), padding='same'))
    # model.add(BatchNormalization(axis=bn_axis))
    # model.add(LeakyReLU())
    # model.add(Conv2DTranspose(64, (5, 5), strides=2, padding='same'))
    # model.add(BatchNormalization(axis=bn_axis))
    # model.add(LeakyReLU())
    # # Because we normalized training inputs to lie in the range [-1, 1],
    # # the tanh function should be used for the output of the generator to ensure
    # # its output also lies in this range.
    # model.add(Conv2D(1, (5, 5), padding='same', activation='tanh'))
    # return model



def make_discriminator(h_dim, z_dim):
    model = Sequential()
    model.add(Dense(h_dim, input_shape=(z_dim,)))
    model.add(LeakyReLU())
    model.add(Dense(h_dim))
    model.add(LeakyReLU())
    model.add(Dense(h_dim))
    model.add(LeakyReLU())
    model.add(Dense(1))
    return model
    """Creates a discriminator model that takes an image as input and outputs a single
    value, representing whether the input is real or generated. Unlike normal GANs, the
    output is not sigmoid and does not represent a probability! Instead, the output
    should be as large and negative as possible for generated inputs and as large and
    positive as possible for real inputs.
    Note that the improved WGAN paper suggests that BatchNormalization should not be
    used in the discriminator."""
    # model = Sequential()
    # if K.image_data_format() == 'channels_first':
    #     model.add(Conv2D(64, (5, 5), padding='same', input_shape=(1, 28, 28)))
    # else:
    #     model.add(Conv2D(64, (5, 5), padding='same', input_shape=(28, 28, 1)))
    # model.add(LeakyReLU())
    # model.add(Conv2D(128, (5, 5), kernel_initializer='he_normal',
    #                         strides=[2, 2]))
    # model.add(LeakyReLU())
    # model.add(Conv2D(128, (5, 5), kernel_initializer='he_normal', padding='same',
    #                         strides=[2, 2]))
    # model.add(LeakyReLU())
    # model.add(Flatten())
    # model.add(Dense(1024, kernel_initializer='he_normal'))
    # model.add(LeakyReLU())
    # model.add(Dense(1, kernel_initializer='he_normal'))
    # return model


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
        alpha = tensorflow.random.uniform((self.batch_size, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]



# class RandomWeightedAverage(_Merge):
#     """Takes a randomly-weighted average of two tensors. In geometric terms, this
#     outputs a random point on the line between each pair of input points.
#     Inheriting from _Merge is a little messy but it was the quickest solution I could
#     think of. Improvements appreciated."""
#
#     def _merge_function(self, inputs):
#         weights = K.random_uniform((BATCH_SIZE, 1, 1, 1))
#         return (weights * inputs[0]) + ((1 - weights) * inputs[1])

def visualise_seq_results(decoded_masks, output_box, num_samples, save_dir=None):
    samples = decoded_masks.shape[0]
    for i in range(samples):
        # decoded_masks[i] = decoded_masks[i] #* np.tile(mask[i], (1, 1, 3))
        canvas_output = make_rgb_mask(decoded_masks[i], output_box[i].numpy())
        plt.imshow(canvas_output)
        plt.savefig('generation_test_' + str(i) + '.jpg')

def generate_images(num_samples, generator_model, output_dir, epoch):
    """Feeds random seeds into the generator and tiles and saves the output to a PNG
    file."""
    generated_encoding = generator_model.predict(np.random.rand(num_samples, n_dim))
    z_dim_half = z_dim//2
    decoder_hidden = [generated_encoding[:,z_dim_half], generated_encoding[:, z_dim_half:]]
    decoder_input = tf.identity(tf.tile(tf.stop_gradient(self.decoder.init_input), ((num_samples, 1, 1))))
    decoder_output, stop_signs = model_seq2seq.decoder(decoder_input, decoder_hidden)
    box_prediction = decoder_output[:, :, -4:]
    decoded_masks = model_seq2seq.part_autoencoder.reconstruct(np.reshape(decoder_output[:, :, :-4], (-1, 128)))
    decoded_masks = np.reshape(decoded_masks,
                               (-1, 24, decoded_masks.shape[1], decoded_masks.shape[2], decoded_masks.shape[3]))
    visualise_seq_results(decoded_masks, box_prediction, num_samples)

tensorflow.compat.v1.disable_eager_execution()

parser = argparse.ArgumentParser(description="Improved Wasserstein GAN "
                                             "implementation for Keras.")
parser.add_argument("--output_dir", "-o", required=True,
                    help="Directory to output generated files to")
args = parser.parse_args()

# # First we load the image data, reshape it and normalize it to the range [-1, 1]
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# X_train = np.concatenate((X_train, X_test), axis=0)
# if K.image_data_format() == 'channels_first':
#     X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1], X_train.shape[2]))
# else:
#     X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
# X_train = (X_train.astype(np.float32) - 127.5) / 127.5

# Now we initialize the generator and discriminator.
generator = make_generator(n_dim, h_dim, z_dim)
discriminator = make_discriminator(h_dim, z_dim)


# The generator_model is used when we want to train the generator layers.
# As such, we ensure that the discriminator layers are not trainable.
# Note that once we compile this model, updating .trainable will have no effect within
# it. As such, it won't cause problems if we later set discriminator.trainable = True
# for the discriminator_model, as long as we compile the generator_model first.
for layer in discriminator.layers:
    layer.trainable = False
discriminator.trainable = False
generator_input = Input(shape=(n_dim,))
generator_layers = generator(generator_input)
discriminator_layers_for_generator = discriminator(generator_layers)
generator_model = Model(inputs=[generator_input],
                        outputs=[discriminator_layers_for_generator])
# We use the Adam paramaters from Gulrajani et al.
generator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                        loss=wasserstein_loss,experimental_run_tf_function=False)

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
real_samples = Input(shape=(z_dim,))
generator_input_for_discriminator = Input(shape=(n_dim,))
generated_samples_for_discriminator = generator(generator_input_for_discriminator)
discriminator_output_from_generator = discriminator(generated_samples_for_discriminator)
discriminator_output_from_real_samples = discriminator(real_samples)

# We also need to generate weighted-averages of real and generated samples,
# to use for the gradient norm penalty.
averaged_samples = RandomWeightedAverage(BATCH_SIZE)([real_samples,
                                            generated_samples_for_discriminator])
# We then run these samples through the discriminator as well. Note that we never
# really use the discriminator output for these samples - we're only running them to
# get the gradient norm for the gradient penalty loss.
averaged_samples_out = discriminator(averaged_samples)

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
discriminator_model = Model(inputs=[real_samples,
                                    generator_input_for_discriminator],
                            outputs=[discriminator_output_from_real_samples,
                                     discriminator_output_from_generator,
                                     averaged_samples_out])
# We use the Adam paramaters from Gulrajani et al. We use the Wasserstein loss for both
# the real and generated samples, and the gradient penalty loss for the averaged samples
discriminator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                            loss=[wasserstein_loss,
                                  wasserstein_loss,
                                  partial_gp_loss], experimental_run_tf_function=False)
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



train_classes = ['dog']
X_train, masks, rgb_masks, rgb_images, classes = make_data(train_classes)
data = data_prep(X_train, masks, rgb_masks, classes, mode='train', model='seq2seq')

for epoch in range(1):

    print("Epoch: ", epoch)
    print("Number of batches: ", int(data['vox2d'].shape[0] // BATCH_SIZE))
    discriminator_loss = []
    generator_loss = []
    minibatches_size = BATCH_SIZE * TRAINING_RATIO

    for i in range(int(X_train.shape[0] // (BATCH_SIZE * TRAINING_RATIO))):
        discriminator_minibatches = {}
        for key in data.keys():
            discriminator_minibatches[key] = data[key][i*minibatches_size:(i+1)*minibatches_size]
        for j in range(TRAINING_RATIO):
            row_vox2d = discriminator_minibatches['vox2d'][j*BATCH_SIZE:(j+1)*BATCH_SIZE]
            affine_input = discriminator_minibatches['affine_input'][j*BATCH_SIZE:(j+1)*BATCH_SIZE]
            cond = discriminator_minibatches['cond'][j*BATCH_SIZE:(j+1)*BATCH_SIZE]
            batch_size, max_n_parts, vox_dim = row_vox2d.shape[0], row_vox2d.shape[1], row_vox2d.shape[2]
            batch_vox2d = tf.reshape(row_vox2d, (-1, vox_dim, vox_dim, 3))
            part_geo_features = model_seq2seq.part_encoder(batch_vox2d)
            part_geo_features = tf.reshape(part_geo_features, (batch_size, max_n_parts, -1))
            part_feature_seq = tf.concat([part_geo_features, affine_input, cond], axis=2)
            hidden1, hidden2 = model_seq2seq.encoder(part_feature_seq)

            real_encoding = tf.concat([hidden1[:,-1,:], hidden2[:,-1,:]], axis=1)
            print(real_encoding.shape)
            # image_batch = discriminator_minibatches[j * BATCH_SIZE:
            #                                                     (j + 1) * BATCH_SIZE]
            noise = np.random.rand(BATCH_SIZE, n_dim).astype(np.float32)
            noise = tf.convert_to_tensor(noise)
            discriminator_loss.append(discriminator_model.train_on_batch(
                [real_encoding, noise],
                [positive_y, negative_y, dummy_y]))
            print("here")
        generator_loss.append(generator_model.train_on_batch(tf.convert_to_tensor(np.random.rand(BATCH_SIZE,
                                                                            n_dim)),
                                                             positive_y))
    # Still needs some code to display losses from the generator and discriminator,
    # progress bars, etc.
    generate_images(generator, args.output_dir, epoch)
