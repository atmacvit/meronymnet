import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2DTranspose, Input, Bidirectional, InputLayer, Conv2D, Dense, \
    BatchNormalization, LeakyReLU, Activation, Dropout, Concatenate, ZeroPadding2D, Reshape, RepeatVector, Flatten, \
    Lambda
import tensorflow.keras.backend as K
from skimage.metrics import structural_similarity as ssim


def PartEncoderModel(shape_size, n_layers, ef_dim=32, z_dim=128):
    model = Sequential()
    out_channels = ef_dim
    for i in range(n_layers - 1):
        if i == 0:
            model.add(ZeroPadding2D(input_shape=shape_size))
        else:
            model.add(ZeroPadding2D())
        model.add(Conv2D(filters=out_channels, kernel_size=(4, 4), strides=(2, 2)))
        model.add(BatchNormalization(momentum=0.1))
        model.add(LeakyReLU(alpha=0.02))
        out_channels = out_channels * 2

    model.add(Conv2D(filters=z_dim, kernel_size=(4, 4), strides=(1, 1)))
    model.add(Activation('sigmoid'))
    model.add(Flatten())
    return model

def PartEncoderModelConditional(shape_size, n_layers, class_size, ef_dim=32, z_dim=128):
    x1 = Input(shape=shape_size)
    x2 = Input(shape=(class_size, ))
    dim1 = shape_size[0]
    dim2 = shape_size[1]
    units = dim1 * dim2
    x3 = Dense(units, activation='relu')(x2)
    x = Dense(units, activation='relu')(x3)
    x = Reshape((dim1, dim2, 1))(x)
    x = Concatenate()([x1, x])
    
    layers = []
    out_channels = ef_dim
    for i in range(n_layers - 1):
        if i == 0:
            layers.append(ZeroPadding2D())
        else:
            layers.append(ZeroPadding2D())
        layers.append(Conv2D(filters=out_channels, kernel_size=(4, 4), strides=(2, 2)))
        layers.append(BatchNormalization(momentum=0.1))
        layers.append(LeakyReLU(alpha=0.02))
        out_channels = out_channels * 2

    layers.append(Conv2D(filters=z_dim, kernel_size=(4, 4), strides=(1, 1)))
    layers.append(Activation('sigmoid'))
    layers.append(Flatten())

    for l in layers:
      x = l(x)

    return Model(inputs=[x1, x2], outputs=x)


def PartReconstructModel(n_layers, f_dim, z_dim):
    in_channels = z_dim
    out_channels = f_dim * (2 ** (n_layers - 2))

    layers = []
    layers.append(Reshape((1, 1, in_channels), input_shape=(in_channels,)))
    layers.append(Conv2DTranspose(filters=out_channels, kernel_size=(4, 4), strides=1))
    for i in range(n_layers - 1):
        out_channels = out_channels // 2
        if i == n_layers - 2:
            out_channels = 3
        layers.append(Conv2DTranspose(filters=out_channels, kernel_size=(4, 4), strides=(2, 2)))
        layers.append(LeakyReLU(alpha=0.02))
        if i % 2 == 0:
            layers.append(Conv2D(filters=out_channels, kernel_size=(4, 4), strides=1))
            layers.append(LeakyReLU(alpha=0.02))
    layers.append(Activation('sigmoid'))
    rgb_reconstruct = Sequential(layers)
    return rgb_reconstruct


def PartReconstructModelConditional(n_layers, f_dim, z_dim, class_size):
    in_channels = z_dim
    out_channels = f_dim * (2 ** (n_layers - 2))

    x = Input(shape=(in_channels,))
    d_1 = Dense(40)(x)
    d_1 = LeakyReLU()(d_1)
    d_1 = Dense(40)(d_1)
    d_1 = LeakyReLU()(d_1)
    d_1 = Dense(class_size, activation='softmax')(d_1)

    layers = []
    layers.append(Reshape((1, 1, in_channels)))
    layers.append(Conv2DTranspose(filters=out_channels, kernel_size=(4, 4), strides=1))
    for i in range(n_layers - 1):
        out_channels = out_channels // 2
        if i == n_layers - 2:
            out_channels = 3
        layers.append(Conv2DTranspose(filters=out_channels, kernel_size=(4, 4), strides=(2, 2)))
        layers.append(LeakyReLU(alpha=0.02))
        if i % 2 == 0:
            layers.append(Conv2D(filters=out_channels, kernel_size=(4, 4), strides=1))
            layers.append(LeakyReLU(alpha=0.02))
    layers.append(Activation('sigmoid'))

    x_2 = x
    for l in layers:
      x_2 = l(x_2)
    
    return Model(inputs=x, outputs=[d_1, x_2])

def PartClassifierModel(shape_size, n_layers, ef_dim=32, z_dim=128):
    model = Sequential()
    out_channels = ef_dim
    for i in range(n_layers - 1):
        if i == 0:
            model.add(ZeroPadding2D(input_shape=shape_size))
        else:
            model.add(ZeroPadding2D())
        model.add(Conv2D(filters=out_channels, kernel_size=(4, 4), strides=(2, 2)))
        model.add(BatchNormalization(momentum=0.1))
        model.add(LeakyReLU(alpha=0.02))
        out_channels = out_channels * 2

    model.add(Conv2D(filters=z_dim, kernel_size=(4, 4), strides=(1, 1)))
    model.add(LeakyReLU())
    model.add(Flatten())
    model.add(Dense(60))
    model.add(LeakyReLU())
    model.add(Dense(30))
    model.add(LeakyReLU())
    model.add(Dense(10, activation='softmax'))
    return model

def PartDecoderModel(n_layers, f_dim, z_dim):
    in_channels = z_dim + 2
    out_channels = f_dim * (2 ** (n_layers - 2))
    x = Input(shape=(in_channels,))
    layers = []
    for i in range(n_layers - 1):
        if i > 0:
            in_channels += z_dim + 2
        if i < 4:
            l = [Dense(out_channels), Dropout(rate=0.4), LeakyReLU()]
        # model.append([nn.Linear(in_channels, out_channels), nn.Dropout(p=0.4), nn.LeakyReLU()])
        else:
            l = [Dense(out_channels), LeakyReLU()]
        # model.append([nn.Linear(in_channels, out_channels), nn.LeakyReLU()])
        in_channels = out_channels
        out_channels = out_channels // 2
        layers.append(Sequential(l))

    l = [Dense(1), Activation('sigmoid')]
    layers.append(Sequential(l))
    out = layers[0](x)
    for i in range(1, n_layers - 1):
        out = layers[i](Concatenate(axis=1)([out, x]))
    out = layers[n_layers - 1](out)

    return Model(inputs=x, outputs=out)


# model.append([nn.Linear(in_channels, 1), nn.Sigmoid()])

class PartAE(tf.keras.Model):
    def __init__(self, input_shape, en_n_layers=5, ef_dim=32, de_n_layers=5, df_dim=32, z_dim=128, is_class_conditioning=False):
        super(PartAE, self).__init__()

        self.z_dim = z_dim
        if is_class_conditioning==False:
          self.encoder = PartEncoderModel(input_shape, en_n_layers, ef_dim, z_dim)
        else:
          self.encoder = PartEncoderModelConditional(input_shape, en_n_layers, 10, ef_dim, z_dim)

        # self.decoder = PartDecoderModel(de_n_layers, df_dim, z_dim)
        if is_class_conditioning==False:
            self.reconstruct = PartReconstructModel(de_n_layers, df_dim, z_dim)
        else:
            self.reconstruct = PartReconstructModelConditional(de_n_layers, df_dim, z_dim, 10)

        self.class_conditioning = is_class_conditioning

    # print(self.encoder.outputs[0].shape)

    def call(self, x):
        if self.class_conditioning == False:
            encoded_out = self.encoder(x)
            reconstruct_out = self.reconstruct(encoded_out)
            return reconstruct_out
        else:
            encoded_out = self.encoder(x)
            classifier_out, reconstruct_out = self.reconstruct(encoded_out)
            return reconstruct_out, classifier_out

        # print(reconstruct_out.shape)
        # encoded_out = tf.expand_dims(encoded_out, axis=1)
        # encoded_out = tf.tile(encoded_out, multiples=(1, x[1].shape[1], 1))
        # encoded_out = tf.reshape(encoded_out, shape=(-1, self.z_dim))
        # point_input = tf.reshape(x[1], (-1, x[1].shape[2]))
        # concat_out = tf.concat([encoded_out, point_input], axis=1)
        # out = self.decoder(concat_out)
        # out = tf.reshape(out, shape=(x[0].shape[0], x[1].shape[1], -1))
        # print(out.shape)
        # print(reconstruct_out.shape)


def custom_loss(y_true, y_pred):
    y_recon = y_true[:, :, :, :-1]
    y_mask = tf.expand_dims(y_true[:, :, :, -1], axis=3)
    y_mask = tf.tile(y_mask, [1, 1, 1, 3])
    print(y_mask.shape)
    print(y_recon.shape)
    print(y_pred.shape)
    loss = K.square(y_recon - y_pred) * y_mask
    return tf.reduce_sum(loss) / tf.reduce_sum(y_mask)


def custom_ssim(y_true, y_pred):
    y_recon = y_true[:, :, :, :-1]
    y_mask = tf.expand_dims(y_true[:, :, :, -1], axis=3)
    y_mask = tf.tile(y_mask, [1, 1, 1, 3])
    print(y_mask.shape)
    print(y_recon.shape)
    print(y_pred.shape)
    #y_pred = y_pred * y_mask
    loss = tf.reduce_mean(tf.image.ssim(y_pred, y_recon, 1.0))
    return 1.0 - loss


if __name__ == "__main__":
    tf.config.experimental_run_functions_eagerly(True)
    # model = PartImNetAE(6, 32, 6, 32, 138)
    model = PartAE((64, 64, 1), en_n_layers=5, de_n_layers=5)
    print(model.encoder.summary())
    print(model.decoder.summary())
    print(model.reconstruct.summary())
    # print(model.summary())

    # model = PartDecoderModel(32, 5, 5, 32, 128)
    # print(model.decoder.summary())
    x = tf.random.normal(shape=(32, 64, 64, 1))  # masks
    y = tf.random.normal(shape=(32, 5, 2))  # points taken from 0-1
    gt_point = tf.random.normal(shape=(32, 5, 1))  # classification for every image-point pair
    gt_reconstruct = tf.random.normal((32, 64, 64, 3))  # rgb reconstruct for image

    gt_reconstruct_final = tf.concat([gt_reconstruct, x], 3)
    print(gt_reconstruct_final.shape)

    print(model.encoder.inputs)

    model.compile(loss=[custom_loss, 'mse'], optimizer='adam')
    model.fit(x=[x, y], y=[gt_reconstruct_final, gt_point], epochs=2)
    model.save_weights('model_partae.h5')
# out = model([x, y])

# print(out)

