from model_partae import *
import numpy as np
from preprocess import *
import matplotlib.pyplot as plt
import os
import cv2
import random

train_object = ['bicycle', 'sheep', 'dog', 'cat', 'horse', 'person', 'cow', 'motorbike', 'aeroplane', 'bird']
test_classes = ['dog']
config = {
    'input_shape': (64, 64, 3),
    'point_batch_size': 200,
    'en_n_layers': 2,
    'ef_dim': 32,
    'de_n_layers': 2,
    'df_dim': 32,
    'z_dim': 128,
    'partae_file': 'model_partae_rgb_ssim_all' + '.h5',
    'part_en_n_layers': 5,
    'part_de_n_layers': 5
}
class_condition = False

train_test_spit = 0.9
tf.config.experimental_run_functions_eagerly(True)
batch_size = 128
test_save_dir = 'Results/latest/all_class/'
train_save_dir = 'Models/partae/all_class/'

checkpoint_filepath = train_save_dir + config['partae_file']
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_filepath, monitor='loss', verbose=1,
    save_best_only=True, mode='auto', period=1)


model = PartAE(config['input_shape'], en_n_layers=config['part_en_n_layers'], de_n_layers=config['part_de_n_layers'],
               is_class_conditioning=class_condition)

x = tf.random.normal(shape=(1, 64, 64, 3))  # rgb masks
if class_condition == True:
    c = tf.random.normal(shape=(1, 10))  # class condition
    out_tmp = model([x, c])
else:
    out_tmp = model(x)

if not os.path.exists(train_save_dir):
    os.makedirs(train_save_dir)
if not os.path.exists(test_save_dir):
    os.makedirs(test_save_dir)

class PlotLosses(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []

        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show()


plot_losses = PlotLosses()

def visualise_part_results(part_rgb_masks, input_masks, part_rgb_pred, save_dir=None):
    fig = plt.figure(figsize=(8, 8))
    columns = 3
    rows = part_rgb_masks.shape[0]
    # plt.show()
    samples = part_rgb_masks.shape[0]
    ax_list = []
    if save_dir is not None:
        for i in range(samples):
            a1 = fig.add_subplot(rows, columns, i * 3 + 1)
            plt.imshow(part_rgb_masks[i])
            a2 = fig.add_subplot(rows, columns, i * 3 + 2)
            plt.imshow(part_rgb_pred[i] * np.tile(input_masks[i], (1, 1, 3)))
            a3 = fig.add_subplot(rows, columns, i * 3 + 3)
            plt.imshow(input_masks[i, :, :, 0])
            ax_list.append([a1, a2, a3])
        print(ax_list)
        headers = ['INPUT', 'RECONSTRUCTED', 'MASK']
        for ax, h in zip(ax_list[0], headers):
            ax.set_title(h)
        plt.savefig(save_dir)

    for i in range(samples):
        plt.imshow(part_rgb_masks[i])
        plt.show()
        plt.imshow(part_rgb_pred[i])
        plt.show()




def train():
    if os.path.exists(checkpoint_filepath):
       model.load_weights(checkpoint_filepath)
    
    X_train, masks, rgb_masks, rgb_images, classes_v = make_data(object_names, train_mode='train')
    part_ae_masks, part_ae_rgb_masks, classes = data_prep(X_train, masks, rgb_masks, classes_v, model='partae')
    train_rgb_gt = tf.concat([part_ae_rgb_masks/255.0, part_ae_masks], 3)

    opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
    if class_condition == True:
        model.compile(loss=[custom_ssim, 'categorical_crossentropy'], optimizer=opt, loss_weights=[5.0, 2.0])
        model.fit(x=[tf.convert_to_tensor(part_ae_rgb_masks/255.0), tf.convert_to_tensor(classes)], y=[tf.convert_to_tensor(train_rgb_gt), tf.convert_to_tensor(classes)], shuffle=True,
                      batch_size=batch_size, epochs=100, callbacks=[model_checkpoint_callback, plot_losses])
    else:
        model.compile(loss=[custom_ssim], optimizer=opt)
        model.fit(x=tf.convert_to_tensor(part_ae_rgb_masks/255.0), y=tf.convert_to_tensor(train_rgb_gt), shuffle=True,
                       batch_size=batch_size, epochs=100, callbacks=[model_checkpoint_callback, plot_losses])
      
    #model.save_weights(checkpoint_filepath)


def test(num_samples, test_classes):
    part_ae_masks_t = None
    part_ae_rgb_masks_t = None
    model.load_weights(checkpoint_filepath)
    for idx in range(len(test_classes)):
        X_train, masks, rgb_masks, rgb_images, classes = make_data([test_classes[idx]])
        part_ae_masks, part_ae_rgb_masks, part_classes = data_prep(X_train, masks, rgb_masks, classes,
                                                                                       config['point_batch_size'], mode='train',
                                                                              model='partae')
        test_mask_input = part_ae_masks[0:num_samples]
        test_rgb_input = part_ae_rgb_masks[0:num_samples]/255.0
        test_classes = part_classes[0:num_samples]
        outputs, output_class = model.predict([tf.convert_to_tensor(test_rgb_input), test_classes], batch_size=1)
        visualise_part_results(test_rgb_input, test_mask_input, outputs,
                           save_dir= test_save_dir + 'all_class_cond_' + str(test_classes[idx]) + '.jpg')


train()
test(100, train_object)
