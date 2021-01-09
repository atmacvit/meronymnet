from model_seq2seq import *
import numpy as np
from preprocess import *
import matplotlib.pyplot as plt
import random
import os

config = {
    'input_shape': (64, 64, 3),
    'point_batch_size': 200,
    'en_n_layers': 2,
    'ef_dim': 32,
    'de_n_layers': 2,
    'df_dim': 32,
    'z_dim': 128,
    'partae_file': 'Models/partae/single_class_background/model_partae_rgb_ssim_dog_background.h5',
    'part_en_n_layers': 5,
    'part_de_n_layers': 5,
    'target_input_prob': 0.5
}


def visualise_seq_results(input_masks, input_box, decoded_masks, stop_signs, output_box, num_samples,
                          save_dir, class_name):
    samples = input_masks.shape[0]
    fig = plt.figure(figsize=(8,8))

    for i in range(input_masks.shape[0]):
        for j in range(input_masks.shape[1]):
            fig.add_subplot((2 * samples), input_masks.shape[1], 2*i*input_masks.shape[1] + j + 1)
            plt.imshow(input_masks[i][j])
            fig.add_subplot((2 * samples), input_masks.shape[1], (2 * i + 1) * input_masks.shape[1] + j + 1)
            plt.imshow(decoded_masks[i][j])

    plt.savefig(save_dir + 'check_sequence_results_' + str(class_name) + '.jpg')
    plt.clf()
    for i in range(samples):
        canvas_input = make_rgb_mask(input_masks[i].numpy(),input_box[i].numpy())
        plt.imshow(canvas_input)
        plt.savefig(save_dir + str(i) + '_check_new_input_test_' + str(class_name) + '.jpg')
        canvas_output = make_rgb_mask(decoded_masks[i], output_box[i].numpy())
        plt.imshow(canvas_output)
        plt.savefig(save_dir + str(i) + '_check_new_output_test_' + str(class_name) + '.jpg')


def data_generator(class_names=None, mode='train', batch_size=32):
    # class_names = range(num_train_parts)
    assert (class_names is not None)
    class_idx = 0
    num_classes = len(class_names)

    X_train, masks, rgb_masks, rgb_images, classes = make_data([class_names[class_idx]])
    data = data_prep(X_train, masks, rgb_masks, classes, mode='train', model='seq2seq')

    num_samples = (data['vox2d'].shape[0])  # //batch_size) * batch_size
    start = -batch_size
    end = 0
    counter = 0
    while True:
        start = start + batch_size
        counter = counter + 1
        if (start + batch_size) >= num_samples:
            class_idx = class_idx + 1
            if class_idx >= num_classes:
                class_idx = 0
            old_data = data

            X_train, masks, rgb_masks, rgb_images, classes = make_data([class_names[class_idx]])
            data = data_prep(X_train, masks, rgb_masks, classes, mode='train', model='seq2seq')
            num_samples = data['vox2d'].shape[0]
            yield_start = start
            start = -batch_size
            yield [old_data['vox2d'][yield_start:], old_data['affine_input'][yield_start:],
                   old_data['cond'][yield_start:], old_data['affine_target'][yield_start:],
                   old_data['sign'][yield_start:], old_data['bce_mask'][yield_start:]]
        else:
            end = start + batch_size
            try:
                yield [data['vox2d'][start:end], data['affine_input'][start:end], data['cond'][start:end],
                       data['affine_target'][start:end], data['sign'][start:end], data['bce_mask'][start:end]]
            except:
                continue


checkpoint_filepath = './models/seq2seq/latest/all_data_woreorder_full_loss_200epochs.h5'

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_filepath, monitor='loss', verbose=1,
    save_best_only=True, mode='auto', period=1)

model_seq2seq = PartSeq2Seq(config)
print("after model init")
objects_list = ['cow', 'sheep', 'bird', 'person', 'cat', 'dog', 'horse', 'motorbike', 'bicycle']

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


def train(classes, train_num=None):
    
    tf.config.experimental_run_functions_eagerly(True)

    batch_size = 64

    opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model_seq2seq.compile(optimizer=opt)
    model_seq2seq.summary()
    if os.path.exists(checkpoint_filepath):
      model_seq2seq.load_weights(checkpoint_filepath)

    X_train, masks, rgb_masks, rgb_images, classes_v = make_data(classes, train_mode='train')
    vox2d, affine_input, cond, affine_target, sign, bce_mask = data_prep(X_train, masks, rgb_masks, classes_v, model='seq2seq')

    model_seq2seq.fit(x=[vox2d, affine_input, cond, affine_target,
                        sign, bce_mask], batch_size=batch_size,
                    callbacks=[model_checkpoint_callback], shuffle=True, epochs=100)

    model_seq2seq.save_weights(checkpoint_filepath)


def test(test_num, test_classes):
    model_seq2seq.load_weights(checkpoint_filepath)
    for cls in test_classes:
        X_train, masks, rgb_masks, rgb_images, classes = make_data([cls], train_mode='test')
        vox2d, affine_input, cond, affine_target, sign, bce_mask = data_prep(X_train, masks, rgb_masks, classes, model='seq2seq')
        decoder_output, stop_signs = model_seq2seq([vox2d, affine_input, cond, affine_target, sign, bce_mask])
        box_prediction = decoder_output[:, :, -4:]
        decoded_masks = model_seq2seq.part_autoencoder.reconstruct(np.reshape(decoder_output[:, :, :-4], (-1, 128)))
        decoded_masks = np.reshape(decoded_masks,
                                   (-1, 24, decoded_masks.shape[1], decoded_masks.shape[2], decoded_masks.shape[3]))
        visualise_seq_results(vox2d, affine_input,  decoded_masks,
                              stop_signs, box_prediction, test_num,
                              'Results/seq2seq/allclass_noreorder/', cls)



#train(['dog', 'sheep', 'horse', 'person', 'motorbike', 'cow', 'bicycle','cat','aeroplane','bird'])
test(5, ['dog', 'sheep', 'horse'])
