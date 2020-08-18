from model_partae import *
import numpy as np
from preprocess import *
import matplotlib.pyplot as plt
import skfmm
import cv2


def visualise_part_results(part_ae_masks, part_rgb_pred, save_dir=None):
	samples = part_ae_masks.shape[0]
	if save_dir is not None:
		for i in range(samples):
			cv2.imwrite(save_dir + '/' + 'input_' + str(i + 1) + '.png',part_ae_masks[i, :, :,0].astype(dtype=np.uint8) * 255.0)
			cv2.imwrite(save_dir + '/' + 'recon_' + str(i + 1) + '.png',part_rgb_pred[i, :, :,0]*255.0)
	for i in range(samples):
		plt.imshow(part_ae_masks[i, :, :,0])
		plt.show()
		plt.imshow(part_rgb_pred[i, :, :,0])
		plt.show()

config = {
'input_shape': (64, 64, 1),
'point_batch_size':200,
'en_n_layers':2,
'ef_dim':32,
'de_n_layers':2,
'df_dim':32,
'z_dim':128,
'partae_file':'models/partae/model_partae_mask_pure_sdf_sigmoid.h5',
'part_en_n_layers':5,
'part_de_n_layers':5
}


X_train, masks, rgb_masks, rgb_images = make_data()
print(masks.shape)
print(X_train.shape)
print(rgb_masks.shape)


part_ae_masks, points, part_ae_gt, part_ae_rgb_masks, seq2seq_masks, seq2seq_pno, seq2seq_bce_mask, seq2seq_affine_input, seq2seq_affine_target, seq2seq_sign = data_prep(X_train, masks, rgb_masks, 200)

checkpoint_filepath = './models/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='max',
    save_best_only=True)

train_test_spit = 0.9

tf.config.experimental_run_functions_eagerly(True)

batch_size = 32

train_size = int(part_ae_masks.shape[0] * train_test_spit)

test_size = part_ae_masks.shape[0] - train_size


model=PartAE(config['input_shape'], en_n_layers=config['part_en_n_layers'], de_n_layers=config['part_de_n_layers'])

train_mask = part_ae_masks[0:train_size] # masks
train_pt = points[0:train_size] # points taken from 0-1
train_pt_gt = part_ae_gt[0:train_size] # classification for every image-point pair
train_rgb = part_ae_masks[0:train_size]#part_ae_rgb_masks[0:train_size]/255.0 # rgb reconstruct for image
train_rgb_mask = tf.concat([train_rgb, train_mask], 3)


test_mask = part_ae_masks[train_size:]
test_pt = points[train_size:]
test_rgb = part_ae_masks[train_size:]#part_ae_rgb_masks[train_size:]/255.0 


#model.compile(loss=['mse', 'mse'], metrics=tf.keras.metrics.MeanSquaredError(), optimizer='adam', loss_weights=[0.5, 0.5])
#model.fit(x = [train_mask, train_pt], y=[train_rgb, train_pt_gt], shuffle=True, batch_size=batch_size, epochs=50, callbacks=[model_checkpoint_callback])
#model.save_weights('model_partae_mask_pure_sdf_sigmoid.h5')
out_tmp = model([test_mask,test_pt])
model.load_weights(config['part_ae_file'])
outputs = model.predict([test_mask,test_pt])
print(outputs[0].shape)


visualise_part_results(test_mask[0:5], outputs[0][0:5], save_dir='Results/partae/mask2mask')

