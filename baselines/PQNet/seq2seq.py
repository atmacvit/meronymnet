from model_seq2seq import *
import numpy as np
from preprocess import *
import matplotlib.pyplot as plt
import random

def visualise_seq_results(input_masks, decoded_masks, num_samples, save_dir=None):
  indices = np.arange(input_masks.shape[0])
  np.random.shuffle(indices)
  indices = indices[:num_samples]
  if save_dir is not None:
    for i in range(num_samples):
      for j in range(input_masks.shape[1]):
        cv2.imwrite(save_dir + '/' + str(i + 1) + '_' + str(j+1) + '_input' + '.png', input_masks[indices[i], j, :, :,0].astype(dtype=np.uint8) * 255.0)
        cv2.imwrite(save_dir + '/' + str(i + 1) + '_' + str(j+1) + '_output' + '.png',decoded_masks[indices[i], j, :, :,0] * 255.0)

  for i in range(num_samples):
      for j in range(input_masks.shape[1]):
        plt.imshow(input_masks[indices[i], j, :, :,0])
        plt.show()
        plt.imshow(decoded_masks[indices[i], j, :, :,0])
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
'part_de_n_layers':5,
'target_input_prob':0.5
}


X_train, masks, rgb_masks, rgb_images = make_data()

print(masks.shape)
print(X_train.shape)
print(rgb_masks.shape)


part_ae_masks, points, part_ae_gt, part_ae_rgb_masks, seq2seq_masks, seq2seq_pno, seq2seq_bce_mask, seq2seq_affine_input, seq2seq_affine_target, seq2seq_sign = data_prep(X_train, masks, rgb_masks, 200)

print("PartAE Masks " + str(part_ae_masks.shape))
print("PartAE Points " + str(points.shape))
print("PartAE GT Point " + str(part_ae_gt.shape))
print("PartAE GT RGB Masks " + str(part_ae_rgb_masks.shape))
print("Seq2Seq Masks " + str(seq2seq_masks.shape))
print("Seq2Seq Affine Input " + str(seq2seq_affine_input.shape))
print("Seq2Seq Part No " + str(seq2seq_pno.shape))
print("Seq2Seq BCE Mask " + str(seq2seq_bce_mask.shape))
print("Seq2Seq Affine Output " + str(seq2seq_affine_target.shape))
print("Seq2Seq stop sign " + str(seq2seq_sign.shape))


checkpoint_filepath = './models/seq2seq/checkpoint/'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='max',
    save_best_only=True)

train_test_spit = 0.9

tf.config.experimental_run_functions_eagerly(True)

batch_size = 32

model_seq2seq = PartSeq2Seq(config)
train_size = int(seq2seq_masks.shape[0] * train_test_spit)
test_size = seq2seq_masks.shape[0] - train_size

#train_size = 32

data = {
'vox2d': seq2seq_masks[0:train_size],   # (B, T, vox_dim, vox_dim, 1),
'sign': seq2seq_sign[0:train_size],
'affine_input' : seq2seq_affine_input[0:train_size],
'affine_target' : seq2seq_affine_target[0:train_size],
'cond':seq2seq_pno[0:train_size],
'bce_mask':seq2seq_bce_mask[0:train_size]
}
test_data = {
'vox2d': seq2seq_masks[train_size:],   # (B, T, vox_dim, vox_dim, 1),
'sign': seq2seq_sign[train_size:],
'affine_input' : seq2seq_affine_input[train_size:],
'affine_target' : seq2seq_affine_target[train_size:],
'cond':seq2seq_pno[train_size:],	
'bce_mask':seq2seq_bce_mask[train_size:]
}

out = model_seq2seq([data['vox2d'], data['affine_input'], data['cond'], data['affine_target'], data['sign'], data['bce_mask']])
#model_seq2seq.load_weights('models/seq2seq/seq2seq_model_50_wobce_tinput.h5')
model_seq2seq.compile(optimizer='adam')
model_seq2seq.fit(x=(data['vox2d'], data['affine_input'], data['cond'], data['affine_target'], data['sign'], data['bce_mask']), validation_data=[test_data['vox2d'], test_data['affine_input'], test_data['cond'], test_data['affine_target'], test_data['sign'], test_data['bce_mask']], batch_size=batch_size, callbacks=[model_checkpoint_callback],shuffle=True,epochs=50)

model_seq2seq.save_weights('models/seq2seq/seq2seq_model_50_wobce_tinput.h5')
decoder_output, stop_signs = model_seq2seq.predict([test_data['vox2d'], test_data['affine_input'], test_data['cond'], test_data['affine_target'], test_data['sign'], test_data['bce_mask']])
decoded_masks = model_seq2seq.part_autoencoder.reconstruct(np.reshape(decoder_output[:,:,:-4],(-1, 128)))
decoded_masks = np.reshape(decoded_masks, (-1, 24, decoded_masks.shape[1], decoded_masks.shape[2], decoded_masks.shape[3]))

visualise_seq_results(test_data['vox2d'], decoded_masks, 3, save_dir='Results/seq2seq/tinput')

print(decoder_output.shape)
print(stop_signs.shape)



