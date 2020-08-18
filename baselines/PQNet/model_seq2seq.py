import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Input, InputLayer, Conv2D, Dense, BatchNormalization, ReLU, LeakyReLU, Activation, Dropout, Concatenate, ZeroPadding2D, Reshape, RepeatVector, Flatten, Lambda
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from model_partae import *
import random
class EncoderGRU(tf.keras.Model):
	def __init__(self, hidden_size, n_layer=1, bidirectional=False):
		super(EncoderGRU, self).__init__()
		self.hidden_size = hidden_size
		self.n_layer = n_layer
		self.bidirectional = bidirectional
		self.num_directions = 2 if bidirectional else 1
		gru_layers = []
		for i in range(n_layer):
			if bidirectional==True:
				gru_layers.append(Sequential(Bidirectional(GRU(hidden_size, dropout=0.2 if n_layer==2 else 0, return_sequences=True))))
			else:
				gru_layers.append(Sequential(GRU(hidden_size, dropout=0.2 if n_layer==2 else 0, return_sequences=True)))

		self.gru_layers = gru_layers

	def call(self, input):
	#	print((self.gru_layers[0](input).shape))
		output1 = self.gru_layers[0](input)
		output2= self.gru_layers[1](output1)
		
		return [output1, output2]#, [hidden1[0], hidden2[0]]

class DecoderGRU(tf.keras.Model):
	def __init__(self, input_size, hidden_size, n_layer=1, bidirectional=False):
		super(DecoderGRU, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.bidirectional = bidirectional
		self.num_directions = 2 if bidirectional else 1
		self.n_units_hidden1 = 256
		self.n_units_hidden2 = 128
		gru_layers = []
		for i in range(n_layer):
			if bidirectional==True:
				gru_layers.append(Bidirectional(GRU(hidden_size, dropout=0.2 if n_layer==2 else 0, return_sequences=True)))
			else:
				gru_layers.append(GRU(hidden_size, dropout=0.2 if n_layer==2 else 0, return_sequences=True))
		self.gru_layers = gru_layers
		self.linear1 = Sequential([Dense(self.n_units_hidden1), LeakyReLU(), Dense(input_size - 4)])
		self.linear2 = Sequential([Dense(self.n_units_hidden2), ReLU(), Dropout(0.2), Dense(4)])
		self.linear3 = Sequential([Dense(self.n_units_hidden2)	, ReLU(), Dropout(0.2), Dense(1)])
		self.dropout_i = 0.2
		self.dropout_o = 0.2
		self.init_input = self.initInput()

	def initInput(self):
		initial_code = tf.zeros((1, 1, self.input_size - 4))
		initial_param = tf.constant([0.5, 0.5, 1, 1])
		initial_param = tf.expand_dims(tf.expand_dims(initial_param, axis=0), axis=0)
		initial = tf.concat([initial_code, initial_param], axis=2)
		return initial

	def call(self, input, hidden):
		#hidden1 = tf.reshape(hidden[0], (hidden[0].shape[0], 2, -1))
	#	hidden2 = tf.reshape(hidden[1], (hidden[0].shape[0], 2, -1))
		hidden1 = hidden[0]
		hidden2 = hidden[1]
		#print("H1" + str(hidden1.shape))
		#print("H2" + str(hidden2.shape))	
		output1 = self.gru_layers[0](input, initial_state=hidden1)
		output2 = self.gru_layers[1](output1, initial_state=hidden2)

		#print("DGRU1" + str(output1.shape))
		#print("DGRU2" + str(output2.shape))	
		
		output1 = tf.squeeze(output1, axis=1)
		output2 = tf.squeeze(output2, axis=1)

		#print("o1" + str(output1.shape))
		#print("o2" + str(output2.shape))
		output_code = self.linear1(output1)
		output_param = self.linear2(output2)
		stop_sign = self.linear3(output1)


		#print("output_code" + str(output_code.shape))
		#print("output_param" + str(output_param.shape))	

		output_seq = tf.concat([output_code, output_param], axis=1)

		return [output1, output2], output_seq, stop_sign

class PartSeq2Seq(tf.keras.Model):
  def __init__(self, config):
    super(PartSeq2Seq, self).__init__()
    input_shape = config['input_shape']
    point_batch_size = config['point_batch_size']
    en_n_layers = config['en_n_layers']
    ef_dim = config['ef_dim']
    de_n_layers = config['de_n_layers']
    df_dim = config['df_dim']
    z_dim = config['z_dim']
    partae_model_file = config['partae_file']
    part_en_n_layers = config['part_en_n_layers']
    part_de_n_layers = config['part_de_n_layers']
    self.target_input_prob = config['target_input_prob']

    self.part_autoencoder = PartAE(input_shape, part_en_n_layers, ef_dim, part_de_n_layers, df_dim, z_dim)
    sample_masks = tf.random.normal((32, input_shape[0], input_shape[1], input_shape[2]))
    sample_points = tf.random.normal((32, point_batch_size, 2))
    self.part_autoencoder([sample_masks, sample_points])
    self.part_autoencoder.load_weights(partae_model_file)
    self.part_encoder = self.part_autoencoder.encoder 
    self.part_decoder = self.part_autoencoder.decoder
    self.part_encoder.trainable = False
    self.encoder =  EncoderGRU(100, n_layer=en_n_layers, bidirectional=True)
    self.decoder = 	DecoderGRU(z_dim + 4, 200, n_layer=de_n_layers, bidirectional=False)
  
  def call(self, input,  training=True):

    row_vox2d = input[0]
    affine_input = input[1]
    cond = input[2]
    affine_target = tf.identity(input[3])
    target_stop = tf.identity(input[4])
    bce_mask = tf.identity(input[5])


    batch_size, max_n_parts, vox_dim = row_vox2d.shape[0], row_vox2d.shape[1], row_vox2d.shape[2]
    #bce_mask = data['mask']

    batch_vox2d = tf.reshape(row_vox2d, (-1, vox_dim, vox_dim, 1))
    part_geo_features = self.part_encoder(batch_vox2d)
    #print(part_geo_features.shape)
    part_geo_features = tf.reshape(part_geo_features, (batch_size, max_n_parts, -1))
    #print(part_geo_features.shape)

    target_part_geo = tf.identity(part_geo_features)
    #print(target_part_geo.shape)
    part_feature_seq = tf.concat([part_geo_features, affine_input, cond], axis=2)
    target_seq = tf.concat([target_part_geo, affine_target], axis=2)

    hidden1, hidden2 = self.encoder(part_feature_seq)
    #print(hidden1.shape)
    #print(hidden2.shape)
    decoder_hidden = [hidden1[:, -1, :], hidden2[:, -1, :]]
    decoder_input = tf.identity(tf.tile(self.decoder.init_input,((batch_size, 1, 1))))
    decoder_outputs = []
    stop_signs = []
    pass_target_input = random.uniform(0,1) > self.target_input_prob

    for di in range(target_part_geo.shape[1]):
      #print("DI" + str(decoder_input.shape))
      #print("DH" + str(decoder_hidden[0].shape))
      decoder_output, output_seq, stop_sign = self.decoder(decoder_input, decoder_hidden)
      #print("DO1" + str(decoder_output[0].shape))
      #print("DO2" + str(decoder_output[1].shape))
      #print(output_seq.shape)
      decoder_outputs.append(output_seq)
      stop_signs.append(stop_sign)
      decoder_input = tf.expand_dims(output_seq, axis=1) if pass_target_input==False else target_seq[:,di:di+1,:]
      #print(decoder_input.shape)
      decoder_hidden = decoder_output

    decoder_outputs = tf.stack(decoder_outputs, axis=1)
    stop_signs = tf.stack(stop_signs, axis=1)

    bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    #print(stop_signs.shape)
    #print(target_stop.shape)

    bce_loss = bce(stop_signs, target_stop)
    code_rec_loss = mse(decoder_outputs[:, :, :-4], target_part_geo)
    param_rec_loss = mse(decoder_outputs[:, :, -4:], affine_target)

    #bce_loss = tf.math.maximum((tf.reduce_sum(bce_loss))/(tf.reduce_sum(bce_mask)), 0.001)
    #code_rec_loss = tf.reduce_sum(code_rec_loss)/(tf.reduce_sum(bce_mask))
    #param_rec_loss = tf.reduce_sum(param_rec_loss)/(tf.reduce_sum(bce_mask))

    loss =  code_rec_loss + param_rec_loss

    # print("bce_loss : " + str(bce_loss))
    # print("code_rec_loss : " + str(code_rec_loss))
    # print("param_rec_loss : " + str(param_rec_loss))
    # print("loss : " + str(loss))
    self.add_loss(loss)

    return decoder_outputs, stop_signs



print(tf.__version__)
#class Seq2SeqAE(tf.keras.Model):
#	def __init__(self, en_input_size, de_input_size, hidden_size):

if __name__ == "__main__": 
	"""
	model_encoder = EncoderGRU((32, 24, 132), 100,n_layer=2, bidirectional=True)
	model_decoder = DecoderGRU(132, 100,n_layer=2, bidirectional=True)
	
	encoder_input = tf.random.normal((32, 24, 132))
	decoder_input = tf.random.normal((32, 1, 128))
	decoder_hidden1 = tf.random.normal((32, 1, 200))
	decoder_hidden2 = tf.random.normal((32, 1, 200))
	out_enc = model_encoder(encoder_input)
	output, output_seq, stop_sign = model_decoder(decoder_input, hidden=[decoder_hidden1, decoder_hidden2])

"""
	config = {
	'input_shape': (64, 64, 1),
	'point_batch_size':5,
	'en_n_layers':2,
	'ef_dim':32,
	'de_n_layers':2,
	'df_dim':32,
	'z_dim':128,
	'partae_file':'model_partae.h5',
	'part_en_n_layers':5,
	'part_de_n_layers':5
	}

	model = PartSeq2Seq(config)

	data = {

		'vox2d': tf.random.normal((32, 24, 64, 64, 1)),   # (B, T, vox_dim, vox_dim, 1),
		'sign': tf.random.normal((32, 24, 1)),
		'affine_input' : tf.random.normal((32, 24, 4)),
		'affine_target' : tf.random.normal((32, 24, 4)),
		'cond':tf.random.normal((32, 24, 24)),
		'bce_mask':tf.random.normal((32, 24))
	}


	output_seq, stop_signs = model((data['vox2d'], data['affine_input'],data['cond'],data['affine_target'],data['sign'], data['bce_mask']))
	#print(output_seq.shape)
	#print(stop_signs.shape)

	model.compile(optimizer='adam')
	model.fit(x=(data['vox2d'], data['affine_input'], data['cond'], data['affine_target'], data['sign'], data['bce_mask']), epochs=2)

	#print(model_encoder.summary())