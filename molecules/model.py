from keras import backend as K
from keras import objectives
from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.layers.core import Dense, Activation, Flatten, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import GRU
from keras.layers.convolutional import Convolution1D

LATENT_REP = 56
MAX_LENGTH = 187

class MoleculeAE():
	autoencoder = None

	def create(self, charset, max_length = MAX_LENGTH,
		latent_rep_size=LATENT_REP, weights_file=None):
		charset_length = len(charset)
		inp = Input(shape=(max_length, charset_length))
		ae_loss, latent = self._buildEncoder()
		outp = self._buildDecoder(latent, latent_rep_size, max_length)
		self.autoencoder = Model(inp, outp)

		if weights_file:
			self.autoencoder.load_weights(weights_file)

		self.autoencoder.compile(optimizer='Adam',
			loss='ae_loss',
			metrics=['accuracy'])

	def _buildEncoder(self, inp, latent_rep_size,
		max_length):
		h = Convolution1D(9, 9, activation='relu', name = 'conv_1')(inp)
		h = Convolution1D(9, 9, activation='relu', name = 'conv_2')(h)
		h = Convolution1D(10, 11, activation='relu', name = 'conv_3')(h)
		h = Flatten(name='flatten')(h)
		h = Dense(435, activation='relu', name='dense_1')(h)
		latent = Dense(56, activation='relu', name='latent')(h)

		def ae_loss(x, x_decoded_mean):
			x = K.flatten(x)
			x_decoded_mean = K.flatten(x_decoded_mean)
			loss = max_length * objectives.binary_crossentropy(x, x_decoded_mean)
			return loss

		return(ae_loss, latent)

	def _buildDecoder(self, latent, latent_rep_size, max_length):
		h = Dense(latent_rep_size, name='latent_input', activation='relu')(latent)
		h = RepeatVector(max_length, name='repeat_vector')(h)
		h = GRU(501, return_sequences=True, name='gru_1')(h)
		h = GRU(501, return_sequences=True, name='gru_2')(h)
		h = GRU(501, return_sequences=True, name='gru_3')(h)
		return TimeDistributed(Dense(charset_length, activation='softmax'), 
			name='decoded_mean')(h)

	def save(self, filename):
		self.autoencoder.save_weights(filename)

	def load(self, charset, weights_file, latent_rep_size = LATENT_REP):
		self.create(self, charset, weights_file = weights_file, 
			latent_rep_size = latent_rep_size)

