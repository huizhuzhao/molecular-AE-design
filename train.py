from __future__ import print_function

import argparse
import os
import h5py
import numpy as np

from molecules.model import MoleculeAE
from molecules.utils import load_dataset
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.callbacks import Callback

NUM_EPOCHS = 1
BATCH_SIZE = 600
LATENT_DIM = 56
RANDOM_SEED = 1337
LOGFILE = "logfile.txt"

class LogHistory(Callback):
	def on_epoch_begin(self, epoch, logs={}):
		logfile = open(LOGFILE, "a")
		logfile.write("At the beginning of epoch %d:\r\n" % epoch)
		logfile.close()

	def on_batch_end(self, batch, logs={}):
		loss = logs.get('loss')
		accuracy = logs.get('acc')
		logfile = open(LOGFILE, "a")
		logfile.write("batch: {:d} loss: {:f} acc: {:f} \r\n".format(
			batch, loss, accuracy))
		logfile.close()

	def on_epoch_end(self, epoch, logs={}):
		loss = logs.get('loss')
		accuracy = logs.get('acc')
		val_loss = logs.get('val_loss')
		val_accuracy = logs.get('val_acc')
		logfile = open(LOGFILE, "a")
		logfile.write("val_loss: {:f} val_acc: {:f} \r\n".format(
			val_loss, val_accuracy))
		logfile.write("At the end of epoch {:d}: loss: {:f} acc: {:f} \r\n".
			format(epoch, loss, accuracy))
		logfile.close()

def get_arguments():
	parser = argparse.ArgumentParser(description='Molecular autoencoder')
	parser.add_argument('data', type=str, help='The HDF5 file containing \
		preprocessed data used for training.')
	parser.add_argument('model', type=str, help = 'Where to save the trained \
		model. If this file exists, it will be opened and resumed.')
	parser.add_argument('--epochs', type=int, metavar='N', default=NUM_EPOCHS,
		help='Number of epochs to run during training.')
	parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT_DIM,
		help='Dimensionality of the latent representation.')
	parser.add_argument('--batch_size', type=int, metavar='N', default=BATCH_SIZE,
		help='Number of samples to process per minibatch during training.')
	parser.add_argument('--random_seed', type=int, metavar='N', default=RANDOM_SEED,
		help='Seed to use to start randomizer for shuffling.')
	return parser.parse_args()

def main():
	args = get_arguments()
	np.random.seed(args.random_seed)

	data_train, data_test, charset = load_dataset(args.data)
	print("Length of charset")
	print(len(charset))
	model = MoleculeAE()
	if os.path.isfile(args.model):
		model.load(charset, args.model, latent_rep_size=args.latent_dim)
	else:
		model.create(charset, latent_rep_size=args.latent_dim)

	checkpointer = ModelCheckpoint(filepath=args.model, verbose=1,
		save_best_only=True)
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
		patience=3, min_lr=0.0001)

	
	if not os.path.isfile(LOGFILE):
		logfile = open(LOGFILE, "w+")
		logfile.close()

	history = LogHistory()

	model.autoencoder.fit(data_train, data_train, shuffle=True, nb_epoch=args.epochs,
		batch_size=args.batch_size, callbacks=[checkpointer, reduce_lr, history],
		validation_data=(data_test, data_test))

	logfile = open(LOGFILE, "a")
	logfile.write("Finished the fitting process.")
	logfile.close()	


if __name__ == '__main__':
	main()