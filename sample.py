from __future__ import print_function

import argparse
import os
import h5py
import numpy as np
import sys

from molecules.model import MoleculeAE
from molecules.utils import load_dataset, from_one_hot_array, decode_smiles_from_indexes

from pylab import figure, axes, scatter, title, show
from rdkit import Chem
from rdkit.Chem import Draw

LATENT_DIM = 56
#TARGET = 'autoencoder'

def get_arguments():
	parser = argparse.ArgumentsParser(description='Molecular autoencoder network')
	parser.add_argument('data', type=str, help='File of latent representation tensors \
		for decoding')
	parser.add_argument('model', type=str, help='Trained Keras model to use.')
	parser.add_argument('--latent_dim', type=int, metavar='N',
		default=LATENT_DIM, help='Dimensionality of the latent representation.')
	return parser.parse_args()

def autoencoder(args, model):
	latent_dim = args.latent_dim
	data, charset = load_dataset(args.data, split=False)

	if os.path.isfile(args.model):
		model.load(charset, args.model, latent_rep_size = latent_dim)
	else:
		raise ValueError("Model file %s does not exist" % args.model)

	sampled = model.autoencoder.predict(data[0].reshape(1, 188, len(charset))).argmax(axis=2)[0]
	mol = decode_smiles_from_indexes(map(from_one_hot_array, data[0]), charset)
	sampled = decode_smiles_from_indexes(sampled, charset)
	print(mol)
	print(sampled)

def main():
	args = get_arguments()
	model = MoleculeAE()
	autoencoder(args, model)

if __name__ == '__main__':
	main()
