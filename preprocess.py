import argparse
import pandas
import h5py
import numpy as np
from molecules.utils import one_hot_array, one_hot_index

from sklearn.model_selection import train_test_split

MAX_LEN_STR = 188
MAX_NUM_ROWS = 1000000

def get_arguments():
	parser = argparse.ArgumentParser(description='Prepare data for training')
	parser.add_argument('infile', type=str, help='Input file name')
	parser.add_argument('outfile', type=str, help='Output file name')
	parser.add_argument('--infile2', type=str, help='Second input file name')
	parser.add_argument('--infile3', type=str, help='Third input file name')
	parser.add_argument('--length', type=int, metavar='N', default=MAX_NUM_ROWS,
		help='Maximum number of rows to include (randomly sampled)')
	return parser.parse_args()

def chunk_iterator(dataset, chunk_size=1000):
	chunk_indices = np.array_split(np.arange(len(dataset)),
		len(dataset)/chunk_size)
	for chunk_ixs in chunk_indices:
		chunk = dataset[chunk_ixs]
		yield (chunk_ixs, chunk)
	raise StopIteration


def main():
	args = get_arguments()
	data1 = open(args.infile, 'r')
	data1 = data1.readlines()
	data2 = open(args.infile2, 'r')
	data2 = data2.readlines()
	data3 = open(args.infile3, 'r')
	data3 = data3.readlines()
	len_train = len(data1)
	len_test = len(data2) + len(data3)

	data = np.array(data1 + data2 + data3)
	train_idx = np.arange(len_train)
	np.random.shuffle(train_idx)
	test_idx = np.arange(len_train, len_train + len_test)
	np.random.shuffle(test_idx)

	if args.length < len(data):
		data = np.random.choice(data, args.length)
	
	charset = list(reduce(lambda x, y: set(y) | x, data, set()))
	one_hot_encoded_fn = lambda row: map(lambda x: one_hot_array(x, len(charset)),
		one_hot_index(row, charset))
	h5f = h5py.File(args.outfile, 'w')
	h5f.create_dataset('charset', data=charset)

	def create_chunck_dataset(h5file, dataset_name, dataset, dataset_shape,
		chunk_size=1000, apply_fn=None):
		new_data = h5file.create_dataset(dataset_name, dataset_shape,
			chunks=tuple([chunk_size]+list(dataset_shape[1:])))
		for (chunk_ixs, chunk) in chunk_iterator(dataset):
			if not apply_fn:
				new_data[chunk_ixs, ...] = chunk
			else:
				#print(chunk)
				#print(type(chunk))
				new_data[chunk_ixs, ...] = apply_fn(chunk)

	create_chunck_dataset(h5f, 'data_train', train_idx, (len(train_idx), MAX_LEN_STR,
		len(charset)), apply_fn=lambda ch: np.array(map(one_hot_encoded_fn, data[ch])))
	create_chunck_dataset(h5f, 'data_test', test_idx, (len(test_idx), MAX_LEN_STR,
		len(charset)), apply_fn=lambda ch: np.array(map(one_hot_encoded_fn, data[ch])))

	h5f.close()

if __name__ == '__main__':
	main()








