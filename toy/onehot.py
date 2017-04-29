import tensorflow as tf
import numpy as np
import math

###
# One hot vector 

char_list = [" ", "#", "(", ")", "+", "-", "/", "1", "2", "3", "4", "5", "6", "7","8", "=", "@", "B", "C", "F", "H", "I", "N", "O", "P", "S", "[", "\\", "]","c", "l", "n", "o", "r", "s"]

with open("s_tr.txt",'r') as f:
	tr_set = f.readlines()
tr_set = [i.strip() for i in tr_set]

cleaned_data = np.zeros((len(tr_set),187,35),dtype=np.float32)
char_lookup = dict((c,i) for i,c in enumerate(char_list))
for i,tr_set in enumerate(tr_set):
	for t,char in enumerate(tr_set):
		cleaned_data[i,t,char_lookup[char]] = 1

#print cleaned_data[0,0,:]

#char_to_index = dict((c, i) for i, c in enumerate(char_list))
#print char_to_index
#index_to_char = dict((i, c) for i, c in enumerate(char_list))
#print index_to_char








