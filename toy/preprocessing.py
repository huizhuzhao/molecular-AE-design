import tensorflow as tf
import numpy as np
import h5py
import random

"""
# Merging all six datasets 

data = open("6.txt", 'r')
lines = data.readlines()
#print lines
data.close()
Smiles = open("test.txt",'a')
for ChemRep in lines:
	smiles = ChemRep.split()
#	print smiles
	Smiles.write(smiles[2] + "\n")
iSmiles.close()

hdf5_file_name = '/home/hossein/Downloads/keras-molecules-master/data/smiles_50k.h5'
fil = h5py.File(hdf5_file_name, 'r')
#with h5py.File(hdf5_file_name, mode='r') as f: 
#	print f

print fil
fil.close()

# Shuffling randomly 

data = open("Data_s.txt",'r')
lines = data.readlines()
leng = len(lines)
print leng
random.shuffle(lines)
open("Data_s.txt",'w').writelines(lines)
"""
"""
# FIND Max 
i = 0
max_len = 0
data = open("Data_s.txt",'r')
lines = data.readlines()
for line in lines:
	
	if len(line)>max_len:
		max_len = len(line)
		max_ind = i
	i = i + 1
print max_len
print max_ind
j = 0
for line in lines:
	if j == max_ind:
		print line	
	j = j + 1
"""

"""
# pad space

max_len = 187
pad = " "


with open("Data_s.txt") as data:
	f = open("data_sh_pad.txt", 'w')
	for line in data:
		line = line.strip()
		while len(line) < max_len:
			line += pad
		#print line
		f.write(line + '\n')
	f.close()

"""
"""
#data = open("test.txt",'r')
#open("data_sh_pad.txt",'w').writelines('start')
#lines = data.readlines()
#f = open("data_sh_pad.txt",'w')
#for line in lines:
	while len(line)<max_len:
		line = line + pad
	#line1 = line.ljust(187)
	print len(line)
	#print line1
	#while len(line)<max_len:
	#	line = line + pad
	#lines2.append('%r\n'%line)
	#print len(line)
	open("data_sh_pad.txt",'a').writelines(line)
	#dat = "\r\n%s"%line
#f.writelines(lines2)
#f.close()

"""
"""
# test padding

data1 = open("data_sh_pad.txt",'r')
line2 = data1.readlines()
#print line2
for linee in line2:
	print len(linee)

"""



#Train, validation and test

data = open("test_pad.txt",'r')
lines = data.readlines()
leng = len(lines)
trainp = round(leng*.7)+1
valp = round(leng*.15)
valindexb = trainp
valindexe = valindexb+valp
testindexb = valindexe
testindexe = testindexb + valp
open("train.txt",'w').writelines("train")
open("validation.txt",'w').writelines("validation")
open("test.txt",'w').writelines("test")
i = 0
for line in lines:
	if i<trainp:
		open("train.txt",'a').writelines(line)
		i = i+1
	elif i<valindexe:
		open("validation.txt",'a').writelines(line)
		i = i+1
	elif i<testindexe: 
		open("test.txt",'a').writelines(line)
		i = i+1
print i 	

# test for length of datasets

tr = open("train.txt",'r')
linetr = tr.readlines()
val = open("validation.txt",'r')
lineval = val.readlines()
ts = open("test.txt",'r')
linets = ts.readlines()
print len(linetr), len(lineval), len(linets)























