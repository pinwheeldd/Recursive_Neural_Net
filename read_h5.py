#!/usr/bin/env python
## USAGE --- python read_h5.py
from __future__ import print_function
import sys
import h5py
import pandas
import sys, os, copy
os.environ['TERM'] = 'linux'
import numpy as np
import pickle


### This function reads the h5 files and saves the jets in numpy arrays
def h5_to_npy(filename,Njets):
    file = pandas.HDFStore(filename)
    jets=np.array(file.select("table",start=0,stop=Njets))
    jets2=jets[:,0:800].reshape((Njets,200,4))
    labels=jets[:,805:806]
    npy_jets=[]
    for i in range(len(jets2)):
        nonzero_entries=jets2[i][~np.all(jets2[i] == 0, axis=1)]
        npy_jets.append([nonzero_entries,0 if labels[i] == 0 else 1])
    return npy_jets

infiles=['train.h5','val.h5','test.h5'] ## these are all the input files for train, validation, test sample
no_of_jets=[1200000,400000,400000]
outfiles=['trainjets.npy','validjets.npy','testjets.npy']
for i in range(len(infiles)):
    np.save(outfiles[i],h5_to_npy(infiles[i],no_of_jets[i]))



