#!/usr/bin/env python
# USAGE
from __future__ import print_function

import sys
import sys, os, copy
import click
os.environ['TERM'] = 'linux'
#pyroot module
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
random.seed(1)
import ROOT as r


# PYTHONPATH TO fastjet WRAPPER
sys.path.append("fastjet-install/lib/python2.7/site-packages")
import fastjet as fj
import analysis_functions as af
import preprocess_functions as pf      
import tree_cluster_hist as cluster_h


def cast(event):
    all_particles=[]
    for i, p in enumerate(event):
        temp=fj.PseudoJet(p[1],p[2],p[3],p[0])
        all_particles.append(temp)
    return all_particles

def cluster(particle_list, Rjet, jetdef_tree):
    
    if jetdef_tree=='antikt':
       tree_jet_def = fj.JetDefinition(fj.antikt_algorithm, Rjet)
    elif jetdef_tree=='kt':
       tree_jet_def = fj.JetDefinition(fj.kt_algorithm, Rjet)
    elif jetdef_tree=='CA':
       tree_jet_def = fj.JetDefinition(fj.cambridge_algorithm, Rjet)
    else:
       print('Missing jet definition')

    jets=[]
    out_jet=fj.sorted_by_pt(tree_jet_def(particle_list))
    for i in range(len(out_jet)):
        if i==0:
           trees, contents=cluster_h._traverse(out_jet[i])
           tree=np.asarray([trees])
           tree=np.asarray([np.asarray(e).reshape(-1,2) for e in tree])
           content=np.asarray([contents])
           content=np.asarray([np.asarray(e).reshape(-1,4) for e in content])
           masses=out_jet[i].m()
           pts=out_jet[i].pt()
           etas=out_jet[i].eta()
           phis=out_jet[i].phi()
           jets.append((tree, content, masses, pts, etas,phis))
    return jets


data=np.load('trainjets.npy')
X=[]
Y=[]
index=0
for entry in data:
    index+=1
    print("train:",index)
    label=entry[1]
    e=cast(entry[0])
    tree, content, mass, pt,eta, phi=cluster(e, 100.0, "kt")[0]
    jet = {}
    jet["root_id"] = 0
    jet["tree"] = tree[0]
    jet["content"] = content[0]
    jet["mass"] = mass
    jet["pt"] = pt
    jet["energy"] = content[0][0, 3]
    jet["eta"] = eta
    jet["phi"] = phi
    X.append(jet)
    Y.append(label)

fd_train = open("train_top.pickle", "wb")
pickle.dump((X, Y), fd_train, protocol=pickle.HIGHEST_PROTOCOL)
fd_train.close()
