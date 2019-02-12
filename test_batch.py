import sys
import os
import numpy as np
import logging
import torch
import glob
import os
import pickle
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import RobustScaler
from rec_net import Predict
from train_batch import pt_order,rewrite_content,extract

n_features=7
n_hidden=50

def wrap(y, dtype='float'):
    y_wrap = Variable(torch.from_numpy(y))
    if dtype=='float':
        y_wrap = y_wrap.float()
    elif dtype == 'long':
        y_wrap = y_wrap.long()
return y_wrap

def unwrap(y_wrap):
    y = y_wrap.data.numpy()
    return y

def wrap_X(X):
    X_wrap = copy.deepcopy(X)
    for jet in X_wrap:
        jet["content"] = wrap(jet["content"])
    return X_wrap



def load_train_file(filename_train, n_train=1200000):
    print("Loading train data")
    fd = open(filename_train, "rb")
    X, y = pickle.load(fd,encoding='latin-1')
    fd.close()
    y = np.array(y)
    indices = torch.randperm(len(X)).numpy()[:n_train]
    X = [X[i] for i in indices]
    y = y[indices]
    print("\tfilename = %s" % filename_train)
    print("\tX size = %d" % len(X))
    print("\ty size = %d" % len(y))
    print("Preprocessing the train data")
    X = [extract(pt_order(rewrite_content(jet))) for jet in X]
    transfer_feature = RobustScaler().fit(np.vstack([jet["content"] for jet in X]))
    
    return transfer_feature

def load_test_file(transfer_feature, filename_test):
    print("Loading test data")
    fd = open(filename_test, "rb")
    X, y = pickle.load(fd,encoding='latin-1')
    fd.close()
    y = np.array(y)
    print("\tfilename = %s" % filename_test)
    print("\tX size = %d" % len(X))
    print("\ty size = %d" % len(y))
    X = [extract(pt_order(rewrite_content(jet))) for jet in X]
    for jet in X:
        jet["content"] = transfer_feature.transform(jet["content"])
    return X, y


def load_model(filename):
    with open(os.path.join("test_top_pt",filename), 'rb') as f:
        print(filename)
        state_dict = torch.load(f,map_location='cpu')
    model = Predict(n_features,n_hidden)
    model.load_state_dict(state_dict)
    return model


def evaluate_models(X, y, model_filenames,batch_size=128):
    rocs = []
    fprs = []
    tprs = []
    
    for filename in os.listdir(model_filenames):
        print(filename)
        if filename.endswith('.pt'):
            model = load_model(filename)
            model.eval()
            offset = 0
            yy, yy_pred, accuracy=[],[],[]
            for i in range(len(X)// batch_size):
                if i == (len(X)// batch_size)-1: 
                   print("offset:",offset, offset+batch_size)
                idx = slice(offset, offset+batch_size)
                Xt, yt = X[idx], y[idx]
                X_var = wrap_X(Xt); y_var = wrap(yt)
                y_pred=unwrap(model(X_var));yt = unwrap(y_var)
                yy_pred.append(y_pred); yy.append(yt)
                accuracy.append(np.sum(np.rint(y_pred).ravel()==yt)/float(len(yt)))
                offset+=batch_size
            roc_auc = roc_auc_score(np.vstack(yy).ravel(), np.vstack(yy_pred).ravel())
            fpr, tpr, _ = roc_curve(np.vstack(yy).ravel(), np.vstack(yy_pred).ravel())
            
            fd = open("ytrue_ypred.pickle", "wb")
            pickle.dump((np.vstack(yy).ravel(), np.vstack(yy_pred).ravel()), fd,protocol=2)
            fd.close() 
            rocs.append(roc_auc)
            fprs.append(fpr)
            tprs.append(tpr)
            accuracy_test=np.mean(np.array(accuracy))
            print("ROC AUC = %.4f" % rocs[-1])
            print("accuracy=",accuracy_test)
    
    return rocs, fprs, tprs


features = load_train_file("train-top.pickle")
X, y = load_test_file(features,"test-top.pickle")
rocs, fprs, tprs = evaluate_models(X, y, "test_top_pt")
    

