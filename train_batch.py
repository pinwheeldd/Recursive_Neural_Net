import torch
from torch.optim import Adam, lr_scheduler
from torch.autograd import Variable
import copy
import numpy as np
import logging
import pickle
import time
import os
import click

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
from rec_net import Predict
from rec_net import log_loss

def wrap(y, dtype='float'):
    y_wrap = Variable(torch.from_numpy(y))
    if dtype=='float':
        y_wrap = y_wrap.float()
    elif dtype == 'long':
        y_wrap = y_wrap.long()
if torch.cuda.is_available():
    y_wrap = y_wrap.cuda()
    return y_wrap


def unwrap(y_wrap):
    if y_wrap.is_cuda:
        y = y_wrap.cpu().data.numpy()
    else:
        y = y_wrap.data.numpy()
    return y


def wrap_X(X):
    X_wrap = copy.deepcopy(X)
    for jet in X_wrap:
        jet["content"] = wrap(jet["content"])
    return X_wrap


def unwrap_X(X_wrap):
    X_new = []
    for jet in X_wrap:
        jet["content"] = unwrap(jet["content"])
        X_new.append(jet)
    return X_new


def _pt(v):
    pz = v[2]
    p = (v[0:3] ** 2).sum() ** 0.5
    eta = 0.5 * (np.log(p + pz) - np.log(p - pz))
    pt = p / np.cosh(eta)
    return pt


def pt_order(jet, root_id=None): # ensure that the left sub-jet has always a larger pt than the right
    
    if root_id is None:
        root_id = jet["root_id"]
    
    if jet["tree"][root_id][0] != -1:
        left = jet["tree"][root_id][0]
        right = jet["tree"][root_id][1]
        
        pt_left = _pt(jet["content"][left])
        pt_right = _pt(jet["content"][right])
        
        if pt_left < pt_right:
            jet["tree"][root_id][0] = right
            jet["tree"][root_id][1] = left
        
        permute_by_pt(jet, left)
        permute_by_pt(jet, right)
    
    return jet


def rewrite_content(jet):
    jet = copy.deepcopy(jet)
    
    if jet["content"].shape[1] == 5:
        pflow = jet["content"][:, 4].copy()

    content = jet["content"]
    tree = jet["tree"]

    def _rec(i):
        if tree[i, 0] == -1:
           pass
        else:
            _rec(tree[i, 0])
            _rec(tree[i, 1])
            c = content[tree[i, 0]] + content[tree[i, 1]]
            content[i] = c

    _rec(jet["root_id"])

    if jet["content"].shape[1] == 5:
       jet["content"][:, 4] = pflow
    
    return jet

def loss(y_pred, y):
    l = log_loss(y, y_pred.squeeze(1)).mean()
    return l


def extract(jet, pflow=False): ## extracting all 7 features
    
    jet = copy.deepcopy(jet)
    s = jet["content"].shape
    if not pflow:
        content = np.zeros((s[0], 7))
    else:
        content = np.zeros((s[0], 7+4))

    for i in range(len(jet["content"])):
        px = jet["content"][i, 0]
        py = jet["content"][i, 1]
        pz = jet["content"][i, 2]
        
        p = (jet["content"][i, 0:3] ** 2).sum() ** 0.5
        eta = 0.5 * (np.log(p + pz) - np.log(p - pz))
        theta = 2 * np.arctan(np.exp(-eta))
        pt = p / np.cosh(eta)
        phi = np.arctan2(py, px)
        
        content[i, 0] = p
        content[i, 1] = eta if np.isfinite(eta) else 0.0
        content[i, 2] = phi
        content[i, 3] = jet["content"][i, 3]
        content[i, 4] = (jet["content"][i, 3] /jet["content"][jet["root_id"], 3])
        content[i, 5] = pt if np.isfinite(pt) else 0.0
        content[i, 6] = theta if np.isfinite(theta) else 0.0
        if pflow:
           if jet["content"][i, 4] >= 0:
              content[i, 7+int(jet["content"][i, 4])] = 1.0

     jet["content"] = content
     return jet

@click.command()
@click.argument("filename_train")
@click.argument("filename_valid")
@click.argument("filename_model")
@click.option("--n_train", default=1200000)
@click.option("--n_valid", default=400000)
@click.option("--n_features", default=7)
@click.option("--n_hidden", default=50)
@click.option("--n_epochs", default=20)
@click.option("--batch_size", default=128)
@click.option("--step_size", default=0.005)
@click.option("--decay", default=0.9)

os.environ['CUDA_VISIBLE_DEVICES'] = "1" ## create the environment


def train(filename_train,filename_valid,filename_model,n_train=1200000,n_valid=400000,n_features=7,
                n_hidden=40,n_epochs=18,batch_size=128,step_size=0.005,decay=0.9):
   
    logging.info("Calling with...")
    logging.info("\tfilename_train = %s" % filename_train)
    logging.info("\tfilename_valid = %s" % filename_valid)
    logging.info("\tfilename_model = %s" % filename_model)
    logging.info("\tn_train = %d" % n_train)
    logging.info("\tn_valid = %d" % n_valid)
    logging.info("\tn_features = %d" % n_features)
    logging.info("\tn_hidden = %d" % n_hidden)
    logging.info("\tn_epochs = %d" % n_epochs)
    logging.info("\tbatch_size = %d" % batch_size)
    logging.info("\tstep_size = %f" % step_size)
    logging.info("\tdecay = %f" % decay)
    ####################### Reading the train data #################################
    logging.info("Loading train data")
    
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



    # Preprocessing  # feature scaling
    logging.info("Preprocessing the train data")
    X = [extract(pt_order(rewrite_content(jet))) for jet in X]
    transfer_feature= RobustScaler().fit(np.vstack([jet["content"] for jet in X]))
    for jet in X:
        jet["content"] = transfer_feature.transform(jet["content"])

    X_train=X
    y_train=y

    '''----------------------------------------------------------------------- '''
    logging.info("Loading validation data")
    
    fd = open(filename_valid, "rb")
    X, y = pickle.load(fd,encoding='latin-1')
    fd.close()
    y = np.array(y)
    
    indices = torch.randperm(len(X)).numpy()[:n_valid]
    X = [X[i] for i in indices]
    y = y[indices]

    print("\tfilename = %s" % filename_valid)
    print("\tX size = %d" % len(X))
    print("\ty size = %d" % len(y))
    logging.info("Preprocessing the train data")
    X = [extract(pt_order(rewrite_content(jet))) for jet in X]
    for jet in X:
        jet["content"] = transfer_feature.transform(jet["content"])
    X_valid=X
    y_valid=y

###########################################Define MODEL ##############################

    logging.info("Initializing model...")
    model = Predict(n_features,n_hidden)
    if torch.cuda.is_available():
       logging.warning("Moving model to GPU")
       model.cuda()
       logging.warning("Moved model to GPU")

###########################OPTIMIZER AND LOSS ##########################################
    logging.info("Building optimizer...")
    optimizer = Adam(model.parameters(), lr=step_size)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=decay)
    
    n_batches = int(len(X_train) // batch_size)
    best_score = [-np.inf]  
    best_model_state_dict = copy.deepcopy(model.state_dict())  # intial parameters of model
    
        
        
###############################VALIDATION OF DATA ########################################
    def callback(epoch, iteration, model):
        
        if iteration % n_batches == 0:
            model.eval()
            offset = 0; train_loss = []; valid_loss = []
            yy, yy_pred, accuracy_train, accuracy_valid = [], [],[],[]
            for i in range(len(X_valid) // batch_size):
                idx = slice(offset, offset+batch_size)
                Xt, yt = X_train[idx], y_train[idx]
                X_var = wrap_X(Xt); y_var = wrap(yt)
                tl = unwrap(loss(model(X_var), y_var)); train_loss.append(tl)
                y_pred_train = model(X_var)
                y = unwrap(y_var); y_pred_train = unwrap(y_pred_train)
                X = unwrap_X(X_var)

                Xv, yv = X_valid[idx], y_valid[idx]
                X_var = wrap_X(Xv); y_var = wrap(yv)
                y_pred = model(X_var)
                vl = unwrap(loss(y_pred, y_var)); valid_loss.append(vl)
                Xv = unwrap_X(X_var); yv = unwrap(y_var); y_pred = unwrap(y_pred)
                yy.append(yv); yy_pred.append(y_pred)
                y_pred=np.column_stack(y_pred).ravel()
                accuracy_valid.append(np.sum(np.rint(y_pred)==yv)/float(len(yv)))
                offset+=batch_size
        
            train_loss = np.mean(np.array(train_loss))
            valid_loss = np.mean(np.array(valid_loss))
            accuracy_valid=np.mean(np.array(accuracy_valid))
            print("accuracy_valid:",accuracy_valid)
            print("train_loss:",train_loss)
            roc_auc = roc_auc_score(np.column_stack(yy).ravel(), np.column_stack(yy_pred).ravel())
            print("roc_auc:",roc_auc)
            if roc_auc > best_score[0]:
               best_score[0]=roc_auc
               best_model_state_dict[0] = copy.deepcopy(model.state_dict())
               with open(filename_model, 'wb') as f:
                    torch.save(best_model_state_dict[0], f)
            scheduler.step(valid_loss)
            model.train()

 ###############################TRAINING ########################################
    logging.warning("Training the data")
    iteration=1
    for i in range(n_epochs):
        print("epoch = %d" % i)
        print("step_size = %.4f" % step_size)
        t0 = time.time()
        for _ in range(n_batches): ## mini batch
            iteration += 1
            model.train()
            optimizer.zero_grad()
            start = torch.round(torch.rand(1) * (len(X_train) - batch_size)).numpy()[0].astype(np.int32)
            idx = slice(start, start+batch_size)
            X, y = X_train[idx], y_train[idx]
            X_var = wrap_X(X); y_var = wrap(y) ## wrap_X, wrap moves to GPU
            l = loss(model(X_var), y_var)
            l.backward()
            optimizer.step()
            X = unwrap_X(X_var); y = unwrap(y_var) ## unwrap_X, unwrap moves to GPU
            callback(i, iteration, model)
            t1 = time.time() ###
        print(f'Epoch took {t1-t0} seconds')
        scheduler.step()
        step_size = step_size * decay


if __name__ == "__main__":
    train()
