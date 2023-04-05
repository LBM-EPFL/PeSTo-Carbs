import os
import sys
import h5py
import json
import numpy as np
import torch as pt
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
from sklearn import metrics

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = 14

#from theme import colors

from src.dataset import StructuresDataset, collate_batch_features, select_by_sid, select_by_max_ba, select_by_interface_types
from src.data_encoding import encode_structure, encode_features, extract_topology, categ_to_resnames, resname_to_categ
from src.structure import data_to_structure, encode_bfactor
from src.structure_io import save_pdb, read_pdb
from src.scoring import bc_scoring, bc_score_names, nanmean

def setup_dataset(config_data, r_types_sel):
    # set up dataset
    dataset = Dataset("datasets/contacts_rr4A_64nn.h5")

    # selected structures
    sids_sel = np.genfromtxt("datasets/subunits_validation_highres_set.txt", dtype=np.dtype('U'))

    # filter dataset
    m = select_by_sid(dataset, sids_sel) # select by sids
    
    # data selection criteria
    m = select_by_sid(dataset, sids_sel) # select by sids
    m &= select_by_max_ba(dataset, config_data['max_ba'])  # select by max assembly count
    #m &= (dataset.sizes[:,0] <= config_data['max_size']) # select by max size
    m &= (dataset.sizes[:,1] >= config_data['min_num_res'])  # select by min size
    m &= select_by_interface_types(dataset, config_data['l_types'], np.concatenate(r_types_sel))  # select by interface type

    # update dataset selection
    dataset.update_mask(m)

    # set dataset types
    dataset.set_types(config_data['l_types'], config_data['r_types'])

    # debug print
    return dataset


def eval_model(model, dataset, ids):
    p_l, y_l = [], []
    with pt.no_grad():
        for i in tqdm(ids):
            # get data
            X, ids_topk, q, M, y = dataset[i]

            # pack data and setup sink (IMPORTANT)
            X, ids_topk, q, M = collate_batch_features([[X, ids_topk, q, M]])

            # run model
            z = model(X.to(device), ids_topk.to(device), q.to(device), M.float().to(device))

            # prediction
            p = pt.sigmoid(z)

            # categorical predictions
            pc = pt.cat([1.0 - pt.max(p, axis=1)[0].unsqueeze(1), p], axis=1).cpu()
            yc = pt.cat([1.0 - pt.any(y > 0.5, axis=1).float().unsqueeze(1), y], axis=1).cpu()

            # data
            p_l.append(pc)
            y_l.append(yc)
    
    return p_l, y_l


# model parameters
# R2
#save_path = "model/save/i_v2_4_2021-03-23_11-51"  # 85
#save_path = "model/save/i_v2_7_2021-05-21_17-33"  # 89
# R3
#save_path = "model/save/i_v3_0_2021-05-27_14-27"  # 89
# R4
#save_path = "model/save/i_v4_0_2021-09-07_11-20"  # 89
save_path = "./"  # 91

# select saved model
model_filepath = os.path.join(save_path, 'model_ckpt.pt')
#model_filepath = os.path.join(save_path, 'model.pt')

# add module to path
if save_path not in sys.path:
    sys.path.insert(0, save_path)
    
# load functions
from config import config_model, config_data
from data_handler import Dataset
from model import Model


# define device
device = pt.device("cuda")

# create model
model = Model(config_model)

# reload model
model.load_state_dict(pt.load(model_filepath, map_location=pt.device("cuda")))

# set model to inference
model = model.eval().to(device)


p_l, y_l = [], []
for i in range(len(config_data['r_types'])):
    # debug print
    print(config_data['r_types'][i])

    # load datasets
    dataset = setup_dataset(config_data, [config_data['r_types'][i]])
    print("dataset: {}".format(len(dataset)))

    # parameters
    N = min(len(dataset), 512)

    # run negative examples
    ids = np.arange(len(dataset))
    np.random.shuffle(ids)
    pi_l, yi_l = eval_model(model, dataset, ids[:N])
    
    # store evaluation results
    p_l.append(pi_l)
    y_l.append(yi_l)


# parameters
#class_names = ["protein", "DNA/RNA", "ion", "ligand", "lipid"]
class_names = ["carbohydrates", "cyclodextrins"]

# compute scores per class
scores = []
for i in range(len(y_l)):
    # extract class
    p = pt.cat(p_l[i], axis=0)[:,i+1]
    y = pt.cat(y_l[i], axis=0)[:,i+1]

    # compute scores
    s = bc_scoring(y.unsqueeze(1), p.unsqueeze(1)).squeeze().numpy()
    
    # compute F1 score
    f1 = metrics.f1_score(y.numpy().astype(int), p.numpy().round())
    
    # compute ratio of positives
    r = pt.mean(y)
    
    # store results
    scores.append(np.concatenate([s, [f1, r]]))
    
# pack data
scores = np.stack(scores).T

# make table
df = pd.DataFrame(data=np.round(scores,2), index=bc_score_names+['F1', 'r'], columns=class_names)

# save dataframe
df.to_csv("results/type_interface_search_scores.csv")

# display
#display(df)

# parameters
#class_names = ["protein", "DNA/RNA", "ion", "ligand", "lipid"]
class_names = ["carbohydrates", "cyclodextrins"]

# plot
plt.figure(figsize=(5,4.5))
for i in range(len(y_l)):
    # get labels and predictions for class
    yi = pt.cat(y_l[i], axis=0)[:,i+1]
    pi = pt.cat(p_l[i], axis=0)[:,i+1]

    # compute roc and roc auc
    fpr, tpr, _ = metrics.roc_curve(yi.numpy(), pi.numpy())
    auc = metrics.auc(fpr, tpr)
    
    # update plot
    plt.plot(fpr, tpr, '-', label="{} (auc: {:.2f})".format(class_names[i], auc))
    
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.legend(loc='best')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.tight_layout()
plt.savefig("results/type_interface_search_roc_auc.svg")
plt.savefig("results/type_interface_search_roc_auc.png", dpi=300)
#plt.show()

# parameters
#class_names = ["protein", "DNA/RNA", "ion", "ligand", "lipid"]

# plot
plt.figure(figsize=(5,4.5))
for i in range(len(y_l)):
    # get labels and predictions for class
    yi = pt.cat(y_l[i], axis=0)[:,i+1]
    pi = pt.cat(p_l[i], axis=0)[:,i+1]

    # compute roc and roc auc
    pre, rec, _ = metrics.precision_recall_curve(yi.numpy(), pi.numpy())
    auc = metrics.auc(rec, pre)
    
    # update plot
    plt.plot(rec, pre, '-', label="{} (auc: {:.2f})".format(class_names[i], auc))
    
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.legend(loc='best')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.tight_layout()
plt.savefig("results/type_interface_search_pr_auc.svg")
plt.savefig("results/type_interface_search_pr_auc.png", dpi=300)
#plt.show()

# pack results
P = pt.cat([pt.cat(p, axis=0) for p in p_l], axis=0)
Y = pt.cat([pt.cat(y, axis=0) for y in y_l], axis=0)

# select only residues at interface
m = pt.any(Y[:,1:] > 0.5, axis=1)
Pi = P[m,1:]
Yi = Y[m,1:]

# pick same sampling of all classes
n = Yi.shape[1]
N = int(pt.min(pt.sum(Yi, axis=0)).item())
ids_unif = pt.from_numpy(np.concatenate([np.random.choice(pt.where(Yi[:,i] > 0.5)[0].numpy(), N, replace=False) for i in range(n)]))

# compute scores for each class
scores = []
for i in range(n):
    # extract class
    p = Pi[ids_unif,i]
    y = Yi[ids_unif,i]

    # compute scores
    s = bc_scoring(y.unsqueeze(1), p.unsqueeze(1)).squeeze().numpy()
    
    # compute F1 score
    f1 = metrics.f1_score(y.numpy().astype(int), p.numpy().round())
    
    scores.append(np.concatenate([s, [f1]]))
    
# pack data
scores = np.stack(scores).T

# make table
df = pd.DataFrame(data=np.round(scores,2), index=bc_score_names+['F1'], columns=class_names)

# save dataframe
df.to_csv("results/type_interface_identification_scores.csv")

# display
#display(df)

# pack results
P = pt.cat([pt.cat(p, axis=0) for p in p_l], axis=0)
Y = pt.cat([pt.cat(y, axis=0) for y in y_l], axis=0)

# select only residues at interface
m = pt.any(Y[:,1:] > 0.5, axis=1)
Pi = P[m,1:]
Yi = Y[m,1:]

# pick same sampling of all classes
n = Yi.shape[1]
N = int(pt.min(pt.sum(Yi, axis=0)).item())
ids_l = [pt.from_numpy(np.random.choice(pt.where(Yi[:,i] > 0.5)[0].numpy(), N, replace=False)) for i in range(n)]

# compute scores for each class
C = pt.zeros((n,n))
for i in range(n):
    ids = pt.argmax(Pi[ids_l[i]], axis=1)
    for j,k in zip(ids, ids_l[i]):
        C[i,j] += Pi[k,j].round()
        

# normalize score
H = (C / pt.sum(C, axis=1).unsqueeze(1)).numpy()
#H = (C / pt.sum(C, axis=0).unsqueeze(0)).numpy()

# plot
plt.figure(figsize=(5, 5))
plt.imshow(H, origin='lower', cmap='BuGn', vmin=0.0, vmax=1.0)
plt.colorbar(fraction=0.046, pad=0.04)
plt.xticks(np.arange(n), class_names, rotation=90)
plt.yticks(np.arange(n), class_names)
for i in range(n):
    for j in range(n):
        v = H[i,j]
        if v > 0.1:
            plt.text(j,i,f"{v:.2f}", ha='center', va='center', color=[np.round(v-0.1)]*3)
plt.xlabel('predicted')
plt.ylabel('actual')
plt.tight_layout()
plt.savefig("results/type_interface_identification_most_confident_confusion_matrix.svg")
plt.savefig("results/type_interface_identification_most_confident_confusion_matrix.png", dpi=300)
#plt.show()

# pack results
P = pt.cat([pt.cat(p, axis=0) for p in p_l], axis=0)
Y = pt.cat([pt.cat(y, axis=0) for y in y_l], axis=0)

# select only residues at interface
m = pt.any(Y[:,1:] > 0.5, axis=1)
Pi = P[m,1:]
Yi = Y[m,1:]

# pick same sampling of all classes
n = Yi.shape[1]
N = int(pt.min(pt.sum(Yi, axis=0)).item())
ids_l = [pt.from_numpy(np.random.choice(pt.where(Yi[:,i] > 0.5)[0].numpy(), N, replace=False)) for i in range(n)]

# compute scores for each class
C = pt.zeros((n,n))
for i in range(n):
    ids = pt.argmax(Pi[ids_l[i]], axis=1)
    for j in range(n):
        for k in ids_l[i]:
            C[i,j] += Pi[k,j].round()
        

# normalize score
H = (C / pt.sum(C, axis=1).unsqueeze(1)).numpy()
#H = (C / pt.sum(C, axis=0).unsqueeze(0)).numpy()

# plot
plt.figure(figsize=(5, 5))
plt.imshow(H, origin='lower', cmap='BuGn', vmin=0.0, vmax=1.0)
plt.colorbar(fraction=0.046, pad=0.04)
plt.xticks(np.arange(n), class_names, rotation=90)
plt.yticks(np.arange(n), class_names)
for i in range(n):
    for j in range(n):
        v = H[i,j]
        if v > 0.1:
            plt.text(j,i,f"{v:.2f}", ha='center', va='center', color=[np.round(v-0.1)]*3)
plt.xlabel('predicted')
plt.ylabel('actual')
plt.tight_layout()
plt.savefig("results/type_interface_identification_most_confident_confusion_matrix.svg")
plt.savefig("results/type_interface_identification_most_confident_confusion_matrix.png", dpi=300)
#plt.show()


