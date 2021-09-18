#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 14:00:37 2020

@author: tianyu
"""
   
import sys, os
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import argparse
import time
import numpy as np

import scipy.sparse as sp
from scipy.sparse import csr_matrix


from sklearn.preprocessing import label_binarize

import pandas as pd
import sys
sys.path.insert(0, 'lib/')


if torch.cuda.is_available():
    print('cuda available')
    dtypeFloat = torch.cuda.FloatTensor
    dtypeLong = torch.cuda.LongTensor
    torch.cuda.manual_seed(1)
else:
    print('cuda not available')
    dtypeFloat = torch.FloatTensor
    dtypeLong = torch.LongTensor
    torch.manual_seed(1)

from coarsening import coarsen, laplacian
from coarsening import lmax_L
from coarsening import perm_data
from coarsening import rescale_L
from layermodel import *
import utilsdata
from utilsdata import *
from train import *
import warnings
warnings.filterwarnings("ignore")
#
#
# Directories.
parser = argparse.ArgumentParser()
parser.add_argument('--dirData', type=str, default='/Users/tianyu/Desktop/scRNAseq_Benchmark_datasets/Intra-dataset/', help="directory of cell x gene matrix")
parser.add_argument('--dataset', type=str, default='Zhengsorted', help="dataset")
parser.add_argument('--dirAdj', type = str, default = '/Users/tianyu/Desktop/scRNAseq_Benchmark_datasets/Intra-dataset/Zhengsorted/', help = 'directory of adj matrix')
parser.add_argument('--dirLabel', type = str, default = '/Users/tianyu/Desktop/scRNAseq_Benchmark_datasets/Intra-dataset/Zhengsorted/', help = 'directory of adj matrix')
parser.add_argument('--outputDir', type = str, default = 'data/output', help = 'directory to save results')
parser.add_argument('--saveResults', type=int, default = 0, help='whether or not save the results')

parser.add_argument('--normalized_laplacian', type=bool, default = True, help='Graph Laplacian: normalized.')
parser.add_argument('--lr', type=float, default = 0.01, help='learning rate.')
parser.add_argument('--num_gene', type=int, default = 1000, help='# of genes')
parser.add_argument('--epochs', type=int, default = 1, help='# of epoch')
parser.add_argument('--batchsize', type=int, default = 64, help='# of genes')
parser.add_argument('--dropout', type=float, default = 0.2, help='dropout value')
parser.add_argument('--id1', type=str, default = '', help='test in pancreas')
parser.add_argument('--id2', type=str, default = '', help='test in pancreas')

parser.add_argument('--net', type=str, default='String', help="netWork")
parser.add_argument('--dist', type=str, default='', help="dist type")
parser.add_argument('--sampling_rate', type=float, default = 1, help='# sampling rate of cells')

args = parser.parse_args()

t_start = time.process_time()


# Load data


print('load data...')    
adjall, alldata, labels, shuffle_index = utilsdata.load_largesc(path = args.dirData, dirAdj=args.dirAdj, dataset=args.dataset, net='String')

# generate a fixed shuffle index
if shuffle_index is not None:
    shuffle_index = shuffle_index.astype(np.int32)
else:
    shuffle_index = np.random.permutation(alldata.shape[0])
    np.savetxt(args.dirData +'/' + args.dataset +'/shuffle_index_'+args.dataset+'.txt')
    
train_all_data, adj = utilsdata.down_genes(alldata, adjall, args.num_gene)
L = [laplacian(adj, normalized=True)]


#####################################################

##Split the dataset into train, val, test dataset. Use a fixed shuffle index to fix the sample order for comparison.
train_data, val_data, test_data, train_labels, val_labels, test_labels = utilsdata.spilt_dataset(train_all_data, labels, shuffle_index)
args.nclass = len(np.unique(labels))
args.train_size = train_data.shape[0] 

# train_labels = label_binarize(train_labels, classes=[0,1,2,3,4,5,6,7,8,9])
# val_labels = label_binarize(val_labels, classes=[0,1,2,3,4,5,6,7,8,9])
# test_labels = label_binarize(test_labels, classes=[0,1,2,3,4,5,6,7,8,9])

## Use the train_data, val_data, test_data to generate the train, val, test loader
train_loader, val_loader, test_loader = utilsdata.generate_loader(train_data,val_data, test_data, 
                                                        train_labels, val_labels, test_labels, 
                                                        args.batchsize)




##Delete existing network if exists
try:
    del net
    print('Delete existing network\n')
except NameError:
    print('No existing network to delete\n')

# Train model
net, t_total_train = train_model(Graph_GCN, train_loader,val_loader, L, args)

## Val
val_acc,confusionGCN, predictions, preds_labels, t_total_test, preds_probs, n_classes = test_model(net, val_loader, L, args)
print('  accuracy(val) = %.3f %%, time= %.3f' % (val_acc, t_total_test))

# Test
test_acc,confusionGCN, predictions, preds_labels, t_total_test, preds_probs, n_classes = test_model(net, test_loader, L, args)
    
print('  accuracy(test) = %.3f %%, time= %.3f' % (test_acc, t_total_test))
fpr, tpr, roc_auc = calculation(preds_labels, predictions.iloc[:,0], preds_probs, args.dataset)
# calculation(preds_labels, predictions.iloc[:,0], preds_probs)

if args.saveResults:
    testPreds4save = pd.DataFrame(preds_labels,columns=['predLabels'])
    testPreds4save.insert(0, 'trueLabels', list(predictions.iloc[:,0]))
    confusionGCN = pd.DataFrame(confusionGCN)
    
    testPreds4save.to_csv(args.outputDir+'/gcn_test_preds_'+ args.dataset+ str(args.num_gene)+'.csv')
    predictions.to_csv(args.outputDir+'/gcn_testProbs_preds_'+ args.dataset+ str(args.num_gene) +'.csv')
    confusionGCN.to_csv(args.outputDir+'/gcn_confuMat_'+ args.dataset+ str(args.num_gene)+'.csv')    
    np.savetxt(args.outputDir+'/newgcn_train_time_'+args.dataset + str(args.num_gene) +'.txt', [t_total_train])   
    np.savetxt(args.outputDir+'/newgcn_test_time_'+args.dataset + str(args.num_gene) +'.txt', [t_total_test])

import matplotlib.pyplot as plt
import itertools
from matplotlib.font_manager import FontProperties

fontP = FontProperties()
fontP.set_size('x-small')

plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.3f})'
               ''.format(roc_auc["micro"]),
         color='r', linestyle=':', linewidth=2)

colors = itertools.cycle(['aqua', 'fuchsia', 'darkorange', 'indigo', 'yellow', 'darkgreen',
                          'teal', 'lime', 'brown', 'cadetblue', 'darkred', 'slategrey', 'purple', 'olive', 
                          'chocolate', 'crimson', 'lawngreen'])

if (args.dataset == 'Zhengsorted'):
  celltypes = ['CD14+ Monocyte', 'CD19+ B', 'CD34+ (Pos_label)', 'CD4+ T Helper2', 'CD4+/CD25 T Reg',
          'CD4+/CD45RA+/CD25- Naive T', 'CD4+/CD45RO+ Memory', 'CD56+ NK', 'CD8+ Cytotoxic T',
          'CD8+/CD45RA+ Naive Cytotoxic']
elif (args.dataset == 'BaronHuman'):
  celltypes = ['acinar', 'activated_stellate', 'alpha', 'beta', 'delta', 'ductal', 'endothelial', 'epsilon',
               'gamma', 'macrophage', 'mast', 'quiescent_stellate', 'schwann', 't_cell']

for i, color, cell in zip(range(n_classes), colors, celltypes):
    lw = 2
    plt.plot(fpr[i], tpr[i], color=color,
            lw=lw, label='{0} (area {1:0.3f})'
            .format(cell, roc_auc[i]))
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right", prop=fontP)
plt.savefig('roc_auc_{}.png'.format(args.dataset))