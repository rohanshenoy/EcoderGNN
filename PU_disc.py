import os
import yaml
import pickle
import numpy as np
import pandas as pd
import itertools
import scipy
import torch
import torch_geometric.datasets as datasets
import torch_geometric.data as Data
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as transforms
import networkx as nx
from torch_geometric.utils.convert import to_networkx
import matplotlib.pyplot as plt
import random

from produce_dataset import *

import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from mpl_toolkits.axes_grid1 import ImageGrid

def build_graph(df_econ, df_gen, phi_min, phi_max, del_R_thresh):

    #only choose wafers in some phi range now
    
    #testing df_econ = df_econ.iloc[:100]
    
    df_econ = df_econ[df_econ.tc_phi>phi_min][df_econ.tc_phi<phi_max];
    
    df_nodes=df_econ[['ECON_0', 'ECON_1', 'ECON_2', 'ECON_3', 'ECON_4', 'ECON_5', 'ECON_6',
           'ECON_7', 'ECON_8', 'ECON_9', 'ECON_10', 'ECON_11', 'ECON_12','ECON_13', 'ECON_14', 'ECON_15', 
            'wafer_energy', 'tc_eta', 'tc_phi']]
    df_nodes.reset_index(inplace=True)
    
    embeddings = (torch.tensor(df_nodes[['ECON_0', 'ECON_1', 'ECON_2', 'ECON_3', 'ECON_4', 'ECON_5', 'ECON_6',
           'ECON_7', 'ECON_8', 'ECON_9', 'ECON_10', 'ECON_11', 'ECON_12','ECON_13', 'ECON_14', 'ECON_15']].values)).to(torch.double)

    eta=df_nodes['tc_eta']
    phi=df_nodes['tc_phi']
    idx = range(len(eta))
    indices_i = np.array([i for i,j in itertools.product(idx,idx)])
    indices_j = np.array([j for i,j in itertools.product(idx,idx)])

    del_R = np.empty([len(eta),len(eta)])
    del_R.shape

    for (i, j) in zip(indices_i,indices_j):
        del_R[i][j]=np.sqrt((eta[i]-eta[j])**2+((phi[i]-phi[j])%(2*np.pi))**2)

    del_R = torch.tensor(del_R)

    adj = np.zeros([len(eta),len(eta)])
    for (i, j) in zip(indices_i,indices_j):
        if del_R[i][j] < del_R_thresh and  del_R[i][j]> 0 :
            adj[i][j]=1.0

    adj=torch.tensor(adj) 

    edge_index = (adj > 0.0).nonzero().t()
    edge_index.shape

    # label PU as 0, electron wafers as 1
    labels = (df_econ['truth_label'])
    features = torch.tensor(labels.values).to(torch.long)

    graph = Data.Data(x=embeddings, edge_index=edge_index, y=features)
    graph.num_classes=2
    return graph,df_nodes

def train_model(ntuple_dir, root_dir, del_R_thresh, numGraphs, num_epochs, num_hidden_1, num_hidden_2, num_layers, outputDir):
    
    output = outputDir + '/del_R_'+ str(del_R_thresh) + '_hid_'+ str(num_hidden_1) +'_'+str(num_hidden_2)+'_'+str(num_layers)+'_'+str(num_epochs)+'/'
    
    if not os.path.exists(output):
        os.makedirs(output)
    
    df_econ = loadEconData(ntuple_dir,root_dir,'no',False)

    df_gen = loadGenData(ntuple_dir,root_dir,'no')
    
    simenergy= df_econ['wafer_energy']
    # label PU as 0, electron wafers as 1
    truth_labels = (simenergy.where(simenergy==0,other=1))
    df_econ['truth_label'] = truth_labels
    
    phi_edges = np.linspace(-np.pi,np.pi,numGraphs+1)

    phi_range = np.empty([len(phi_edges)-1,2])

    for i in range(len(phi_edges)-1):
        phi_range[i] = (phi_edges[i],phi_edges[i+1])
    
    graphs = []
    cut_event_dfs=[]
    
    for (phi_min,phi_max) in phi_range:
    
        # only build graph in 30 degree range covering detector with no overlap
        graph, cut_event_df = build_graph(df_econ,df_gen,phi_min,phi_max, del_R_thresh)
        graphs.append(graph);
        cut_event_dfs.append(cut_event_df)
    
    #randomly train and validate on two different graphs
    ind = random.sample(range(0,len(graphs)),2)
    
    device = torch.device('cuda')
    
    train_data = graphs[ind[0]].to(device)
                        
    val_data = graphs[ind[1]].to(device)
    
    #weight training and validation accordingly
   
    class GCN_2(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(train_data.num_features, num_hidden_1)
            self.conv2 = GCNConv(num_hidden_1, train_data.num_classes)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index

            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)

            return F.log_softmax(x, dim=1)
        
    class GCN_3(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(train_data.num_features, num_hidden_1)
            self.conv2 = GCNConv(num_hidden_1, num_hidden_2)
            self.conv3 = GCNConv(num_hidden_2, train_data.num_classes)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index

            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
            x = self.conv3(x, edge_index)

            return F.log_softmax(x, dim=1)
    
    model = GCN_2().to(device)
    if num_layers == 3:
        model = GCN_3().to(device) 
    model = model.double()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    train_weight  = (len(cut_event_dfs[ind[0]]['wafer_energy']) - (cut_event_dfs[ind[0]]['wafer_energy'] != 0).sum())/(cut_event_dfs[ind[0]]['wafer_energy'] != 0).sum()


    val_weight  = (len(cut_event_dfs[ind[1]]['wafer_energy']) - (cut_event_dfs[ind[1]]['wafer_energy'] != 0).sum())/(cut_event_dfs[ind[1]]['wafer_energy'] != 0).sum()
    

    train_loss_arr = []
    val_loss_arr = []
    model.train()
    for epoch in range(num_epochs):
        
        optimizer.zero_grad()
        train_pred = model(train_data)
        
        train_loss = F.nll_loss(train_pred,train_data.y,weight=torch.tensor(np.asarray([1.0,train_weight]).astype('double')).to(device))
        
        train_loss_arr.append(train_loss)
        
        val_pred = model(val_data)
        val_loss = F.nll_loss(val_pred,val_data.y,weight=torch.tensor(np.asarray([1.0,val_weight]).astype('double')).to(device))
        
        val_loss_arr.append(val_loss)
        
        #loss = F.cross_entropy(torch.sigmoid(out), data.y, weight=torch.tensor(np.asarray([1,2.6]).astype('double')).to(device))
               
        train_loss.backward()
        optimizer.step()
    
    plt.plot(range(num_epochs),train_loss_arr,label = 'train loss')
    plt.plot(range(num_epochs),val_loss_arr, label ='val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(output+'loss.png')
    plt.close()
    
    model.eval()
    train_pred = model(train_data).argmax(dim=1)
    correct = (train_pred == train_data.y).sum()
    train_acc = int(correct)/train_data.y.size(dim=0)
    print(f'Train accuracy: {train_acc:.4f}')
    
    #testing dataset
    
    acc_array=[]
    pred_array=[]
    for graph in graphs:
        val_data=graph.to(device)
        pred = model(val_data).argmax(dim=1)
        pred_array.append(pred)
        correct = (pred == val_data.y).sum()
        acc = int(correct)/val_data.y.size(dim=0)
        acc_array.append(acc)
    
    acc_array = np.asarray(acc_array)
    test_acc_array = np.delete(acc_array,[ind[0],ind[1]])
    
    test_acc = test_acc_array.mean()

    print(f'Accuracy: {np.asarray(acc_array).mean():.4f}')
    
    for i in range(len(graphs)):
        cut_event_dfs[i]['pred_label'] =  pred_array[i].cpu().numpy()

    GNN_PU_removed_event_dfs=[]
    for i in range(len(graphs)):
        app = cut_event_dfs[i]
        GNN_PU_removed_event_dfs.append(app[app.pred_label==1])

    GNN_PU_removed_event=GNN_PU_removed_event_dfs[0]
    for i in range(1,len(graphs)):
        app = GNN_PU_removed_event_dfs[i]
        GNN_PU_removed_event=pd.concat([GNN_PU_removed_event,app])
        
    cols = ['ECON_0', 'ECON_1', 'ECON_2', 'ECON_3', 'ECON_4', 'ECON_5', 'ECON_6',
       'ECON_7', 'ECON_8', 'ECON_9', 'ECON_10', 'ECON_11', 'ECON_12','ECON_13', 'ECON_14', 'ECON_15']

    GNN_PU_removed_event['sum'] = GNN_PU_removed_event[cols].sum(axis=1);
    df_econ['sum'] = df_econ[cols].sum(axis=1);
    
    fig, axes = plt.subplots(1,2,figsize=(20,5))
    plt.subplot(1,2,1)
    h2 = plt.hist2d(x=(GNN_PU_removed_event['tc_eta']),y=GNN_PU_removed_event['tc_phi'],bins=(15,12),weights=GNN_PU_removed_event['sum'],vmin=0,vmax=400000)
    fig.colorbar(h2[3])
    plt.xlabel(r'$\eta$');
    plt.title('GNN pred');
    plt.subplot(1,2,2)
    h3 = plt.hist2d(x=(df_econ['tc_eta']),y=df_econ['tc_phi'],bins=(15,12),weights=df_econ['sum'],vmin=0,vmax=400000)
    fig.colorbar(h3[3])
    plt.xlabel(r'$\eta$');
    plt.title('truth');
    
    plt.savefig(output+'sum_econ.png')
    plt.close()
    
    fig, axes = plt.subplots(1,2,figsize=(20,5))
    plt.subplot(1,2,1)
    h2 = plt.hist2d(x=(GNN_PU_removed_event['tc_eta']),y=GNN_PU_removed_event['tc_phi'],bins=(20,15),weights=GNN_PU_removed_event['pred_label'],vmax=100)
    fig.colorbar(h2[3])
    plt.xlabel(r'$\eta$');
    plt.title('GNN pred');

    plt.subplot(1,2,2)
    h3 = plt.hist2d(x=(df_econ['tc_eta']),y=df_econ['tc_phi'],bins=(20,15),weights=df_econ['truth_label'],vmax=100)
    fig.colorbar(h3[3])
    plt.xlabel(r'$\eta$');
    plt.title('truth');
    
    plt.savefig(output+ 'labels.png')
    plt.close()
    
    
    file = open(output+"/acc.txt", "w+")
 
    # Saving the acc in a text file
    content = str([train_acc,test_acc])
    file.write(content)
    file.close()
    
    return train_acc, test_acc

