import os
import yaml
import pickle
import numpy as np
import pandas as pd

import argparse
from itertools import product

from PU_disc import train_model

parser = argparse.ArgumentParser()
parser.add_argument('-i',"--inputRootDir", type=str, default='/ecoderemdvol/econ_ntuples/qk/', dest="inputRootDir",
                    help="path to root files")

parser.add_argument('-t',"--rootTree", type=str, default='FloatingpointAEttbarDummyHistomaxGenmatchGenclustersntuple', 
                    dest="rootTree", help="AE tree to read HGCAL ECON data from")

#parser.add_argument('--delRThresh', type = float64, default = 0.02, dest = delRThresh,help = 'delta R threshold for adj matrix')

#parser.add_argument('-N', '--num_graphs', default = 12, dest = numGraphs, help = 'number of detector sections')

parser.add_argument('-o',"--outputDir", type=str, default='/ecoderemdvol/GNN/', dest="outputDir",
                    help="path to output dir")

def main(args):
    
    inputRootDir = args.inputRootDir
    rootTree = args.rootTree
    outputDir  = args.outputDir

    del_R_threshs = [0.02,0.03,0.04]
    numGraphss = [12,24]
    num_epochss = [100,500]
    num_hidden_1s = [64, 48, 32, 24, 16]
    num_hidden_2s = [32, 24, 16]
    num_layerss = [2,3]

    
    prod = list(product(del_R_threshs, numGraphss, num_epochss, num_hidden_1s, num_hidden_2s, num_layerss))
    
    results = np.empty([len(prod),8])
    
    for it, (del_R_thresh, numGraphs, num_epochs, num_hidden_1, num_hidden_2, num_layers) in enumerate(prod):
        
        train_acc, test_acc = train_model(inputRootDir, rootTree, del_R_thresh, numGraphs, num_epochs, num_hidden_1, num_hidden_2, num_layers, outputDir)
        
        results[it]=[del_R_thresh, numGraphs, num_epochs, num_hidden_1, num_hidden_2, num_layers, train_acc,test_acc]
    
    df = pd.DataFrame(results,columns = ['del_R_thresh', 'numGraphs', 'num_epochs', 'num_hidden_1', 'num_hidden_2', 'num_layers','train_acc','test_acc'])
        
    df.to_csv(outputDir+'/results.csv')
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)