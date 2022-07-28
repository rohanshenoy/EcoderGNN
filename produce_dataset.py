import os
import uproot
import numpy as np
import pandas as pd
import awkward as ak

def loadEconData(inputRoot,
                 rootFileTDirectory='FloatingpointAutoEncoderEMDAEMSEttbarDummyHistomaxGenmatchGenclustersntuple',
                 outputFileName='econ_data.csv',
                threshSimEnergy=True):
    current_dir=os.getcwd()
    mergeTrainingData = pd.DataFrame()
    if os.path.isdir(inputRoot):
        
        for infile in os.listdir(inputRoot):
            if os.path.isdir(inputRoot+infile): continue
            inputRootFile = os.path.join(inputRoot,infile)
            
            ev_dict = uproot.open(inputRootFile)[rootFileTDirectory+'/HGCalTriggerNtuple']
            
            arrays_toread = [
                "econ_index","econ_data",
                "econ_subdet","econ_zside","econ_layer","econ_waferu","econ_waferv","econ_wafertype",
                "tc_simenergy",
                "tc_subdet","tc_zside","tc_layer","tc_waferu","tc_waferv","tc_wafertype",
                "gen_pt","gen_energy","gen_eta","gen_phi",
                "genpart_pt","genpart_energy",
            ]
            events = ev_dict.arrays(arrays_toread)

            econ = ak.zip({
                "index": events['econ_index'],
                "data": events["econ_data"],
                "subdet": events["econ_subdet"],
                "zside": events["econ_zside"],
                "layer": events["econ_layer"],
                "waferu": events["econ_waferu"],
                "waferv": events["econ_waferv"],
            })
            tc = ak.zip({
                "simenergy": events["tc_simenergy"],
                "subdet": events["tc_subdet"],
                "zside": events["tc_zside"],
                "layer": events["tc_layer"],
                "waferu": events["tc_waferu"],
                "waferv": events["tc_waferv"],
            })
            gen = ak.zip({
                "pt": events["gen_pt"],
                "energy": events["gen_energy"],
                "eta": events["gen_eta"],
                "phi": events["gen_phi"],
            })

            # find wafers that we want to save
            # the problem is that the number of wafers from trigger cells: trigger cells/48 
            # is not the same as the number of wafers from econ data: econ_data/16
            df_tc = ak.to_pandas(tc)
            df_econ = ak.to_pandas(econ)
            df_gen = ak.to_pandas(gen)

            df_simtotal = df_tc.groupby(['entry','subdet','zside','layer','waferu','waferv'])["simenergy"].sum()
            df_econ.index.names
            df_econ.reset_index(inplace=True)
            df_econ.set_index(['entry','subdet','zside','layer','waferu','waferv'],inplace=True)
            df_econ['simenergy'] = df_simtotal
            df_econ.drop(columns='subentry',inplace=True)
            
            df_econ_wsimenergy=pd.DataFrame()
            if threshSimEnergy:
                df_econ_wsimenergy = df_econ[df_econ.simenergy > 0]
            else:
                df_econ_wsimenergy = df_econ
                
            df_econ_wsimenergy = df_econ_wsimenergy.rename(columns={"index": "econ_index", "data": "econ_data", "simenergy": "wafer_energy"})
            df_econ_wsimenergy.reset_index(inplace=True)
            df_econ_wsimenergy.set_index(['entry'],inplace=True)
            df=df_econ_wsimenergy
            df['WaferEntryIdx'] = (df.layer*10000 + df.waferu*100 + df.waferv)*df.zside
            dfTrainData = df.pivot_table(index='WaferEntryIdx',columns='econ_index',values='econ_data').fillna(0).astype(int)
            dfTrainData.columns = [f'ECON_{i}' for i in range(16)]

            dfTrainData[['subdet','zside','layer','waferu','waferv','wafer_energy']] = df.groupby(['WaferEntryIdx'])[['subdet','zside','layer','waferu','waferv','wafer_energy']].mean()
            dfTrainData[['subdet','zside','layer','waferu','waferv']] = dfTrainData[['subdet','zside','layer','waferu','waferv']].astype(int)
            
            #Mapping wafer_u,v to physical coordinates
            dfEtaPhi=pd.read_csv('WaferEtaPhiMap.csv')
            dfTrainData=dfTrainData.merge(dfEtaPhi, on=['subdet','layer','waferu','waferv'])
            dfTrainData.reset_index(drop=True,inplace=True)
            mergeTrainingData=pd.concat([mergeTrainingData,dfTrainData])
    #map abs(eta) to physical eta: zside*eta
    
    mergeTrainingData['tc_eta'] = mergeTrainingData['zside']*mergeTrainingData['tc_eta']

    if '.csv' in outputFileName:
        mergeTrainingData.to_csv(outputFileName,index=False)
    if '.pkl' in outputFileName:
        mergeTrainingData.to_pickle(outputFileName)
    if '.h5' in outputFileName:
        mergeTrainingData.to_hdf(outputFileName, key='df', mode='w')

    return mergeTrainingData

def loadGenData(inputRoot,
                 rootFileTDirectory='FloatingpointAutoEncoderEMDAEMSEttbarDummyHistomaxGenmatchGenclustersntuple',
                 outputFileName='gen_data.csv'):
    current_dir=os.getcwd()
    mergeGenData = pd.DataFrame()
    if os.path.isdir(inputRoot):
        
        for infile in os.listdir(inputRoot):
            if os.path.isdir(inputRoot+infile): continue
            inputRootFile = os.path.join(inputRoot,infile)
            
            ev_dict = uproot.open(inputRootFile)[rootFileTDirectory+'/HGCalTriggerNtuple']
            
            arrays_toread = ["gen_pt","gen_energy","gen_eta","gen_phi","genpart_pt","genpart_energy", "genpart_eta","genpart_phi"]
            
            events = ev_dict.arrays(arrays_toread)
            
            gen = ak.zip({
                "pt": events["gen_pt"],
                "energy": events["gen_energy"],
                "eta": events["gen_eta"],
                "phi": events["gen_phi"],
                })

            genpart = ak.zip({
                "part_pt": events["genpart_pt"],
                "part_energy": events["genpart_energy"],
                "part_eta": events["genpart_eta"],
                "part_phi": events["genpart_phi"],
                })

            df_gen=ak.to_pandas(gen)
            df_genpart=ak.to_pandas(genpart)
            df_gen.reset_index(drop=True,inplace=True)
            
            mergeGenData=pd.concat([mergeGenData,df_gen])

    if '.csv' in outputFileName:
        mergeGenData.to_csv(outputFileName,index=False)
    if '.pkl' in outputFileName:
        mergeGenData.to_pickle(outputFileName)
    if '.h5' in outputFileName:
        mergeGenData.to_hdf(outputFileName, key='df', mode='w')

    return mergeGenData
            
            
if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input',dest='inputRoot', default='econ_ntuple.root', help="TPG Ntuple directory/file to process")
    parser.add_argument('-d','--dir',dest='rootFileTDirectory', default='FloatingpointAutoEncoderEMDAEMSEttbarDummyHistomaxGenmatchGenclustersntuple', help="Directory within input root file to find HGCalTriggerNtuple TTree")
    parser.add_argument('-o','--output',dest='outputFileName',default='econ_data.csv',help='Output file name (either a .csv or .pkl file name)')
    parser.add_argument('-t',action='store_true',dest='threshSimEnergy',default=False,help='only choose wafers with simenergy)')

    args = parser.parse_args()

    df_econ = loadEconData(inputRoot = args.inputRoot,
                          rootFileTDirectory = args.rootFileTDirectory,
                          outputFileName = args.outputFileName)
    df_gen = loadGenData(inputRoot = args.inputRoot,
                          rootFileTDirectory = args.rootFileTDirectory,
                          outputFileName = args.outputFileName)
