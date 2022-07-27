###########################################
# Auxiliary functions for the MBS analysis#
import pandas as pd
import tqdm
import glob2
import numpy as np

def readEBAS(path=None):

    table = pd.read_table(path)
        
    k=2
    
    for s in table.iloc[:,0]:
        
        if '1 1' == s:
            
            p = k

        if 'starttime' in s:
                
            break
            
        k=k+1

    reference = pd.to_datetime(table.iloc[p-1][0][:10])
    
    # Fix duplicate names

    names = table.iloc[k-2][0].split(' ')
    
    b = [x for x in names if names.count(x) > 1]
        
    i = 0
    for nb in list(set(b)):
        for l,name in enumerate(names):
            if nb == name:
                if i >= 1:
                    names[l] = name+'_'
                    i = 0
                    continue
                i=i+1
    #
    
    
    
    df = pd.read_fwf(path,infer_nrows=300,skiprows=k,names=names)
    
    df = df.astype('float')
    
    df.starttime = pd.to_datetime(reference)+pd.to_timedelta(df.starttime.astype('str')+'D')
    
    df.endtime = pd.to_datetime(reference)+pd.to_timedelta(df.endtime.astype('str')+'D')
    
    df.index = df.starttime+(df.starttime-df.endtime)/2
    
    return df

def readMAAP(path=None):
    header = 'EPOCH,DOY,F1_A31,BaR_A31,ZSSAR_A31,P_A31,BacR_A31,Ff_A31,IfR_A31,IpR_A31,IrR_A31,Is1_A31,Is2_A31,L_A31,Pd1_A31,Pd2_A31,Q_A31,Qt_A31,T1_A31,T2_A31,T3_A31,XR_A31,ZIrR_A31,ZPARAMETERS_A31,ZSPOT_A31'
    
    header = header.split(',')
    
    maap_list = glob2.glob(path+'*')
    
    maap = pd.DataFrame()
    
    for item in maap_list:
        
        maap_ = pd.read_csv(item,names=header,skiprows=7)
        
        maap = maap.append(maap_)
        
    maap = maap.iloc[:,-4]
    
    maap.index = pd.to_datetime(maap.index)
    
    return maap
