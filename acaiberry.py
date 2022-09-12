###########################################
# Auxiliary functions for the MBS analysis#
import pandas as pd
import tqdm
import glob2
import numpy as np
import datetime as dt

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
                    names[l] = name+str(i)#'_'
#                     i = 0
#                     continue
                i=i+1

    
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

def readDMPS(path=None, flist=None, identifier=None):
    ''' Function to read Nestor DMPS data

    Takes either the argument path = path to data folder as a string
    or flist = a list of paths for chosen data files as strings

    Returns a dictionary with fields: 'time', 'Ntot_int', 'Ntot_cpc',
                                      'dNdlogD', and 'diam'
    '''
    # Check if a path is given, else look for file list
    if path is not None:
        flist = glob2.glob(path+'*'+str(identifier)+'*_.sum')
        flist.sort()
    elif flist is not None:
        flist.sort()
    else:
        print('You need to provide a path or a list of files...')
    print(flist)
    # Create lists
    time = []
    Ntot_int = []
    Ntot_cpc = []
    dNdlogD = []
    diam = []

    # # Main loop
    for file in range(len(flist)):
        # Read data file
        df = pd.read_csv(flist[file], sep='\t', header=None)

        # Check for the error with zeros at the end of the diameter array
        if df.iloc[0, -1] == 0:
            print('Excluding file: ', flist[file].split('/')[-1], 'because' +
                  'the last bin diameter is zero')
            # Wrtie excluded file to log
            with open('log.txt', 'a') as f:
                f.write('Excluding file: ' + flist[file].split('/')[-1] +
                        ' because the last bin diameter is zero \n')
            continue

        # Check if the file has too many columns
        if len(df.columns) > 40:
            print('Excluding file: ', flist[file].split('/')[-1], 'because' +
                  'it has more than 40 columns (not standard)')
            # Wrtie excluded file to log
            with open('log.txt', 'a') as f:
                f.write('Excluding file: ' + flist[file].split('/')[-1] + ' because ' +
                  'it has more than 40 columns (not standard)' + '\n')
            continue

        # Print name of data file
        print(flist[file].split('/')[-1])

        # Set the year to 2018
        y = int(flist[file][-18:-14])-1
        year = dt.datetime(y, 12, 31)
        timestamp = [year + dt.timedelta(item) for item in df.iloc[1:, 0]]

        # Append data to lists
        time.append(pd.DataFrame(timestamp))
        Ntot_int.append(pd.DataFrame(df.iloc[1:, 1].values, index=timestamp))
        Ntot_cpc.append(pd.DataFrame(df.iloc[1:, 2].values, index=timestamp))
        dNdlogD.append(pd.DataFrame(df.iloc[1:, 3:].values, index=timestamp,
                                    columns=df.iloc[0, 3:].values))
        diam.append(pd.DataFrame(np.tile(df.iloc[0, 3:].values,
                                         (df.shape[0]-1, 1)), index=timestamp))

    # Concatenate data into dataframes
    time = pd.concat(time)
    Ntot_int = pd.concat(Ntot_int)
    Ntot_cpc = pd.concat(Ntot_cpc)
    dNdlogD = pd.concat(dNdlogD)
    diam = pd.concat(diam)

    # Return data as a dictionary of pandas DataFrame objects
    DMPS = {'time': time.reset_index(drop=True),
            'Ntot_int': Ntot_int,
            'Ntot_cpc': Ntot_cpc,
            'dNdlogD': dNdlogD,
            'diam': diam}
    return DMPS
