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

def readCVI(path=None,startdate=None,enddate=None,EF=None,resample=None):
    
    dum_row = 7
    
    dum_row2 = 10
    
    repdict = {'On': 1., 'Off': 0.}
        
    gcvi = pd.DataFrame()
    
    if startdate == None:
        
        folder_list = glob2.glob(path+'*')
    
    else:
        
        folder_list = []
        
        times = pd.date_range(startdate,enddate,freq='1d')
       
        for t in times:
            
            folder_list_date = glob2.glob(path+'*'+t.strftime("%y%m%d"))
            
            folder_list = folder_list + folder_list_date
            
    for k,folder in tqdm.tqdm(enumerate(folder_list)):
         
        file_list = glob2.glob(folder+'\\*' + 'GCVI*' + '.dat')
        
        file_list.sort()
        
        df = pd.DataFrame()
        
        for i,file in enumerate(file_list):
            
            df1 = pd.read_csv(file, sep="\s+", skiprows=dum_row, usecols = ['HR:MN:SC', 'visiblty', 'cvi_stat',\
            'compstat', 'vac_stat', 'blwrstat', 'humidity'])
            
            starttime = pd.to_datetime(times[k],format='%d/%m/%Y %H:%M:%S\n') 
            
            df1['time'] = starttime + pd.to_timedelta(df1['HR:MN:SC'])
            
            df1.drop(['HR:MN:SC'], axis=1, inplace=True) 
            
            df1.set_index('time', inplace=True) 
            
            df = df.append(df1)            
            

        df['compstat'] = df['compstat'].map(repdict)
        
        df['vac_stat'] = df['vac_stat'].map(repdict)
        
        df['blwrstat'] = df['blwrstat'].map(repdict)
        
        df['cvi_stat'] = df['cvi_stat'].map(repdict)
        
        df['sum_stat'] = (df.cvi_stat + df.compstat + df.vac_stat + df.blwrstat)
        
        df['cloud']    = np.nan
        
        df.loc[df[df.sum_stat == 4].index,'cloud'] = 1
        
        df.loc[df[(df.sum_stat > 0)&(df.sum_stat < 4)].index,'cloud'] = 0.5
        
        df.loc[df[df.sum_stat == 0].index,'cloud'] = 0
        
        df = df.drop(['cvi_stat', 'compstat','vac_stat', 'blwrstat', 'sum_stat'],axis=1)
        
        if resample != None:
            
            df = df.resample(resample).median()
            
        

        #ENRICHMENT FACTOR 
        
        file_list = glob2.glob(folder+'\\' + 'CVI*' + '.dat')
        
        file_list.sort()
        
        ef = pd.DataFrame()
        
        for i, file in enumerate(file_list):
            
            ef1 = pd.read_csv(file, sep="\s+", skiprows=dum_row2, usecols = ['HR:MN:SC', 'tosmflow','airspeed','cnt_flow'])
            
            starttime = pd.to_datetime(times[k],format='%d/%m/%Y %H:%M:%S\n') 
            
            ef1['time'] = starttime + pd.to_timedelta(ef1['HR:MN:SC'])
            
            ef1.drop(['HR:MN:SC'], axis=1, inplace=True) 
            
            ef1.set_index('time', inplace=True) 
            
            ef = ef.append(ef1)
            
        if resample != None:
            
            ef = ef.resample(resample).median()
        
        ef['EF']=(1.67e-5*ef['airspeed'])/(ef['tosmflow']/(60*1000))
        
        ef = ef.replace([np.inf, -np.inf, np.nan], 0)
            
        df = df.join(ef)  
        
        df = df.drop(['tosmflow','airspeed','cnt_flow'],axis=1)
        
        gcvi = gcvi.append(df)      
    
    cs = gcvi.cloud
    
    cs = cs.replace(0.5,0)
    
    CE = ((cs[1:]-cs[:-1].values)[(cs[1:]-cs[:-1].values)!=0]).dropna()
    
    CEt = pd.DataFrame(columns=['S','E'])
    
    CEt.S = CE[CE == 1].index; CEt.E = CE[CE == -1].index
    
    CEt['D'] = CE[CE == -1].index-CE[CE == 1].index
    
    return gcvi, CEt
