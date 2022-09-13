###########################################
# Functions           for the CVI analysis#
import pandas as pd
import tqdm
import glob2
import numpy as np

def readCVI(path=None,startdate=None,enddate=None,EF=None,resample=None):
    
    if enddate == None:
        
        enddate = startdate
        
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
            
            df_1 = df.loc[:,['visiblty','humidity']].resample(resample).median()
            
            df_2 = df.loc[:,'cloud'].resample(resample).max()
            
            df = df_1.join(df_2)
            
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

    if len(CE[CE ==  1].index) > len(CE[CE == -1].index):

        CEt = pd.DataFrame(index=np.arange(len(CE[CE ==  1].index)),columns=['S','E'])

        CEt.loc[:len(CE[CE ==  1].index),'S'] = CE[CE ==  1].index

        CEt.loc[:len(CE[CE ==  -1].index)-1,'E'] = CE[CE ==  -1].index

    if len(CE[CE ==  1].index) < len(CE[CE == -1].index):

        CEt = pd.DataFrame(index=np.arange(len(CE[CE ==  -1].index)),columns=['S','E'])

        CEt.loc[1:len(CE[CE ==  1].index),'S'] = CE[CE ==  1].index

        CEt.loc[:len(CE[CE ==  -1].index),'E'] = CE[CE ==  -1].index

    if len(CE[CE ==  1].index) == len(CE[CE == -1].index):

        if CE[CE ==  1].index[0] < CE[CE == -1].index[0]:

            CEt = pd.DataFrame(index=np.arange(len(CE[CE ==  -1].index)),columns=['S','E'])

            CEt.loc[:len(CE[CE ==  1].index),'S'] = CE[CE ==  1].index

            CEt.loc[:len(CE[CE ==  -1].index),'E'] = CE[CE ==  -1].index

        else:

            CEt = pd.DataFrame(index=np.arange(len(CE[CE ==  -1].index)+1),columns=['S','E'])

            CEt.loc[1:len(CE[CE ==  1].index),'S'] = CE[CE ==  1].index

            CEt.loc[:len(CE[CE ==  -1].index)-1,'E'] = CE[CE ==  -1].index    


    CEt['D'] = CEt.E-CEt.S
    
    return gcvi, CEt
