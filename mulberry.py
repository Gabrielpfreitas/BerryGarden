#__________________________________________________________________________________________________________________________________________#
# MULTIPARAMETER BIOAEROSOL SPECTROMETER PYTHON FUNCTIONS
# author: Gabriel Pereira Freitas
# contact: gabriel.freitas@aces.su.se

import numpy as np
import pandas as pd
import juniperberry as jb
import glob2
import tqdm
import matplotlib.pyplot as plt
from time import sleep
from IPython.display import clear_output
from sklearn.cluster import KMeans

# PRE PROCESS MBS # 

def processMBS(path_raw=None, path_hdf=None, min_col = True, use_lut=None, dum_row = 33, size_T = 0.8):
    
    clear_output()

    if path_raw == None:
        raise ValueError('Raw files path not given, use path_raw = ')
        
    if path_hdf == None:
        print('Output not given, H5 files will be written in raw folder')

    if use_lut == None:
        print('No external LUT path was given, using embedded LUT')
    else:
        print('Embedded LUT overriden, using external LUT')
    if dum_row == 33:
        print('Dum row has not been overriden, using 33')
        
    if size_T == 0.8:
        print('Selecting particles above 0.8 µm')
    else:
        print('Selecting particles above '+str(size_T)+' µm')
        print('WARNING! Size threshold not ideal')
    
    print('Looking for files in '+path_raw)

    files = glob2.glob(path_raw+'*.csv')

    if len(files) == 0:
        raise ValueError('No files found, check raw path and identifiers')
    else:
        print(str(len(files)) +' MBS files found')

    print('Reading days of measurement...')
    days = [file[-19:-11] for file in files]

    days = list(dict.fromkeys(days))

    print(str(len(days))+' day(s) of measurement found within this folder')

    print('Starting processing in ')
    print('3')
    sleep(1)
    print('2')
    sleep(1)
    print('1')
    sleep(1)
    clear_output()

    d = 0
    for day in days:
        
        jb.printc('Working on day: '+day)
        files = glob2.glob(path_raw+'*'+day+'*.csv')

        df = pd.DataFrame()
        ft = pd.DataFrame()

        for i in tqdm.tqdm(range(len(files))):
            if use_lut == None:
                if min_col == False:
                    df1 = pd.read_csv(files[i], skiprows=dum_row, usecols = ['Time(ms)', 'FT', 'Size', 'Total', 'Measured', 'AsymLR%', \
                                                                    'AsymLRinv%', 'MeanL','MeanR','KurtosisL','KurtosisR', \
                                                                    'PeakMeanL','PeakMeanR','SumL','SumR','VarianceL', \
                                                                    'VarianceR','MirrorL%','MirrorR%',\
                                                                    'SkewL', 'SkewR', 'XE1_1', 'XE1_2', 'XE1_3', 'XE1_4', \
                                                                    'XE1_5', 'XE1_6', 'XE1_7', 'XE1_8'])
                else:
                    df1 = pd.read_csv(files[i], skiprows=dum_row, usecols = ['Time(ms)', 'FT', 'Size','Total', 'Measured', 'AsymLR%', \
                                                                    'MeanR','PeakMeanR', 'XE1_1', 'XE1_2', 'XE1_3', 'XE1_4', \
                                                                    'XE1_5', 'XE1_6', 'XE1_7', 'XE1_8'])
            else:
                if min_col == False:
                    df1 = pd.read_csv(files[i], skiprows=dum_row, usecols = ['Time(ms)', 'FT', 'LASER', 'Total', 'Measured', 'AsymLR%', \
                                                                    'AsymLRinv%', 'MeanL','MeanR','KurtosisL','KurtosisR', \
                                                                    'PeakMeanL','PeakMeanR','SumL','SumR','VarianceL', \
                                                                    'VarianceR','MirrorL%','MirrorR%',\
                                                                    'SkewL', 'SkewR', 'XE1_1', 'XE1_2', 'XE1_3', 'XE1_4', \
                                                                    'XE1_5', 'XE1_6', 'XE1_7', 'XE1_8'])
                else:
                    df1 = pd.read_csv(files[i], skiprows=dum_row, usecols = ['Time(ms)', 'FT', 'LASER', 'Total', 'Measured', 'AsymLR%', \
                                                                    'MeanR','PeakMeanR', 'XE1_1', 'XE1_2', 'XE1_3', 'XE1_4', \
                                                                    'XE1_5', 'XE1_6', 'XE1_7', 'XE1_8'])
            fp = open(files[i]) 
            for i, line in enumerate(fp):
                if i == dum_row-1:
                    starttime = pd.to_datetime(line[17:],format='%d/%m/%Y %H:%M:%S\n') 
                elif i > dum_row-1:
                    break                        
            fp.close()

            df1['time'] = starttime + pd.to_timedelta(df1['Time(ms)'], unit='ms')
            
            df1.drop(['Time(ms)'], axis=1, inplace=True) #
            
            df1.set_index('time', inplace=True)
            
            df1 = df1[df1.FT == 0]
        
            df1 = df1.drop('FT',axis=1) 
            
            if use_lut != None:
            
                lut = pd.read_csv(use_lut,sep='\t')

                lut['Size (um)'] = lut['Size (um)'].round(3)

                df1['Size'] = df1['LASER'].map(lut['Size (um)'])

                df1 = df1.drop(df1['LASER'],axis=1)
            
            df1 = df1[df1.Size >= size_T]
            
            df = df.append(df1) 
      
        df['count'] = 1
        
        jb.printc('Saving file')
        if path_hdf == None:
            df.to_hdf(path_raw+'MBS_'+day+'.h5','df_key', format='t', data_columns=True)
        else:
            df.to_hdf(path_hdf+'MBS_'+day+'.h5','df_key', format='t', data_columns=True)
        jb.printc('Finished working on this day, moving on...')
        clear_output()
    print('Pre-processing completed.')
        
    
    
def readMBS(path=None,fluo=None,header=None,date_sel=None):
    if path == None:
        raise ValueError('Please insert path!')
        
    if fluo != None:
        print('Reading fluorescent particles')
        files = glob2.glob(path+'*FL*.h5')
        if date_sel != None:
            files = glob2.glob(path+'*FL*'+date_sel+'*.h5')
    else:
        print('Reading coarse particles')
        files = glob2.glob(path+'*TC*.h5')
        if date_sel != None:
            files = glob2.glob(path+'*TC*'+date_sel+'*.h5')
        
    clear_output()    
    
    df = pd.DataFrame()
    
    for file in tqdm.tqdm(files):
        if header == None:
            df1 = pd.read_hdf(file)
        else:
            df1 = pd.read_hdf(file,header=header)
        df = df.append(df1)
    
    df = df.sort_index()
    
    df.index = pd.to_datetime(df.index)
    
    df = df[~df.index.duplicated()]
    
    clear_output()
    
    print('Done!')
    return df
    
def kmeansMBS(df=None,n=None,cv=None,rs=0):
    
    if n == None:
        raise ValueError('Choose a number of clusters!')
        
    if rs != None:
        print('Using random state = '+str(rs))
        
    if cv == None:
        print('No cluster variables were given, clustering using all variables...')
    
    df = df.astype('float')
    
    df = df.dropna()
    
    df = df[~df.index.duplicated()]   
    
    if cv != None:
        df_ = df.loc[:,cv]
    else:
        df_ = df
    print('Starting clustering...')
    
    if rs == None:
        kmeans = KMeans(n_clusters=n).fit(df_)
    else:
        kmeans = KMeans(n_clusters=n,random_state=rs).fit(df_)
    
    prediction = kmeans.predict(df_)
    
    df_['prediction']=prediction
    
    if cv != None:
        df = df.drop(cv,axis=1)
        df = df.join(df_)
    else:
        df = df_
        
    
    clear_output()
    
    print('Done!')
    
    return df
    
def getconc(df=None,time=None,flow=None):
    
    #Count the number of particles per time window
    conc = df['count'].resample(time).sum()
   
    #Calculate concentration accounting for sample/sheath flow and amount of time sampled
    conc = conc/((0.315/flow)*(pd.to_timedelta(time).total_seconds()/60))
    
    #Account for loss of particles
    conc = conc*(df['Total'].resample(time).mean()/df['Measured'].resample(time).mean()) 
    
    return conc
    
    

# Plotting functions

def vis_clus(df=None,scale=1.5):
    
    n_c = df.prediction.max()+1
    
    if n_c <= 10:
        cmap = plt.get_cmap('tab10',10)
    else:
        cmap = plt.get_cmap('tab10',20)    
     
    nm_x = [315,364,414,461,508,552,595,640]
    
    total = df['count'].sum()
    
    plt.figure(figsize=(12,5*scale))
   
    for i in range(n_c):
               
        ax = plt.subplot(n_c//3,4,i+1); ax.set_facecolor('ivory')

        cl = df[df.prediction == i]

        cl = cl.loc[:,['XE1_1','XE1_2','XE1_3','XE1_4','XE1_5','XE1_6','XE1_7','XE1_8']]

        plt.plot(nm_x,cl.median(),color=cmap(i))

        plt.plot(nm_x,cl.mean(),'--',color=cmap(i))

        plt.fill_between(nm_x,cl.quantile(q=0.25),cl.quantile(q=0.75),color=cmap(i),alpha=0.5)

        plt.ylim(0,2000)

        plt.title('Cluster '+str(i)+': '+str((len(cl.index)/total)*1e2)[:4]+' %',color=cmap(i),fontsize=14)

        plt.ylabel('FL signal (a.u.)')

        plt.xlabel('Wavelength (nm)')
        
    plt.tight_layout()
        
    plt.show()

def distfl_pcolor(host=None,df=None,bins=np.logspace(1.8,3.3)/100,vmin=None,vmax=None,cmap='bone_r',title=None,\
                         pad=0.17,dndlogdscale=None,fltotalmax=5000):
    '''
    Input variables: 
    host: A subplot of a figure, e.g. plt.subplot(2,1,2)
    df: A MBS dataframe containing at least size and fluroescence signals
    bins: predetermined as np.logspace(1.8,3.3)/100, you may override this by filling this variable
    vmin and vmax: left as None, you can overwrite to constrain the colormap
    cmap: change cmap of the colorplot, default is bone_r
    title: overwritting this variable with a string will create a title
    pad: correct colorbar placemente, default is 0.17
    dndlogdscale: place log if dn/dlogd desired in log scale 
    '''   
    #Necessary groups
    nm_x = [315,364,414,461,508,552,595,640]
    xe   = ['XE1_1','XE1_2','XE1_3','XE1_4','XE1_5','XE1_6','XE1_7','XE1_8']
    
    #Perform digitization over provided bins
    pc = pd.DataFrame(index=bins[:-1]+(bins[1:]-bins[:-1])/2,columns=xe)
    for i in range(len(pc.index)):
        if len(df[np.digitize(df.Size,bins) == i].index) > 5:
            pc.iloc[i] = df[np.digitize(df.Size,bins) == i][xe].mean()
    
    
    #Host axis parameters
    host.set_xlim(0.8,15)
    host.set_xscale('log')
    host.set_xlabel('Size (µm)')
    host.set_ylabel('Central wavelength (nm)')
    
    #Perform plotting of colorplot
    cl = host.pcolor(bins[:-1]-(bins[1:]-bins[:-1])/2,pc.columns,pc.T.replace(np.nan,0),\
                     shading='auto',vmin=vmin,vmax=vmax,cmap=cmap)
    host.set_yticks(ticks=xe,minor=False)
    host.set_yticklabels(nm_x)
    plt.colorbar(cl,shrink=1,pad=pad,label='Fluorescent signal (a.u.)')

    #First parasite axis
    par1 = host.twinx()
    a1 = par1.hist(df['Size'],bins=bins); counts, bins, bars = a1;_ = [b.remove() for b in bars]
    par1.step(bins[:-1]+(bins[1:]-bins[:-1])/2,(a1[0]/(np.log10(bins[1:])-np.log10(bins[:-1]))),\
              where='mid',color='mediumvioletred',lw=1)
    par1.set_ylim(1e0,)
    if dndlogdscale != None:
        par1.set_yscale(dndlogdscale)
    par1.set_ylabel('dNd/dlogD',color='mediumvioletred')
    for i in par1.get_yticklabels():
        i.set_color('mediumvioletred')
    par1.spines["right"].set_color('mediumvioletred')
    
    #second parasite axis
    par2 = host.twinx()
    par2.spines['right'].set_position(('outward', 60))
    par2.set_ylabel('Mean FL signal')
    par2.yaxis.label.set_color('forestgreen')
    par2.spines["right"].set_color('forestgreen')
    for i in par2.get_yticklabels():
        i.set_color('forestgreen')
    par2.plot(bins[:-1]-(bins[1:]-bins[:-1])/2,pc.sum(axis=1).replace(0,np.nan),'o',color='forestgreen')
    par2.set_ylim(0,fltotalmax)

    plt.title(title)

def distmo_pcolor(host=None,df=None,bins=np.logspace(1.8,3.3)/100,vmin=None,vmax=None,cmap='bone_r',title=None,\
                         pad=0.17,dndlogdscale=None):
    '''
    Input variables: 
    host: A subplot of a figure, e.g. plt.subplot(2,1,2)
    df: A MBS dataframe containing at least size and fluroescence signals
    bins: predetermined as np.logspace(1.8,3.3)/100, you may override this by filling this variable
    vmin and vmax: left as None, you can overwrite to constrain the colormap
    cmap: change cmap of the colorplot, default is bone_r
    title: overwritting this variable with a string will create a title
    pad: correct colorbar placemente, default is 0.17
    dndlogdscale: place log if dn/dlogd desired in log scale 
    '''   
    #Necessary groups
    nm_x = [315,364,414,461,508,552,595,640]
    xe   = ['XE1_1','XE1_2','XE1_3','XE1_4','XE1_5','XE1_6','XE1_7','XE1_8']
    
    #Perform digitization over provided bins
    pc = pd.DataFrame(index=bins[:-1]+(bins[1:]-bins[:-1])/2,columns=xe)
    for i in range(len(pc.index)):
        if len(df[np.digitize(df.Size,bins) == i].index) > 5:
            pc.iloc[i] = df[np.digitize(df.Size,bins) == i][xe].mean()
    
    
    #Host axis parameters
    host.set_xlim(0.8,15)
    host.set_xscale('log')
    host.set_xlabel('Size (µm)')
    host.set_ylabel('Central wavelength (nm)')
    
    #Perform plotting of colorplot
    cl = host.pcolor(bins[:-1]-(bins[1:]-bins[:-1])/2,pc.columns,pc.T.replace(np.nan,0),\
                     shading='auto',vmin=vmin,vmax=vmax,cmap=cmap)
    host.set_yticks(ticks=xe,minor=False)
    host.set_yticklabels(nm_x)
    plt.colorbar(cl,shrink=1,pad=pad,label='Fluorescent signal (a.u.)')

    #First parasite axis
    par1 = host.twinx()
    a1 = par1.hist(df['Size'],bins=bins); counts, bins, bars = a1;_ = [b.remove() for b in bars]
    par1.step(bins[:-1]+(bins[1:]-bins[:-1])/2,(a1[0]/(np.log10(bins[1:])-np.log10(bins[:-1]))),\
              where='mid',color='mediumvioletred',lw=1)
    par1.set_ylim(1e0,)
    if dndlogdscale != None:
        par1.set_yscale(dndlogdscale)
    par1.set_ylabel('dNd/dlogD',color='mediumvioletred')
    for i in par1.get_yticklabels():
        i.set_color('mediumvioletred')
    par1.spines["right"].set_color('mediumvioletred')
    
    #second parasite axis
    par2 = host.twinx()
    par2.spines['right'].set_position(('outward', 60))
    par2.set_ylabel('Mean FL signal')
    par2.yaxis.label.set_color('forestgreen')
    par2.spines["right"].set_color('forestgreen')
    for i in par2.get_yticklabels():
        i.set_color('forestgreen')
    par2.plot(bins[:-1]-(bins[1:]-bins[:-1])/2,pc.sum(axis=1).replace(0,np.nan),'o',color='forestgreen')
    par2.set_ylim(0,5000)

    plt.title(title)

def hist_time(df=None,freq='1H',param='Size',bins=np.logspace(1.6,3.2)/100,norm=False):

    df_ = pd.DataFrame(index=pd.date_range(df.index[0],df.index[-1],freq=freq),columns=bins[1:]+(bins[1:]-bins[:-1])/2)

    for i in df_.index:

        a0 = np.histogram(df[(df.index > i.to_pydatetime())&\
                                     (df.index < i.to_pydatetime()+pd.to_timedelta(freq))][param],bins=bins)[0]

        a0 = a0/(np.log10(bins[1:])-np.log10(bins[:-1]))
        
        if norm == True:
            
            a0 = a0/a0.sum()

        df_.loc[i] = a0

        clear_output()
        
    return df_

def hist_to_plot(df=None,color=None,bins=np.logspace(1.6,3.2)/100):
    
    a1 = plt.hist(df,bins=bins,color=color)

    x = a1[0]/(np.log10(bins[1:])-np.log10(bins[:-1]))
    
    plt.step(bins[1:]+(bins[1:]-bins[:-1])/2,x/x.sum(),where='mid',color=color)
    
    counts, bins, bars = a1
    
    _ = [b.remove() for b in bars]

        
        
def create_gamma_file(path=None,path_gamma=None):
    
    files = glob2.glob(path+'*.csv')

    drow = 33

    TD = pd.DataFrame(index=np.arange(0,len(files),1),columns=['XE1_'+str(x)+'ME' for x in range(1,9)]+
                    ['XE1_'+str(x)+'SD' for x in range(1,9)]+['XE1_'+str(x)+'3S' for x in range(1,9)]+
                    ['XE1_'+str(x)+'9S' for x in range(1,9)]+['T ME','T SD','T 3S','T 9S'])
    import os
    import time
    for i in np.arange(0,len(TD.index),1).tolist():

        jb.printc('Working on file: '+files[i]+' which is file number '+str(i+1)+' out of '+str(len(files)+1))

        df_ = pd.read_csv(files[i],skiprows=drow,usecols = ['FT']+['XE1_'+str(x) for x in range(1,9)],engine='python')

        df_ = df_[df_.FT == 1]

        for CH in ['XE1_'+str(x) for x in range(1,9)]:
            
            TD.loc[i,CH+'ME'] = df_[CH].mean()
            
            TD.loc[i,CH+'SD'] = df_[CH].std()
            
            TD.loc[i,CH+'3S'] = df_[CH].std()*3

            TD.loc[i,CH+'9S'] = df_[CH].std()*9

            TD.loc[i,'T ME']  = df_.loc[:,['XE1_'+str(x) for x in range(1,9)]].sum(axis=1).mean()

            TD.loc[i,'T SD']  = df_.loc[:,['XE1_'+str(x) for x in range(1,9)]].sum(axis=1).std()   

            TD.loc[i,'T 3S']  = df_.loc[:,['XE1_'+str(x) for x in range(1,9)]].sum(axis=1).std()*3

            TD.loc[i,'T 9S']  = df_.loc[:,['XE1_'+str(x) for x in range(1,9)]].sum(axis=1).std()*9 

        TD.to_csv(path_gamma+'gamma.csv')
        jb.printc('Finishing work on file: '+files[i])
        
        
        
def getparticles(path=None,gamma=None):
    
    cv = ['XE1_'+str(x) for x in range(1,9)]
    
    files = glob2.glob(path+'MBS*')
    
    fl = pd.DataFrame()
    
    tc = pd.DataFrame()
    
    for file in tqdm.tqdm(files):
        
        raw = pd.read_hdf(file)
        
        fl_ = raw[['XE1_'+str(x) for x in range(1,9)]]
        
        fl_[cv] = fl_[cv]-gamma.iloc[:,16:24].mean().values
        
        fl_[fl_ < 0] = np.nan
        
        fl_ = fl_.dropna(axis = 0, how = 'all')
        
        fl_ = fl_.replace(np.nan,0)
        
        fl = fl.append(fl_)
        
        tc = tc.append(raw[['Size','AsymLR%','PeakMeanR','MeanR','Total','Measured']])
        
        del raw
    
    clear_output()
    
    fl['count'] = 1
    
    tc['count'] = 1
 
    df = tc[tc.index.isin(fl.index)]
        
    fl.loc[df.index,'Size'] = df.loc[:,'Size']
        
    fl.loc[df.index,'AsymLR%'] = df.loc[:,'AsymLR%']
        
    fl.loc[df.index,'PeakMeanR'] = df.loc[:,'PeakMeanR']
        
    fl.loc[df.index,'MeanR'] = df.loc[:,'MeanR']
        
    fl.loc[df.index,'Measured'] = df.loc[:,'Measured']
        
    fl.loc[df.index,'Total'] = df.loc[:,'Total']
    
    fl['Group'] = ''
    
    del df
    
    for i,channel in enumerate(cv):
        
        fl.loc[fl[channel] > gamma.iloc[:,24:36].mean().values[i]-gamma.iloc[:,16:24].mean().values[i],'Group']  += 'ABCDEFGH'[i]
    
    clear_output()
    
    return fl, tc
        
        
        
def pbaptree(df=None):
    
    #ALl particles with 9sigma threshold in B
    pbap = df[['B' in Group for Group in df.Group]]

    #All those whose highest signal is B
    pbap = pbap[pbap.loc[:,['XE1_'+str(x) for x in range(1,9)]].idxmax(axis=1) == 'XE1_2']
    
    #
    return pbap
        
        
def create_lut(df=None):
    
    from colorcet.plotting import swatch, swatches, candy_buttons
    
    lut = pd.DataFrame(index=df.Group.unique(),columns=['CMAP N'])

    lut['CMAP N'] = np.arange(0,len(lut.index),1)
    
    cmap = plt.get_cmap('cet_glasbey_hv',len(lut.index))
    
    return {'lut':lut,'cmap':cmap}
        
def class_time(df=None,time=None,lut=None,th=1):
    
    #create bottom vector
    bottom        = df['count'].resample(time).sum()
    
    bottom.loc[:] = 0
    
    #create sum vector
    
    total        = df[df.Group != '']['count'].resample(time).sum()
    
    for i,Group in tqdm.tqdm(enumerate(np.sort(df[df['Group'] != ''].Group.unique()))):
        
        if ((df[df.Group == Group]['count'].resample(time).sum()/total)*1e2).max() > th:
        
            y = (df[df.Group == Group]['count'].resample(time).sum()/total)*1e2

            y = y.replace(np.nan,0)

            plt.bar(x=y.index,\
                    height=y,\
                    bottom=bottom,\
                    color=lut['cmap'](lut['lut'].loc[Group].values[0]),\
                    width=pd.to_timedelta(time),\
                    align='edge')

            bottom    = bottom + y    
    
    plt.xlim(y.index[0],y.index[-1])

    plt.ylim(0,100)

    plt.ylabel('Composition to \n Highly Fluorescent particles (%)')
        
        
def class_size(df=None,lut=None,bins=np.logspace(1.8,3.3)/100,ax=None,th=1000):

    df = df[df.Group != '']
    
    pc = pd.DataFrame(index=bins[:-1]+(bins[1:]-bins[:-1])/2,columns=df.sort_values(by='Group').Group.unique())
    
    for i in tqdm.tqdm(range(len(pc.index))):
    
        for k in pc.columns:
    
            pc.loc[pc.index[i],k] = df[(np.digitize(df.Size,bins) == i)&(df.Group == k)]['count'].sum()
    
    pc_safe = pc.copy()
    
    pc = (pc.T/pc.sum(axis=1)).replace(np.nan,0)
    
    bottom = pc.iloc[0]

    bottom.loc[:] = 0

    if ax != None:
        
        ax.set_facecolor('peachpuff')

    for i,Group in enumerate(pc.index):
        
        if pc_safe[Group].sum() > th:
        
            plt.bar(pc.loc[Group].index,\
                    
                    height=pc.loc[Group].values,\
                    
                    bottom=bottom,width=np.diff(bins),\
                    
                    label=Group,\
                    
                    color=lut['cmap'](lut['lut'].loc[Group].values[0]))  
            
            bottom = bottom + pc.loc[Group].values

    plt.xscale('log')
    
    plt.bar(0,0,label='Others',color='peachpuff')
    
    
    plt.legend(bbox_to_anchor =(1.025, -0.1),ncol=5,title='Classes:')
    
    plt.ylabel('Contribution to signal')
    
    plt.xlabel('Optical diameter (µm)',labelpad=0.5)

    if ax != None:
        
        ax.tick_params('both',which='both',length=5)
    
    plt.xlim(0.95,15)

    plt.ylim(0,1)
    
    if ax != None:
        
        box = ax.get_position()

        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
        
def corr_class(df=None,time=None,th=100):

    df = df[df.Group != '']
    
    corr = pd.DataFrame(index=df['count'].resample(time).sum().index,columns=np.sort(df[df['Group'] != ''].Group.unique()))
    
    for Group in corr.columns:
        
        if df[df.Group == Group]['count'].sum() > th:
            
            corr[Group] = df[df.Group == Group]['count'].resample(time).sum()
        
    corr = corr.corr()
    
    return corr

def corr_plot(corr=None,ax=None):
    
    mesh = plt.pcolormesh(corr.index,corr.columns,corr.T,vmin=-1,vmax=1,cmap='RdYlGn',shading='nearest')


    plt.gca().add_patch(plt.Polygon(\
        [[-0.5,-0.5],[len(corr.columns)-0.5,len(corr.columns)-0.5],[len(corr.columns)-0.5,-0.5]], \
        color='white'))
    k = 0
    for x in corr.columns:
        for y in corr.index[k:]:
               plt.text(x=x,y=y,s=str(corr.loc[x,y])[0:4],fontsize=8,ha='center')
        k = k+1
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    plt.xticks(rotation=90)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        