import numpy as np
import pandas as pd
import glob2
import tqdm
import matplotlib.pyplot as plt
from time import sleep
from IPython.display import clear_output
from sklearn.cluster import KMeans
import os

def printc(s=None,os_n = 1, skip=False):
        
    if skip == False:
    
        os.write(os_n, (s+'\n').encode())
        
def qfig(x=10,y=5,x1=1,x2=1,n=1):
    
    fig = plt.figure(figsize=(x,y))
    
    pl  = plt.subplot(x1,x2,n)
    
    return fig, pl
    
def month_ticks(date1=None,date2=None,rotation=0):
    
    plt.xticks((pd.date_range(date1,date2,freq='1M')+pd.to_timedelta('1d')),
            (pd.date_range(date1,date2,freq='1M')+pd.to_timedelta('1d')).astype('str').str.slice(5,7)+'/'+\
            (pd.date_range(date1,date2,freq='1M')+pd.to_timedelta('1d')).astype('str').str.slice(2,4),
           rotation=rotation)
    
def sel_rem(to_do='Sel',df=None,date1=None,date2=None):
    
    if to_do == 'Sel':
        
        df = df[(df.index > date1)&(df.index < date2)]
    
    if to_do == 'Rem':
        
        df = df[(df.index < date1)|(df.index > date2)]
    
    return df
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
