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
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
