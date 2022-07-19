import sys
import os
from typing import List, Tuple
#sys.path.append('C:\\Users\\GabrielFreitas\\Desktop\\Python\\MULBERRY\\')
sys.path.append('C:\\Users\\GabrielFreitas\\Desktop\\Python\\CloudBerry\\')
import matplotlib as mpl
import matplotlib.pyplot as plt
import mulberry as mb
import auxiliary as aux
import pandas as pd
import numpy as np
import glob2
import cloudberry as cb
import juniperberry as jb
import auxberry as ab
jb.printc('Packages initiliazed')


# mpl.rc('font',family='consolas')
mpl.rc('font',size=14)
jb.printc('Matplotlib common commands given')




############ DIRECTORY GETTER #######################################
jb.printc('Defining file info getter')
def collect_fileinfos(path_directory: str, filesurvey: List[Tuple]):

    content_dir: List[str] = os.listdir(path_directory)

    for filename in content_dir:

        path_file = os.sep.join([path_directory, filename])

        if os.path.isdir(path_file):

            collect_fileinfos(path_file, filesurvey)

        else:

            stats = os.stat(path_file)

            filesurvey.append((path_directory, filename, stats.st_mtime, stats.st_size))

path_dir: str = r'C:\Users\GabrielFreitas\Desktop\PhD\Data'  

jb.printc('Getting folders from computer...')

filesurvey: List[Tuple] = []

collect_fileinfos(path_dir, filesurvey)

direc: pd.DataFrame = pd.DataFrame(filesurvey, columns=('path_directory', 'filename', 'st_mtime', 'st_size'))

path_dir: str = r'C:\Users\GabrielFreitas\Desktop\PhD\Data'   

filesurvey: List[Tuple] = []

collect_fileinfos(path_dir, filesurvey)

direc: pd.DataFrame = pd.DataFrame(filesurvey, columns=('path_directory', 'filename', 'st_mtime', 'st_size'))

path_dir: str = r'D:\Data'

filesurvey: List[Tuple] = []

collect_fileinfos(path_dir, filesurvey)

direc = direc.append(pd.DataFrame(filesurvey, columns=('path_directory', 'filename', 'st_mtime', 'st_size')))

jb.printc('Function search_dir established')

def search_dir(str1=None):
    res = [string for string in direc['path_directory'].unique() if str1 in string]
    return res

jb.printc('Python initiliazed... done!')