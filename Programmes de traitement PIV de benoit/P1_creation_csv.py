# 15/06/2022
# pour sauvegarder des fichiers Lavision en .csv


#importation des librairies
import lvpyio as lv
from lvpyio import read_set, is_multiset
from lvpyio import read_buffer
from pprint import pprint
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

import os
from scipy import signal
import time
import matplotlib.pylab as pylab
import re
import xlsxwriter
import pandas as pd

start = time.time()
params={'legend.fontsize':'20','figure.figsize':(18,10),'axes.labelsize':'20','axes.titlesize':'20','xtick.labelsize':'20','ytick.labelsize':'20'}
pylab.rcParams.update(params)


#--------------------------------
# DEBUT DES PARAMETRES
Date='240301'
Temps='101626'

# Encore un peu d'amelioration necessaire

#path=r'D:\\sillage_sphere24\plans_YZ\Project_Piv_220510_124821\Recording_Date='+Date+r'_Time='+Temps+r'\\PIV_MPd(3x32x32_50%ov_ImgCorr)_01\\'

#path=r'E:\\sillage_sphere24\plans_YZ\2209_x6\Recording_Date='+Date+r'_Time='+Temps+r'\\PIV_MPd(3x32x32_50%ov_ImgCorr)\\'

#path=r'E:\\sillage_sphere14\plans_YZ\2211x_obj70\Recording_Date='+Date+r'_Time='+Temps+r'\\PIV_MPd(3x32x32_50%ov_ImgCorr)\\'

path=r'E:\\sillage_sphere14\plans_YZ\2024_plansYZ\Recording_Date='+Date+r'_Time='+Temps+r'\\PIV_MPd(3x32x32_50%ov_ImgCorr)\\'

#path=r'E:\\caracterisation_canal\Project_Piv_231205_092745\Recording_Date='+Date+r'_Time='+Temps+r'\\PIV_MPd(3x32x32_50%ov_ImgCorr)\\'

#path=r'E:\\sillage_cube24\plans_YZ\2302_X\Recording_Date='+Date+r'_Time='+Temps+r'\\PIV_MPd(3x32x32_50%ov_ImgCorr)\\'

#path=r'E:\\sillage_cube12\Plans_YZ\Recording_Date='+Date+r'_Time='+Temps+r'\\PIV_MPd(3x32x32_50%ov_ImgCorr)\\'

#path=r'G:\\2022_plansYZ\\2205_x6\\Recording_Date='+Date+r'_Time='+Temps+r'\\PIV_MPd(3x32x32_50%ov_ImgCorr)\\'

#path=r'E:\\flaps\essai\Recording_Date='+Date+r'_Time='+Temps+r'\\PIV_MPd(3x32x32_50%ov_ImgCorr)\\'


# Chemin du dossier pour l'enregistrement

#pathsave=r'D:\\sillage_sphere24\plans_YZ\Project_Piv_220510_124821\Recording_Date='+Date+r'_Time='+Temps+r'\\'+Date+r'_Time='+Temps+r'_csv\\'

pathsave=r'E:\\sillage_sphere14\plans_YZ\2024_plansYZ\Recording_Date='+Date+r'_Time='+Temps+r'\\'+Date+r'_Time='+Temps+r'_csv\\'

#pathsave=r'E:\\flaps\essai\Recording_Date='+Date+r'_Time='+Temps+r'\\'+Date+r'_Time='+Temps+r'_csv\\'

#pathsave=r'E:\\caracterisation_canal\Project_Piv_231205_092745\Recording_Date='+Date+r'_Time='+Temps+r'\\'+Date+r'_Time='+Temps+r'_csv\\'

#pathsave=r'E:\\sillage_cube24\plans_YZ\2302_X\Recording_Date='+Date+r'_Time='+Temps+r'\\'+Date+r'_Time='+Temps+r'_csv\\'

#pathsave=r'E:\\sillage_cube12\Plans_YZ\Recording_Date='+Date+r'_Time='+Temps+r'\\'+Date+r'_Time='+Temps+r'_csv\\'

#pathsave=r'G:\\2022_plansYZ\\2205_x6\\Recording_Date='+Date+'_Time='+Temps+'\\'+Date+'_Time='+Temps+'_csv\\'

# selection de la region d'interet
xx0=20
xx1=69
yy0=10
yy1=61

fs=10 # frequence en Hertz

#Nimage=4200 # Nombre d'images a traiter
Nimage=4200 # Nombre d'images a traiter

# FIN DES PARAMETRES
#--------------------------------

# pour avoir le bon nombre de 0 dans le nom du fichier
def filename(i):  
    if i<10:
        return('B0000'+str(i))
    if i>=10 and i<100:
        return('B000'+str(i))
    if i>=100 and i<1000:
        return('B00'+str(i))
    if i>=1000 and i<10000:
        return('B0'+str(i))
    
# Creation du dossier pour l'enregistrement s'il n'existe pas encore    
if not os.path.exists(pathsave):
    os.makedirs(pathsave)
    
print('Dimensions du csv : '+str(yy1-yy0)+','+str(xx1-xx0))

# ------------------
# Un exemple d'image, pour verification 
filename0='B00007'
buffer = read_buffer(path+filename0+'.vc7')

ma_arr = buffer.as_masked_array()
ma_arr_x=ma_arr["v"]
ma_arr_y=ma_arr["u"]
ma_arr_x1=ma_arr_x[yy0:yy1,xx0:xx1]
ma_arr_y1=ma_arr_y[yy0:yy1,xx0:xx1]

# figures dans un cas pour verification
fig = plt.figure(figsize=(17,6))
plt.pcolor(ma_arr_x1,cmap='RdYlBu')
plt.title('u')
plt.colorbar()
plt.show()

fig = plt.figure(figsize=(17,6))
plt.pcolor(ma_arr_y1,cmap='RdYlBu')
plt.title('v')
plt.colorbar()
plt.show()

# ------------------
#Sauvegarde des .csv

# boucle sur les images
for ii in range(1,Nimage+1):
    
    # Affichage regulier de la progression dans la console
    if round(ii/500)==ii/500:
        print(ii)
        
    # Lecture des fichiers Lavision    
    buffer = read_buffer(path+filename(ii)+'.vc7')
    ma_arr = buffer.as_masked_array()
    ma_arr_x=ma_arr["v"]
    ma_arr_y=ma_arr["u"]
    ma_arr_x1=ma_arr_x[yy0:yy1,xx0:xx1]
    ma_arr_y1=ma_arr_y[yy0:yy1,xx0:xx1]

    # Sauvegarde des fichier .csv
    np.savetxt(pathsave+filename(ii)+"_u.csv",ma_arr_x1, delimiter=",")
    np.savetxt(pathsave+filename(ii)+"_v.csv",ma_arr_y1, delimiter=",")
    
    
end = time.time()
print("temps passé: ", end - start)