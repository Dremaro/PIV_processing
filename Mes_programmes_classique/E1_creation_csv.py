
#importation des librairies
import sys
sys.path.insert(0, 'C:/Users/pc1/Leviia/Documents/1_Savoir et Apprentissage/Programmation/PythonKnowledge/mes_outils')

import lvpyio as lv                       # type: ignore
from lvpyio import read_set, is_multiset  # type: ignore
from lvpyio import read_buffer            # type: ignore
from pprint import pprint
from tqdm import tqdm
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

import FonctionsPerso as Fp  # type: ignore


# params={'legend.fontsize':'20','figure.figsize':(18,10),'axes.labelsize':'20','axes.titlesize':'20','xtick.labelsize':'20','ytick.labelsize':'20'}
# pylab.rcParams.update(params)

##################################################################################################################################################
#!#############################################     FONCTIONS       ##############################################################################
##################################################################################################################################################

def filename(i):
    """pour avoir le bon nombre de 0 dans le nom du fichier

    Args:
        i (int): numéro du fichier dont on veut le nom avec le bon nombre de 0
    """
    if i<10:
        return('B0000'+str(i))
    if i>=10 and i<100:
        return('B000'+str(i))
    if i>=100 and i<1000:
        return('B00'+str(i))
    if i>=1000 and i<10000:
        return('B0'+str(i))



##################################################################################################################################################
#!#############################################      PARAMETRES       ############################################################################
##################################################################################################################################################

# path = r'C:\Users\pc1\Leviia\Documents\2_ENS-Paris-Saclay\3A__M2_BME2_IPP\Stage_PMMH\PIV_processing\test\\'
# pathsave = r'C:\Users\pc1\Leviia\Documents\2_ENS-Paris-Saclay\3A__M2_BME2_IPP\Stage_PMMH\PIV_processing\1_test\\'

# path = r'C:\Users\pc1\Leviia\Documents\2_ENS-Paris-Saclay\3A__M2_BME2_IPP\Stage_PMMH\PIV_processing\essais\essai_test\VC7_data_test\\'
# pathsave = r'C:\Users\pc1\Leviia\Documents\2_ENS-Paris-Saclay\3A__M2_BME2_IPP\Stage_PMMH\PIV_processing\essais\essai_test\VC7_output_test\\'

path = r'C:\Users\pc1\Leviia\Documents\2_ENS-Paris-Saclay\3A__M2_BME2_IPP\Stage_PMMH\PIV_processing\essais\essai_65Hz_threshold3100\00_rawimages\\'
pathsave = r'C:\Users\pc1\Leviia\Documents\2_ENS-Paris-Saclay\3A__M2_BME2_IPP\Stage_PMMH\PIV_processing\essais\essai_65Hz_threshold3100\01_csvimages\\'

os.makedirs(pathsave, exist_ok=True)
images_paths = os.listdir(path)

pourcent = 0.26
Nimage=int((len(images_paths)-1)*pourcent)       # Nombre d'images a traiter
print('Nombre d\'images à traiter : '+str(Nimage)+'/'+str(len(images_paths)-1))
input("Appuyez sur une touche pour continuer, ctrl+C pour arrêter")



##################################################################################################################################################
#!######################################################      MAIN      ##########################################################################
##################################################################################################################################################
#region : affichage d'un exemple... on s'en fiche un peu
# print("Affichage d'un EXEMPLE d'image pour vérification")
# # ------------------
# # Un exemple d'image, pour verification 
# filename0='B00001'
# buffer = lv.read_buffer(path+filename0+'.vc7')
# ma_arr = buffer.as_masked_array()
# ma_arr_x=ma_arr["u"]
# ma_arr_y=ma_arr["v"]

# x,y = np.meshgrid(np.arange(ma_arr_x.shape[1]),np.arange(ma_arr_x.shape[0]))
# plt.quiver(x,y,ma_arr_x,ma_arr_y)
# plt.show()

# fig = plt.figure(figsize=(8,6))
# plt.pcolor(ma_arr_x,cmap='RdYlBu')
# plt.title('u')
# plt.colorbar()
# plt.show()

# fig = plt.figure(figsize=(8,6))
# plt.pcolor(ma_arr_y,cmap='RdYlBu')
# plt.title('v')
# plt.colorbar()
# plt.show()
#endregion

# ------------------
#Sauvegarde des .csv
input("Procéder à la sauvegarde des fichiers .csv ?      ctrl+C pour arrêter")
# boucle sur les images
for ii in tqdm(range(1,Nimage+1)):
    
    # Lecture des fichiers Lavision    
    buffer = lv.read_buffer(path+filename(ii)+'.vc7')
    ma_arr = buffer.as_masked_array()
    ma_arr_x=ma_arr["u"] # ! Attention il y a une inversion entre u et v, à voir si c'est exprès ou une erreur
    ma_arr_y=ma_arr["v"] # ! Je pense que c'était une erreur, j'ai inversé u et v
    x,y = np.meshgrid(np.arange(ma_arr_x.shape[1]),np.arange(ma_arr_x.shape[0]))
    plt.quiver(x,y,ma_arr_x,ma_arr_y)
    plt.show()

    # Sauvegarde des fichier .csv
    np.savetxt(pathsave+filename(ii)+"_u.csv",ma_arr_x, delimiter=",")
    np.savetxt(pathsave+filename(ii)+"_v.csv",ma_arr_y, delimiter=",")
    










