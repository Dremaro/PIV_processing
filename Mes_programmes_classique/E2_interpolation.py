'''
Fichier de traitement des CSV donnée en output par le script P1_creation_csv.py
'''
import os
import sys
sys.path.insert(0, 'C:/Users/pc1/Leviia/Documents/1_Savoir et Apprentissage/Programmation/PythonKnowledge/mes_outils')
# sys.path is a list
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from tqdm import tqdm
import itertools

import tkinter as tk
from PIL import Image, ImageTk

from FonctionsPerso import ImageBrowser # type: ignore

################################## FONCTIONS ################################################################################################################################################
#############################################################################################################################################################################################

def compter_voisins_communiquant(data_csv, i, j, l_voisins):
    '''
    Cette fonction recursive permet de compter le nombre de voisins nuls communiquant avec la case (i,j)
    '''
    l_voisins_nul_proches = []
    # conditions are a bit complexe because we need to avoid index on the borders
    if i > 0 and data_csv[i-1,j] == 0:
        l_voisins_nul_proches.append((i-1,j))
    if i < len(data_csv)-1 and data_csv[i+1,j] == 0:
        l_voisins_nul_proches.append((i+1,j))
    if j > 0 and data_csv[i,j-1] == 0:
        l_voisins_nul_proches.append((i,j-1))
    if j < len(data_csv[0])-1 and data_csv[i,j+1] == 0:
        l_voisins_nul_proches.append((i,j+1))

    # on ajoute les voisins nuls proches qui ne sont pas encore dans la liste l_voisins pour ne pas les inspecter plusieurs fois
    l_voisins_nul_non_repertorie = [xy for xy in l_voisins_nul_proches if xy not in l_voisins]
    l_voisins = l_voisins + l_voisins_nul_non_repertorie

    # si on ne repère pas de nouveaux voisins nuls, on stop la branche de récursion actuelle (return), sinon on continue
    if len(l_voisins_nul_non_repertorie) != 0:
        for ij in l_voisins_nul_non_repertorie:
            l_voisins = compter_voisins_communiquant(data_csv, ij[0], ij[1], l_voisins)
    return l_voisins

def reperer_trous(data_csv, critical_size=10):
    '''
    Cette fonction permet de repérer les trous dans les données
    '''
    l_trous = []       # liste des trous qui sont des listes de positions de zéros (liste de liste)
    l_known_zeros = [] # cette liste contient en pernance les zeros contenu dans l_trous (plus les gros trous) mais sous forme de liste simple
    mobile_parts = []  # liste des parties mobiles (trous de taille supérieur à critical_size)

    for i in range(1,len(data_csv)-1):         # parcour des y (attention y pointe vers le bas dans les csv)
        for j in range(1,len(data_csv[0])-1):  # parcour des x
            # si on trouve un zero qui n'est pas encore répertorié, on compte ses voisins et on le répertorie
            if (data_csv[i,j]==0) and ((i,j) not in l_known_zeros):
                l_voisins = compter_voisins_communiquant(data_csv, i, j, [(i,j)]) # on trouve les voisins nuls communiquant avec la case (i,j)
                l_known_zeros = l_known_zeros + l_voisins                         # on ajoute les voisins nuls à la liste des zeros repertoriés
                if len(l_voisins) < critical_size: # si le trou est petit, ce n'est pas une pièce mobile (l'aile)
                    l_trous.append(l_voisins)
                else : # si le trou est grand, c'est une pièce mobile
                    mobile_parts.append(l_voisins)
    return l_trous, mobile_parts

 
def trou_plus_proche(trou, l_trous):
    '''
    Cette fonction permet de calculer la distance entre un pixel (i,j) et le trou le plus proche
    return :
    - les coordonnées du pixel central du trou étudié
    - les coordonnées du pixel de trou le plus proche
    '''
    im,jm = np.mean(trou, axis=0)
    autres_trous = [a_trou for a_trou in l_trous if a_trou != trou]
    coord_zero_plus_proche = None
    for a_trou in autres_trous:
        for ij in a_trou:
            distance = np.sqrt((ij[0]-im)**2 + (ij[1]-jm)**2)
            if distance < distance_min:
                distance_min = distance
                coord_zero_plus_proche = ij
    return (im,jm), coord_zero_plus_proche

def contour_trou(data_csv, trou):
    '''
    Cette fonction permet de calculer le contour d'un trou
    '''
    contour = []
    for i,j in trou:
        if i > 0 and data_csv[i-1,j] != 0:
            contour.append((i-1,j))
        if i < len(data_csv)-1 and data_csv[i+1,j] != 0:
            contour.append((i+1,j))
        if j > 0 and data_csv[i,j-1] != 0:
            contour.append((i,j-1))
        if j < len(data_csv[0])-1 and data_csv[i,j+1] != 0:
            contour.append((i,j+1))
    
    return list(set(contour))



def moyenne_progressive(data_csv, trou):
    '''
    Cette fonction permet de combler un trou par technique de moyenne progressive
    '''
    contour = contour_trou(data_csv, trou)
    filled_pixels = []
    # TODO : à finir
    for i,j in trou:
        value = np.mean([data_csv[i,j] for i,j in contour])
        data_csv[i,j] = value

def combler_zones(data_csv, zones_a_combler):
    '''
    Cette fonction permet de combler une zone vide par la valeur moyenne du contour
    '''
    for zone in zones_a_combler:
        contour = contour_trou(data_csv, zone)
        value = np.mean([data_csv[i,j] for i,j in contour])
        for i,j in zone:
            data_csv[i,j] = value

def apply_zero_as_mask(data_csv, masks):
    '''
    Cette fonction permet d'appliquer un masque sur une image
    '''
    for mask in masks:
        for point in mask:
            data_csv[point[0],point[1]] = 0
    return data_csv




def compute_vorticity(u, v, dx, dy, parties_mobiles):
    parties_mobiles = list(itertools.chain(*parties_mobiles))
    # Initialize the derivatives
    dv_dx = np.zeros_like(v)
    du_dy = np.zeros_like(u)

    # Compute the derivatives only at the points where both neighboring points are not part of the mobile parts
    for i in range(1, u.shape[0]-1):
        for j in range(1, u.shape[1]-1):
            l_voisins = [(i-1, j), (i+1, j), (i, j-1), (i, j+1), (i-1, j-1), (i-1, j+1), (i+1, j-1), (i+1, j+1)]
            if all(voisin not in parties_mobiles for voisin in l_voisins):
                du_dy[i, j] = (u[i+1, j] - u[i-1, j]) / (2*dy)
                dv_dx[i, j] = (v[i, j+1] - v[i, j-1]) / (2*dx)

    # Compute the vorticity
    vorticity = dv_dx - du_dy

    return vorticity

def index_to_coords(mobile_parts, shape_img):
    '''
        -------------------> j
      | [[ .  .  .  .  .  .],
      |  [ .  .  .  .  .  .],
      |  [ .  .  .  .  .  .],
    i |  [ .  .  .  .  .  .],
      |  [ .  .  .  .  .  .],
      v  [ .  .  .  .  .  .]]
    
    We need to turn it into:

       ^  y = max_i - i
       |
       |
       |
       |                 x = j
       |_ _ _ _ _ _ _ _ _ _> 

    on a besoin de shape_img pour correctement renverser les lignes, cf fonction
    '''
    out = []
    max_i, max_j = shape_img
    for part in mobile_parts:
        reverted_part = [(j,max_i-i) for i,j in part]
        out.append(reverted_part)
    return out

################################## MAIN CODE ################################################################################################################################################
#############################################################################################################################################################################################

############### ! Variables Globales et adresse de sauvegarde ####################
ratio_pixel_mm = 7.3/41 # = 0.178 en mm/pixel7
dl = 0.178

# path_csv_rawcsv = r'essais\essai_test\VC7_output_test\\'
# path_save_interpolated_csv = r'essais\essai_test\VC7_test_interpolated_UV_csv\\'
# path_save_absspeed_csv = r'essais\essai_test\VC7_test_absspeed_csv\\'

path_csv_rawcsv = r'essais\essai_65Hz_threshold3100\01_csvimages\\'

path_save_U_interpolated_csv = r'essais\essai_65Hz_threshold3100\02_U_interpolated_csv\\'
path_save_V_interpolated_csv = r'essais\essai_65Hz_threshold3100\02_V_interpolated_csv\\'
path_save_mobile_parts = r'essais\essai_65Hz_threshold3100\02_MP_mobile_parts\\'
path_save_absspeed_csv = r'essais\essai_65Hz_threshold3100\03_absspeed_csv\\'
path_save_vorticity_csv = r'essais\essai_65Hz_threshold3100\04_vorticity_csv\\'


# Listes de stockage des grandeurs d'intérêt
speed_fieldes = []
vorticity_fieldes = []

# Make sure the folder exists and create the list of directories
os.makedirs(path_save_U_interpolated_csv, exist_ok=True)
os.makedirs(path_save_V_interpolated_csv, exist_ok=True)
os.makedirs(path_save_absspeed_csv, exist_ok=True)
os.makedirs(path_save_vorticity_csv, exist_ok=True)
os.makedirs(path_save_mobile_parts, exist_ok=True)
input_csvs = os.listdir(path_csv_rawcsv)



################## ! Boucle de traitement principale ############################
last_u_speed = []  # permet de garder en mémoire un fichier sur l'autre
id=0
for file in tqdm(input_csvs):
    ################# CONVERTIR EN ARRAY ##########
    datafram_csv = pd.read_csv(path_csv_rawcsv + file, sep=',') # lie le fichier csv
    data_csv = datafram_csv.values[1:-1,1:-1]                   # convertit en array et enlève les bords (qui sont nulls ici)

    #region ########## REPERER LES TROUS ##########
    l_trous, mobile_parts = reperer_trous(data_csv, critical_size=15) # repère les trous et les pièces mobiles (gros trous : size > critical_size)
    combler_zones(data_csv, mobile_parts)                             # combler les parties mobiles pour ne pas les interpoler.
    #endregion

    #region ####### COMBLER LES TROUS PAR INTERPOLATION ########## On utilise la fonction griddata de scipy
    non_zero_coords = np.argwhere(data_csv != 0)                               # on récupère les coordonnées des points non nuls
    zero_coords = np.argwhere(data_csv == 0)                                   # et les coordonnées des points nuls
    non_zero_values = data_csv[non_zero_coords[:, 0], non_zero_coords[:, 1]]   # Get the values of the non-zero points
                                                                               # Use griddata to interpolate the zero values
    interpolated_values = sp.interpolate.griddata(non_zero_coords, non_zero_values, zero_coords, method='linear') # * Interpolation linéaire
    data_csv[zero_coords[:, 0], zero_coords[:, 1]] = interpolated_values       # Fill the zero values in the original array with the interpolated values
    apply_zero_as_mask(data_csv, mobile_parts)                                 # On remet la partie mobile à zéro
    #endregion

    #region ########## CALCUL de la VITESSE et SAUVEGARDE ##########
    if "u" in file:                                            # si le fichier est un fichier de vitesse u
        last_u_speed = data_csv                                  # on garde en mémoire pour construire le tableau des vitesses
    else:                                                      # si le fichier est un fichier de vitesse v  
        speed = np.stack((last_u_speed, data_csv), axis=-1)      # tableau de vitesse (u,v) (axis=-1 pour ajouter la dimension à la fin)
        U = speed[:,:,0]  # vitesse u
        V = speed[:,:,1]  # vitesse v
        shape_img = U.shape

        # grandeurs d'intérêt :
        abs_speed = np.sqrt(speed[:,:,0]**2 + speed[:,:,1]**2)   # calcul de la norme de la vitesse
        vorticity = compute_vorticity(speed[:,:,0], speed[:,:,1], dl, dl, mobile_parts) # et ici la vorticité
        
        # stock tout ça dans des listes pour post-traitement éventuel dans ce programme
        speed_fieldes.append(speed)
        vorticity_fieldes.append(vorticity)

        l = list(itertools.chain.from_iterable(mobile_parts))

        # sauvegarde au format csv
        # np.savetxt(path_save_absspeed_csv + "AV" + file[1:-6] + ".csv", abs_speed, delimiter=',')
        # np.savetxt(path_save_U_interpolated_csv + "U" + file[1:-6] + ".csv", speed[:,:,0], delimiter=',')
        # np.savetxt(path_save_V_interpolated_csv + "V" + file[1:-6] + ".csv", speed[:,:,1], delimiter=',')
        # np.savetxt(path_save_vorticity_csv + "V" + file[1:-6] + ".csv", vorticity, delimiter=',')
        # mobile_parts = index_to_coords(mobile_parts, shape_img) # inversion des coordonnées que maintenant car on avait encore besoin de l'ordre d'indexation (i,j) pour le mask, maintenant on a besoin de l'ordre de coordonnées pixel (j,i) 
        # np.savetxt(path_save_mobile_parts + "MP" + file[1:-6] + ".csv", list(itertools.chain.from_iterable(mobile_parts)), delimiter=',')
    #endregion




