
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
import cv2

import tkinter as tk
from PIL import Image, ImageTk

from FonctionsPerso import ImageBrowser # type: ignore





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

def trouver_trous(image, critical_size=10):
    """Autre fonction pour trouver les trous dans une image mais qui utilise OpenCV.
    Fonctionne aussi mais pas spécialement plus rapide que la fonction reperer_trous héhéhé.

    Args:
        image (np.array): l'image à trous
        critical_size (int, optional): threshold pour considérer le trou comme une partie mobile. Defaults to 10.

    Returns:
        coordonnées des trous, coordonnées des parties mobiles 
    """
    # Convert the image to grayscale and uint8 if it's not already
    image = np.uint8(image)
    _, thresh = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY_INV) # Threshold the image, let's assume the holes are black
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours in the thresholded image
        
    # Get the coordinates of the pixels in each hole
    holes_coords = []
    for hole in contours:
        # Create an empty image to draw the contour on
        image = np.zeros_like(thresh)
        # Draw the contour
        cv2.drawContours(image, [hole], -1, (255), thickness=cv2.FILLED)
        # Find the coordinates of the pixels in the contour
        coords = np.where(image == 255)
        # Convert the coordinates to a list of [x, y] pairs
        coords = list(zip(coords[1], coords[0]))
        holes_coords.append(coords)
    
    mobile_parts = [hole for hole in holes_coords if len(hole) > critical_size]
    
    return holes_coords, mobile_parts

    # holes_coords = [cv2.boundingRect(hole) for hole in holes]
    # Print the coordinates of the holes
    # for hole in holes_coords:
    #     x, y, w, h = hole
    #     print(f'Hole at ({x}, {y}), width: {w}, height: {h}')




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

def normalize_to_255(array):
    array_min, array_max = np.min(array), np.max(array)
    normalized_array = ((array - array_min) / (array_max - array_min)) * 255
    return normalized_array.astype(np.uint8)







if __name__ == "__main__":
    print("Les fonction se chargent bien")

