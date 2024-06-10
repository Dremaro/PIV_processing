# x (m), y (m), u (m/s), v (m/s), dp/dx (Pa/m), dp/dy (Pa/m), p (Pa), |p| (Pa)
# F = -grad(P)
import os
import sys
sys.path.insert(0, 'C:/Users/pc1/Leviia/Documents/1_Savoir et Apprentissage/Programmation/PythonKnowledge/mes_outils')
# sys.path is a list
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from tqdm import tqdm
import cv2
import itertools

import tkinter as tk
from PIL import Image, ImageTk

from FonctionsPerso import ImageBrowser # type: ignore



#################### FONCTIONS ###############################################################################################################
##############################################################################################################################################


def get_shape(X,Y):
    Lx = len(np.unique(X))
    Ly = len(np.unique(Y))
    return Lx, Ly

def parties_mobiles(X,Y,P,grad_P):
    MP = []
    indexes = []
    for i in range(len(P)):
        if np.isnan(P[i]) or np.isnan(grad_P[0][i]) or np.isnan(grad_P[1][i]):
            MP.append([X[i], Y[i]])
            indexes.append(i)
    return MP, indexes

def put_MP_to_nan(X,Y,P,MP):
    for i in range(len(X)):
        if [X[i], Y[i]] in MP:
            P[i] = np.nan
    return P

def find_anchor(X,Y,MP,mp_index):
    x_min_mp = (min(MP, key=lambda x: x[0]))[0]
    l_x_min_index = []
    for k,el in enumerate(MP):
        i = mp_index[k]
        if el[0] == x_min_mp:
            l_x_min_index.append(i)

    y_max = -100000 ; i_anchor = 0
    for index in l_x_min_index:
        if Y[index] > y_max:
            y_max = Y[index]
            i_anchor = index
    
    return i_anchor

def find_sonde(X,Y,MP,mp_index,position_relative, i_anchor, Lx, Ly):
    x_anchor = X[i_anchor]
    y_anchor = Y[i_anchor]




path = r"essais\essai_88Hz_threshold3100\01_images_dat\\"
l_dir = os.listdir(path)
l_dir = [f for f in l_dir if 'queen2' in f]
pr_sonde1 = [20,4]    # [x,y] relative position to anchor point in pixels
pr_sonde2 = [20,-8]
P_sonde1 = []
P_sonde2 = []
time = []
H_wing = []
l_grad_P = []
l_X = []
l_Y = []

for n,file in enumerate(l_dir):
    data_pressure = pd.read_csv(path + file, sep=',', header=None)
    data_pressure.columns = ['x', 'y', 'u', 'v', 'dpdx', 'dpdy', 'p', 'abs_p']
    # data_pressure = data_pressure.sort_values(by=['y', 'x'])
    # data_pressure = data_pressure.reset_index(drop=True)
    X = data_pressure['x'].values
    Y = data_pressure['y'].values
    Lx,Ly = get_shape(X,Y)
    P = data_pressure['p'].values
    dPdx = data_pressure['dpdx'].values
    dPdy = data_pressure['dpdy'].values
    # grad_P = np.array([[dPdx[i], dPdy[i]] for i in range(len(dPdx))])
    grad_P = [dPdx, dPdy]
    MP, index = parties_mobiles(X,Y,P,grad_P)
    P_naned = put_MP_to_nan(X,Y,P,MP)

    i_anchor = find_anchor(X,Y,MP,index)
    h = Y[i_anchor]
    H_wing.append(h)
    i_sonde1 = i_anchor + pr_sonde1[0]*Ly - pr_sonde1[1] # quand on augmente l'indice dans le tableau, on diminue y, quand on ajoute Ly, on augmente x de 1.
    i_sonde2 = i_anchor + pr_sonde2[0]*Ly - pr_sonde2[1]
    P_sonde1.append(P_naned[i_sonde1])
    P_sonde2.append(P_naned[i_sonde2])
    l_grad_P.append(grad_P)

    
    time.append(n*1.81e-4)
    if n == 0:
        l_X = X
        l_Y = Y
        plt.scatter(X, Y, c=P, cmap='viridis') # viridis, plasma, inferno, magma 
        plt.scatter([mp[0] for mp in MP], [mp[1] for mp in MP], c='red')
        plt.scatter(X[i_anchor], Y[i_anchor], c='black')
        plt.scatter(X[i_sonde1], Y[i_sonde1], c='blue')
        plt.colorbar()
        plt.show()



############################## Calcul de la force de poussée local et moyenne ########################################################################

# l_F = -l_grad_P
Fx = 0 ; Fy = 0
for i,grad in enumerate(l_grad_P):
    l_fx = [] ; l_fy = []
    for dpdx, dpdy in zip(grad[0], grad[1]):
        if not np.isnan(dpdx) and not np.isnan(dpdy):
            l_fx.append(-dpdx)
            l_fy.append(-dpdy)
    l_fx = np.array(l_fx)
    l_fy = np.array(l_fy)
    mean_fx = np.mean(l_fx)
    mean_fy = np.mean(l_fy)
    Fx += mean_fx
    Fy += mean_fy
mean_Fx = Fx/len(l_grad_P)
mean_Fy = Fy/len(l_grad_P)
print('Force moyenne de poussée : Fx = '+str(mean_Fx)+' N, Fy = '+str(mean_Fy)+' N')


############################## Affichage ########################################################################################################

show_pressure = True
if show_pressure:
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Pressure (Pa)', color=color)
    ax1.plot(time, P_sonde1, color=color, label='extrados')
    ax1.plot(time, P_sonde2, color='tab:green', label='intrados')
    ax1.axhline(y=0, color='k', linestyle='-')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(['extrados', 'intrados'])

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Hauteur (m)', color=color)  # we already handled the x-label with ax1
    ax2.plot(time, H_wing, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Pressure at sonde, relative position to anchor point: '+str(pr_sonde1)+' and '+str(pr_sonde2))
    fig.text(0.5, 0.02, 'Force moyenne de poussée : Fx = '+str(mean_Fx)+' N, Fy = '+str(mean_Fy)+' N', ha='center')

    plt.grid()
    plt.show()


plt.quiver(l_X, l_Y, -l_grad_P[0][0], -l_grad_P[0][1])
plt.grid()
plt.show()



