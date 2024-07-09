import os
import sys
import pickle
sys.path.insert(0, 'C:/Users/pc1/Leviia/Documents/1_Savoir et Apprentissage/Programmation/PythonKnowledge/mes_outils')

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from math import *

import fonctions_utiles as fu

from NanoFly_Objects import ImageProcessing, GestionnaireDonnees, ImageBrowser
from NanoFly_Objects import Visualization as Vz
from NanoFly_Objects import PressureFieldAnalyser

path_to_working_directory = r'C:\Users\pc1\Leviia\Documents\2_ENS-Paris-Saclay\3A__M2_BME2_IPP\Stage_PMMH\PIV_processing'

path_freq3D = r'essais_3D\3D_82Hz'

path_face_horiz = os.path.join(path_to_working_directory, os.path.join(path_freq3D, r'n_face_horiz'))
path_profil_horiz = os.path.join(path_to_working_directory, os.path.join(path_freq3D, r'p_profil_horiz'))

folders_face = os.listdir(path_face_horiz)
folders_profil = os.listdir(path_profil_horiz)
print(folders_face)

data_positions_face = [GestionnaireDonnees(os.path.join(path_face_horiz, folder)) for folder in folders_face]
for data in data_positions_face:
    print(data)












