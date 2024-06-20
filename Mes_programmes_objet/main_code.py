
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
# np.set_printoptions(threshold=np.inf)

###############################################################################################################
########################### Variables globales ################################################################
###############################################################################################################
dl = 0.178e-3


raw_im = r"00_rawimages\\"
# path_dossier = r'essais\essai_60Hz_threshold3000'
# path_dossier = r'essais\essai_65Hz_threshold3100'
# path_dossier = r'essais\essai_70Hz_threshold3000'
# path_dossier = r'essais\essai_73Hz_threshold3000'
# path_dossier = r'essais\essai_76Hz_threshold3000'
# path_dossier = r'essais\essai_79Hz_threshold3000'
# path_dossier = r'essais\essai_82v5Hz_threshold3000'
# path_dossier = r'essais\essai_82v5Hz_threshold3000_1v1V'
# path_dossier = r'essais\essai_85Hz_threshold3000'
# path_dossier = r'essais\essai_88Hz_threshold3000'
path_dossier = r'essais\essai_95Hz_threshold3000'
path = os.path.join(path_dossier, raw_im)


# ! Commandes et variables globales
create_data = 0 ; save_session = 1     # be careful with the save_session variable, 
save_data_ii = 0                       # it will overwrite the previous saved session if there is one
comput_pressure_i = 1
GD = GestionnaireDonnees(path)

freq_thresh = path_dossier.split('_')[1] + '_' + path_dossier.split('_')[2]
dico = {
    "essai_60Hz_threshold3000" : [10,917.5],
    "essai_65Hz_threshold3100" : [10,847],
    "essai_70Hz_threshold3000" : [15,1182],
    "essai_73Hz_threshold3000" : [10,753],
    "essai_76Hz_threshold3000" : [15,1086],
    "essai_79Hz_threshold3000" : [12,837],
    "essai_82v5Hz_threshold3000" : [18,1201],
    "essai_82v5Hz_threshold3000_1v1V" : [18,1201],
    "essai_85Hz_threshold3000" : [18,1166],
    "essai_88Hz_threshold3000" : [6,375],
    "essai_95Hz_threshold3000" : [18,1042]
}



# ! 1ère étape : Créer les tableaux de vitesses interpolés
if create_data:
    # Charger les vitesses U et V
    GD.vc7_en_vitesseUV(pourcentage = 0.6)
    l_U, l_V = GD.l_U, GD.l_V

    # Remplacer les valeurs masquées '_' à zéro
    GD.masked_to_zero(l_U)
    GD.masked_to_zero(l_V)

    # Bouchage des trous
    IP = ImageProcessing(l_U, l_V)
    IP.interpolation_trous()
    mobile_parts = IP.l_mobile_parts
    vitesses = IP.vitesses
    IP.l_U_i = [vitesses[i][0] for i in range(len(vitesses))]
    IP.l_V_i = [vitesses[i][1] for i in range(len(vitesses))]

    if save_session:
        # sauvegarder les données de la session (attention à ne pas écraser une sauvegarde du même nom)
        GD.save_session(save_path=path_dossier, name='session_'+freq_thresh+'.db')


# ! 2ème étape : moyenner les images sur une période puis sauvegarder au format dat
if save_data_ii:
    # Charger les variables et objets sauvegardées dans le fichier .db
    GD.load_session(find_path=path_dossier ,name='session_'+freq_thresh+'.db')  ;  print('session loaded')

    dico_infos = dico
    # Récupérer le nombre d'image par période (stocké dans IP.T_mean)
    experiment = path_dossier.split("\\")[-1]
    l_info = dico_infos[str(experiment)]
    IP.mesurer_periode(IP.l_U_i, l_info, show_name=True)

    # Calculer la période moyenne
    l_U_par_periode , l_mp_U_par_periode = IP.moyenner_les_images_sur_le_temps(IP.l_U_i, IP.flatten_mp)
    l_V_par_periode , l_mp_V_par_periode = IP.moyenner_les_images_sur_le_temps(IP.l_V_i, IP.flatten_mp)

    # Réparation de la partie mobile de chaque image de la période
    l_mp_reparee = IP.reparer_mp(l_mp_V_par_periode, l=1)

    # sauvegarder les données au format dat pour le calcul de la pression par la suite.
    dat_im = "01_images_dat\\"
    dat_mp = "01_mobile_parts_dat\\"
    save_path_speed = os.path.join(path_dossier, dat_im)
    save_path_mp = os.path.join(path_dossier, dat_mp)
    os.makedirs(save_path_speed, exist_ok=True)
    os.makedirs(save_path_mp, exist_ok=True)
    IP.sauver_UV_au_format_dat((l_U_par_periode,l_V_par_periode), save_path_speed)
    IP.sauver_MP_au_format_dat(l_mp_reparee, save_path_mp)
    

    print('Extracting pressure data...')
    PFA = PressureFieldAnalyser(save_path_speed, key='queen2', dl= 0.178e-3)
    GD.save_session(save_path=path_dossier, name='session_pressure_analysis.db')


# ! 3ème étape : Calculer le champs de pression
#################################################################################################
###################### Activation du programme matlab Queen2 ####################################
#################################################################################################



# ! 4ème étape : traîter le champs de pression puis calculer les forces de poussées 

if comput_pressure_i:
    GD.load_session(find_path=path_dossier ,name='session_pressure_analysis.db')
    Vz.animate_images(PFA.l_P, save_dir=path_dossier, save_as='pressure_field.mp4')

    #region : test
    n = 10
    X = PFA.l_X[n]
    Y = PFA.l_Y[n]
    P = PFA.l_P[n]
    p_sans_nan = P[~np.isnan(P)]
    p_max = np.max(p_sans_nan)
    p_min = np.min(p_sans_nan)
    MP = np.array(PFA.l_MP[n])

    # Vz.show_2d_array(P, show=False)
    # plt.title('Pressure field 82x82')
    # plt.colorbar()
    # plt.text(0, 90, 'Max : {} Pa  ;  Min : {} Pa'.format(p_max, p_min), color='black', fontsize=12)
    # plt.show()

    F, M, centre, contour, forces_locales = PFA.calculer_force(X,Y,P, MP)
    contour = np.array(contour)
    forces_locales = np.array(forces_locales)
    print(forces_locales[0])
    print('Force calculated : ', F)
    print('Moment calculated : ', M)

    Vz.show_2d_array(P, show=False)
    Vz.scatter_plot(MP[:,0], MP[:,1] , show=False)
    Vz.scatter_plot(contour[:,0], contour[:,1], show=False, label_vect='Contour')
    Vz.scatter_plot(centre[0], centre[1], show=False, label_vect='Centre')
    Vz.vector_plot(contour[:,0], contour[:,1], forces_locales[:,0], forces_locales[:,1], show=False, label_vect='Forces locales')
    Vz.vector_plot(centre[0], centre[1], F[0], F[1], color='r', show=False, label_vect='Force totale')
    plt.title('Pressure forces and moments')
    plt.legend()
    plt.show()
    #endregion

    L_Fx = []
    L_Fy = []
    L_M = []
    for k in range(len(PFA.l_P)):
        F, M, centre, contour, forces_locales = PFA.calculer_force(PFA.l_X[k], PFA.l_Y[k], PFA.l_P[k], np.array(PFA.l_MP[k]))
        L_Fx.append(F[0])
        L_Fy.append(F[1])
        L_M.append(M)
    
    Vz.plot_forces(PFA.time, L_Fx, L_Fy, L_M, save_dir=path_dossier)

    # L_forces = []

    # with open('F_x.pkl', 'rb') as f:
    #     print('loading forces...')
    #     L_forces = pickle.load(f)

    # L_forces.append(L_Fx)

    # with open('F_x.pkl', 'wb') as f:
    #     pickle.dump(L_forces, f)




















