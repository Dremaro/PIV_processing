
# Pour lancer ce programme, veuillez activer l'environnement virtuel dans le terminale
import os
import sys
sys.path.insert(0, 'C:/Users/pc1/Leviia/Documents/1_Savoir et Apprentissage/Programmation/PythonKnowledge/mes_outils')

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk

import fonctions_utiles as fu

from NanoFly_Objects import ImageProcessing, GestionnaireDonnees, ImageBrowser
from NanoFly_Objects import Visualization as Vz


# np.set_printoptions(threshold=np.inf)


raw_im = r"00_rawimages\\"
# path_dossier = r'essais\essai_60Hz_threshold3000'
# path_dossier = r'essais\essai_65Hz_threshold3100'
# path_dossier = r'essais\essai_70Hz_threshold3000'
# path_dossier = r'essais\essai_73Hz_threshold3000'
path_dossier = r'essais\essai_76Hz_threshold3000'
# path_dossier = r'essais\essai_79Hz_threshold3000'
# path_dossier = r'essais\essai_82v5Hz_threshold3000'
# path_dossier = r'essais\essai_82v5Hz_threshold3000_1v1V'
# path_dossier = r'essais\essai_85Hz_threshold3000'
# path_dossier = r'essais\essai_88Hz_threshold3000'
# path_dossier = r'essais\essai_95Hz_threshold3000'
path = os.path.join(path_dossier, raw_im)


# ! Commandes et variables globales
create_data = True ; save_session = True     # be careful with the save_session variable, 
save_data_ii = False                         # it will overwrite the previous saved session if there is one
GD = GestionnaireDonnees(path)

freq_thresh = path_dossier.split('_')[1] + '_' + path_dossier.split('_')[2]
dico = {
    "essai_60Hz_threshold3000" : [20,1835],
    "essai_65Hz_threshold3100" : [10,847],
    "essai_70Hz_threshold3000" : [15,1182],
    "essai_73Hz_threshold3000" : [20,1506],
    "essai_76Hz_threshold3000" : [],
    "essai_79Hz_threshold3000" : [],
    "essai_82v5Hz_threshold3000" : [],
    "essai_82v5Hz_threshold3000_1v1V" : [],
    "essai_85Hz_threshold3000" : [],
    "essai_88Hz_threshold3000" : [6,375],
    "essai_95Hz_threshold3000" : []
}




# ! 1ère étape : Créer les tableaux de vitesses interpolés
if create_data:
    GD.vc7_en_vitesseUV(pourcentage = 0.5)
    l_U, l_V = GD.l_U, GD.l_V

    GD.masked_to_zero(l_U)
    GD.masked_to_zero(l_V)

    Vz.show_2d_array(l_U[0])
    
    # root = tk.Tk()
    # browser = ImageBrowser(root, l_images = l_U, size=500, show_name=True)
    # root.mainloop()

    IP = ImageProcessing(l_U, l_V)
    IP.interpolation_trous()
    mobile_parts = IP.l_mobile_parts
    vitesses = IP.vitesses
    l_U_i = [vitesses[i][0] for i in range(len(vitesses))]
    l_V_i = [vitesses[i][1] for i in range(len(vitesses))]

    Vz.show_2d_array(l_U_i[0])

    if save_session:
        # sauvegarder les données de la session (attention à ne pas écraser une sauvegarde du même nom)
        GD.save_session(save_path=path_dossier, name='session_'+freq_thresh+'.db')


# ! 2ème étape : moyenner les images sur une période puis sauvegarder au format dat
if save_data_ii:
    # Charger les variables et objets sauvegardées dans le fichier .db
    GD.load_session(find_path=path_dossier ,name='session_'+freq_thresh+'.db')  ; print('session loaded')

    dico_infos = dico
    # Récupérer le nombre d'image par période (stocké dans IP.T_mean)
    experiment = path_dossier.split("\\")[-1]
    print(experiment)
    l_info = dico_infos[str(experiment)]
    IP.mesurer_periode(l_U, l_info, show_name=True)

    # Calculer la période moyenne
    l_U_par_periode , l_mp_U_par_periode = IP.moyenner_les_images_sur_le_temps(l_U, IP.flatten_mp)
    l_V_par_periode , l_mp_V_par_periode = IP.moyenner_les_images_sur_le_temps(l_V, IP.flatten_mp)

    # Visualiser les images moyennées si nécessaire pour vérifier que tout est correct
    root = tk.Tk()
    browser = ImageBrowser(root, l_images = l_U_par_periode, size=500, show_name=True)
    root.mainloop()

    # sauvegarder les données au format dat pour le calcul de la pression par la suite.
    dat_im = "01_images_dat\\"
    dat_mp = "01_mobile_parts_dat\\"
    save_path_speed = os.path.join(path_dossier, dat_im)
    save_path_mp = os.path.join(path_dossier, dat_mp)
    os.makedirs(save_path_speed, exist_ok=True)
    os.makedirs(save_path_mp, exist_ok=True)
    IP.sauver_UV_au_format_dat((l_U_par_periode,l_V_par_periode), save_path_speed)
    IP.sauver_MP_au_format_dat(l_mp_V_par_periode, save_path_mp)
    



