import os
import sys
sys.path.insert(0, 'C:/Users/pc1/Leviia/Documents/1_Savoir et Apprentissage/Programmation/PythonKnowledge/mes_outils')
# sys.path is a list
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp

import sysconfig
print(sysconfig.get_paths()["purelib"])
print("")

from tqdm import tqdm

import tkinter as tk
from PIL import Image, ImageTk

import FonctionsPerso as Fp  # type: ignore

################################### FONCTIONS ##################################
################################################################################


def calcul_mp_moyen(l_mp_par_periode, threshold=0.3):
    """Il faut imaginer que l'on superpose les différentes versions (pour chaque période) de l'image de la partie mobile
    et que l'on ne conserve que les points de partie mobile qui se sont superposé dans suffisament de périodes (d'où le threshold qui est un pourcentage)

    Args:
        - l_mp_par_periode (liste): liste de liste de liste(de listes pour les points = [x,y]), chaque sous-liste contient autant de sous-sous-liste que \n
        d'instants dans la période, c'est la partie mobile aux différentes périodes (sous-liste) et aux différents instants (sous-sous-liste).
        - threshold (float, optional): pourcentage de présence d'un pixel de la partie mobile à une coordonnée donnée au dela \n
        de laquel le pixel est conservé dans la partie mobile sur une période renvoyé. Defaults to 0.3.

    Returns:
        - l_mp_par_periode_mean : ce n'est donc pas vraiment la moyenne des valeurs des pixels, mais en quelque sorte la forme moyenne pondéré de la partie \n
        mobile. C'est une liste, représentant une unique période, chaque élément est un tableau de pixel contenant les coordonnées des pixels de la partie mobile.
    """
    l_mp_par_periode_mean = [[]]                        # liste qui va contenir les points de la partie mobile moyenne à chaque instant

    for i in range(len(l_mp_par_periode[0])):       # parcoure les instants dans la période
        mps_at_t = [periode[i] for periode in l_mp_par_periode]    # on prend la ième partie mobile de chaque période (mobile parts at time i)
        points_mp = []                                 # liste de tous les points des parties mobiles de chaque image de l'instant i
        points_differents = []                         # identique au précédent mais sans doublons

        # le but de cette première boucle est de créer une liste contenant tous les points de la partie mobile à l'instant i pour chaque periode
        # on obtient donc environ (n_periode * n_point_par_partie_mobile) points
        for p in range(len(mps_at_t)):                 # parcours la partie mobile aux différentes periodes
            for point in mps_at_t[p]:
                if (point not in points_differents):
                    points_differents.append(point)
                points_mp.append(point)
        
        # Cette seconde boucle va nous permettre de ne garder que les points qui sont présents dans une proportion supérieure à threshold
        # on élimine ainsi les abhérations qui n'apparaissent que quelques fois.
        for point in points_differents:
            proportion = points_mp.count(point) / len(mps_at_t)
            if proportion > threshold:
                l_mp_par_periode_mean[-1].append(point)
            # l_mp_par_periode_mean[-1].append(point)
        l_mp_par_periode_mean[-1] = np.array(l_mp_par_periode_mean[-1])

        # on ajoute une liste vide à l'élément suivant de l_mp_par_periode_mean si on a pas atteint le dernier instant de la période
        # pour construire la prochaine partie mobile moyenne
        if i < len(l_mp_par_periode[0])-1:
            l_mp_par_periode_mean.append([])
        # print(f"Instant n°{i}")
    return l_mp_par_periode_mean
        
def moyenner_les_images_sur_le_temps(S_folder, MP_folder):
    l_info = []
    experiment_folder = S_folder.split("\\")[-2]
    l_info = d_infos[str(experiment_folder)] # on récupère les infos de l'expérience dans le dictionnaire si elle y sont présente
    # TODO : ajouter un try except pour gérer les erreurs de clé et remplacer la structure if else çi après

    #!####### UI pour calculer la période en nombre d'images ########
    if len(l_info) != 0:
        n_images = l_info[0]
        n_periodes = l_info[1]
        T_mean = int(n_images) / int(n_periodes)
    else:
        print("La liste d'information de ce fichier est vide, vous devez la remplir manuellement.")
        print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
        print("Vous pouvez maintenant compter le nombre d'images ayant défilé et le nombre de périodes,\nveillez à compter les images pour un nombre entier de périodes")
        print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
        root = tk.Tk()
        browser = Fp.ImageBrowser(root, S_folder)
        root.mainloop()
        n_images = input("Nombre d'images ayant défilé : ")
        n_periodes = input("Nombre de périodes : ")
        T_mean = int(n_images) / int(n_periodes)
    print(f"Nombre d'images moyen par période : {T_mean}")
    input("Appuyez sur une touche pour valider le nombre d'images moyen par période et continuer... ")

    #!####### Mise en forme des données ########
    l_images_par_periode = [[]]
    l_mp_par_periode = [[]]
    images_path = os.listdir(S_folder)
    MP_path = os.listdir(MP_folder)
    i_periodes = 0
    min_taille = 1000000000

    print("moyennage en cour....")
    for i,path in enumerate(images_path):
        # on ajoute une liste vide à l_images_par_periode si on a atteint un multiple du nombre d'images moyen par période
        if i == int(T_mean*(i_periodes+1)):
            taille = len(l_images_par_periode[-1])
            if taille < min_taille:             # pour pouvoir moyenner, toutes nos périodes doivent
                min_taille = taille             # contenir le même nombre d'images ==> on garde le minimum
            i_periodes += 1                           # on incrémente le nombre de périodes
            print(f"Période n°{i_periodes} contenant {taille} images")
            if i_periodes == int(n_periodes):         # on arrête la boucle si on a atteint le nombre de périodes qu'on avait compté
                break
            l_images_par_periode.append([])           # si on a pas bouclé la dernière période, on initialise la suivante
            l_mp_par_periode.append([])               # de même pour les parties mobiles

        # on ajoute le fichier suivant au dernier élément de l_images_par_periode dans tous les cas
        file_S_path = os.path.join(S_folder, path)
        S_data = pd.read_csv(file_S_path, sep=',')
        S_data_array = S_data.to_numpy()
        l_images_par_periode[-1].append(S_data_array)

        file_mp_path = os.path.join(MP_folder, MP_path[i])
        mp_data = pd.read_csv(file_mp_path, sep=',')
        mp_data_array = mp_data.values.tolist()        # on convertit en liste car toutes les sous-listes n'ont pas la même taille (array impossible)
        l_mp_par_periode[-1].append(mp_data_array)

    # On met le même nombre d'images dans chaque période
    for i,el in enumerate(l_images_par_periode):       # TODO : peut être qu'on pourrait conserver toutes les images et moyenner malgré tous les quelques une qui restent, comme ça la période finale est complète
        l_images_par_periode[i] = el[:min_taille]
        l_mp_par_periode[i] = l_mp_par_periode[i][:min_taille]


    # len(l_mp_par_periode) = 8
    # len(l_mp_par_periode[0]) = 84
    # len(l_mp_par_periode[0][0]) = 119
    # len(l_mp_par_periode[0][0][0]) = 2
    # for nuage_point in l_mp_par_periode[0]:
    #     Y,X, = zip(*nuage_point)
    #     plt.scatter(X,Y)
    #     plt.show()
    #!####### Moyenne des images par période et renvoie ########
    l_images_par_periode = np.array(l_images_par_periode)
    images_periodes_mean = np.mean(l_images_par_periode, axis=0)
    #l_mp_par_periode = np.array(l_mp_par_periode)
    mp_periodes_mean = calcul_mp_moyen(l_mp_par_periode)

    return images_periodes_mean, mp_periodes_mean





def sauver_UV_au_format_dat(UandVperiodes_mean, save_path):
    U_periodes_mean, V_periodes_mean = UandVperiodes_mean
    mes_images = []

    for n in range(len(U_periodes_mean)):
        # on fait défiler les images
        x_position = [] ; y_position = [] ; u_speed = [] ; v_speed = []
        data_U = U_periodes_mean[n]
        data_V = V_periodes_mean[n]
        size_i = len(data_U)

        for i in range(size_i):
            ligne_U = data_U[i]
            ligne_V = data_V[i]
            size_j = len(ligne_U)
            for j in range(size_j):
                x_position.append(j*dl)  # x est l'axe horizontal => colonnes
                y_position.append((size_i - i)*dl)  # y est l'axe vertical => lignes      # ! Attention j'ai reinversé i et j ici pour obtenir une image dans le bon sens, doit y avoir une inversion dans le programme précédent.
                u_speed.append(ligne_U[j])
                v_speed.append(ligne_V[j])
        x_position = np.array(x_position) ; y_position = np.array(y_position) ; u_speed = np.array(u_speed) ; v_speed = np.array(v_speed)
        speed_data = np.column_stack((x_position, y_position, u_speed, v_speed))
        # plt.quiver(x_position, y_position, u_speed, v_speed)
        # plt.show()

        # Save the data to a .dat file
        np.savetxt(save_path + f'mean_speed_{Fp.numeros_dans_lordre(n,5)}.dat', speed_data, delimiter=',')
        # print(f"mean_speed_{Fp.numeros_dans_lordre(n,3)}.dat saved in {save_path_to_dat}")
        mes_images.append(speed_data)
    print("Les données de vitesse UV ont été sauvegardées dans le dossier : ", save_path)
    return np.array(mes_images)

def sauver_MP_au_format_dat(mp_periodes_mean, save_path):
    for n in range(len(mp_periodes_mean)):
        mp_data = mp_periodes_mean[n]
        for point in mp_data:    # on convertit les coordonnées des points en mètre
            point[0] = point[0]*dl
            point[1] = point[1]*dl
        # X,Y = zip(*mp_data)
        # plt.scatter(X,Y)
        # plt.show()
        np.savetxt(save_path + f'mean_mp_{Fp.numeros_dans_lordre(n,5)}.dat', mp_data, delimiter=',')
        # print(f"mean_mp_{Fp.numeros_dans_lordre(i,3)}.dat saved in {save_path_to_dat}")
    print("La position de la partie mobile a été sauvegardées dans le dossier : ", save_path)
    return mp_periodes_mean

############################## VARIABLES AND PATHS ##############################
#################################################################################
ratio_pixel_mm = 7.3/41 # = 0.178 en mm/pixel7
dl = 0.178e-3  # en m


# folder = r"VC7_output_test"
# folder = r"VC7_test_absspeed_csv"
# folder = r'essais\essai_65Hz_threshold3100\03_absspeed_csv\\'
folder65Hz_U = r"essais\essai_65Hz_threshold3100\02_U_interpolated_csv"
folder65Hz_V = r"essais\essai_65Hz_threshold3100\02_V_interpolated_csv"
folder65Hz_MP = r"essais\essai_65Hz_threshold3100\02_MP_mobile_parts"
l_info_65Hz = [678,8,121.860]
d_infos = {
    "essai_65Hz_threshold3100": [678,8,121.860],
}
save_path = r'essais\essai_65Hz_threshold3100\05_U_temporal_mean\\'
save_path_to_dat = r'essais\essai_65Hz_threshold3100\06_1_temporal_mean_as_dat\\'
save_mp_to_dat = r'essais\essai_65Hz_threshold3100\06_2_temporal_mean_MP_as_dat\\'
os.makedirs(save_path, exist_ok=True)
os.makedirs(save_path_to_dat, exist_ok=True)
os.makedirs(save_mp_to_dat, exist_ok=True)



################################### MAIN #######################################
################################################################################

U_periodes_mean = moyenner_les_images_sur_le_temps(folder65Hz_U, folder65Hz_MP)[0]
V_periodes_mean, mp_periode_mean = moyenner_les_images_sur_le_temps(folder65Hz_V, folder65Hz_MP)
img = V_periodes_mean[0]
plt.imshow(img)
plt.show()


result = sauver_UV_au_format_dat((U_periodes_mean, V_periodes_mean), save_path_to_dat)
mp_restult = sauver_MP_au_format_dat(mp_periode_mean, save_mp_to_dat)



showAndSave = False
if showAndSave == True:
    for i in tqdm(range(len(U_periodes_mean))):
        np.savetxt(save_path + f'temporal_mean_periode_{Fp.numeros_dans_lordre(i,3)}.csv', U_periodes_mean[i], delimiter=',')
    # root = tk.Tk()
    # browser = Fp.ImageBrowser(root, save_path, show_file_name=True)
    # root.mainloop()






