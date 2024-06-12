
######################################### BIBLIOTHEQUE ############################################################
###################################################################################################################
import os
import sys
import dill
from tqdm import tqdm
import copy

sys.path.insert(0, 'C:/Users/pc1/Leviia/Documents/1_Savoir et Apprentissage/Programmation/PythonKnowledge/mes_outils')
# sys.path is a list

import lvpyio as lv                       # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as plc
import pandas as pd
import fonctions_utiles as Fu
import scipy as sp
import itertools
import cv2

import time
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

import FonctionsPerso as Fp              # type: ignore




######################################### OBJECTS ############################################################
##############################################################################################################

class GestionnaireDonnees:
    """Cette classe permet de gérer les données de l'expérience PIV.
    On peut notamment :
    - Convertir les fichiers .vc7 en fichiers .csv contenant les composantes de vitesse u et v.
    - et pleins d'autres choses à venir...
    """
    def __init__(self, path):
        self.access_path = path
        # self.save_path = save_path
        l_dir = os.listdir(self.access_path)   ;   n_dir = len(l_dir)
        self.image_paths = l_dir                  # on prend le pourcentage indiqué
        self.Nimage = len(self.image_paths)
        self.l_U = []
        self.l_V = []

    def vc7_en_vitesseUV(self, pourcentage=1):
        """
        Cette méthode permet de convertir les fichiers .vc7 en fichiers .csv
        contenant les composantes de vitesse u et v.
        """
        # os.makedirs(self.save_path, exist_ok=True)
        # boucle sur les images
        for ii in tqdm(range(1,int(self.Nimage*pourcentage))):
            # Lecture des fichiers Lavision   
            nom_fichier = 'B'+self.numeros_dans_lordre(ii, 5)
            buffer = lv.read_buffer(self.access_path + nom_fichier + '.vc7')
            ma_arr = buffer.as_masked_array()
            ma_arr_x=ma_arr["u"]
            ma_arr_y=ma_arr["v"]
            self.l_U.append(ma_arr_x)
            self.l_V.append(ma_arr_y)

    def masked_to_zero(self, l_image):
        """Cette fonction permet de convertir les valeurs masked en zéro.
        On a besoin de le faire car l'algorithme de remplissage des trous
        détecte des clusters de zéros et non des valeurs masked.

        Args:
            l_image (liste d'array): liste des images à traiter
        """
        for i, image in enumerate(l_image):
            l_image[i] = np.ma.filled(image, float(0))

    def numeros_dans_lordre(self,i,odg):
        """Cette fonction renvoie un nombre avec un nombre de chiffres 
        valant odg en ajoutant des zéros à gauche si nécessaire.

        Args:
            i (int): numéro que l'on souhaite modifier
            odg (int): Nombre de chiffre totale à obtenir

        Returns:
            string: nombre remplieavec des zéros à gauche
        """
        for k in range(1,odg+1):
            if i<10**k:
                return (odg-k)*'0' + str(i)
        
    def load_session(self, find_path=None, name="session.db"):
        if find_path is not None:
            with open(os.path.join(find_path, name), 'rb') as file:
                dill.load(file)
        else:
            dill.load_session(name)

    def save_session(self, save_path=None, name="session.db"):
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            path = os.path.join(save_path, name)
            dill.dump_session(path)
            print(f"Session saved in {path}, as {name}")
        else:
            dill.dump_session('session.db')
            print(f"Session saved in {save_path} as session.db")


class ImageProcessing:
    """Permet de faire tout les traitements d'image nécessaire, notamment :
    - Interpolation des trous dans les images de vitesse u et v
    - Calcul de la moyenne temporelle des champs de vitesse
    - Calcul de la moyenne temporelle des pièces mobiles
    - et pleins d'autres choses à venir...

    Args:
            l_U (liste d'array): liste les tableau de vitesse u
            l_V (liste d'array): liste les tableau de vitesse v
    """
    
    def __init__(self, l_U, l_V):
        self.ratio_pixel_mm = 7.3/41 # = 0.178 en mm/pixel7
        self.dl = 0.178e-3  # en m
        self.vitesses = [[u, v] for u, v in zip(l_U, l_V)] # liste de listes de tableaux de vitesses u et v
        self.l_mobile_parts = []
        self.flatten_mp = []

    def interpolation_trous(self, critical_size=10):
        """Cette fonction permet de repérer les trous qui ne sont pas des pièces mobiles
        et de les combler par interpolation.
        Elle charge ses résultats dans les attribut d'objet :
        - self.vitesses : liste de listes de tableaux de vitesses u et v
        - self.l_mobile_parts : liste de listes de listes de coordonnées des pièces mobiles

        Args:
            critical_size (int, optional): taille au dela de laquelle un trou est considéré comme une pièce mobile. Defaults to 10.
        """
        vitesses_interpolees = []
        for i,vitesse in tqdm(enumerate(self.vitesses)):
            l_mp_uv = []
            vitesses_interpolees.append([])
            for j,uv in enumerate(vitesse):
                # on retire les bords
                uv = uv[1:-1,1:-1]
                # print(uv)

                # repérer les trous
                l_trous, mobile_parts = Fu.reperer_trous(uv, critical_size) # repère les trous et les pièces mobiles (gros trous : size > critical_size)
                # l_trous, mobile_parts = Fu.trouver_trous(uv, critical_size)
                Fu.combler_zones(uv, mobile_parts)                          # combler les parties mobiles pour ne pas les interpoler.

                # Combler les trous par interpolation
                non_zero_coords = np.argwhere(uv != 0)                                     # on récupère les coordonnées des points non nuls
                zero_coords = np.argwhere(uv == 0)                                         # on récupère les coordonnées des points nuls
                non_zeros_values = uv[non_zero_coords[:,0], non_zero_coords[:,1]]          # on récupère les valeurs des points non nuls
                valeurs_interpolees = sp.interpolate.griddata(non_zero_coords, non_zeros_values, zero_coords, method='linear') # interpolation des valeurs avec griddata
                uv[zero_coords[:,0], zero_coords[:,1]] = valeurs_interpolees               # on remplace les valeurs nulles par les valeurs interpolées
                Fu.apply_zero_as_mask(uv, mobile_parts)                                    # on remet les parties mobiles à zéro
                
                # mise à jour des vitesses interpolées
                vitesses_interpolees[i].append(uv)

                im_shape = uv.shape
                # récupération des parties mobiles (trous de taille sup à critical_size)
                mobile_parts = Fu.index_to_coords(mobile_parts, im_shape)
                mobile_parts = list(itertools.chain.from_iterable(mobile_parts))
                l_mp_uv.append(mobile_parts)
            
            # On stock les trous de la vitesse i à la ième position de l_mobile_parts
            self.flatten_mp.append(l_mp_uv)
        self.vitesses = vitesses_interpolees

    def mesurer_periode(self,l_image, l_info, show_name=False):
        # Pour éviter de devoir refaire le calcul à chaque fois, un utilise des listes d'informations
        # pour stocker les résultats.
        if len(l_info) != 0:
            print("Une liste d'informations a été fournie :", l_info)
            self.n_periodes = l_info[0]
            self.n_images = l_info[1]
            self.T_mean = self.n_images / self.n_periodes
            print(f"Nombre d'images moyen par période : {self.T_mean}")
            input("Confirmer ?")

        else :
            print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
            print("Vous pouvez maintenant compter le nombre d'images ayant défilé et le nombre de périodes,\nveillez à compter les images pour un nombre entier de périodes")
            print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
            l_image_255 = Fu.normalize_to_255(l_image)
            # print(np.amax(l_image_255), np.amin(l_image_255.all()))
            
            root = tk.Tk()
            browser = ImageBrowser(root, l_images = l_image_255, show_name=show_name)
            root.mainloop()
            self.n_images = int(input("Nombre d'images ayant défilé : "))
            self.n_periodes = int(input("Nombre de périodes : "))
            self.T_mean = self.n_images / self.n_periodes
            print(f"Nombre d'images moyen par période : {self.T_mean}")
            input("Appuyez sur une touche pour valider le nombre d'images moyen par période et continuer... ")
            

    def moyenner_les_images_sur_le_temps(self, l_image, l_mp):
        """Cette fonction permet de moyenner les images sur le temps."""

        #!############ Mise en forme des données ##############
        l_images_par_periode = [[]]
        l_mp_par_periode = [[]]
        i_periodes = 0
        min_taille = 1000000000

        print("moyennage en cour....")
        for i,image in enumerate(l_image):
            # on ajoute une liste vide à l_images_par_periode si on a atteint un multiple du nombre d'images moyen par période
            if i == int(self.T_mean*(i_periodes+1)):
                taille = len(l_images_par_periode[-1])
                if taille < min_taille:             # pour pouvoir moyenner, toutes nos périodes doivent
                    min_taille = taille             # contenir le même nombre d'images ==> on garde le minimum
                i_periodes += 1                           # on incrémente le nombre de périodes
                print(f"Période n°{i_periodes} contenant {taille} images")
                if i_periodes == int(self.n_periodes):         # on arrête la boucle si on a atteint le nombre de périodes qu'on avait compté
                    break
                l_images_par_periode.append([])           # si on a pas bouclé la dernière période, on initialise la suivante
                l_mp_par_periode.append([])               # de même pour les parties mobiles

            # on ajoute le fichier suivant au dernier élément de l_images_par_periode dans tous les cas
            image = np.array(image)
            l_images_par_periode[-1].append(image)
            mp = l_mp[i]        # on conserve en liste car toutes les sous-listes n'ont pas la même taille (array impossible)
            l_mp_par_periode[-1].append(mp)

        # On met le même nombre d'images dans chaque période
        for i,el in enumerate(l_images_par_periode):       # TODO : peut être qu'on pourrait conserver toutes les images et moyenner malgré tous les quelques une qui restent, comme ça la période finale est complète
            l_images_par_periode[i] = el[:min_taille]
            l_mp_par_periode[i] = l_mp_par_periode[i][:min_taille]

        #!####### Moyenne des images par période et renvoie ########
        l_images_par_periode = np.array(l_images_par_periode)
        images_periodes_mean = np.mean(l_images_par_periode, axis=0)
        #l_mp_par_periode = np.array(l_mp_par_periode)
        mp_periodes_mean = self.calcul_mp_moyen(l_mp_par_periode)

        return images_periodes_mean, mp_periodes_mean

    def calcul_mp_moyen(self, l_mp_par_periode, threshold=0.3):
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
            mps_at_t = [periode[i] for periode in l_mp_par_periode]    # on prend les PM du ième instant de chaque période (mobile parts at time i)
            points_mp = []                                 # liste de tous les points des parties mobiles de chaque image de l'instant i
            points_differents = []                         # identique au précédent mais sans doublons

            # le but de cette première boucle est de créer une liste contenant tous les points de la partie mobile à l'instant i pour chaque periode
            # on obtient donc environ (n_periode * n_point_par_partie_mobile) points
            for p in range(len(mps_at_t)):                 # parcours la partie mobile aux différentes periodes
                for mp in mps_at_t[p]:
                    for point in mp:
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


    def sauver_UV_au_format_dat(self, UandVperiodes_mean, save_path):
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
                    x_position.append(j*self.dl)  # x est l'axe horizontal => colonnes
                    y_position.append((size_i - i)*self.dl)  # y est l'axe vertical => lignes      # ! Attention j'ai reinversé i et j ici pour obtenir une image dans le bon sens, doit y avoir une inversion dans le programme précédent.
                    u_speed.append(ligne_U[j])
                    v_speed.append(ligne_V[j])
            x_position = np.array(x_position) ; y_position = np.array(y_position) ; u_speed = np.array(u_speed) ; v_speed = np.array(v_speed)
            speed_data = np.column_stack((x_position, y_position, u_speed, v_speed))

            # Save the data to a .dat file
            np.savetxt(save_path + f'mean_speed_{Fp.numeros_dans_lordre(n,5)}.dat', speed_data, delimiter=',')
            # print(f"mean_speed_{Fp.numeros_dans_lordre(n,3)}.dat saved in {save_path_to_dat}")
            mes_images.append(speed_data)
        print("Les données de vitesse UV ont été sauvegardées dans le dossier : ", save_path)
        return np.array(mes_images)

    def sauver_MP_au_format_dat(self, mp_periodes_mean, save_path):
        xy_mp_position = []
        for n in range(len(mp_periodes_mean)):
            mp_data = mp_periodes_mean[n]
            xy_mp_position = []
            for point in mp_data:    # on convertit les coordonnées des pixels en mètres
                a = point[0]*self.dl
                b = point[1]*self.dl
                xy_point = [a,b]
                xy_mp_position.append(xy_point)
            np.savetxt(save_path + f'mean_mp_{Fp.numeros_dans_lordre(n,5)}.dat', xy_mp_position, delimiter=',')
            # print(f"mean_mp_{Fp.numeros_dans_lordre(i,3)}.dat saved in {save_path_to_dat}")
        
        print("La position de la partie mobile a été sauvegardées dans le dossier : ", save_path)
        return mp_periodes_mean


class Visualization:
    @staticmethod
    def show_image(image):
        plt.imshow(image)
        plt.show()

    @staticmethod
    def scatter_plot(X, Y, C):
        plt.scatter(X, Y, c=C, cmap='viridis')
        plt.colorbar()
        plt.show()
    
    @staticmethod
    def show_2d_array(array):
        plt.imshow(array, cmap='viridis')
        plt.colorbar()
        plt.show()
    

class ImageBrowser:
    '''
    A simple image browser that allows you to navigate through a folder/list of images using Next and Previous buttons.
    ## How to use:
    
    root = tk.Tk()\n
    folder = "path_to_your_folder"       # replace with your folder\n
    root = tk.Tk()\n
    browser = ImageBrowser(root, folder) # root is the tkinter object\n
    root.mainloop()
    '''
    def __init__(self, master, folder=None, l_images=None, size=400, show_name=False):
        self.size = size
        self.folder = folder
        self.l_images = l_images.copy()  # we make a copy to avoid modifying the original list

        if self.folder is None and self.l_images is None:
            raise ValueError("Either 'folder' or 'list' must be provided")
        if self.folder is not None and self.l_images is not None:
            raise ValueError("Only one of 'folder' or 'list' should be provided")
        if self.l_images is not None:
            self.master = master
            self.show_n = show_name
            self.files = [f"indice_{i}.png" for i in range(len(self.l_images))]
            self.index = 0
            self.l_images = Fu.normalize_to_255(self.l_images)
            self.img_obj = Image.fromarray(self.l_images[self.index]).resize((self.size, self.size))
            self.img = ImageTk.PhotoImage(self.img_obj)
        
        if self.folder is not None:
            self.master = master
            self.show_n = show_name
            self.files = [f for f in os.listdir(folder) if f.endswith('.csv')]
            self.index = 0
            self.img_obj = self.load_image(self.files[self.index])
            self.img = ImageTk.PhotoImage(self.img_obj)
        
        # Show the image, its number and the time
        # self.master.geometry("1200x1200")
        self.label = tk.Label(master, image=self.img)
        self.label.grid(row=0, column=1)
        self.number_label = tk.Label(master, text=f"Image: {self.index + 1}/{len(self.files)}")
        self.number_label.grid(row=0, column=0)
        self.time_label = tk.Label(master, text=f"Time: {self.index*180e-3:.3f} ms")
        self.time_label.grid(row=0, column=2)

        # Previous and 10 previous buttons
        prev_button = tk.Button(master, text="<< Prev", command=self.prev)
        prev_button.grid(row=1, column=0)
        prev10_button = tk.Button(master, text="<< Prev 10", command=self.prev10)
        prev10_button.grid(row=2, column=0)
        prev100_button = tk.Button(master, text="<< Prev 100", command=self.prev100)
        prev100_button.grid(row=3, column=0)

        # Next and 10 next buttons
        next_button = tk.Button(master, text="Next >>", command=self.next)
        next_button.grid(row=1, column=2)
        next10_button = tk.Button(master, text="Next 10 >>", command=self.next10)
        next10_button.grid(row=2, column=2)
        next100_button = tk.Button(master, text="Next 100 >>", command=self.next100)
        next100_button.grid(row=3, column=2)

        # Show the file name file_name_label
        if self.show_n:
            if folder is not None:
                self.name_label = tk.Label(master, text=self.files[self.index])
                self.name_label.grid(row=3, column=1)
            else :
                self.name_label = tk.Label(master, text="indice n°"+str(self.index))
                self.name_label.grid(row=3, column=1)

    def load_image(self, filename):
        data = pd.read_csv(os.path.join(self.folder, filename), sep=',').values
        vmin = data.min()
        vmax = data.max()
        vcenter = (vmin + vmax) / 2
        norm = plc.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax) # Create a two-slope normalization
        cmap = plt.get_cmap('RdBu')                                    # and a colormap
        data = cmap(norm(data))                    # Normalize the data and apply the colormap
        np_data = np.uint8(data*255)               # Convert the data to 8-bit values
        img = Image.fromarray(np_data)  # Convert the data to an image
        img = img.resize((500, 500))
        return img

    def next(self):
        self.index += 1
        if self.index >= len(self.files):
            self.index = 0
        self.update()
    
    def next10(self):
        self.index += 10
        if self.index >= len(self.files):
            self.index = 0
        self.update()

    def next100(self):
        self.index += 100
        if self.index >= len(self.files):
            self.index = 0
        self.update()

    def prev(self):
        self.index -= 1
        if self.index < 0:
            self.index = len(self.files) - 1
        self.update()
    
    def prev10(self):
        self.index -= 10
        if self.index < 0:
            self.index = len(self.files) - 1
        self.update()

    def prev100(self):
        self.index -= 100
        if self.index < 0:
            self.index = len(self.files) - 1
        self.update()

    def update(self):
        if self.folder is not None:
            self.img_obj = self.load_image(self.files[self.index])
        else :
            self.img_obj = Image.fromarray(self.l_images[self.index]).resize((self.size, self.size))
        self.img = ImageTk.PhotoImage(self.img_obj)
        self.label.config(image=self.img)
        self.number_label.config(text=f"Image: {self.index + 1}/{len(self.files)}")
        self.time_label.config(text=f"Time: {self.index*180e-3:.3f} ms")
        if self.show_n:
            if self.folder is not None:
                self.name_label.config(text=self.files[self.index])
            else:
                self.name_label.config(text="indice n°"+str(self.index))



class MobilePartProcessing:
    def __init__(self, folder_path, threshold=0.3):
        self.folder_path = folder_path
        self.threshold = threshold
        self.mobile_parts = self.load_mobile_parts()

    def load_mobile_parts(self):
        mobile_parts = []
        for file_name in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path, file_name)
            data = pd.read_csv(file_path, sep=',').values.tolist()
            mobile_parts.append(data)
        return mobile_parts

    def calculate_mean_mobile_parts(self, periods):
        # Implement your mean calculation logic here
        pass

    def save_mobile_parts(self, save_path, mobile_parts):
        os.makedirs(save_path, exist_ok=True)
        for i, part in enumerate(mobile_parts):
            np.savetxt(os.path.join(save_path, f'mean_mobile_part_{i}.csv'), part, delimiter=',')

class UIHandler:
    def __init__(self):
        self.root = tk.Tk()

    def show_image_browser(self, folder_path):
        browser = Fp.ImageBrowser(self.root, folder_path)
        self.root.mainloop()




def main():
    # Initialize data and processing objects
    experiment_data = ExperimentData(d_infos["essai_65Hz_threshold3100"])
    img_proc = ImageProcessing(folder65Hz_U)
    mp_proc = MobilePartProcessing(folder65Hz_MP)

    # Process data
    U_mean_images = img_proc.calculate_mean_images(experiment_data.periods)
    mean_mobile_parts = mp_proc.calculate_mean_mobile_parts(experiment_data.periods)

    # Save results
    img_proc.save_images(save_path, U_mean_images)
    mp_proc.save_mobile_parts(save_mp_to_dat, mean_mobile_parts)

    # Visualize results
    Visualization.show_image(U_mean_images[0])
















    # def load_images(self):
    #     images = []
    #     for file_name in os.listdir(self.folder_path):
    #         file_path = os.path.join(self.folder_path, file_name)
    #         data = pd.read_csv(file_path, sep=',').to_numpy()
    #         images.append(data)
    #     return images
    
    # def load_vc7_data(self):
    #     fields = []

    # def calculate_mean_images(self, periods):
    #     # Implement your mean calculation logic here
    #     pass

    # def save_images(self, save_path, images):
    #     os.makedirs(save_path, exist_ok=True)
    #     for i, image in enumerate(images):
    #         np.savetxt(os.path.join(save_path, f'mean_image_{i}.csv'), image, delimiter=',')


if __name__ == "__main__":
    print("Hello World")


