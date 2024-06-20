
######################################### BIBLIOTHEQUE ############################################################
###################################################################################################################
import os
import sys
import dill
from tqdm import tqdm
import copy
from math import *

sys.path.insert(0, 'C:/Users/pc1/Leviia/Documents/1_Savoir et Apprentissage/Programmation/PythonKnowledge/mes_outils')
# sys.path is a list

import lvpyio as lv                       # type: ignore
import numpy as np
import pandas as pd
import fonctions_utiles as Fu
import scipy as sp
import itertools
import cv2

import time
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib.colors as plc
import matplotlib.animation as animation
import matplotlib.ticker as ticker
from PIL import Image, ImageTk

import FonctionsPerso as Fp              # type: ignore
         



######################################### OBJETS #############################################################
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
    def __init__(self, l_U, l_V, dl = 0.178e-3):
        self.ratio_pixel_mm = 7.3/41 # = 0.178 en mm/pixel7
        self.dl = dl  # en m
        self.vitesses = [[u, v] for u, v in zip(l_U, l_V)] # liste de listes de tableaux de vitesses u et v
        self.l_U_i = []
        self.l_V_i = []
        self.l_mobile_parts = []
        self.flatten_mp = []
        self.coeffs = []

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
            for j,vit in enumerate(vitesse):

                # on retire les bords
                uv = vit[2:-2,2:-2]  # (premier bord null, deuxième bord de mauvaise qualité donc 2:-2)

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


    def reparer_mp(self, mean_periode, l=0):
        max_length = 0

        # On commence par trouver la longueur maximale, pour avoir la longueur de la partie mobile
        for mp in mean_periode:
            x, y = zip(*mp)
            x = np.array(x)
            x_min = min(x)
            x_max = max(x)
            if x_max - x_min > max_length:
                max_length = x_max - x_min

        l_repared_mp = []
        for mp in mean_periode:
            x, y = zip(*mp)
            x = np.array(x) ; y = np.array(y)
            x_min = min(x)

            coeff = np.polyfit(x, y, 1)
            poly_y = np.polyval(coeff, x)
            std = np.std(y - poly_y)
            ecart_to_droite = 2   # round(std) + l
            if ecart_to_droite == 0:
                ecart_to_droite = 1
            self.coeffs.append(coeff)

            artificel_mp_points = []
            for i in range(x_min, x_min + max_length - 1):
                y_droite = np.polyval(coeff, i)
                for j in range(-ecart_to_droite, ecart_to_droite):
                    h = round(y_droite) # obtenir la hauteur en pixels
                    artificel_mp_points.append([i, h + j])

            artificel_mp_points = np.array(artificel_mp_points)
            mp_reparee = np.vstack((mp, artificel_mp_points))
            mp_reparee = np.unique(mp_reparee, axis=0)

            l_repared_mp.append(mp_reparee)

        return l_repared_mp


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
            np.savetxt(save_path + f'mean_speed_{Fu.numeros_dans_lordre(n,5)}.dat', speed_data, delimiter=',')
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
            max_b = max([point[1] for point in xy_mp_position])
            np.savetxt(save_path + f'mean_mp_{Fu.numeros_dans_lordre(n,5)}.dat', xy_mp_position, delimiter=',')
            # print(f"mean_mp_{Fp.numeros_dans_lordre(i,3)}.dat saved in {save_path_to_dat}")
        
        print("La position de la partie mobile a été sauvegardées dans le dossier : ", save_path)
        return mp_periodes_mean

    def compute_vorticity(self, U, V):
        dUdx, dUdy = np.gradient(U, self.dl)
        dVdx, dVdy = np.gradient(V, self.dl)
        return dVdx - dUdy


class Visualization:
    @staticmethod
    def show_image(image, cb = False, show=True):
        plt.imshow(image)
        if show:
            plt.show()

    @staticmethod
    def scatter_plot(X, Y, label_vect=None, C=None, show = True):
        if label_vect is not None:
            plt.scatter(X, Y, c=C, cmap='viridis', label=label_vect)
        else :
            plt.scatter(X, Y, c=C, cmap='viridis')
        
        if show:
            plt.show()
    
    @staticmethod
    def show_2d_array(array, show=True, cb = False):
        plt.imshow(array, cmap='viridis')
        if cb:
            plt.colorbar()
        if show:
            plt.show()
    
    @staticmethod
    def vector_plot(X, Y, U, V, label_vect=None, show=True, color=None, cb=False):
        if label_vect is not None:
            plt.quiver(X, Y, U, V, color=color, label=label_vect)
        else:
            plt.quiver(X, Y, U, V, color=color)
        if cb:
            plt.colorbar()
        if show:
            plt.show()
    
    @staticmethod
    def animate_images(l_images, save_dir = None, show=True, save_as='animation.mp4'):

        # Initialize a figure and axis for the animation
        fig, ax = plt.subplots()
        cmap = plt.cm.viridis
        cmap.set_bad(color='black')

        im = [None]

        # Define the init function for the animation
        def init():
            image = np.ma.array(l_images[0], mask=np.isnan(l_images[0]))
            im[0] = ax.imshow(image, cmap=cmap)
            return [im[0]]

        # Define the update function for the animation
        def update(i):
            image = np.ma.array(l_images[i], mask=np.isnan(l_images[i]))
            im[0].set_array(image)
            return [im[0]]

        # Create the animation
        ani = animation.FuncAnimation(fig, update, interval=20, frames=len(l_images), init_func=init, blit=True)

        # Save the animation
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_as = os.path.join(save_dir, save_as)
            ani.save(save_as, writer='ffmpeg')

        # Show the animation
        if show:
            plt.show()

    @staticmethod
    def plot_forces(t, L_Fx, L_Fy, L_M, show=True, save_dir=None, save_as='efforts.png'):
        fig, ax = plt.subplots()

        ax.plot(t, L_Fx, color='r', label='Force X')
        ax.plot(t, L_Fy, color='orange', label='Force Y')
        ax.plot(t, L_M, color='b', label='Moment')
        ax.set_xlabel('Temps (s)')

        # Définir le nombre de graduations sur les axes
        ax.xaxis.set_major_locator(ticker.MultipleLocator(base=0.002))  # Pour l'axe des x
        ax.yaxis.set_major_locator(ticker.MultipleLocator(base=0.005))  # Pour l'axe des y

        ax.set_title("Efforts globaux exercé sur l'aile")
        ax.grid()
        ax.legend()
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_as = os.path.join(save_dir, save_as)
            plt.savefig(save_as)
        if show:
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


class PressureFieldAnalyser:
    def __init__(self, path, key='queen2', dl=0.178e-3):
        """gather output data of the matlab program in the following columns
        x (m),  y (m),  u (m/s),  v (m/s),  dp/dx (Pa/m),  dp/dy (Pa/m),  p (Pa),  |p| (Pa)

        Args:
            path (string): path to the folder containing the data
            keyword (str, optional): string that's inside every file name of interest. Defaults to 'queen2'.
        """
        self.dl = dl
        self.l_dir = os.listdir(path)
        self.l_path = [os.path.join(path, f) for f in self.l_dir if key in f]
        if len(self.l_path) == 0:
            print(f"No file with the keyword '{key}' found in the folder {path}")
        self.time = []
        self.l_X = []
        self.l_Y = []
        self.l_U = []
        self.l_V = []
        self.l_P = []
        self.l_dPdx = []
        self.l_dPdy = []
        self.extraire_donnees() # conversion en tableaux 2D
        self.l_grad_P = list(zip(self.l_dPdx, self.l_dPdy))
        self.l_MP = []
        for i in range(len(self.l_P)):
            n_x_i = len(self.l_X[i])
            n_y_i = len(self.l_Y[i])
            MP = Fu.parties_mobiles(n_x_i, n_y_i, self.l_P[i], self.l_grad_P[i])
            self.l_MP.append(MP)
        print("Initialisation done")

    def extraire_donnees(self): 
        for n,path in enumerate(tqdm(self.l_path)):
            data_pressure = pd.read_csv(path, sep=',', header=None)
            data_pressure.columns = ['x', 'y', 'u', 'v', 'dpdx', 'dpdy', 'p', 'abs_p']
            X = data_pressure['x'].values
            Y = data_pressure['y'].values
            X = np.unique(X)  ;   n_x = len(X)
            Y = np.unique(Y)  ;   n_y = len(Y)
            tab_P = np.zeros((n_x, n_y))
            tab_U = np.zeros((n_x, n_y))
            tab_V = np.zeros((n_x, n_y))
            tab_dPdx = np.zeros((n_x, n_y))
            tab_dPdy = np.zeros((n_x, n_y))

            # On remplis les tableaux 2D
            for index, row in data_pressure.iterrows():
                j = np.where(X == row['x'])[0][0] # on constitue le tableau contenant les indices des x et on prend le premier élément
                i = np.where(Y == row['y'])[0][0] 
                i = n_y - i - 1                   # on inverse l'axe y pour avoir l'origine en bas à gauche
                tab_P[i,j] = row['p']
                tab_U[i,j] = row['u']
                tab_V[i,j] = row['v']
                tab_dPdx[i,j] = row['dpdx']
                tab_dPdy[i,j] = row['dpdy']
            
            self.l_X.append(X)
            self.l_Y.append(Y)
            self.l_P.append(tab_P)
            self.l_dPdx.append(tab_dPdx)
            self.l_dPdy.append(tab_dPdy)
            self.time.append(n*1.80e-4)

    def calculer_force(self, X, Y, P, MP):
        ### D'abord on trouve le contour de la partie mobile ###
        centre = [0,0] ; perimetre = 0
        contour = []
        for k, p_mp in enumerate(MP):
            [i,j] = p_mp
            voisins = Fu.voisinnage(P, i,j, large = False)
            for voisin in voisins:
                P_voisin = P[voisin[1], voisin[0]]
                if not isnan(P_voisin) and (voisin not in contour):
                    centre[0] += i ; centre[1] += j ; perimetre += 1
                    contour.append(voisin)                   # on obtient ainsi le contour extérieur de la partie mobile
        centre = [centre[0]/perimetre, centre[1]/perimetre]  # on obtient le centre de la partie mobile pour le calcul des moments
        distances = [Fu.distance(centre, point) for point in contour]

        ### On calcule les forces locales et leurs directions ###
        forces_locales = []
        for point in contour:
            f = [0,0]
            [i,j] = point
            voisins = Fu.voisinnage(P, i,j, large = False)
            # On cherche la direction de la force air --> pièce mobile
            for k,voisin in enumerate(voisins):
                if isnan(P[voisin[1], voisin[0]]):
                    if k==0:
                        f[0]-=1  # si un pixel de pièce mobile est à gauche, composante x négative
                    if k==2:
                        f[0]+=1  # si un pixel de pièce mobile est à droite, composante x positive
                    if k==1:
                        f[1]+=1  # si un pixel de pièce mobile est en haut, composante y positive
                    if k==3:
                        f[1]-=1  # si un pixel de pièce mobile est en bas, composante y négative
            # On calcule la force
            norme = sqrt(f[0]**2 + f[1]**2)
            Intensite = P[j,i]*self.dl/norme
            f = [Intensite*f[0], Intensite*f[1]]
            forces_locales.append(f)
        
        ### On intègre pour calculer les efforts totaux ###
        Resultante = [0,0]
        Moment = 0
        for k,point in enumerate(contour):
            f = forces_locales[k]
            d = distances[k]
            Resultante[0] += f[0]
            Resultante[1] += f[1]

            theta = atan2(point[1]-centre[1], point[0]-centre[0])
            f_ortho = f[1]*cos(theta) - f[0]*sin(theta)
            Moment += f_ortho*d

        return Resultante, Moment, centre, contour, forces_locales
    



            
class LObjetUnique:
    """
    Un objet pour les gouverner tous.
    Un objet pour les trouver.
    Un objet pour les amener tous et dans les ténèbres, les lier.
    """

    def __init__(self, path):
        variable = 0
        
        








if __name__ == "__main__":
    print("Hello World")


