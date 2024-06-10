import os
import sys
sys.path.insert(0, 'C:/Users/pc1/Leviia/Documents/1_Savoir et Apprentissage/Programmation/PythonKnowledge/mes_outils')
# sys.path is a list
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp

import tkinter as tk
from PIL import Image, ImageTk


from FonctionsPerso import ImageBrowser # type: ignore




########## VISUALISATION ##########
root = tk.Tk()
folder1 = r"essais\essai_65Hz_threshold3100\01_csvimages"
# folder1 = r'essais\essai_65Hz_threshold3100\02_U_interpolated_csv\\'
# folder1 = r'essais\essai_65Hz_threshold3100\02_MP_mobile_parts\\'
# folder1 = r'essais\essai_65Hz_threshold3100\05_U_temporal_mean\\'

#folder = r"VC7_test_absspeed_csv"
browser = ImageBrowser(root, folder = folder1, show_name = True)
l = [np.random.rand(500, 500)*255 for _ in range(10)]
# browser = ImageBrowser(root, l_images=l, show_name = True)
root.mainloop()














