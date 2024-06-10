# 17/01/2024
# Permet de calculer les taux de turbulence dans le canal

#from lvreader import read_buffer
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import matplotlib.pylab as pylab
import numpy.ma as ma
import os
import time as t 
import pandas as pd
import openpyxl


start = t.time()

params={'legend.fontsize':'20','figure.figsize':(18,10),'axes.labelsize':'20','axes.titlesize':'20','xtick.labelsize':'20','ytick.labelsize':'20'}
pylab.rcParams.update(params)

###############################

Date='230420'
Temps='113359'

#path=r'/mnt/6a509136-da71-4a58-b0d8-02e60d6215a1/01_Recherche/Canal_sillage/#Python_Lavision_donnees/'+Date+'_Time='+Temps+'_csv/'

#path='E:\\sillage_sphere24\plans_YZ\\2209_x6\Recording_Date='+Date+r'_Time='+Temps+r'\\'+Date+'_Time='+Temps+'_csv\\'

path='E:\\sillage_cube12\Plans_YZ\\Recording_Date='+Date+r'_Time='+Temps+r'\\'+Date+'_Time='+Temps+'_csv\\'

#path=r'E:\\caracterisation_canal\Project_Piv_231205_092745\Recording_Date='+Date+r'_Time='+Temps+'\\'


#path=path+Date+'_Time='+Temps+'_csv\\'
print(path)


fs=10 # frequence en Hertz

nid=1 #1ere image traitee
nif=600 #derniere image traitee

# selection de la region d'interet
xx0=1
xx1=100
yy0=10
yy1=50

u='u'
v='v'


#FIN DES PARAMETRES
#--------------------------------------------

def filename(i):
#for i  in range(1,99,1):    
    if i<10:
        return('B0000'+str(i))
    if i>=10 and i<100:
        return('B000'+str(i))
    if i>=100 and i<1000:
        return('B00'+str(i))
    if i>=1000 and i<10000:
        return('B0'+str(i))

def decoupage(n,m,df):
    """coupe l'image en n*m parties (n lignes et m colonnes) et renvoie un tableau contenant la moyenne des vitesses dans chaque partie, un selon x et un selon y"""
    ma_arr=ma.asarray(df)
    arr=ma_arr.data
    (ligne,colonne)=np.shape(arr)
    tab=np.empty((n,m),dtype=np.ndarray)
    step_n=ligne//n
    step_m=colonne//m
    for i in range(n):
        for j in range(m):
            tab[i][j]=np.mean(arr[i*step_n:(i+1)*step_n,j*step_m:(j+1)*step_m])
    return(tab)

#--------------------------------
pathresultat=path+Date+'_'+Temps+'resultats/'

# creation du dossier de resultats
if not os.path.exists(pathresultat):
    os.makedirs(pathresultat)

# Trace de la premiere image pour verification, v uniquement pour l'instant
data=np.loadtxt(path+'B00001_u.csv', delimiter=',')
#
fig1 = plt.figure(figsize=(17,6))
plt.pcolor(data,cmap='RdYlBu')
plt.title('B00001_u, direction perpendiculaire')
plt.colorbar()
plt.savefig(pathresultat+'B00001_u.png')
plt.savefig(pathresultat+'B00001_u.svg')
plt.show()

# Trace de la premiere image pour verification, v uniquement pour l'instant
data=np.loadtxt(path+'B00001_v.csv', delimiter=',')
#
fig1 = plt.figure(figsize=(17,6))
plt.pcolor(data,cmap='RdYlBu')
plt.title('B00001_v, direction principale')
plt.colorbar()
plt.savefig(pathresultat+'B00001_v.png')
plt.savefig(pathresultat+'B00001_v.svg')
plt.show()

# Domaine reduit
datar=data[yy0:yy1,xx0:xx1]
#
fig1 = plt.figure(figsize=(17,6))
plt.pcolor(datar,cmap='RdYlBu')
plt.title('B00001_v,  direction principale, domaine reduit')

plt.colorbar()
plt.show()

#--------------------------------
# Boucle sur les images

VV=np.zeros((nif-nid,np.shape(datar)[0],np.shape(datar)[1]))
VVv=np.zeros((nif-nid,np.shape(datar)[0],np.shape(datar)[1]))
print(VVv.shape)


# Chargement des csv et remplissage des tableaux VV et VVv
for i in range(nid,nif):
    data=np.loadtxt(path+filename(i)+'_'+u+'.csv', delimiter=',')
    VV[i-1,:,:]=data[yy0:yy1,xx0:xx1]
    data=np.loadtxt(path+filename(i)+'_'+v+'.csv', delimiter=',')
    VVv[i-1,:,:]=data[yy0:yy1,xx0:xx1]

Vavg=np.average(VVv, axis=0)
print(Vavg.shape)
Vstd=np.std(VVv, axis=0)



# Vitesse principale, valeur moyenne
fig1 = plt.figure(figsize=(17,6))
plt.pcolor(Vavg,cmap='RdYlBu')
plt.title('Vavg, valeur moyenne en temps, direction principale, domaine reduit')
plt.colorbar()
plt.savefig(pathresultat+'Vavg.png')
plt.savefig(pathresultat+'Vavg.svg')
plt.show()
# Vitesse principale, ecart-type
fig1 = plt.figure(figsize=(17,6))
plt.pcolor(Vstd,cmap='RdYlBu')
plt.title('Vstd, ecart-type en temps, direction principale, domaine reduit')
plt.colorbar()
plt.savefig(pathresultat+'Vstd.png')
plt.savefig(pathresultat+'Vstd.svg')
plt.show()

Uavg=np.average(VV, axis=0)
Ustd=np.std(VV, axis=0)

# Vitesse perpendiculaire, valeur moyenne
fig1 = plt.figure(figsize=(17,6))
plt.pcolor(Uavg,cmap='RdYlBu')
plt.title('Uavg, valeur moyenne en temps, direction perpendiculaire, domaine reduit')
plt.colorbar()
plt.savefig(pathresultat+'Uavg.png')
plt.savefig(pathresultat+'Uavg.svg')
plt.show()
# Vitesse perpendiculaire, ecart-type
fig1 = plt.figure(figsize=(17,6))
plt.pcolor(Ustd,cmap='RdYlBu')
plt.title('Ustd, ecart-type en temps, direction principale, domaine reduit')
plt.colorbar()
plt.savefig(pathresultat+'Ustd.png')
plt.savefig(pathresultat+'Ustd.svg')
plt.show()


Iturb=np.sqrt(Vstd**2+2*Ustd**2)/Vavg

# Intensite de la turbulence
fig1 = plt.figure(figsize=(17,6))
plt.pcolor(Iturb,cmap='RdYlBu')
plt.title('Iturb, Intensite de la turbulence')
plt.colorbar()
plt.savefig(pathresultat+'Iturb.png')
plt.savefig(pathresultat+'Iturb.svg')
plt.show()

Iturbavg=np.average(Iturb)
print("Intensite de la turbulence : ",Iturbavg)
np.savetxt(pathresultat+"Intensite_turbulence.csv",np.array([Iturbavg]), delimiter=",")


# --- --- --- ---
end = t.time()
print("temps passé : ", end - start,"s")

