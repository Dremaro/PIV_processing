# 26/03/2024
# Trace la vorticite a partir des .csv, avec les changements u,v,y

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
# import skimage as ski


start = t.time()

params={'legend.fontsize':'20','figure.figsize':(18,10),'axes.labelsize':'20','axes.titlesize':'20','xtick.labelsize':'20','ytick.labelsize':'20'}
pylab.rcParams.update(params)

###############################


Date='230420'
Temps='113359'

#path='E:\\sillage_sphere14\plans_YZ\\2024_plansYZ\Recording_Date='+Date+r'_Time='+Temps+r'\\'+Date+'_Time='+Temps+'_csv\\'

path='E:\\sillage_cube12\Plans_YZ\Recording_Date='+Date+r'_Time='+Temps+r'\\'+Date+'_Time='+Temps+'_csv\\'


print(path)

optionV=0 # 0 si donnees de vitesse, 1 si donnees de vorticite

fs=10 # frequence en Hertz

nid=1 #1ere image traitee
nif=4200 #derniere image traitee
nstep=1 # ecart entre 2 images

# selection de la region d'interet
yy0=0
yy1=100
xx0=0
xx1=100

# Choix des images tracees et enregistrees
nif_disp=2 #derniere image representee

# limite pour les frequences (en Hz) pour le trace du periodogramme
flim0=0
flim1=1

# A ajouter : un coefficient pour avoir les distances en mm pour Gx

u='u'
v='v'

#FIN DES PARAMETRES
#--------------------------------------------
#--------------------------------------------
pathresultat=path+Date+'_'+Temps+'resultats\\'

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

# calcul des coordonnees du barycentre
def barycentre(vorticite):

    m1=np.ones(np.shape(vorticite))
    x1=np.cumsum(m1,axis=1)
    y1=np.cumsum(m1,axis=0)
    Gx=np.sum(x1*abs(vorticite))/np.sum(abs(vorticite))
    Gy=np.sum(y1*abs(vorticite))/np.sum(abs(vorticite))
    G=[Gx,Gy]

    return G


#--------------------------------
# Gestion des dossiers

if not os.path.exists(pathresultat):
    os.makedirs(pathresultat)

#--------------------------------
# frequence effective
fseff=fs/nstep 
    
#--------------------------------
# trace du champ pour u/ v pour la premiere image
if optionV==0:

    data1=np.loadtxt(path+filename(nid)+'_v.csv', delimiter=',')
    data2=np.loadtxt(path+filename(nid)+'_u.csv', delimiter=',')

    # restriction des donnees a une zone
    data1=data1[yy0:yy1,xx0:xx1]
    data2=data2[yy0:yy1,xx0:xx1]
    
    data1=np.flip(data1,0)
    data2=np.flip(data2,0)

    # calcul de la vorticite
    [du_dy, du_dx] = np.gradient(data1)
    [dv_dy, dv_dx] = np.gradient(data2)
    vorticite=-dv_dx-du_dy

elif optionV==1:
    vorticite=np.loadtxt(path+filename(nid)+'_vorticite.csv', delimiter=',')



G=barycentre(vorticite)
# print(G)

x = np.arange(np.shape(vorticite)[1])  
y = np.arange(np.shape(vorticite)[0])
fig0, ax0 = plt.subplots()
im=ax0.pcolormesh(x, y, vorticite, shading='nearest', vmin=vorticite.min(), vmax=vorticite.max(),cmap=plt.set_cmap('RdYlBu'))
fig0.colorbar(im, ax=ax0)
plt.plot(G[0],G[1],'ok')
plt.title('Vorticite, unité arbitraire image '+str(nid))
plt.show()


#--------------------------------
#--------------------------------
# La boucle sur les images
Gimage=np.zeros([int((nif-nid)/nstep),2]) # contiendra les coordonnees des barycentres

kk=-1

for i in np.arange(nid,nif,nstep).astype(int):

    kk=kk+1

    # chargement des donnees
    if optionV==0:

        data1=np.loadtxt(path+filename(nid)+'_v.csv', delimiter=',')
        data2=np.loadtxt(path+filename(nid)+'_u.csv', delimiter=',')

        # restriction des donnees a une zone
        data1=data1[yy0:yy1,xx0:xx1]
        data2=data2[yy0:yy1,xx0:xx1]
    
        data1=np.flip(data1,0)
        data2=np.flip(data2,0)


        # calcul de la vorticite
        [du_dy, du_dx] = np.gradient(data1)
        [dv_dy, dv_dx] = np.gradient(data2)
        vorticite=-dv_dx-du_dy

    elif optionV==1:
        vorticite=np.loadtxt(path+filename(i)+'_vorticite.csv', delimiter=',')

    G=barycentre(vorticite)
    Gimage[kk,:]=G

    if i<nif_disp:

        x = np.arange(np.shape(vorticite)[1])  
        y = np.arange(np.shape(vorticite)[0])
        fig1, ax1 = plt.subplots()
        im=ax1.pcolormesh(x, y, vorticite, shading='nearest', vmin=vorticite.min(), vmax=vorticite.max(), cmap=plt.set_cmap('RdYlBu'))
        plt.plot(G[0],G[1],'ok')
        plt.title('Vorticite, unité arbitraire, image '+str(i))
        fig1.colorbar(im, ax=ax1)
        plt.savefig(pathresultat+filename(i)+'_vorticite.jpg')
        #plt.savefig(pathresultat+filename(i)+'_vorticite.svg')
        plt.close(fig1)
        

        fig1, ax1 = plt.subplots()
        im=ax1.pcolormesh(x, y, data1, shading='nearest', vmin=vorticite.min(), vmax=vorticite.max(), cmap=plt.set_cmap('RdYlBu'))
        plt.plot(G[0],G[1],'ok')
        plt.title('data1, u , unité arbitraire, image '+str(i))
        fig1.colorbar(im, ax=ax1)
        plt.savefig(pathresultat+filename(i)+'_data1.jpg')
        #plt.close(fig1)
        
        print(i)
        print(data1[0,0])
        print(data1[40,0])
        print(data1[0,40])
        print('************************')
        
        fig1, ax1 = plt.subplots()
        im=ax1.pcolormesh(x, y, data2, shading='nearest', vmin=vorticite.min(), vmax=vorticite.max(), cmap=plt.set_cmap('RdYlBu'))
        plt.plot(G[0],G[1],'ok')
        plt.title('data2, v , unité arbitraire, image '+str(i))
        fig1.colorbar(im, ax=ax1)
        plt.savefig(pathresultat+filename(i)+'_data2.jpg')
        plt.close(fig1)

        fig1, ax1 = plt.subplots()
        im=ax1.pcolormesh(x, y, dv_dx, shading='nearest', vmin=vorticite.min(), vmax=vorticite.max(), cmap=plt.set_cmap('RdYlBu'))
        plt.plot(G[0],G[1],'ok')
        plt.title('dv_dx , unité arbitraire, image '+str(i))
        fig1.colorbar(im, ax=ax1)
        plt.savefig(pathresultat+filename(i)+'_dv_dx.jpg')
        plt.close(fig1)
        
        fig1, ax1 = plt.subplots()
        im=ax1.pcolormesh(x, y,du_dy, shading='nearest', vmin=vorticite.min(), vmax=vorticite.max(), cmap=plt.set_cmap('RdYlBu'))
        plt.plot(G[0],G[1],'ok')
        plt.title('du_dy , unité arbitraire, image '+str(i))
        fig1.colorbar(im, ax=ax1)
        plt.savefig(pathresultat+filename(i)+'_du_dy.jpg')
        plt.close(fig1)
        

#-------------------------------------------------------------------        
end = t.time()
print("temps passé: ", end - start)
