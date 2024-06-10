# 14/03/2024, 09/04/2024
# Permet de calculer le barycentre de la vorticite a partir des .csv
# echelle en mm pour les distances

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
# importing shutil module
import shutil
from scipy import signal


start = t.time()

params={'legend.fontsize':'20','figure.figsize':(18,10),'axes.labelsize':'20','axes.titlesize':'20','xtick.labelsize':'20','ytick.labelsize':'20'}
pylab.rcParams.update(params)

###############################


Date='230427' #'230613'
Temps='092249' #'100628'


#path='E:\\sillage_sphere14\plans_YZ\\2024_plansYZ\Recording_Date='+Date+r'_Time='+Temps+r'\\'+Date+'_Time='+Temps+'_csv\\'

path='E:\\sillage_cube12\Plans_YZ\Recording_Date='+Date+r'_Time='+Temps+r'\\'+Date+'_Time='+Temps+'_csv\\'


print(path)

optionV=0 # 0 si donnees de vitesse, 1 si donnees de vorticite

fs=10 # frequence en Hertz

nid=1 #1ere image traitee
nif=4200 #derniere image traitee
nstep=1 # ecart entre 2 images

# selection de la region d'interet
yy0=10
yy1=40
xx0=10
xx1=40

pint2mm=3.73/2 # conversion des pixels de la fenetre d'interrogation vers des mm

# Choix des images tracees et enregistrees
nif_disp=30 #derniere image representee
debugimg=0 # 1 pour tracer toutes les images, 0 pour tracer seulement la vorticite

# limite pour les frequences (en Hz) pour le trace du periodogramme
flim0=0
flim1=1

ordrefiltre=6
frequence_coupure=1

# nombre de points exclus de chaque cotes pour la representation des donnees filtrees
nbexc=10 



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

    data1=np.loadtxt(path+filename(nid)+'_v.csv', delimiter=',') # vitesse u en realite
    data2=-np.loadtxt(path+filename(nid)+'_u.csv', delimiter=',') # vitesse v en realite

    # restriction des donnees a une zone
    data1=data1[yy0:yy1,xx0:xx1]
    data2=data2[yy0:yy1,xx0:xx1]

    # calcul de la vorticite
    [du_dy, du_dx] = np.gradient(data1)
    du_dy=-du_dy
    [dv_dy, dv_dx] = np.gradient(data2)
    dv_dy=-dv_dy
    vorticite=dv_dx-du_dy

elif optionV==1:
    vorticite=np.loadtxt(path+filename(nid)+'_vorticite.csv', delimiter=',')



G=barycentre(vorticite)
# print(G)

x = np.arange(np.shape(vorticite)[1])  
y = np.arange(np.shape(vorticite)[0])
fig0, ax0 = plt.subplots()
im=ax0.pcolormesh(x, y, abs(vorticite), shading='nearest', vmin=vorticite.min(), vmax=vorticite.max(),cmap=plt.set_cmap('RdYlBu'))
fig0.colorbar(im, ax=ax0)
plt.plot(G[0],G[1],'ok')
plt.title('Valeur absolue de la vorticite, unité arbitraire image '+str(nid))
plt.show()


fig0, ax0 = plt.subplots()
im=ax0.pcolormesh(x, y, np.flipud(vorticite), shading='nearest', vmin=vorticite.min(), vmax=vorticite.max(),cmap=plt.set_cmap('RdYlBu'))
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

            data1=np.loadtxt(path+filename(i)+'_v.csv', delimiter=',') # vitesse u en realite
            data2=-np.loadtxt(path+filename(i)+'_u.csv', delimiter=',') # vitesse v en realite

            # restriction des donnees a une zone
            data1=data1[yy0:yy1,xx0:xx1]
            data2=data2[yy0:yy1,xx0:xx1]

            # calcul de la vorticite
            [du_dy, du_dx] = np.gradient(data1)
            du_dy=-du_dy
            [dv_dy, dv_dx] = np.gradient(data2)
            dv_dy=-dv_dy
            vorticite=dv_dx-du_dy


    elif optionV==1:
        vorticite=np.loadtxt(path+filename(i)+'_vorticite.csv', delimiter=',')

    G=barycentre(vorticite)
    Gimage[kk,:]=G

    if i<nif_disp:

        x = np.arange(np.shape(vorticite)[1])  
        y = np.arange(np.shape(vorticite)[0])
        fig1, ax1 = plt.subplots()
        im=ax1.pcolormesh(x, y, abs(vorticite), shading='nearest', vmin=vorticite.min(), vmax=vorticite.max(), cmap=plt.set_cmap('RdYlBu'))
        plt.plot(G[0],G[1],'ok')
        plt.title('Valeur absolue de la vorticite, unité arbitraire, image '+str(i))
        fig1.colorbar(im, ax=ax1)
        plt.savefig(pathresultat+filename(i)+'_abs_vorticite.jpg')
        #plt.savefig(pathresultat+filename(i)+'_abs_vorticite.svg')
        plt.close(fig1)
        
        
        fig1, ax1 = plt.subplots()
        vorticitedim=np.flipud(vorticite)/pint2mm*1000
        im=ax1.pcolormesh(x, y,vorticitedim , shading='nearest', vmin=vorticitedim.min(), vmax=vorticitedim.max(), cmap=plt.set_cmap('RdYlBu'))
        plt.plot(G[0],G[1],'ok')
        plt.title('Vorticite image '+str(i))
        fig1.colorbar(im, ax=ax1)
        plt.savefig(pathresultat+filename(i)+'_vorticite.jpg')
        plt.close(fig1)
        
        if debugimg==1:
        
            fig1, ax1 = plt.subplots()
            im=ax1.pcolormesh(x, y, np.flipud(data1), shading='nearest', vmin=data1.min(), vmax=data1.max(), cmap=plt.set_cmap('RdYlBu'))
            plt.plot(G[0],G[1],'ok')
            plt.title('Vitesse u '+str(i))
            fig1.colorbar(im, ax=ax1)
            plt.savefig(pathresultat+filename(i)+'_vitesse_u.jpg')
            plt.close(fig1)
        
            fig1, ax1 = plt.subplots()
            im=ax1.pcolormesh(x, y, np.flipud(data2), shading='nearest', vmin=data2.min(), vmax=data2.max(), cmap=plt.set_cmap('RdYlBu'))
            plt.plot(G[0],G[1],'ok')
            plt.title('Vitesse v '+str(i))
            fig1.colorbar(im, ax=ax1)
            plt.savefig(pathresultat+filename(i)+'_vitesse_v.jpg')
            plt.close(fig1)
        
            fig1, ax1 = plt.subplots()
            im=ax1.pcolormesh(x, y, np.flipud(du_dx), shading='nearest', vmin=du_dx.min(), vmax=du_dx.max(), cmap=plt.set_cmap('RdYlBu'))
            plt.plot(G[0],G[1],'ok')
            plt.title('du_dx '+str(i))
            fig1.colorbar(im, ax=ax1)
            plt.savefig(pathresultat+filename(i)+'_du_dx.jpg')
            plt.close(fig1)
        
            fig1, ax1 = plt.subplots()
            im=ax1.pcolormesh(x, y, np.flipud(du_dy), shading='nearest', vmin=du_dy.min(), vmax=du_dy.max(), cmap=plt.set_cmap('RdYlBu'))
            plt.plot(G[0],G[1],'ok')
            plt.title('du_dy '+str(i))
            fig1.colorbar(im, ax=ax1)
            plt.savefig(pathresultat+filename(i)+'_du_dy.jpg')
            plt.close(fig1)
        
            fig1, ax1 = plt.subplots()
            im=ax1.pcolormesh(x, y, np.flipud(dv_dx), shading='nearest', vmin=dv_dx.min(), vmax=dv_dx.max(), cmap=plt.set_cmap('RdYlBu'))
            plt.plot(G[0],G[1],'ok')
            plt.title('dv_dx '+str(i))
            fig1.colorbar(im, ax=ax1)
            plt.savefig(pathresultat+filename(i)+'_dv_dx.jpg')
            plt.close(fig1)
        
            fig1, ax1 = plt.subplots()
            im=ax1.pcolormesh(x, y, np.flipud(dv_dy), shading='nearest', vmin=dv_dy.min(), vmax=dv_dy.max(), cmap=plt.set_cmap('RdYlBu'))
            plt.plot(G[0],G[1],'ok')
            plt.title('dv_dy '+str(i))
            fig1.colorbar(im, ax=ax1)
            plt.savefig(pathresultat+filename(i)+'_dv_dy.jpg')
            plt.close(fig1)
        

# Copie des donnees de la position du barycentre
df = pd.DataFrame({'Gx':Gimage[:,0],'Gy':Gimage[:,1]})
with pd.ExcelWriter(pathresultat+'Barycentre.xlsx') as writer:
    df.to_excel(writer)
# Trace de la position du barycentre
fig0, ax0 = plt.subplots()
plt.plot(Gimage[:,0],Gimage[:,1],marker='o',linestyle='--',color='k', markersize=2, linewidth=0.2)
plt.title('Barycentre de la valeur absolue de la vorticite, images '+str(nid)+' a '+ str(nif))
plt.ylabel(r'$G_y$ (pixel)')
plt.xlabel(r'$G_x$ (pixel)')
plt.axis('equal')
fig0.tight_layout()
plt.savefig(pathresultat+'Barycentre.jpg')
plt.savefig(pathresultat+'Barycentre.svg')
plt.show()

fig1, ax1 = plt.subplots()
plt.plot(Gimage[:,0]*pint2mm,Gimage[:,1]*pint2mm,marker='o',linestyle='--',color='k', markersize=2, linewidth=0.2)
plt.title('Barycentre de la valeur absolue de la vorticite, images '+str(nid)+' a '+ str(nif))
plt.ylabel(r'$G_y$ (mm)')
plt.xlabel(r'$G_x$ (mm)')
plt.axis('equal')
fig0.tight_layout()
plt.savefig(pathresultat+'Barycentre_mm.jpg')
plt.savefig(pathresultat+'Barycentre_mm.svg')
plt.show()

sos =  signal.butter(ordrefiltre, frequence_coupure, 'lp', analog=False,fs=fs, output='sos')
Gx_filtered = signal.sosfiltfilt(sos, Gimage[:,0])

fig1, ax1 = plt.subplots()
plt.plot(Gimage[:,0],marker='o',linestyle='--',color='k', markersize=2, linewidth=0.2)
plt.plot(Gx_filtered,marker='d',linestyle=':',color='g', markersize=2, linewidth=0.2)
plt.title('Gx barycentre de la valeur absolue de la vorticite, images '+str(nid)+' a '+ str(nif))
plt.ylabel(r'$G_x$ (pix)')
#plt.xlabel(r'$G_x$ (mm)')
#plt.axis('equal')
fig0.tight_layout()
plt.savefig(pathresultat+'Barycentre_Gx.jpg')
#plt.savefig(pathresultat+'Barycentre_Gx_mm.svg')
plt.show()


sos = signal.bessel(ordrefiltre, frequence_coupure, 'lp', analog=False,fs=fs, output='sos')
Gy_filtered = signal.sosfiltfilt(sos, Gimage[:,1])

fig1, ax1 = plt.subplots()
plt.plot(Gimage[:,1],marker='o',linestyle='--',color='k', markersize=2, linewidth=0.2)
plt.plot(Gy_filtered,marker='d',linestyle=':',color='g', markersize=2, linewidth=0.2)
plt.title('Gy barycentre de la valeur absolue de la vorticite, images '+str(nid)+' a '+ str(nif))
plt.ylabel(r'$G_y$ (pix)')
#plt.xlabel(r'$G_x$ (mm)')
#plt.axis('equal')
fig0.tight_layout()
plt.savefig(pathresultat+'Barycentre_Gy.jpg')
#plt.savefig(pathresultat+'Barycentre_Gx_mm.svg')
plt.show()

fig4, ax4 = plt.subplots()
plt.plot(Gx_filtered[10:-10],Gy_filtered[10:-10],marker='o',linestyle='--',color='g', markersize=2, linewidth=0.2)
plt.title('Barycentre de la valeur absolue de la vorticite, filtre, images '+str(nid)+' a '+ str(nif))
plt.ylabel(r'$G_y$ (pixel)')
plt.xlabel(r'$G_x$ (pixel)')
plt.axis('equal')
fig0.tight_layout()
plt.savefig(pathresultat+'Barycentre_filtre.jpg')
plt.savefig(pathresultat+'Barycentre_filtre.svg')
plt.show()



# Calcul du periodogramme de la position du barycentre selon x
fx, Pxx_den = signal.periodogram(Gimage[:,0], fseff)

df = pd.DataFrame({'fx':fx,'Pxx_den':Pxx_den})
with pd.ExcelWriter(pathresultat+'FTT_Gx.xlsx') as writer:
    df.to_excel(writer)

# Calcul du periodogramme de la position du barycentre selon x, signal filtre
fxf, Pxx_denf = signal.periodogram(Gx_filtered, fseff)

# Trace du spectrogramme calcule
fig2 = plt.figure(figsize=(17,6))
plt.plot(fx, Pxx_den,'bx--')
plt.plot(fxf, Pxx_denf,'gd:')
plt.ylabel('Densité spectrale')
plt.xlabel('Fréquence (Hz)')
plt.title('Gx')
plt.legend()
plt.grid()
plt.xlim([flim0,flim1])
fig2.tight_layout()
plt.savefig(pathresultat+'Gx_periodogramme.jpg')
plt.savefig(pathresultat+'Gx_periodogramme.svg')
plt.show()


# Calcul du periodogramme de la position du barycentre selon y
fy, Pyy_den = signal.periodogram(Gimage[:,1], fseff)

df = pd.DataFrame({'fy':fy,'Pyy_den':Pyy_den})
with pd.ExcelWriter(pathresultat+'FTT_Gy.xlsx') as writer:
    df.to_excel(writer)


# Trace du spectrogramme calcule
fig3 = plt.figure(figsize=(17,6))
plt.plot(fy, Pyy_den,'bx--')
plt.ylabel('Densité spectrale')
plt.xlabel('Fréquence (Hz)')
plt.title('Gy')
plt.legend()
plt.grid()
plt.xlim([flim0,flim1])
fig3.tight_layout()
plt.savefig(pathresultat+'Gy_periodogramme.jpg')
plt.savefig(pathresultat+'Gy_periodogramme.svg')
plt.show()



#--------------------------------
# La boucle sur les images, pour verification de la position du barycentre filtre
kk=-1

for i in np.arange(nid,nif,nstep).astype(int):

    kk=kk+1
    
    if  (i> nbexc)*(i<nif_disp) :

        x = np.arange(np.shape(vorticite)[1])  
        y = np.arange(np.shape(vorticite)[0])
        fig1, ax1 = plt.subplots()
        im=ax1.pcolormesh(x, y, abs(vorticite), shading='nearest', vmin=vorticite.min(), vmax=vorticite.max(), cmap=plt.set_cmap('RdYlBu'))
        plt.plot(Gimage[i,0],Gimage[i,1],'ok')
        plt.plot(Gx_filtered[i],Gy_filtered[i],'dg')
        plt.title('Valeur absolue de la vorticite, unité arbitraire, image '+str(i))
        fig1.colorbar(im, ax=ax1)
        plt.savefig(pathresultat+filename(i)+'_abs_vorticite_avec_G_filtre.jpg')
        #plt.savefig(pathresultat+filename(i)+'_abs_vorticite.svg')
        plt.close(fig1)
        


# copie du programme
shutil.copyfile('./P2bis_detection_barycentre_v05.py',pathresultat+'/P2bis_detection_barycentre_v05.py')



#-------------------------------------------------------------------        
end = t.time()
print("temps passé: ", end - start)
