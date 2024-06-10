# 15/06/2022
# calcule l'amplitude de la frequence principale et de la basse frequence, en moyennant les valeurs sur  les pixels du .csv
# Important : la valeur de la frequence principale et de la basse frequence doivent etre donnees par l'utilisateur. Si elles sont fausses, le resulat sera faux

#from lvreader import read_buffer
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import matplotlib.pylab as pylab
import time
#from pydmd import SpDMD,DMD, HODMD

start = time.time()

params={'legend.fontsize':'20','figure.figsize':(13,19),'axes.labelsize':'20','axes.titlesize':'20','xtick.labelsize':'20','ytick.labelsize':'20'}
pylab.rcParams.update(params)


#--------------------------------------------
# DEBUT DES PARAMETRES

Date='230322'
Temps='120753'
fs=10 # frequence en Hertz

nid=1 #1ere image traitee
nif=4200 #derniere image traitee

# Frequence8s principales et basse frequence
f1=0.131
f2=0.026197

#FIN DES PARAMETRES
#--------------------------------------------

#---------------
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


# Chemin vers les .csv

path='E:\\sillage_cube24\plans_YZ\\2302_X\Recording_Date='+Date+r'_Time='+Temps+r'\\'+Date+'_Time='+Temps+'_csv\\'

#path='E:\\sillage_sphere14\plans_YZ\\2301_obj85_X\Recording_Date='+Date+r'_Time='+Temps+r'\\'+Date+'_Time='+Temps+'_csv\\'

#path=r'D:\\sillage_sphere24\plans_YZ\Project_Piv_220510_124821\Recording_Date='+Date+r'_Time='+Temps+r'\\'+Date+r'_Time='+Temps+r'_csv\\'

#path='E:\\sillage_sphere24\plans_YZ\\2209_x6\Recording_Date='+Date+r'_Time='+Temps+r'\\'+Date+'_Time='+Temps+'_csv\\'

#path=r'G:\\2022_plansYZ\\2205_x6\\Recording_Date='+Date+'_Time='+Temps+'\\'+Date+'_Time='+Temps+'_csv\\'

#Hpath='E:\\sillage_sphere14\plans_YZ\\2211x_obj70\Recording_Date='+Date+r'_Time='+Temps+r'\\'+Date+'_Time='+Temps+'_csv\\'


#==========================================

#text_file = open(r'D:\\sillage_sphere24\plans_YZ\Project_Piv_220510_124821\Recording_Date='+Date+r'_Time='+Temps+r'\\'+Date+r'_Time='+Temps+r'_csv\\'+'0_valeurs_frequence.txt', "w")

#text_file = open(r'E:\sillage_sphere24\plans_YZ\\2209_x6\Recording_Date='+Date+r'_Time='+Temps+r'\\'+Date+'_Time='+Temps+'_csv\\'+'0_valeurs_frequence.txt', "w")

#text_file = open(r'G:\\2022_plansYZ\\2205_x6\Recording_Date='+Date+r'_Time='+Temps+r'\\'+Date+r'_Time='+Temps+r'_csv\\'+'0_valeurs_frequence.txt', "w")

#text_file = open(r'E:\sillage_sphere14\plans_YZ\\2301_obj85_X\Recording_Date='+Date+r'_Time='+Temps+r'\\'+Date+'_Time='+Temps+'_csv\\'+'0_valeurs_frequence.txt', "w")

text_file = open(r'E:\sillage_cube24\plans_YZ\\2302_X\Recording_Date='+Date+r'_Time='+Temps+r'\\'+Date+'_Time='+Temps+'_csv\\'+'0_valeurs_frequence.txt', "w")

                 

text_file.write('Frequence principale='+str(f1)+', basse frequence='+str(f2)+'\n')

#-------------------------------------------#
# Composante u
#-------------------------------------------#

#--------------------------------
# Trace de la premiere image pour verification, v uniquement pour l'instant
data=np.loadtxt(path+'B00001_u.csv', delimiter=',')

#plt.subplot(211)
fig1 = plt.figure(figsize=(17,6))
plt.pcolor(data,cmap='RdYlBu')
plt.colorbar()
plt.show()

# Creation du tableau (temps,(image) )
VV=np.zeros((nif-nid,np.shape(data)[0],np.shape(data)[1]))
print(VV.shape)

text_file.write('Dimension du tableau u :'+str(VV.shape)+'\n')

# Pour la frequence principale
Pic1=np.zeros((np.shape(data)[0],np.shape(data)[1]))
# Pour la basse frequence
Pic2=np.zeros((np.shape(data)[0],np.shape(data)[1]))

# Boucle sur les images, chargement des donnees du csv
for i in range(nid,nif):
    
    data=np.loadtxt(path+filename(i)+'_u.csv', delimiter=',')
    VV[i-1,:,:]=data


# Calcul du spectrogram sur un exemple
fx, Pxx_den = signal.periodogram(VV[:,20,17], fs)

# determination du numero du tableau correspondant a la frequence principale et a la basse frequence
f1x=f1+np.zeros(fx.shape)
af1=np.argmin((fx-f1x)**2)
f2x=f2+np.zeros(fx.shape)
af2=np.argmin((fx-f2x)**2)
#print(af1)
#print(Pxx_den[af1])

# Trace du spectrogramme calcule comme exemple
fig2 = plt.figure(figsize=(17,6))
plt.plot(fx, Pxx_den,'bx--')
plt.ylabel('Densité spectrale')
plt.title('Vitesse moyenne selon x dans un rectangle')
plt.legend()
plt.grid()
plt.show()

# boucle sur les sous-domaines, pour 
# - calculer le spectrogram
# - obtenir une amplitude de la frequence principale et de la basse frequence  
for i in range(0,np.shape(data)[0]):
    for j in range(0,np.shape(data)[1]):

        fx, Pxx_den = signal.periodogram(VV[:,i,j], fs)
# Modif
        Pic1[i,j]=Pxx_den[af1]+Pxx_den[af1-1]+Pxx_den[af1+1]
        Pic2[i,j]=Pxx_den[af2]+Pxx_den[af2-1]+Pxx_den[af2+1]

# Image de l'amplitude de la frequence principale
fig = plt.figure(figsize=(17,6))
plt.pcolor(Pic1,cmap='RdYlBu')
plt.title('Frequence principale u')
plt.colorbar()
plt.savefig(path+'\\'+str(nif)+'_f_principale_u.pdf')
plt.show()

A1=np.average(Pic1)
print("Moyenne sur les pixels du .csv de la puissance de la frequence principale, u : ",A1)
text_file.write("Moyenne sur les pixels du .csv de la puissance de la frequence principale, u : "+str(A1)+'\n')

# Image de l'amplitude de la basse frequence
fig = plt.figure(figsize=(17,6))
plt.pcolor(Pic2,cmap='RdYlBu')
plt.title('Basse frequence u')
plt.colorbar()
plt.savefig(path+'\\'+str(nif)+'_basse_f_u.pdf')
plt.show()

A2=np.average(Pic2)
print("Moyenne sur les pixels du .csv  de la puissance de la basse frequence, u : ",A2)
text_file.write("Moyenne sur les pixels du .csv  de la puissance de la basse frequence, u : "+str(A2)+'\n')

#-------------------------------------------#
# Composante v
#-------------------------------------------#

#--------------------------------
# Trace de la premiere image pour verification, u uniquement pour l'instant
data=np.loadtxt(path+'B00001_v.csv', delimiter=',')

#plt.subplot(211)
fig1 = plt.figure(figsize=(17,6))
plt.pcolor(data,cmap='RdYlBu')
plt.colorbar()
plt.show()

# Creation du tableau (temps,(image) )
VVv=np.zeros((nif-nid,np.shape(data)[0],np.shape(data)[1]))
print(VVv
      .shape)

text_file.write('Dimension du tableau v :'+str(VVv.shape)+'\n')

# Pour la frequence principale
Picv1=np.zeros((np.shape(data)[0],np.shape(data)[1]))
# Pour la basse frequence
Picv2=np.zeros((np.shape(data)[0],np.shape(data)[1]))

# Boucle sur les images, chargement des donnees du csv
for i in range(nid,nif):
    
    data=np.loadtxt(path+filename(i)+'_v.csv', delimiter=',')
    VVv[i-1,:,:]=data


# Calcul du spectrogram sur un exemple
fy, Pyy_den = signal.periodogram(VVv[:,20,17], fs)

# determination du numero du tableau correspondant a la frequence principale et a la basse frequence
f1y=f1+np.zeros(fy.shape)
af1=np.argmin((fy-f1y)**2)
f2y=f2+np.zeros(fy.shape)
af2=np.argmin((fy-f2y)**2)
#print(af1)
#print(Pxx_den[af1])

# Trace du spectrogramme calcule comme exemple
fig2 = plt.figure(figsize=(17,6))
plt.plot(fy, Pyy_den,'bx--')
plt.ylabel('Densité spectrale')
plt.title('Vitesse moyenne selon x dans un rectangle')
plt.legend()
plt.grid()
plt.show()

# boucle sur les sous-domaines, pour 
# - calculer le spectrogram
# - obtenir une amplitude de la frequence principale et de la basse frequence  
for i in range(0,np.shape(data)[0]):
    for j in range(0,np.shape(data)[1]):

        fy, Pyy_den = signal.periodogram(VVv[:,i,j], fs)
# Modif
        Picv1[i,j]=Pyy_den[af1]+Pyy_den[af1-1]+Pyy_den[af1+1]
        Picv2[i,j]=Pyy_den[af2]+Pyy_den[af2-1]+Pyy_den[af2+1]

# Image de l'amplitude de la frequence principale
fig = plt.figure(figsize=(17,6))
plt.pcolor(Picv1,cmap='RdYlBu')
plt.title('Frequence principale')
plt.colorbar()
plt.savefig(path+'\\'+str(nif)+'_f_principale_v.pdf')
plt.show()

Av1=np.average(Picv1)
print("Moyenne sur les pixels du .csv de la puissance de la frequence principale, v : ",Av1)
text_file.write("Moyenne sur les pixels du .csv de la puissance de la frequence principale, v : "+str(Av1)+'\n')

# Image de l'amplitude de la basse frequence
fig = plt.figure(figsize=(17,6))
plt.pcolor(Picv2,cmap='RdYlBu')
plt.title('Basse frequence v')
plt.colorbar()
plt.savefig(path+'\\'+str(nif)+'_basse_f_v.pdf')
plt.show()

Av2=np.average(Picv2)
print("Moyenne sur les pixels du .csv  de la puissance de la basse frequence, v : ",Av2)
text_file.write("Moyenne sur les pixels du .csv  de la puissance de la basse frequence, v : "+str(Av2)+'\n')

#-------------------------------------------#
# Fermeture du fichier de resultat texte
text_file.close()

end = time.time()
print("temps passé: ", end - start)


