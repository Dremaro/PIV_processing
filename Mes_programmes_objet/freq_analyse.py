import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import deque

import fonctions_utiles as fu


with open('F_x.pkl', 'rb') as f:
    L_forces = pickle.load(f)




l_frq = [60,65,70,73,76,79,82.5,85,88,95]
times = [[t*1.8e-4 for t in range(0,len(F_x))] for F_x in L_forces]
colors = cm.coolwarm(np.linspace(0, 1, len(L_forces)))
i = 10
for i in range(0, len(L_forces)-1):
    if i != 7:
        frq = l_frq[i]
        time = times[i]
        p = time[-1]
        plt.plot(np.array(time)/p, L_forces[i], color = colors[i], label='{}Hz'.format(l_frq[i]))
plt.legend()
plt.title('X-axis Force component over time')
plt.xlabel('Time over period t/T')
plt.ylabel('Force along x-axis (N)')
plt.grid()
plt.show()



def rouler(L, shift):
    if shift > 0:
        for i in range(shift):
            last = L.pop(-1)
            L.insert(0, last)
    else:
        shift = -shift
        for i in range(shift):
            first = L.pop(0)
            L.append(first)
    return L



# Trouver l'index du maximum de la première courbe
max_index = np.argmax(L_forces[-1])

for i in range(0, len(L_forces)-1):
    if i != 7:
        # Trouver l'index du maximum de la courbe actuelle
        current_max_index = np.argmax(L_forces[i])
        
        # Calculer le décalage nécessaire pour aligner le maximum de la courbe actuelle avec celui de la première courbe
        shift = max_index - current_max_index
        
        # Décaler les éléments de la courbe et du temps
        rolled_force = rouler(L_forces[i], shift)
        
        # Normaliser le temps pour qu'il soit entre 0 et 1
        p = times[i][-1]
        normalized_time = np.array(times[i]) / p
        
        # Tracer la courbe
        plt.plot(normalized_time, rolled_force, color = colors[i], label='{}Hz'.format(l_frq[i]))

plt.legend()
plt.title('X-axis Force component over time')
plt.xlabel('Time over period t/T')
plt.ylabel('Force along x-axis (N)')
plt.grid()
plt.show()









l_max_Fx = [max(L_forces[i]) for i in range(len(L_forces)) if i != 7]
l_mean_Fx = [np.mean(L_forces[i]) for i in range(len(L_forces)) if i != 7]
plt.plot(l_frq, l_max_Fx, 'o-', label='Max Force')
plt.plot(l_frq, l_mean_Fx, 'o-', label='Mean Force')
plt.legend()
plt.title('Max and Mean X-axis Force components')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Force along x-axis (N)')
plt.grid()
plt.show()






