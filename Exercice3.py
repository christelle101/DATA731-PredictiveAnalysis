"""
Created on Wed Dec 29 19:37:18 2021

@author: nomen
"""

import random as aleas
import matplotlib.pyplot as plt
from scipy.signal import freqz
import numpy as np
import pandas as pd
import statsmodels.api as sm

###########################################################################
#               EXERCICE 3 - Identification de modèle AR
###########################################################################

"""
    Creation de trois series temporelles y1, y2, y3 par simulation stohchastique
"""
#Generation des coefficients
a = [- 0.0707, 0.2500]
b = [- 1.6674, 0.9025]
c = [1.7820, 0.8100]

#Donnees
n = 1536

#REMPLACER TOUS LES t PAR DES i
i = range(- 2, n - 1)

y = [k*0 for k in i]

#Creation des series
y1 = []
y2 = []
y3 = []

for k in range(1, int(n/3)):
    y[k] = -a[0]*y[k - 1] - a[1]*y[k - 2] + aleas.gauss(0, 1)
    y1.append(y[k])
    
for k in range(int(n/3) + 1, 2*int(n/3)):
    y[k] = -b[0]*y[k - 1] - b[1]*y[k - 2] + aleas.gauss(0, 1)
    y2.append(y[k])
    
for k in range(2*int(n/3) + 1, n):
    y[k] = -c[0]*y[k - 1] - c[1]*y[k - 2] + aleas.gauss(0, 1)
    y3.append(y[k])
    
y=y[3:]  # suppression des donnees transitoires
i=i[3:]
    
#Visualisation de la série 1
plt.plot(i[0 : int(n/3)], y[0 : int(n/3)], color = '#EC3874')
plt.grid()
plt.title("Serie 1")
plt.show()

#Visualisation de la série 2
plt.plot(i[int(n/3) + 1 : 2*int(n/3)], y[int(n/3) + 1 : 2*int(n/3)], y[0:int(n/3)])
plt.grid()
plt.title("Serie 2")
plt.show()

#Visualisation de la série 3
plt.plot(i[2*int(n/3) + 1 : n], y[2*int(n/3) + 1:n], color ='#4CAE58')
plt.grid()
plt.title("Serie 3")
plt.show()

def estimer_autocorrelation(x):
    """
        Fonction qui permet d'estimer l'autocorrelation
    """
    m = len(x)
    variance = x.var()
    x = x - x.mean()
    r = np.correlate(x, x, 'full')
    resultat = r/(variance*m)
    return resultat

y= np.array(y)
acfy = estimer_autocorrelation(y)

def spectre(*args): 
    """
    Fonction qui permet de calculer les spectres des sous-series
        Np : nombre de points du spectre
        f : recuperation des echantillons de frequence (abscisses)
        mag : hauteurs des frequences observables correspondantes (ordonnees)
    """
    Np = 256 
    f=freqz(1,args[0],Np)[0] 
    mag=[]   
    for arg in args:
        mag.append(abs(freqz(1,arg,Np)[1])) # calcul du spectre de chaque sous-serie
    return (f,mag)
 
f,mag=spectre([1]+a,[1]+b,[1]+c)
spectre1 = mag[0]
spectre2 = mag[1]
spectre3 = mag[2]
plt.semilogy(
	f,mag[0],'-g',
	f,mag[1],':b',
	f,mag[2],'--r'
)
plt.grid() 
plt.legend(['spectre1', 'spectre2','spectre3'])
plt.title("Spectres")
plt.show()

sm.graphics.tsa.plot_acf(y[0:int(n/3)+1], lags=25)
plt.title("Autocorrelation de la serie 1")
plt.grid() 
plt.show()