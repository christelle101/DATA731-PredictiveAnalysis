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
        #Fonction qui permet d'estimer l'autocorrelation
    """
    m = len(x)
    variance = x.var()
    x = x - x.mean()
    r = np.correlate(x, x, 'full')
    resultat = r/(variance*m)
    return resultat

y= np.array(y)
acfy = estimer_autocorrelation(y)


"""
    QUESTION 2 - Visualisation des spectres des sous-series
"""
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


"""
    QUESTION 2 - Visualisation de l'autocorrelation pour chaque serie temporelle
"""
#Visualisation de la serie 1
sm.graphics.tsa.plot_acf(y[0:int(n/3)+1], lags = 40, color = '#EC3874')
plt.title("Autocorrelation de la serie 1")
plt.grid() 
plt.show()

#Visualisation de la serie 2
sm.graphics.tsa.plot_acf(y[int(n/3)+1:2*int(n/3)], lags = 40)
plt.grid() 
plt.title("Autocorrelation de la serie 2")
plt.show()

#Visualisation de la serie 3
sm.graphics.tsa.plot_acf(y[2*int(n/3)+1:n], lags = 40, color = '#4CAE58')
plt.grid() 
plt.title("Autocorrelation de la serie 3")
plt.show()


"""
    QUESTION 3 - Creation d'une serie temporelle constituee par la somme des series
    synthetisees precedemment.
"""
#Visualisation de y
somme = []
for j in range(len(y1)):
  somme.append(y1[j] + y2[j] + y3[j])
plt.plot(range(len(y1)),somme[:])
plt.grid()
plt.title("y : somme de y1, y2 et y3")
plt.show()

#Tracé de l'autocorrélation de y
sm.graphics.tsa.plot_acf(somme, lags = 40)
plt.grid() 
plt.title("Autocorrelation de y")
plt.show()

#Tracé de la densité spectrale de puissance de y
plt.psd(somme[:])
plt.title("Densité spectrale de puissance de y")
plt.show()


"""
    QUESTION 4 - Modélisation de y par un processus AR d'ordre 2.
    L'objectif de cette etape est d'estimer les coefficients de ce modele et de comparer 
    les autocorrélations/densites spectrales de y et du modele estime.
"""
t=range(-2,n-1)

y=[k*0 for k in t]
y1 = []
y2 = []
y3 = []
for k in range(1,int(n/3)):
    y[k]=-a[0]*y[k-1]-a[1]*y[k-2]+aleas.gauss(0,1)
    y1.append(y[k])

for k in range(int(n/3)+1,2*int(n/3)):
    y[k]=-b[0]*y[k-1]-b[1]*y[k-2]+aleas.gauss(0,1)
    y2.append(y[k])

for k in range(2*int(n/3)+1,n):
    y[k]=-c[0]*y[k-1]-c[1]*y[k-2]+aleas.gauss(0,1)
    y3.append(y[k])

y=y[3:]  # suppression des donnees transitoires
t=t[3:]

def AR_model_somme(debut, fin, serie, vrai_spectre):
    """
        : parametre debut : debut de l'intervalle
        : parametre fin : fin de l'intervalle
        : parametre serie : nom de la serie à modéliser
        : parametre vrai_spectre : vrai spectre à comparer aux résultats 
        : type debut : int
        : type fin : int
        : type serie : string
        : type vrai_spectre : spectre
        : return : la serie temporelle et la comparaison entre les spectres
        : type return : plt.show
    """
    D = np.cov([
        y[debut : fin] + [0, 0, 0, 0],
        [0] + y[debut : fin] + [0, 0, 0],
        [0, 0] + y[debut : fin] + [0, 0],
        [0, 0, 0] + y[debut : fin] + [0],
        [0, 0, 0, 0] + y[debut : fin]])
    
    E = - np.linalg.inv(D[0:2, 0:2]) @ D[0, 1:3].reshape(2, 1)  # car on veut l'avoir à l'ordre 2
    H = - np.linalg.inv(D[0:3, 0:3]) @ D[0, 1:4].reshape(3, 1)  # car on veut l'avoir à l'ordre 3
    E1 = np.append([1], E)  # vecteur de coefficients incluant a0(ordre 4)
    H1 = np.append([1], H)
    
    #on trace la série entre 0 et le début de l'intervalle
    plt.plot(t[debut : fin], y[debut : fin])
    plt.title(serie)
    plt.show()
    
    #on trace les spectres (estimation)
    f, mag = spectre(E1, H1)
    
    #on calcule les valeurs correspondants aux spectres des 3 sous-series
    plt.semilogy(
    	f, mag[0],
    	f, mag[1],
    	':r',
        f, vrai_spectre,':b',
        linewidth = 2,
    )
    plt.title('Spectre / Calcul sur l intervalle [{} {}]'.format(debut, fin))
    plt.legend(['ordre2', 'ordre3',"vrai spectre"])
    return plt.show()

AR_model_somme(0,int(n/3),"série 1",spectre1)
AR_model_somme(int(n/3),2*int(n/3),"série 2",spectre2)
AR_model_somme(0,n,"serie 3",spectre3)

#Spectre de la somme de y1, y2, y3
s=[]
for i in range(2):
  s.append(a[i]+b[i]+c[i])

f,mag=spectre([1]+s)
spectreS = mag[0]

plt.semilogy(
	f,mag[0],
)
plt.grid() 
plt.legend('spectre1')
plt.title("Spectre de la somme")
plt.show()