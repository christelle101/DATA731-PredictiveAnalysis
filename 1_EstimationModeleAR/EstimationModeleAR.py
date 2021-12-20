# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 11:14:14 2020
@contact: atto / abatt@univ-smb.fr
"""
#%%
####################################################################
#################################   EXERCICE 3   ###################
####################################################################
#                    Modèles autorégressifs
####################################################################
####################################################################
## Etude d-une serie temporelle (non-stationnaire globalement)
##    mais composee par juxtaposition de 3 blocs stationnaires
####################################################################
####################################################################
# Utiliser systematique l-aide de python (fonction help), Exemple :
#    help(int) pour obtenir l-aide sur la fonction "int"
#%% 
import random as aleas  # pour generer des nombres aleatoires et +
import matplotlib.pyplot as plt # pour les graphiques
from scipy.signal import freqz   # pour avoir le TF de l'autocorrelation
import numpy as np #pour gerer les moyennes + covariances +++
####################################################################
#############   EXERCICE 3 / Partie 1            ###################
####################################################################
n=1536 #  nombre de donnees a genener
#% Sequences des parametres des 3 modeles AR du second ordre
a=[
    [1,-0.1344,0.9025], # coefficients du premier processus AR
    [1,-1.6674,0.9025], # Ccoefficients du second processus AR
    [1, 1.7820,0.8100]  # coefficients du troisieme processus AR
]
####################################################################
## Generer une serie temporelle (non-stationnaire globalement)
##    composee de 3 blocs stationnaires
####################################################################
####################################################################
## Generation et juxtaposition des 3 blocs de largeur n/3 chacun
t=range(-2,n-1)
y=[k*0 for k in t]
for k in range(1,int(n/3)):
    y[k+1]=-a[0][1]*y[k]-a[0][2]*y[k-1]+aleas.gauss(0,1)
for k in range(int(n/3)+1,2*int(n/3)):
    y[k+1]=-a[1][1]*y[k]-a[1][2]*y[k-1]+aleas.gauss(0,1)
for k in range(2*int(n/3)+1,n):
    y[k+1]=-a[2][1]*y[k]-a[2][2]*y[k-1]+aleas.gauss(0,1)
y=y[3:]  # suppression des donnees transitoires
t=t[3:]
# Trace- de la serie
plt.plot(t,y,label='Data = juxtapososition de 3 sous-series stationnaires')
plt.title("Série = juxtaposition de 3 sous-séries stationnaires")
plt.show()
#
###########################################################################
#%%
###########################################################################
# Calcul et trace-s des spectres des trois sous-series a partir de freqz
#  Puisque l on connait les coefficients, le calcul est fait directement
#  a- partir des coefficients (ne depend donc pas du nombre d-echantillons) 
# 
def spectre(*args): 
	Np = 256 # nombre de points du spectre
	f=freqz(1,args[0],Np)[0] # recuperer les echantillons de frequences (abscisses)
	mag=[]   # hauteurs des frequences observables correspondantes (ordonnees)
	for arg in args:
		mag.append(abs(freqz(1,arg,Np)[1])) # calcul du spectre de chaque sous-serie
	return (f,mag)

f,mag=spectre(a[0],a[1],a[2])
## Calcul des spectres des trois sous-series 
plt.semilogy(
	f,mag[0],'-g',
	f,mag[1],':b',
	f,mag[2],'--r'
)
## Traces des spectres des trois sous-series
plt.title("Spectres des trois sous-séries") 
plt.show()
#
###########################################################################
#%%
# On choisit de decrire y par un modele AR d-ordre 3, puis d-ordre 4.
#    Estimation des coefficients des modeles AR d-ordres 3 et 4
# re-utiliser la partie deja ecrite pour superposer les estimations locales de spectres avec le resultat escompte-
x = aleas.randint(1, 7)
Y1 = [np.mean(y), 0, 0, 0, 0]
#print(Y1)
Y2 = [0, np.mean(y), 0, 0, 0]
Y3 = [0, 0, np.mean(y), 0, 0]
Y4 = [0, 0, 0, np.mean(y), 0]
Y5 = [0, 0, 0, 0, np.mean(y)]
A = np.array([Y1, Y2, Y3, Y4, Y5])
print(A)
D = np.cov(A)
#print(D)

E = np.matmul(-np.linalg.inv(D[0:4, 0:4]),np.transpose(D[0,1:5]))
H = np.matmul(-np.linalg.inv(D[0:3, 0:3]),np.transpose(D[0,1:4]))
print(E)
#print(H)
E1 = [1, np.transpose(E)]
H1 = [1, np.transpose(H)]
#print(E1)
