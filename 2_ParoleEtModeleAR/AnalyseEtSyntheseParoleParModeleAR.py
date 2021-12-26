"""
    MODULES
"""
from scipy.io import loadmat
import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import warnings
import statsmodels.formula.api as smf 
from statsmodels.tsa.ar_model import AR
from scipy.signal import lfilter, hamming, freqz, deconvolve, convolve
import random as aleas

warnings.filterwarnings("ignore")

#Chargement de la donnée de parole
DataParole = loadmat('DataParole.mat')
DataParole = DataParole['DataParole']
wait = input("Ajuster le volume - Puis Appuyer sur une touche du clavier pour continuer.")
sd.play(DataParole, 8192) # son emis via haut parleur externe

#Visualisation de la donnée de parole
plt.plot(DataParole)
plt.ylabel('Data Parole')
plt.show() 

#n1 et n2 sont le debut et la fin de la serie a analyser
n1 = 200 
n2 = len(DataParole)

y = DataParole[n1 : n2]

#longueur de chaque trame d'analyse
m = 150

nb_trames = np.floor((n2 - n1 + 1)/m)

#ordre du modele AR
ordreAR = 8 

y1 = y[1 : m]

plt.plot(y1)
plt.ylabel('Data Parole')
plt.title("Trame 1")
plt.show()

#récupération des coefficients du modele
modele = AR(y1)
modele_fitted = modele.fit()
coeffAR1 = modele_fitted.params
print('Les coefficients du modele sont :\n %s' % coeffAR1)

z = [k*0 for k in range(len(y1))]
for k in range (1, len(y1)):
    z[k]=-coeffAR1[0]*y[k - 1] - coeffAR1[1]*y[k - 2]-coeffAR1[2]*y[k - 3]-coeffAR1[3]*y[k - 4]-coeffAR1[4]*y[k - 5]-coeffAR1[5]*y[k - 6]-coeffAR1[6]*y[k - 7]-coeffAR1[7]*y[k - 8]

plt.plot(range(len(y1[4:])),z[4:],label='Data =series stationnaires 1')
plt.title("Serie stationnaire 1")
plt.show()

print(modele_fitted.sigma2)
yf1 = lfilter(coeffAR1[0:9],1,y1)

"""Synth2=lfilter(1,a[0:9],rand)
Synth3 = lfilter(1,a[0:9],yf1)"""

plt.plot(y1[4:], 'b-', label='data')
plt.plot(modele_fitted.fittedvalues[4:], 'r-', label='data')
plt.show()

plt.plot(y1[0:], 'b-', label='data')
plt.plot(yf1[0:], 'g-', label='data')
plt.show()
