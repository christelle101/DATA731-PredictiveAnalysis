"""
    MODULES
"""
from scipy.io import loadmat
import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np 
from scipy.signal import lfilter, hamming, freqz, deconvolve, convolve

#Chargement de la donnée de parole
DataParole = loadmat('DataParole.mat')
DataParole = DataParole['DataParole']
wait = input("Ajuster le volume - Puis Appuyer sur une touche du clavier pour continuer.")
sd.play(DataParole, 8192) # son emis via haut parleur externe

#Visualisation de la donnée de parole
z = DataParole
plt.plot(z)
plt.ylabel('Data Parole')
plt.show() 

#n1 et n2 sont le debut et la fin de la serie a analyser
n1 = 200 
n2 = len(z)

y = z[n1 : n2]

#longueur de chaque trame d'analyse
m = 150

nb_trames = np.floor([(n2 - n1 + 1)/m])

#ordre du modele AR
ordreAR = 8 

y1 = y[1 : m]

plt.plot(y1)
plt.ylabel('Data Parole')
plt.title("Trame 1")
plt.show()


