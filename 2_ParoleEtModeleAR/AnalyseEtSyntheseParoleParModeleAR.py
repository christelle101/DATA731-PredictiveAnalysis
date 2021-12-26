"""
    MODULES
"""
from scipy.io import loadmat
import sounddevice as sd
import matplotlib.pyplot as plt 
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


