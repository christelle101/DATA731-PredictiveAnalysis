"""
    MODULES
"""
from scipy.io import loadmat
import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
import warnings
from statsmodels.tsa.ar_model import AR
from scipy.signal import lfilter
import time

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

plt.plot(y1[4:], 'b-', label='data')
plt.plot(modele_fitted.fittedvalues[4:], 'r-', label='data')
plt.show()

plt.plot(y1[0:], 'b-', label='data')
plt.plot(yf1[0:], 'g-', label='data')
plt.show()

n = 150
res = y1- yf1

m1=ordreAR+1
k=1
residuel = y1-yf1

NbTrames = int((n2-n1+1)/m)
for k in range(1,NbTrames-1):
    y2 = y[k*m -m1 + 1 : (k+1)*m]
    model = AR(y2)
    model_fitted = model.fit()
    coeffsAR = model_fitted.params
    yf2 = lfilter(coeffsAR[1:8],1,y2)
    residuel2 = y2[m1:m1+m-1]-yf2[m1:m1+m-1]
    residuel = np.concatenate((residuel,residuel2), axis=0)
    
    if k< 10:
        
        plt.plot(yf2[m1:m1+m-1], 'g-', label='data')
        plt.plot(y2[m1:m1+m-1], 'b-', label='data')
        plt.title("Trame %d, Estimée vs Réalité "%k)
        plt.legend('estimee','Vraie')
        plt.grid()
        plt.show()

plt.plot(residuel) 
plt.title("Paroles estimées")
plt.grid()
plt.show()
plt.plot(y) 
plt.title("Vraies Paroles")
plt.grid()
plt.show()

sd.play(residuel, 8192)
time.sleep(3)
sd.play(residuel, 8192)