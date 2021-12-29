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