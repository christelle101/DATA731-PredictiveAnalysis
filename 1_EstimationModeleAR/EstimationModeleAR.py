import random as aleas  
import matplotlib.pyplot as plt 
from scipy.signal import freqz   
import numpy as np 
"""
    random : pour generer des nombres aleatoires
    matplotlib.pyplot : pour generer des graphiques et gerer leur construction
    scipy.signal : pour avoir le TF de l'autocorrelation
    numpy : pour implementer les moyennes et les covariances
"""

#On commence par générer n données
n=1536 

# Sequences des parametres des 3 modeles AR du second ordre
a=[
    [1,-0.1344,0.9025], # coefficients du premier processus AR
    [1,-1.6674,0.9025], # Ccoefficients du second processus AR
    [1, 1.7820,0.8100]  # coefficients du troisieme processus AR
]



"""
    On génère une série temporelle (non-stationnaire globalement),
    et composée de 3 blocs stationnaires. Ensuite, on juxtapose les
    3 blocs de largeur n/3 chacun. Ensuite, on trace la série globale.
"""
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
plt.plot(t,y)
plt.title("Data = juxtapososition de 3 sous-series stationnaires")
plt.show()



""""
    On utilise freqz pour calculer et tracer les spectres des trois sous-series.
    Puisqu'on connaît les coefficients, le calcul se fait directement à partir 
    des coefficients et ne dépend donc pas du nombre d'échantillons.
""" 
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
plt.show()



"""
    On choisit de décrire y par un modèle AR d'ordre 3, puis d'ordre 4.
"""
#    Estimation des coefficients des modeles AR d-ordres 3 et 4
def AR_model(debut, fin, serie, vrai_spectre):
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
    
    E = - np.linalg.inv(D[0:4, 0:4]) @ D[0, 1:5].reshape(4, 1)  # car on veut l'avoir à l'ordre 4
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
    plt.legend(['ordre4', 'ordre3',"vrai spectre"])
    return plt.show()



"""
    EXECUTION :
        
    On exécute la fonction AR_model sur différents intervalles, ainsi que sur
    différents spectres, conformément au fichier MATLAB associé.
"""
#calcul sur l'intervalle de 1 à 1536
AR_model(1, 1536, "serie 1", mag[0])

#calcul sur l'intervalle 1 à 256
AR_model(1, 256, "Série 1 sur l'intervalle [1, 256]", mag[0])

#calcul sur l'intervalle de 350 à 605
AR_model(350, 605, "Série 1 sur l'intervalle [350, 605]", mag[0])

#calcul sur l'intervalle [606, 681]
AR_model(606, 681, "Série 2 sur l'intervalle [606 ; 681]", mag[1])

#calcul sur l'intervalle [700; 955]
AR_model(700, 955, "Série 2 sur l'intervalle [700 ; 955]", mag[1])

#calcul sur l'intervalle [956 ; 1211]
AR_model(956, 1211, "Série 2 sur l'intervalle [956 ; 1211]", mag[1])
AR_model(956, 1211, "Série 2 sur l'intervalle [956 ; 1211]", mag[2])

#calcul sur l'intervalle [1212; 1467]
AR_model(1212, 1467, "Série 2 sur l'intervalle [956 ; 1211]", mag[2])