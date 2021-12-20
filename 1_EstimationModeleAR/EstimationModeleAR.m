%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% -*- coding: utf-8 -*-
%Created on Sun Sep 20 11:14:14 2020
%    @contact: atto / abatt@univ-smb.fr
%%%%%%%%%%%%%   EXERCICE 2 / PARTIE 1            %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Etude d-une serie temporelle (non-stationnaire globalement)
%%%%    composee de 3 blocs stationnaires
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Utiliser systematique l-aide de matlab (fonction help), Exemple :
%    >>help mean %% pour obtenir l-aide sur la fonction "mean"
% Apres pause: appuyer sur une touche quelconque pour poursuivre
%% Etude d'une serie temporelle non stationnaire par bloc de m1 données
%   et un décalage avec recouvrement tous les m2 pas
%   la non stationnarité est située aux points 512 et 1024
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
close all;
clc;
%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n=1536; % nombre de donnees a genener
% Sequences des parametres des 3 modeles AR du second ordre
% coefficients du premier processus AR
a1(1)=1;						
a1(2)=-0.1344;
a1(3)=0.9025;
% coefficients du second processus AR
a2(1)=1;						
a2(2)=-1.6674;
a2(3)=0.9025;
% coefficients du troisieme processus AR
a3(1)=1;						
a3(2)=1.7820;
a3(3)=0.8100;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
% Generation du 1er bloc de la serie (largeur n/3)
t=-2:1:n-1;
y=0*t;
for k=3:n/3
    z=randn;
    y(k)=-a1(2)*y(k-1)-a1(3)*y(k-2)+z;
end
%
%  Generation du 2nd bloc de la serie (largeur n/3)
for k=n/3+1:2*n/3
    z=randn;
    y(k+2)=-a2(2)*y(k+1)-a2(3)*y(k)+z;
end
%
%  Generation du 3e bloc de la serie (largeur n/3)
for k=2*n/3+1:n
    z=randn;
    y(k+2)=-a3(2)*y(k+1)-a3(3)*y(k)+z;
end
echo on
% Trace- de la serie
t = t(3:end);
y = y(3:end);
figure
plot(t,y);
title('Serie = juxtapososition de 3 sous-series stationnaires');
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pause
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%% Calcul et trace-s des spectres des trois sous-series a partir de freqz
%  Puisque l on connait les coefficients, le calcul est fait directement
%  a- partir des coefficients (ne depend donc pas du nombre d-echantillons) 
Np = 256; % nombre de points du spectre
[hh1,w]=freqz(1,a1,Np); % calcul du spectre de la sous-série 1
mag11=abs(hh1);
[hh2,w]=freqz(1,a2,Np); % calcul du spectre de la sous-série 2
mag12=abs(hh2);
[hh3,w]=freqz(1,a3,Np); % calcul du spectre de la sous-série 3
mag13=abs(hh3);
f=w; %/(2*pi)*1000;		% échantillonnage de l axe des fréquences
figure
semilogy(f,mag11,'-g',f,mag12,':b',f,mag13,'--r'); % tracé du spectre
title('Spectres des sous-séries');
legend('Sous-série 1', 'Sous-série 2', 'Sous-série 3');
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pause
%% On choisit de decrire y par un modele AR d-ordre 3, puis d'ordre 4.
%   Estimation des coefficients des modeles AR d-ordres 3 et 4
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calcul global sur n données 
%
% Calcul des coefficients des modèles AR d 'ordre 3 et 4
% On crée les versions translatées de y (style autocorrélation)
y1=[y 0 0 0 0]';
y2=[0 y 0 0 0]';
y3=[0 0 y 0 0]';
y4=[0 0 0 y 0]';
y5=[0 0 0 0 y]';
A=[y1 y2 y3 y4 y5];
D=cov(A);			% matrice de covariance / d'autocorrélation 5*5
%
% Calcul des coefficients par Yule-Walker
E=-inv(D(1:4,1:4))*D(1,2:5)'; % ordre 4
H=-inv(D(1:3,1:3))*D(1,2:4)'; % ordre 3
E1=[1 E']'; % vecteur de coefficients incluant a0(ordre 4)
H1=[1 H']'; % vecteur de coefficients incluant a0(ordre 3)
%
% Tracé des spectres estimés
[h11,w]=freqz(1,E1,Np); % ordre 4
[h12,w]=freqz(1,H1,Np); % ordre 3
mag1=abs(h11);
mag2=abs(h12);
figure
semilogy(f,mag1,f,mag2,':r', 'LineWidth',4);% tracé du spectre
title(['Spectre / Calcul sur l intervalle [1 ',num2str(n) ']']);
legend('ordre4', 'ordre3');


%% Calculs locaux sur m1 données avec m1 << n 
pause;
%
m1=256; % largeur d une trame d analyse
m2=350; % paramètre utilisé pour définir le début d une trame
%
% Calcul des coefficients des modèles AR d 'ordre 3 et 4
%   à partir de la donnée 1 sur m1 données
%
%calcul de l' autocorrelation
y1=[y(1:m1) 0 0 0 0]';
y2=[0 y(1:m1) 0 0 0]';
y3=[0 0 y(1:m1) 0 0]';
y4=[0 0 0 y(1:m1) 0]';
y5=[0 0 0 0 y(1:m1)]';
A=[y1 y2 y3 y4 y5];
D=cov(A);             % matrice de covariance / d'autocorrélation    
%
% Calcul des coefficients par Yule-Walker
E=-inv(D(1:4,1:4))*D(1,2:5)'; % ordre 4
H=-inv(D(1:3,1:3))*D(1,2:4)'; % ordre 3
E1=[1 E']'; % vecteur de coefficients incluant a0 (ordre 4)
H1=[1 H']'; % vecteur de coefficients incluant a0 (ordre 3)
[h11,w]=freqz(1,E1,Np);
[h12,w]=freqz(1,H1,Np);
mag1=abs(h11);
mag2=abs(h12);
figure
plot(y(1:m1));				%tracé de la trame (sous-série)
title(['\{y(1) ,..., y(' num2str(m1) ')\}']);
pause
figure
semilogy(f,mag11,f,mag1,'-r',f,mag2,'--k');%trace du spectre
title(['Spectre / Calcul sur l intervalle [1 ',num2str(m1) ']']);
legend('vrai spectre-y1', 'ordre4', 'ordre 3');
pause;

%calcul des coefficients des modèles AR d 'ordre 3 et 4
%à partir de la donnée m2 sur m1 données
y1=[ y(m2:m1+m2) 0 0 0 0]';
y2=[0 y(m2:m1+m2) 0 0 0]';
y3=[0 0 y(m2:m1+m2) 0 0]';
y4=[0 0 0 y(m2:m1+m2) 0]';
y5=[0 0 0 0 y(m2:m1+m2)]';
A=[y1 y2 y3 y4 y5];
D=cov(A);             % matrice de covariance / d'autocorrélation    
E=-inv(D(1:4,1:4))*D(1,2:5)'; % ordre 4
H=-inv(D(1:3,1:3))*D(1,2:4)'; % ordre 3
E1=[1 E']'; % vecteur de coefficients incluant a0 (ordre 4)
H1=[1 H']'; % vecteur de coefficients incluant a0 (ordre 3)
[h11,w]=freqz(1,E1,Np);
[h12,w]=freqz(1,H1,Np);
mag1=abs(h11);
mag2=abs(h12);
figure
plot(y(m2:m1+m2-1));			%tracé
title(['\{y(' num2str(m2) ') ,..., y(' num2str(m2+m1-1) ')\}']);
pause
figure
semilogy(f,mag11,'r',f,mag12,'g',f,mag1,'--r',f,mag2,'--g');% tracé du spectre
title(['Spectre / Calcul sur l intervalle [' num2str(m2) ',' num2str(m2+m1-1) ']']);
legend('vrai spectre-y1', 'vrai spectre-y2', 'ordre4', 'ordre 3');
pause;

%calcul des coefficients des modèles AR d 'ordre 3 et 4
%ˆ partir de la donnée m2+m1 sur m1 donné?es
y1=[ y(m2+m1:2*m1+m2) 0 0 0 0]';
y2=[0 y(m2+m1:2*m1+m2) 0 0 0]';
y3=[0 0 y(m2+m1:2*m1+m2) 0 0]';
y4=[0 0 0 y(m2+m1:2*m1+m2) 0]';
y5=[0 0 0 0 y(m2+m1:2*m1+m2)]';
A=[y1 y2 y3 y4 y5];
D=cov(A);             % matrice de covariance / d'autocorrélation    
E=-inv(D(1:4,1:4))*D(1,2:5)'; % ordre 4
H=-inv(D(1:3,1:3))*D(1,2:4)'; % ordre 3
E1=[1 E']'; % vecteur de coefficients incluant a0 (ordre 4)
H1=[1 H']'; % vecteur de coefficients incluant a0 (ordre 3)
[h11,w]=freqz(1,E1,Np);
[h12,w]=freqz(1,H1,Np);
mag1=abs(h11);
mag2=abs(h12);
figure
plot(y(m2+m1:2*m1+m2-1));				%tracé
title(['\{y(' num2str(m2+m1) ') ,..., y(' num2str(m2+2*m1-1) ')\}']);
pause
figure
semilogy(f,mag12,f,mag1,'--r',f,mag2,'--g');%tracé du spectre
title(['Spectre / Calcul sur l intervalle [' num2str(m2+m1) ',' num2str(m2+2*m1-1) ']']);
legend('vrai spectre-y2', 'ordre4', 'ordre 3');
pause

%calcul des coefficients des modèles AR d 'ordre 3 et 4
%à partir de la donnée 2*m2 sur m1 données
y1=[ y(2*m2:m1+2*m2) 0 0 0 0]';
y2=[0 y(2*m2:m1+2*m2) 0 0 0]';
y3=[0 0 y(2*m2:m1+2*m2) 0 0]';
y4=[0 0 0 y(2*m2:m1+2*m2) 0]';
y5=[0 0 0 0 y(2*m2:m1+2*m2)]';
A=[y1 y2 y3 y4 y5];
D=cov(A);             % matrice de covariance / d'autocorrélation    
E=-inv(D(1:4,1:4))*D(1,2:5)'; % ordre 4
H=-inv(D(1:3,1:3))*D(1,2:4)'; % ordre 3
E1=[1 E']'; % vecteur de coefficients incluant a0 (ordre 4)
H1=[1 H']'; % vecteur de coefficients incluant a0 (ordre 3)
[h11,w]=freqz(1,E1,Np);
[h12,w]=freqz(1,H1,Np);
mag1=abs(h11);
mag2=abs(h12);
figure
plot(y(2*m2:m1+2*m2-1));			%tracé
title(['\{y(' num2str(2*m2) ') ,..., y(' num2str(2*m2+m1-1) ')\}']);
pause
figure
semilogy(f,mag12,'b',f,mag1,'--r',f,mag2,'--g');%tracé du spectre
title(['Spectre / Calcul sur l intervalle [' num2str(2*m2) ',' num2str(2*m2+m1-1) ']']);
legend('vrai spectre-y2', 'ordre4', 'ordre 3');
pause

%calcul des coefficients des modèles AR d 'ordre 3 et 4
%à partir de la donnée 2*m2+m1 sur m1 données
y1=[ y(2*m2+m1:2*m1+2*m2) 0 0 0 0]';
y2=[0 y(2*m2+m1:2*m1+2*m2) 0 0 0]';
y3=[0 0 y(2*m2+m1:2*m1+2*m2) 0 0]';
y4=[0 0 0 y(2*m2+m1:2*m1+2*m2) 0]';
y5=[0 0 0 0 y(2*m2+m1:2*m1+2*m2)]';
A=[y1 y2 y3 y4 y5];
D=cov(A);             % matrice de covariance / d'autocorrélation    
E=-inv(D(1:4,1:4))*D(1,2:5)'; % ordre 4
H=-inv(D(1:3,1:3))*D(1,2:4)'; % ordre 3
E1=[1 E']'; % vecteur de coefficients incluant a0 (ordre 4)
H1=[1 H']'; % vecteur de coefficients incluant a0 (ordre 3)
[h11,w]=freqz(1,E1,Np);
[h12,w]=freqz(1,H1,Np);
mag1=abs(h11);
mag2=abs(h12);
figure
plot(y(2*m2+m1:2*m1+2*m2-1));			%tracé 
title(['\{y(' num2str(2*m2+m1) ') ,..., y(' num2str(2*m2+2*m1-1) ')\}']);
pause
figure
semilogy(f,mag12,'r',f,mag13,'g',f,mag1,'--r',f,mag2,'--g');%tracé du spectre
title(['Spectre / Calcul sur l intervalle [' num2str(2*m2+m1) ',' num2str(2*m2+2*m1-1) ']']);
legend('vrai spectre-y2', 'vrai spectre-y3', 'ordre4', 'ordre 3');
pause

%calcul des coefficients des modèles AR d 'ordre 3 et 4
%à partir de la donnée 2*m2+2*m1 sur m1 données
y1=[ y(2*m2+2*m1:3*m1+2*m2) 0 0 0 0]';
y2=[0 y(2*m2+2*m1:3*m1+2*m2) 0 0 0]';
y3=[0 0 y(2*m2+2*m1:3*m1+2*m2) 0 0]';
y4=[0 0 0 y(2*m2+2*m1:3*m1+2*m2) 0]';
y5=[0 0 0 0 y(2*m2+2*m1:3*m1+2*m2)]';
A=[y1 y2 y3 y4 y5];
D=cov(A);             % matrice de covariance / d'autocorrélation    
E=-inv(D(1:4,1:4))*D(1,2:5)'; % ordre 4
H=-inv(D(1:3,1:3))*D(1,2:4)'; % ordre 3
E1=[1 E']'; % vecteur de coefficients incluant a0 (ordre 4)
H1=[1 H']'; % vecteur de coefficients incluant a0 (ordre 3)
[h11,w]=freqz(1,E1,Np);
[h12,w]=freqz(1,H1,Np);
mag1=abs(h11);
mag2=abs(h12);
figure
plot(y(2*m2+2*m1:3*m1+2*m2-1));		%tracé
title(['\{y(' num2str(2*m2+2*m1) ') ,..., y(' num2str(2*m2+3*m1-1) ')\}']);
pause
figure
semilogy(f,mag13,f,mag1,'--r',f,mag2,'--g');%tracé du spectre
title(['Spectre / Calcul sur l intervalle [' num2str(2*m2+2*m1) ',' num2str(2*m2+3*m1-1) ']']);
legend('vrai spectre-y3','ordre4', 'ordre 3');




