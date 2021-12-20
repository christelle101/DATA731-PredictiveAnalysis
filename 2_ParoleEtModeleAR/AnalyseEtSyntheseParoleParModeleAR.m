%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% -*- coding: utf-8 -*-
%Created on Sun Sep 20 11:14:14 2020
%    @contact: atto / abatt@univ-smb.fr
%%%%%%%%%%%%%   EXERCICE 2 / PARTIE 2            %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Apres pause: frapper n importe quelle touche pour poursuivre
%% Analyse et synthese de parole par un modele AR(n)
clear all;clc;close all;

% Chargement de la donnée de parole dans y
%load('DataParole.mat')
%z = DataParole;


pause


% Ecoute de la donnée de parole 
soundsc(z)

% Visualisation de la donnée de parole 
figure
plot(z);
title('Données de parole');
pause

%% Nous allons proceder a l' analyse par trames (fenetres) sur y
%   PARAMETRES D ANALYSE
% n1 et n2 sont les debut et fin de la série a analyser
n1 = 200;
n2 = length(z);
%
y= z(n1:n2)';
%
m=150; % longueur de chaque trame d analyse est m
%
NbTrames = floor((n2-n1+1)/m);
%
ordreAR=8;   % ordre du modele AR
%
% Premiere trame
y1=[y(1:m)]';
%%%%%%%%%%%%%%%%%%%%%%%%%
[coeffsAR1]=real(lpc(y1,ordreAR));
yf1=filter(coeffsAR1,1,y1);
residuel = y1-yf1; residuel = residuel'; % erreur residuelle d estimation
%%%%%%%%%%%%%%%%%%%%%%%%%
figure
subplot(221);
plot(y1);
title(['Trame 1 (T1)']);
subplot(222);
plot(yf1)
title(['Erreur sur T1']);
subplot(223);
plot(y1-yf1)
title(['Estimee sur T1']);
subplot(224);
plot(xcorr(yf1,'coeff'))
title(['Correlation erreur sur T1']);
%%%%%%%%%%%%%%%%%%%%%%%%%


%%
%%%%%%%%%%%%%%%%%%%%%%%%%
pause
%%%%%%%%%%%%%%%%%%%%%%%%%
Synth2=filter(1,coeffsAR1,randn(size(y1)))';
Synth3 = filter(1,coeffsAR1,yf1)';
figure
plot(Synth2/max(abs(Synth2))); 
hold on; 
plot(Synth3/max(abs(Synth3)),'r'); 
hold off
title('Syntheses');
legend('Synthèse par BBG', 'Synthèse par Erreur-residuelle');

%%%%%%%%%%%%%%%%%%%%%%%%%
pause
%
NbTramesAffichees = 10;  % doit etre inferieur a  NbTrames
m1=ordreAR+1; 
%%%%%%%%%%%%%%%%%%%%%%%%%
for k = 1:NbTrames-1
    y2 = y(k*m -m1 + 1 : (k+1)*m);
    % calcul des coefficients d'un AR d'ordre n
    [coeffsAR2]=real(lpc(y2,ordreAR));
    yf2=filter(coeffsAR2,1,y2);
    residuel2 = y2(m1:m1+m-1)-yf2(m1:m1+m-1);
    residuel = [residuel residuel2];
    synth2=filter(1,coeffsAR2,randn(size(y2)));
    synth3=filter(1,coeffsAR2,yf2);
    synth2 = synth2(m1:m1+m-1);
    synth3 = synth3(m1:m1+m-1);
    Synth2 = [Synth2 synth2];
    Synth3 = [Synth3 synth3];
    if (k <= NbTramesAffichees-1)
        figure
        subplot(221);
        plot(y2(m1:m1+m-1));
        title(['Trame ' num2str(k)]);
        subplot(222);
        plot(yf2(m1:m1+m-1))
        title(['Erreur']);
        subplot(223);
        plot(residuel2)
        title(['Estimee']);
        subplot(224);
        plot(xcorr(yf2,'coeff'))
        title(['Correlation erreur']);
        %%%%%%
        pause
        %%%%%%
        figure
        plot(synth2/max(abs(synth2))); hold on; 
        plot(synth3/max(abs(synth3)),'r'); hold off
        legend('Synthèse par BBG', 'Synthèse par Erreur-residuelle');
        title('Syntheses de parole');
        %%%%%%
        pause
    end
end

echo on
%%%%%%
soundsc(y) % parole originelle
pause
%%%%%%
soundsc(residuel) % parole estimée
pause  
%%%%%%
soundsc(Synth3) % parole synthétisée avec l'erreur d'estimation
pause
%%%%%%
soundsc(Synth2) % parole synthétisée avec bruit blanc gaussien


%% Affichage de l'ensemble des traitements
L = length(residuel);
%%%%%%
figure
subplot(221);
plot((1:L),y(1:L)/max(abs(y(1:L))));
title('Parole originelle');
subplot(222);
plot((1:L),residuel/max(abs(residuel)));
title('Estimee');
subplot(223);
plot((1:L),Synth3/max(abs(Synth3)));
title('Synthese par erreur-residuelle');
subplot(224);
plot((1:L),Synth2/max(abs(Synth2)));
title('Synthese par BBG');