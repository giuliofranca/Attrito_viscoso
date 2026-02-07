##################################################################################
# Giulio Franca                                                                  #
#                                                                                #
# Università degli studi di Perugia                                              #
# Progetto di Metodi Computazionali per la Fisica                                #
#                                                                                #
# Modulo 2 - Studio statistico della gittata per un insieme di corpi,            #
# data la distribuzione della velocità iniziale                                  #                                            
#                                                                                #
##################################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import integrate
from scipy import optimize

import moto_balistico

# Definizione funzione di probabilità
def prob(v,C):
    """
    Funzione che rappresenta la distribuzione di probabilità legata alla
    velocità iniziale in modulo dei corpi in caduta 
    --------------
    Parametri
    v : modulo della velocità iniziale
    C : costante arbitraria
    --------------
    Restituisce
    Valore probabilità della velocità iniziale, associata a ogni lancio
    """
    return C * np.sin(v/47.7)

# Definizione inversa della cumulativa
def cum_inv(u):
    """
    Funzione che rappresenta l'inverso della cumulativa della funzione di probabilità
    --------------
    Parametri
    u : Valore della probabilità cumulativa
    --------------
    Restituisce
    v : velocità associata alla probabilità cumulativa

    """
    v = 47.7 * np.arccos(1 - u * (1 - np.cos(150/47.7)))
    return v

# Definizione funzione per trovare la distribuzione che segue la lege sopra descritta
def distribution(_N):
    """
    Funzione per generare N valori di velocità iniziale secondo la legge indicata
    --------------
    Parametri
    N : numero di valori che si vuole estrarre
    --------------
    Restituisce
    distr : distribuzione delle velocità iniziali
    """
    distr_cum = np.random.random(_N)

    distr = cum_inv(distr_cum)
    return distr

# Definizione distribuzione che estrae la gittata da v
def gittata_da_v(_t, _x0, _y0, _v, _theta, _k, _m):
    """
    Funzione che, data la velocità iniziale in modulo, restituisce la gittata
    --------------
    Parametri
    _t : array dei tempi su cui calcolo soluzione dell'eq. diff.
    (_x0, _y0) : posizioni iniziali
    _v : modulo velocità iniziale
    _theta : angolo formato dalla velocità con la verticale
    _k : coefficiente di attrito
    _m : massa corpo
    --------------
    Restituisce
    gittata del corpo
    """
    vx_iniziale = _v * np.sin(_theta)
    vy_iniziale = _v * np.cos(_theta)
    sol = moto_balistico.soluzione(_t, _x0, _y0, vx_iniziale, vy_iniziale, _k, _m)
    impatto = moto_balistico.evento_impatto(_t, sol[:,0], sol[:,1], sol[:,2], sol[:,3])
    gittata = impatto[1] - _x0

    return gittata

# Definizione gittata nel caso senza attrito
def R_ideale(v0, y0, theta0):
    """
    Funzione che calcola la gittata nel caso ideale, trascurando l'attrito
    --------------
    Parametri
    v0 : velocità iniziale in modulo
    y0 : altezza di partenza
    theta0 : angolo formato con la verticale
    --------------
    Restituisce
    R : gittata ideale
    """
    v0x = v0 * np.sin(theta0)
    v0y = v0 * np.cos(theta0)
    R = v0x/9.81 * (v0y + np.sqrt(v0y**2 + 2*9.81*y0))
    return R

# Determinazione distribuzione statistica delle gittate
def main():

    # Definizione parametri comuni

    # angolo

    theta_deg = 30
    theta_rad = np.deg2rad(theta_deg)

    # massa
    massa = 2

    # posizione iniziale
    x0 = 0
    y0 = 3

    # costanti di attrito viscoso
    k = np.array([0, 0.1, 3, 10])

    # Numero di oggeti generati
    N = 5000 

    # Genero la distribuzione delle velocità iniziali
    v_iniziali = distribution(N)

    # Genero i tempi su cui risolvere l'equazione differenziale
    vy_max = np.max(v_iniziali * np.cos(theta_rad))
    T_max = moto_balistico.T_volo_max(vy_max, y0)
    tempi = np.linspace(0, T_max, 2000)

    # Distribuzione velocità iniziali
    n, bins, p = plt.hist(v_iniziali, bins = 70, color= 'royalblue', label = 'Velocità iniziali')
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    # Confronto con distribuzione teorica attesa
    params, params_covariance = optimize.curve_fit(prob, bin_centers, n / N)
    C = params[0]

    v_equispaziate = np.linspace(0,150,500)
    plt.plot(v_equispaziate, prob(v_equispaziate, C) * N, color= 'darkred', label= 'Valore atteso')
    plt.title('Velocità iniziali')
    plt.xlabel('Velocità iniziali [m/s]')
    plt.ylabel('Eventi')
    plt.legend(loc= 'upper left')

    plt.show()

    # Determino le gittate ideali nel caso senza attrito
    gittate_ideali = R_ideale(v_iniziali, y0, theta_rad)

    # Determino le gittate per ogni valore di k
    for ki in k:
        gittate = []

        for vi in v_iniziali:
            gittata = gittata_da_v(tempi, x0, y0, vi, theta_rad, ki, massa)
            gittate.append(gittata)

        array_gittate = np.array(gittate)

        # Istogramma con gittate, senza confronto con distribuzione ideale
        
        plt.hist(array_gittate, bins = 70, color= 'gold', label = 'Gittate, k = {:.2f} kg/s'.format(ki))
        plt.title('Distribuzione gittate per K = {:.2f} kg/s'.format(ki))
        plt.xlabel('Gittate [m]')
        plt.ylabel('Eventi')
        plt.legend(loc= 'upper right')

        plt.show()

        # Istogramma con gittate, confronto con distribuzione ideale
        bins_comuni = np.linspace(min(array_gittate.min(), gittate_ideali.min()),
                          max(array_gittate.max(), gittate_ideali.max()), 70)
        plt.hist(array_gittate, bins = bins_comuni, color= 'gold', label = 'Gittate, k = {:.2f} kg/s'.format(ki))
        plt.hist(gittate_ideali, bins = bins_comuni, color= 'darkgreen', label = 'Gittate ideali')
        plt.title('Distribuzione gittate per K = {:.2f} kg/s'.format(ki))
        plt.xlabel('Gittate [m]')
        plt.ylabel('Eventi')
        plt.legend(loc= 'upper right')

        plt.show()

if __name__ == '__main__':

    main()
    



