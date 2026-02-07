##################################################################################
# Giulio Franca                                                                  #
#                                                                                #
# Università degli studi di Perugia                                              #
# Progetto di Metodi Computazionali per la Fisica                                #
#                                                                                #
# Modulo 1 - Risoluzione equazione differenziale con parametri configurabili     #
#                                                                                #
##################################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import integrate
import argparse

# Definizione costante gravitazionale, parametro non configurabile
g = 9.81

# Funzione per equazione differenziale grave in campo gravitazionale + attrito
def drdt(r, t, k, m):
    """
    Funzione che descrive l'equazione differenziale del moto del grave in campo
    gravitazionale terrestre in presenza di attrito viscoso
    --------------
    Parametri
    r : vettore in ingresso contenente coordinate spaziali e velocità (x,y,vx,vy) 
    t : variabile temporale
    k : coefficiente di attrito viscoso
    m : massa del grave
    --------------
    Restituisce
    velocità x, velocità y, accelerazione x, accelerazione y

    """
    dxdt = r[2]
    dvxdt = -(k/m)*r[2] 
    dydt = r[3]
    dvydt = -(k/m)*r[3] -g

    return (dxdt, dydt, dvxdt, dvydt)

# Funzione che risolve l'equazione differenziale in funzione di parametri configurabili
def soluzione(_tempi, _x0, _y0, _vx0, _vy0, _k, _m):
    """
    Funzione che risolve l'equazione differenziale
    --------------
    Parametri
    _tempi : array dei tempi su cui risolvo numericamente l'equazione differenziale
    (_x0, _y0, __vx0, _vy0) : condizioni iniziali inseribili dall'utente
    (_k, _m) : parametri caratteristici equazione
    --------------
    Restituisce
    array con coordinate spaziali e velocità del moto (x, y, vx, vy)

    """

    r_solved = integrate.odeint(drdt, (_x0, _y0, _vx0, _vy0), _tempi, args = (_k, _m))

    return r_solved

# definizione funzione che crea parser e immagazzina i tipi di dati ricevuti dall'utente
def parse_arguments():
    """
    Funzione che genera il parser che contiene gli argomenti inseriti dall'utente
    --------------
    Argomenti obbligatori : coefficiente k, massa, velocità iniziali
    --------------
    Argomenti facoltativi : posizione iniziale. Se non specificato (x0, y0) = (0,0)
    """
    parser = argparse.ArgumentParser(description = 'Simula il moto balistico di un corpo con attrito viscoso lineare, calcolando traiettoria, gittata, altezza massima e velocità massima. I parametri da configurare sono coefficiente di attrito (k), massa (m), velocità iniziali (vx0, vy0) e posizioni iniziali (x0, y0). Se non altrimenti specificato, la posizione iniziale sarà (0,0)')

    parser.add_argument('--k', type= float, required= True, help = 'Coefficiente di attrito viscoso')
    parser.add_argument('--m', type= float, required= True, help= 'Massa del grave')
    parser.add_argument('--vx0', type = float, required= True, help= 'Velocità iniziale x' )
    parser.add_argument('--vy0', type= float, required= True, help='Velocità iniziale y')
    parser.add_argument('--x0', type= float, default= 0, help='Ascissa iniziale x')
    parser.add_argument('--y0', type= float, default = 0, help='Ordinata iniziale y')
    return parser.parse_args()

# definzione funzione per trovare parametri al punto di impatto col suolo
def evento_impatto(t, x, y, vx, vy):
    """
    Funzione per determinare gittata e velocità massime, usando un'interpolazione
    lineare tra i due punti in cui y cambia segno, y>= e y<0
    --------------
    Parametri
    t : array dei tempi su cui ho risolto l'equazione
    (x, y, vx, vy) : array che contengono la soluzione dell'equazione differenziale
    --------------
    Restituisce
    tempo di volo, gittata, velocità x e y al tempo di atterraggio (valori massimi nel moto)
    """
    y = np.array(y)
    for i in range(len(y) - 1):
        if(y[i] > 0 and y[i+1] < 0):
            alpha = y[i]/(y[i] - y[i+1])
            t_atterraggio = t[i] + alpha * (t[i+1] - t[i])
            x_atterraggio = x[i] + alpha * (x[i+1] - x[i])
            vx_atterraggio = vx[i] + alpha * (vx[i+1] - vx[i])
            vy_atterraggio = vy[i] + alpha * (vy[i+1] - vy[i])
            return t_atterraggio, x_atterraggio, vx_atterraggio, vy_atterraggio

# definizione funzione per trovare parametri al punto di massima altezza
def evento_massimo(t, x, y, vx, vy):
    """
    Funzione per determinare altezza massima, attraverso un'interpolazione
    lineare tra i due punti in cui vy cambia segno, vy > 0 e vy < 0
    --------------
    Parametri
    t : array dei tempi su cui ho risolto l'equazione
    (x, y, vx, vy) : array che contengono la soluzione dell'equazione differenziale
    --------------
    Restituisce
    t_massimo : tempo impiegato per raggiungere altezza massima
    y_ massimo : altezza massima

    """
    vy = np.array(vy)
    for i in range(len(vy) - 1):
        if(vy[i] > 0 and vy[i+1] < 0):
            alpha = vy[i]/(vy[i] - vy[i+1])
            t_massimo = t[i] + alpha * (t[i+1] - t[i])
            y_massimo = y[i] + alpha * (y[i+1] - y[i])
            return t_massimo, y_massimo

# Definizione funzione che sovrastima il tempo di volo, non considerando l'attrito
def T_volo_max(_vy0, _y0):
    """
    Funzione che sovrastima il tempo di volo non considerando l'attrito
    --------------
    Parametri
    _vy0 : velocità iniziale y
    _y0 : coordinata iniziale y
    safety_param = parametro per incrementare ulteriormente tempo di volo
    --------------
    Restituisce
    Tempo di volo nel caso senza attrito

    """
    safety_param = 2
    if _y0 == 0:
        T_max = (2*_vy0)/g

    else:
        T_max = (_vy0 + np.sqrt(_vy0**2 + 2*g*_y0))/g

    return T_max * safety_param

# Risoluzione equazione differenziale con parametri stabiliti dall'utente
def main():
    # Includo i parametri in variabili
    argomenti = parse_arguments()
    k = argomenti.k
    m = argomenti.m 
    vx0 = argomenti.vx0
    vy0 = argomenti.vy0
    x0 = argomenti.x0
    y0 = argomenti.y0
    
    # Sovrastima del tempo di volo considerando il caso senza attrito
    tempo_volo = T_volo_max(vy0, y0)

    # Creazione array tempi con numero alto di valori per permettere interpolazione lineare
    tempi = np.linspace(0, tempo_volo, 2000)

    # Risoluzione equazione differenziale
    sol = soluzione(tempi, x0, y0, vx0, vy0, k, m)
    x_solved = sol[:,0]
    y_solved = sol[:,1]
    vx_solved = sol[:,2]
    vy_solved = sol[:,3]

    # Studio traiettoria: x(t) e y(t)

    mask = y_solved >= 0

    fig, ax = plt.subplots(1, 2, figsize=(12,6))

    ax[0].plot(tempi[mask], x_solved[mask], color = 'royalblue', linewidth=2, label = 'x(t)')
    ax[0].set_xlabel('t [s]')
    ax[0].set_ylabel('x [m]')
    ax[0].legend(loc = 'upper left')
    ax[0].grid(True)
    ax[0].set_title('Coordinata x nel tempo ')

    ax[1].plot(tempi[mask], y_solved[mask], color = 'darkgreen', linewidth=2, label = 'y(t)')
    ax[1].set_xlabel('t [s]')
    ax[1].set_ylabel('y [m]')
    ax[1].legend(loc = 'upper right')
    ax[1].grid(True)
    ax[1].set_title('Coordinata y nel tempo')

    plt.show()

    # Studio traiettoria: y(x)
    plt.plot(x_solved[mask], y_solved[mask], color= 'darkred', linewidth=2, label = 'y(x)')
    plt.title('Traiettoria del moto')
    plt.xlabel(' x [m]')
    plt.ylabel('y [m]')
    plt.grid(True)
    plt.legend(loc = 'upper left')
    
    plt.show()

    # Studio gittata e massima velocità di caduta
    impatto = evento_impatto(tempi, x_solved, y_solved, vx_solved, vy_solved)

    gittata = impatto[1] - x0

    vx_max = impatto[2]
    vy_max = impatto[3]
    v_max = np.sqrt(vx_max**2 + vy_max**2)


    # Studio altezza massima
    massimo = evento_massimo(tempi, x_solved, y_solved, vx_solved, vy_solved)

    h_max = massimo[1]


    # Racchiudo dati raccolti in un Data Frame
    mydict = { 'Gittata [m]' : np.round(gittata, 2), 'Altezza massima [m]' : np.round(h_max, 2), 'Velocità massima [m/s]' : np.round(v_max, 2)}
    mydf = pd.DataFrame([mydict])

    print(mydf)

if __name__ == '__main__':

    main()