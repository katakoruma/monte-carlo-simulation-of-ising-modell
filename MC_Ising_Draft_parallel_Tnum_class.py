#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 14:43:29 2022

@author: leon
"""
import numpy as np
import matplotlib.pyplot as plt
from MC_Ising import MCI

from joblib import Parallel, delayed
import multiprocessing

num_cores = multiprocessing.cpu_count()


# plot Eigenschaften
live_plotten = False
plot = True
colorm = 'seismic'

# Definiere Systemgroesse
N = 10    #Kantenlänge

# Definiere Temperaturen immer in Einheiten von J/k
T_min = 1.5
T_max = 3.5
T_Anzahl = 21
Temperaturen = np.linspace(T_min,T_max,T_Anzahl)


# Definiere die Anzahl der Realisierungen zu einer Temperatur
N_realis = 10

# Definiere die Anzahl der Thermalisierungsschritte
N_therm = 100

# Definiere die Anzahl an Monte Carlo Updates um einen Gleichgewichtszustand
# zu samplen
N_MC = 100

# aeusseres Magnetfeld
B = 0


# Initialisierung des Pseudo-Zufallszahlengenerators
np.random.seed(None)

mci = MCI(Temperaturen, N_realis, N_therm, N_MC, T_max, T_min, T_Anzahl, N, B, colorm, live_plotten)


"""
Führe die Monte Carlo Simulation für verschiedene Temperaturen durch
"""

if __name__ == '__main__':

    # Setze container fuer die Messgroessen zur Abspeicherung auf
    M_Abs = np.ones((len(Temperaturen), N_realis))   # Magnetisierung
    chi_Abs = np.ones((len(Temperaturen), N_realis))
    u_Abs = np.ones((len(Temperaturen), N_realis))
    cv_Abs = np.ones((len(Temperaturen), N_realis))

    # Iteriere druch Temperaturen
    results = Parallel(n_jobs=num_cores)(delayed(mci.mainloop_T)(T_ind) for T_ind in range(T_Anzahl))


    for T_ind in range(T_Anzahl):

        M_Abs[T_ind,:]    =  results[T_ind][0]
        chi_Abs[T_ind,:]  =  results[T_ind][1]
        u_Abs[T_ind,:]    =  results[T_ind][2]

    cv_Abs = mci.berechne_Waermekap(u_Abs[0:-1,:], u_Abs[1::,:])


    if plot:

        mci.plotten(M_Abs, chi_Abs, u_Abs, cv_Abs)
