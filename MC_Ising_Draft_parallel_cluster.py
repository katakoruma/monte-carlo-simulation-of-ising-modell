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
import os

jobid = os.environ["SLURM_JOB_ID"]
num_cores = int(os.environ["SLURM_JOB_CPUS_PER_NODE"]) - 1

# plot Eigenschaften
live_plotten = False
plot = False
colorm = 'seismic'

# Definiere Systemgroesse
N = 10    #Kantenlänge

# Definiere Temperaturen immer in Einheiten von J/k
T_min = 1.5
T_max = 10.5
T_Anzahl = num_cores
Temperaturen = np.linspace(T_min,T_max,T_Anzahl)


# Definiere die Anzahl der Realisierungen zu einer Temperatur
N_realis = 10

# Definiere die Anzahl der Thermalisierungsschritte
N_therm = 1000

# Definiere die Anzahl an Monte Carlo Updates um einen Gleichgewichtszustand
# zu samplen
N_MC = 10000

# aeusseres Magnetfeld
B = 0

# Initialisierung des Pseudo-Zufallszahlengenerators
np.random.seed(None)

mci = MCI(Temperaturen, N_realis, N_therm, N_MC, T_max, T_min, T_Anzahl, N, B, colorm, live_plotten)


"""
Führe die Monte Carlo Simulation für verschiedene Temperaturen durch
"""

if __name__ == '__main__':

    print('N = ', N)
    print('T_min = ', T_min, ', T_max = ', T_max, ', T_Anzahl = ', T_Anzahl)
    print('N_realis = ', N_realis, ', N_therm = ', N_therm, ', N_MC = ', N_MC)

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


    print('M_Abs')
    print(M_Abs)

    print('chi_Abs')
    print(chi_Abs)

    print('u_Abs')
    print(u_Abs)

    print('cv_Abs')
    print(cv_Abs)


    os.mkdir('data/{}'.format(jobid))

    np.save('data/{}/M_{}'.format(jobid,jobid),M_Abs)
    np.save('data/{}/chi_{}'.format(jobid,jobid),chi_Abs)
    np.save('data/{}/u_{}'.format(jobid,jobid),u_Abs)
    np.save('data/{}/cv_{}'.format(jobid,jobid),cv_Abs)
    np.save('data/{}/temp_{}'.format(jobid,jobid),Temperaturen)

    if plot:

        mci.plotten(M_Abs, chi_Abs, u_Abs, cv_Abs)
