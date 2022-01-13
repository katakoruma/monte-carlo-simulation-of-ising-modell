#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 02:04:30 2022

@author: leon
"""
import numpy as np

from MC_Ising import MCI

jobid = 25043066


# Definiere Systemgroesse
N = 10    #Kantenlänge


# Definiere die Anzahl der Realisierungen zu einer Temperatur
N_realis = 10

# Definiere die Anzahl der Thermalisierungsschritte
N_therm = 1000

# Definiere die Anzahl an Monte Carlo Updates um einen Gleichgewichtszustand
# zu samplen
N_MC = 10000

# aeusseres Magnetfeld
B = 0




chi           =    np.load('data/{}/chi_{}.npy'.format(jobid,jobid))
cv            =    np.load('data/{}/cv_{}.npy'.format(jobid,jobid))
M             =    np.load('data/{}/M_{}.npy'.format(jobid,jobid))
u             =    np.load('data/{}/u_{}.npy'.format(jobid,jobid))
temperatures  =    np.load('data/{}/temp_{}.npy'.format(jobid,jobid))


T_num    = len(temperatures)
T_max    = max(temperatures)
T_min    = min(temperatures)


mci = MCI(temperatures, T_max, T_min, T_num, N, N_realis, N_therm, N_MC, B)


mci.plot(M, chi, u, cv)
