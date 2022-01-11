#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 02:04:30 2022

@author: leon
"""
import numpy as np

from MC_Ising import MCI

jobid = 25028694


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




chi_Abs      =    np.load('data/{}/chi_{}.npy'.format(jobid,jobid))
cv_Abs       =    np.load('data/{}/cv_{}.npy'.format(jobid,jobid))
M_Abs        =    np.load('data/{}/M_{}.npy'.format(jobid,jobid))
u_Abs        =    np.load('data/{}/u_{}.npy'.format(jobid,jobid))
Temperaturen =    np.load('data/{}/temp_{}.npy'.format(jobid,jobid))


T_Anzahl = len(Temperaturen)
T_max    = max(Temperaturen)
T_min    = min(Temperaturen)


mci = MCI(Temperaturen, N_realis, N_therm, N_MC, T_max, T_min, T_Anzahl, N, B)


mci.plot(M_Abs, chi_Abs, u_Abs, cv_Abs)
