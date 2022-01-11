#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 15:36:04 2022

@author: leon
"""
import numpy as np
from MC_Ising import MCI


# plot Eigenschaften
live_plotten = False
plot = True
colorm = 'seismic'

# Definiere Systemgroesse
N = 10    #Kantenlänge

# Definiere Temperaturen immer in Einheiten von J/k
T_min = 1.5
T_max = 3.5
T_Anzahl = 5
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


if __name__ == '__main__':
     

    M_Abs, chi_Abs, u_Abs, cv_Abs = mci.mainloop_lin()
        
        
    if plot:
    
        mci.plotten(M_Abs, chi_Abs, u_Abs, cv_Abs)   