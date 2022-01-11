#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 15:06:08 2022

@author: leon
"""
import numpy as np
from MC_Ising import MCI


# plot Eigenschaften
live_plotten = True
plot = False
colorm = 'seismic'

# Definiere Systemgroesse
N = 100    #Kantenlänge

# Definiere Temperaturen immer in Einheiten von J/k
T_min = 4
T_max = 6.0
T_Anzahl = 4
Temperaturen = np.linspace(T_min,T_max,T_Anzahl)

# Definiere die Anzahl der Realisierungen zu einer Temperatur
N_realis = 1

# Definiere die Anzahl der Thermalisierungsschritte
N_therm = 100
      
# Definiere die Anzahl an Monte Carlo Updates um einen Gleichgewichtszustand
# zu samplen
N_MC = 30

# aeusseres Magnetfeld
B = 0


# Initialisierung des Pseudo-Zufallszahlengenerators
np.random.seed(None)

mci = MCI(Temperaturen, N_realis, N_therm, N_MC, T_max, T_min, T_Anzahl, N, B, colorm, live_plotten)


if __name__ == '__main__':
     

    M_Abs, chi_Abs, u_Abs, cv_Abs = mci.mainloop_lin()
        
        
    if plot:
    
        mci.plotten(M_Abs, chi_Abs, u_Abs, cv_Abs)   
            