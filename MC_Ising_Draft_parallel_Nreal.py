#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 14:30:37 2022

@author: leon
"""
import numpy as np
from MC_Ising import MCI

from joblib import Parallel, delayed
import os


#basic configuration
jobid = 123             # tag for saved files, gets replaced by jobid if SLURM job
num_cores = 10          # number of workers, gets replaced by available cores if SLURM job

live_plotten = False    # live plot of the spin configuration
plot = True             # create plots of magnetization, susceptibility, energy and heat capacity
save = False            # save numpy arrays
num_cor_N_realis = False   # number of temperature steps equal to number of workers

colorm = 'seismic'

# Size of sytem
N = 10    #edge length

# Define temperatures in units of J/k
T_min = 1.5
T_max = 10.5
T_num = 5

# Define the number of realizations to a temperature
N_realis = 10       #replaced by num_cores if num_cor_N_realis = True

# Define the number of thermalization steps
N_therm = 100

# Define the number of Monte Carlo updates to sample an equilibrium state.
N_MC = 100

# external magnetic field
B = 0


if __name__ == '__main__':

    np.random.seed(None)

    if 'SLURM_JOB_ID' in os.environ:
        jobid = os.environ['SLURM_JOB_ID']

    if 'SLURM_JOB_CPUS_PER_NODE' in os.environ:
        num_cores = int(os.environ['SLURM_JOB_CPUS_PER_NODE']) - 1

    if num_cor_N_realis:
        N_realis = num_cores

    temperatures = np.linspace(T_min,T_max,T_num)

    mci = MCI(temperatures, N_realis, N_therm, N_MC, T_max, T_min, T_num, N, B, colorm, live_plotten)

    print('live_plotten: ',live_plotten,', plot: ', plot,', save: ', save)
    print('N = ', N, ', num_cores = ',num_cores)
    print('T_min = ', T_min, ', T_max = ', T_max, ', T_num = ', T_num)
    print('N_realis = ', N_realis, ', N_therm = ', N_therm, ', N_MC = ', N_MC)

    # Setze container fuer die Messgroessen zur Abspeicherung auf
    M = np.ones((len(temperatures), N_realis))   # Magnetisierung
    chi = np.ones((len(temperatures), N_realis))
    u = np.ones((len(temperatures), N_realis))
    cv = np.ones((len(temperatures), N_realis))

    # Iteriere druch temperatures
    for T_ind in range(T_num):

        # Setze Temperatur
        T = temperatures[T_ind]

        # Als Orientierung fue die Laufzeit
        print("Wir befinden uns bei Temperatur {:.1f}".format(T))

        results = Parallel(n_jobs=num_cores)(delayed(mci.mainloop_N_realis)(realis,T) for realis in range(N_realis))

        for realis in range(N_realis):

            M[T_ind,realis]    =  results[realis][0]
            chi[T_ind,realis]  =  results[realis][1]
            u[T_ind,realis]    =  results[realis][2]

        cv = mci.heat_capacity(u[0:-1,:], u[1::,:])


    print('M:')
    print(M)

    print('chi:')
    print(chi)

    print('u:')
    print(u)

    print('cv:')
    print(cv)


    if save:

        os.mkdir('data/{}'.format(jobid))

        np.save('data/{}/M_{}'.format(jobid,jobid),M)
        np.save('data/{}/chi_{}'.format(jobid,jobid),chi)
        np.save('data/{}/u_{}'.format(jobid,jobid),u)
        np.save('data/{}/cv_{}'.format(jobid,jobid),cv)
        np.save('data/{}/temp_{}'.format(jobid,jobid),temperatures)


    if plot:

        mci.plot(M, chi, u, cv)
