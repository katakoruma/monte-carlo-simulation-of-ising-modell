#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 14:43:29 2022

@author: leon
"""

import numpy as np
from MC_Ising import MCI

from joblib import Parallel, delayed
import os


#basic configuration
jobid = 42             # tag for saved files, gets replaced by jobid if SLURM job
num_cores = 10          # number of workers, gets replaced by available cores if SLURM job

live_plotten = False    # live plot of the spin configuration
plot = True             # create plots of magnetization, susceptibility, energy and heat capacity
save = False            # save numpy arrays
num_cor_T_num = False   # number of temperature steps equal to number of workers

colorm = 'seismic'

# Size of sytem
N = 10    #edge length

# Define temperatures in units of J/k
T_min = 1.5
T_max = 10.5
T_num = 5               #replaced by num_cores if num_cor_T_num = True

# Define the number of realizations to a temperature
N_realis = 10

# Define the number of thermalization steps
N_therm = 100

# Define the number of Monte Carlo updates to sample an equilibrium state.
N_MC = 100

# external magnetic field
B = 0


if __name__ == '__main__':

    np.random.seed(None)

    # check if SLURM job
    if 'SLURM_JOB_ID' in os.environ:
        jobid = os.environ['SLURM_JOB_ID']

    if 'SLURM_JOB_CPUS_PER_NODE' in os.environ:
        num_cores = int(os.environ['SLURM_JOB_CPUS_PER_NODE']) - 1

    if num_cor_T_num:
        T_num = num_cores

    temperatures = np.linspace(T_min,T_max,T_num)

    mci = MCI(temperatures, T_max, T_min, T_num, N, N_realis, N_therm, N_MC, B, colorm, live_plotten)

    print('live_plotten: ',live_plotten,', plot: ', plot,', save: ', save)
    print('N = ', N, ', num_cores = ',num_cores)
    print('T_min = ', T_min, ', T_max = ', T_max, ', T_num = ', T_num)
    print('N_realis = ', N_realis, ', N_therm = ', N_therm, ', N_MC = ', N_MC)

    M = np.zeros((T_num, N_realis))
    chi = np.zeros((T_num, N_realis))
    u = np.zeros((T_num, N_realis))
    cv = np.zeros((T_num, N_realis))

    # Execute the Monte Carlo simulation for different temperatures
    results = Parallel(n_jobs=num_cores)(delayed(mci.mainloop_T)(T_ind) for T_ind in range(T_num))


    for T_ind in range(T_num):

        M[T_ind,:]    =  results[T_ind][0]
        chi[T_ind,:]  =  results[T_ind][1]
        u[T_ind,:]    =  results[T_ind][2]

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

        mci.save_arrays(M, chi, u, cv, jobid = jobid)

    if plot:

        mci.plot(M, chi, u, cv)
