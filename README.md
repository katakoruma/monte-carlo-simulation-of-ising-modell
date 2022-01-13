# Monte Carlo Simulation of Ising Modell

## General

This project is a Monte Carlo simulation of Ising Modell. For further information check the [Wikipedia entry](https://en.wikipedia.org/wiki/Ising_model) or the [Bachelor thesis](http://www.wiese.itp.unibe.ch/theses/wirz_bachelor.pdf) of Marcel Wirtz (in german). 
It was inspired and partly adopted by an exercise in the course "Theoretical Physics IV : Statistical Physics" at RWTH University.
Some functions were added to the original script. 
The most important extension is the parallelization, which massively reduces the runtime of the computationally expensive algorithm.
The parallelized code was also optimized for execution on a HPC system with a SLURM scheduler.
Furthermore I put the most important features into a python class and used the advantages to simplify
In the context of physics I added the possibility of applying an external magnetic field.

Depending on the setting, the scripts can be used to perform 2 essential functions. 
A live plot of an NxN matrix of a 2-spin system, showing the different behavior of the spins below and above the Curie temperature in a vividly way.
In addition, the simulated values of energy, heat capacity, magnetization and susceptibility per spin in the thermalized state can be determined, stored and plotted. 
Sometimes this can be CPU-intensive if several temperatures and several realizations are applied.   


## The different versions

### MC_Ising
Contains the class MCI and the most important functions.
Additionally, this script features the serial realization. 
It should only be used for the live plotting function if possible, since data collection can be performed much faster with the parallelized versions.


### MC_Ising_Draft_parallel_Tnum.py
Parallelized code. Parallelization for the different temperature steps.


### MC_Ising_Draft_parallel_Nreal.py
Parallelized code. Parallelization for the different reallizations steps at each temperature step.


### MC_Ising_plotten.py
Immport and plot of saved numpy arrays.  


