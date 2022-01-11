#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 13:12:02 2022

@author: leon
"""

import numpy as np
import matplotlib.pyplot as plt


class MCI:

    def __init__(self, temperatures, N_realis, N_therm, N_MC, T_max, T_min, T_num, N, B, colorm = 'seismic', live_plotten = False):

        self.temperatures   = temperatures
        self.N_realis       = N_realis
        self.N_therm        = N_therm
        self.N_MC           = N_MC
        self.T_max          = T_max
        self.T_min          = T_min
        self.T_num          = T_num
        self.N              = N
        self.B              = B
        self.colorm         = colorm
        self.live_plotten   = live_plotten

        if live_plotten:
            self.fig, self.ax = plt.subplots()

    def initialisiere_Zustand(self):
        """
        Create a random N x N array of spins
        """
        #binäre Schreibweise mit 0 und 1, eignet sich genauso zur Charakterisierung
        start_zustand = np.random.randint(0,2,size=((self.N,self.N)))

        #skaliere um auf -1,1
        start_zustand = -np.ones((self.N,self.N)) + 2 * start_zustand

        return start_zustand


    def MC_metro_update(self, config, temp):
        """
        Update of spin states with respect to temperature
        """
        # Probiere Ordnung NxN Spin flips
        for flip in range(self.N*self.N):

            # bestimme random Spin zum flippen
            x_s = np.random.randint(0,self.N)
            y_s = np.random.randint(0,self.N)


            x_smin,x_smax,y_smin,y_smax = x_s-1 , x_s+1 , y_s-1 , y_s+1

            if x_s == self.N - 1:
                x_smax = 0
            if y_s == self.N - 1:
                y_smax = 0

            E_o =   config[x_s,y_s] * (config[x_smin,y_s] + config[x_smax,y_s] + config[x_s,y_smin] + config[x_s,y_smax] + self.B)
            E_n = - config[x_s,y_s] * (config[x_smin,y_s] + config[x_smax,y_s] + config[x_s,y_smin] + config[x_s,y_smax] + self.B)


            Delta_E = E_o - E_n

            if Delta_E <= 0:
                p = 1
            else:
                p = np.exp(-Delta_E / temp)

           #Dummy update
            if np.random.rand() < p:
                config[x_s,y_s] = - config[x_s,y_s]

        return config


    def thermalisiere_Zustand(self, config, temp):
        """
        Führe auf dem initialisierten Zustand Ther_Schritte mal das MC update
        durch um möglichst den initiierten random Zustand in den
        Gleichgewichtsszustand zu überführen
        """
        for t_schritt in range(self.N_therm):

            # Update das Spin system
            self.MC_metro_update(config, temp)

            # Visualisierung (kann für den Algorithmus ignoriert werden)
            if self.live_plotten:
                self.ax.clear()
                self.ax.set_title("MC Schritte fuer T={:.1f} und B={}".format(temp,self.B))
                self.ax.imshow(config, cmap = self.colorm)
                plt.pause(0.01)

        return config


    def berechne_Magnetisierung(self, config):
        """
        Calculate the magnetization per spin for the given configuration
        """
        # Bestimme den Mittelwert der Spins des ganzen Gitters
        mag = np.mean(config)

        return mag

    def berechne_Suszeptibilitaet(self, mag):
        """
        Calculate the susceptibility per spin for the given magnetization
        """
        # Bestimme den Mittelwert der Spins des ganzen Gitters

        Susz = mag**2 * (self.N**2 - 1) / self.N**2

        return Susz

    def berechne_Energie(self, config):
        """
        Calculate the energy per spin for the given configuration
        """

        E = 2 * self.N**2

        for x_s in range(self.N):
            for y_s in range(self.N):

                x_smax , y_smax =  x_s-1 , y_s-1

                E -=  config[x_s,y_s] * (config[x_smax,y_s] + config[x_s,y_smax] + self.B)

        E = E / self.N**2

        return E

    def berechne_Waermekap(self, E1, E2):
        """
        Calculation of heat capacity.
        """

        dE = E2 - E1
        dT = (self.T_max - self.T_min) / self.T_num

        cv = dE / dT

        return cv

    def plotten(self, M_Abs, chi_Abs, u_Abs, cv_Abs):
        """
        Error plot of magnetization, susceptibility, energy and heat capacity per spin
        """

        plt.figure()
        plt.errorbar(self.temperatures,np.mean(M_Abs,1),yerr=np.std(M_Abs,1),ecolor='red')
        ax = plt.subplot()
        plt.xlabel('Temperatures J/K')
        plt.ylabel('M(T)')
        plt.title('Magnetization per spin')
        plt.text(0.7, 0.7, ' N = {} \n N_realis = {} \n N_therm = {} \n N_MC = {} \n  T_num = {}'.format(self.N,self.N_realis,self.N_therm,self.N_MC,self.T_num), transform = ax.transAxes,
            bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})
        plt.show()

        plt.figure()
        plt.errorbar(self.temperatures,np.mean(chi_Abs,1),yerr=np.std(chi_Abs,1),ecolor='red')
        ax = plt.subplot()
        plt.xlabel('Temperatures J/K')
        plt.ylabel('Chi(T)')
        plt.title('Magnetic susceptibility per spin')
        plt.text(0.7, 0.7, ' N = {} \n N_realis = {} \n N_therm = {} \n N_MC = {} \n  T_num = {}'.format(self.N,self.N_realis,self.N_therm,self.N_MC,self.T_num), transform = ax.transAxes,
            bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})
        plt.show()

        plt.figure()
        plt.errorbar(self.temperatures,np.mean(u_Abs,1),yerr=np.std(u_Abs,1),ecolor='red')
        ax = plt.subplot()
        plt.xlabel('Temperatures J/K')
        plt.ylabel('U(T)')
        plt.title('Energy per spin')
        plt.text(0.7, 0.05, ' N = {} \n N_realis = {} \n N_therm = {} \n N_MC = {} \n  T_num = {}'.format(self.N,self.N_realis,self.N_therm,self.N_MC,self.T_num), transform = ax.transAxes,
            bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})
        plt.show()

        plt.figure()
        plt.errorbar(self.temperatures[0:-1],np.mean(cv_Abs,1),yerr=np.std(cv_Abs,1),ecolor='red')
        ax = plt.subplot()
        plt.xlabel('Temperatures J/K')
        plt.ylabel('cv(T)')
        plt.title('Heat coefficient per spin')
        plt.text(0.7, 0.05, ' N = {} \n N_realis = {} \n N_therm = {} \n N_MC = {} \n  T_num = {}'.format(self.N,self.N_realis,self.N_therm,self.N_MC,self.T_num), transform = ax.transAxes,
            bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})
        plt.show()

        return


    def mainloop_lin(self):
        """
        Iteration loop without parallelization.
        Used in
        """


        # Setze container fuer die Messgroessen zur Abspeicherung auf
        M_Abs = np.ones((len(self.temperatures), self.N_realis))   # Magnetisierung
        chi_Abs = np.ones((len(self.temperatures), self.N_realis))
        u_Abs = np.ones((len(self.temperatures), self.N_realis))
        cv_Abs = np.ones((len(self.temperatures), self.N_realis))

        # Iteriere druch temperatures
        for T_ind in range(self.T_num):

             # Setze Temperatur
             T = self.temperatures[T_ind]

             # Als Orientierung fue die Laufzeit
             print("Wir befinden uns bei Temperatur {:.1f}".format(T))

             # Erstelle mehrere Realisierungen für eine Temperatur um ungünstige Start
             # configurationen, welche schlecht thermalisieren aufzufangen

             M,chi,u,cv = [],[],[],[]

             for realis in range(self.N_realis):

                 # Initialisiere Messgroesse pro Realisierung
                 M_realis = np.zeros(self.N_MC)
                 chi_realis = np.zeros(self.N_MC)
                 u_realis = np.zeros(self.N_MC)

                 # Initialisiere spin zustand
                 zustand = self.initialisiere_Zustand()

                 # Bringe zustand in thermisches Gleichgewicht
                 self.thermalisiere_Zustand(zustand, T)

                 # Sample durch Mikrozustände im Gleichgewicht
                 for MC_iter in range(self.N_MC):

                     # Update das Spin system in den nächsten Mikrozustand
                     self.MC_metro_update(zustand, T)

                     # Visualisierung (kann für den Algorithmus ignoriert werden)
                     if self.live_plotten:
                         self.ax.clear()
                         self.ax.set_title("MC Schritte fuer T={:.1f} und B={}".format(T,self.B))
                         self.ax.imshow(zustand, cmap = self.colorm)
                         plt.pause(0.01)

                     # Speichere alle Mikrozustände dieser Realisierung
                     M_realis[MC_iter] = self.berechne_Magnetisierung(zustand)
                     chi_realis[MC_iter] = self.berechne_Suszeptibilitaet(M_realis[MC_iter])
                     u_realis[MC_iter] = self.berechne_Energie(zustand)

             M_Abs[T_ind, realis] = np.mean(np.abs(M_realis))
             chi_Abs[T_ind, realis] = np.mean(chi_realis)
             u_Abs[T_ind, realis] = np.mean(u_realis)

             cv_Abs[T_ind, realis] = self.berechne_Waermekap(u_Abs[T_ind-1, realis],u_Abs[T_ind, realis])


        return M_Abs, chi_Abs, u_Abs, cv_Abs

    def mainloop_T(self, T_ind):
        """
        Iteration loop with parallelization of the different temperatures
        Used in
        """

        # Setze Temperatur
        T = self.temperatures[T_ind]

        # Erstelle mehrere Realisierungen für eine Temperatur um ungünstige Start
        # configurationen, welche schlecht thermalisieren aufzufangen

        M,chi,u,cv = [],[],[],[]

        for realis in range(self.N_realis):

            # Initialisiere Messgroesse pro Realisierung
            M_realis = np.zeros(self.N_MC)
            chi_realis = np.zeros(self.N_MC)
            u_realis = np.zeros(self.N_MC)

            # Initialisiere spin zustand
            zustand = self.initialisiere_Zustand()

            # Bringe zustand in thermisches Gleichgewicht
            self.thermalisiere_Zustand(zustand, T)

            # Sample durch Mikrozustände im Gleichgewicht
            for MC_iter in range(self.N_MC):

                # Update das Spin system in den nächsten Mikrozustand
                self.MC_metro_update(zustand, T)

                # Visualisierung (kann für den Algorithmus ignoriert werden)
                if self.live_plotten:
                    self.ax.clear()
                    self.ax.set_title("MC Schritte fuer T={:.1f} und B={}".format(T,self.B))
                    self.ax.imshow(zustand, cmap = self.colorm)
                    plt.pause(0.01)

                # Speichere alle Mikrozustände dieser Realisierung
                M_realis[MC_iter] = self.berechne_Magnetisierung(zustand)
                chi_realis[MC_iter] = self.berechne_Suszeptibilitaet(M_realis[MC_iter])
                u_realis[MC_iter] = self.berechne_Energie(zustand)

            M.append(np.mean(np.abs(M_realis)))
            chi.append(np.mean(chi_realis))
            u.append(np.mean(u_realis))

        return M, chi, u

    def mainloop_N_realis(self, realis, T):
        """
        Iteration loop with parallelization of the different realizations per temperature
        Used in
        """

        # Initialisiere Messgroesse pro Realisierung
        M_realis = np.zeros(self.N_MC)
        chi_realis = np.zeros(self.N_MC)
        u_realis = np.zeros(self.N_MC)

        # Initialisiere spin zustand
        zustand = self.initialisiere_Zustand()

        # Bringe zustand in thermisches Gleichgewicht
        self.thermalisiere_Zustand(zustand, T)

        # Sample durch Mikrozustände im Gleichgewicht
        for MC_iter in range(self.N_MC):

            # Update das Spin system in den nächsten Mikrozustand
            self.MC_metro_update(zustand, T)

            # Visualisierung (kann für den Algorithmus ignoriert werden)
            if self.live_plotten:
                self.ax.clear()
                self.ax.set_title("MC Schritte fuer T={:.1f} und B={}".format(T,self.B))
                self.ax.imshow(zustand, cmap = self.colorm)
                plt.pause(0.01)

            # Speichere alle Mikrozustände dieser Realisierung
            M_realis[MC_iter] = self.berechne_Magnetisierung(zustand)
            chi_realis[MC_iter] = self.berechne_Suszeptibilitaet(M_realis[MC_iter])
            u_realis[MC_iter] = self.berechne_Energie(zustand)

            M    = np.mean(np.abs(M_realis))
            chi  = np.mean(chi_realis)
            u    = np.mean(u_realis)

        return M, chi, u
