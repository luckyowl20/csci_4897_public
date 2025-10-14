import numpy as np
import matplotlib.pyplot as plt

class SIRModel:

    def __init__(self, s0, i0, r0, beta, gamma, tmax, stepsize):
        """
        Initializes the SIR (Susceptible-Infected-Recovered) model parameters and state variables.

        Args:
            s0 (float): Initial number of susceptible individuals.
            i0 (float): Initial number of infected individuals.
            r0 (float): Initial number of recovered individuals.
            beta (float): Transmission rate of the infection.
            gamma (float): Recovery rate of infected individuals.
            tmax (float): Maximum time for simulation.
            stepsize (float): Time step size for simulation.

        Attributes:
            s0 (float): Initial susceptible count.
            i0 (float): Initial infected count.
            r0 (float): Initial recovered count.
            beta (float): Infection transmission rate.
            gamma (float): Recovery rate.
            tmax (float): Simulation end time.
            stepsize (float): Simulation time step.
            N (float): Total population size.
            time (np.ndarray): Array of time points for simulation.
            S (np.ndarray): Array to store susceptible counts over time.
            I (np.ndarray): Array to store infected counts over time.
            R (np.ndarray): Array to store recovered counts over time.
        """

        self.s0 = s0
        self.i0 = i0
        self.r0 = r0
        self.beta = beta
        self.gamma = gamma
        self.tmax = tmax
        self.stepsize = stepsize

        self.N = s0 + i0 + r0

        # Time points
        self.time = np.arange(0, tmax + stepsize, stepsize)
        self.S = np.zeros(len(self.time))
        self.I = np.zeros(len(self.time))
        self.R = np.zeros(len(self.time))
    
    def run_model(self):
        for idx, t in enumerate(self.time):
            if idx == 0:
                self.S[0] = self.s0   # initial number of susceptible individuals     
                self.I[0] = self.i0   # initial number of infected individuals
                self.R[0] = self.r0   # initial number of recovered individuals
            else:

                dS_dt = -self.beta * self.S[idx-1] * self.I[idx-1] / self.N
                dI_dt = self.beta * self.S[idx-1] * self.I[idx-1] / self.N - self.gamma * self.I[idx-1]
                dR_dt = self.gamma * self.I[idx-1]

                # update equations for eulers method
                self.S[idx] = self.S[idx-1] + dS_dt * self.stepsize
                self.I[idx] = self.I[idx-1] + dI_dt * self.stepsize
                self.R[idx] = self.R[idx-1] + dR_dt * self.stepsize

        return self.S, self.I, self.R, self.time



    def plot(self, title='SIR Model', show=False):
        # turn off auto rendering of plots if we dont want to show it
        # additional plot changes outside of this still render 
        if not show:
            plt.ioff()  

        fig, ax = plt.subplots()
        ax.plot(self.time, self.S, color='b', label='Susceptible')
        ax.plot(self.time, self.I, color='r', label='Infected')
        ax.plot(self.time, self.R, color='k', label='Recovered')  
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel('Population')
        ax.legend()
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0) 

        return fig, ax
