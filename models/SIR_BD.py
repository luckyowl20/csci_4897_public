from .SIR import SIRModel
import numpy as np
import matplotlib.pyplot as plt

class SIRBDModel(SIRModel):
    def __init__(self, N0, s0, i0, r0, beta, gamma, epsilon, delta, tmax, stepsize):
        """
        Initialize the SIR model with birth and death dynamics.
        Parameters
        ----------
        N0 : int or float
            Initial total population size.
        s0 : int or float
            Initial number of susceptible individuals.
        i0 : int or float
            Initial number of infected individuals.
        r0 : int or float
            Initial number of recovered individuals.
        beta : float
            Transmission rate of the infection.
        gamma : float
            Recovery rate of infected individuals.
        epsilon : float
            Birth rate (rate at which new individuals enter the population).
        delta : float
            Death rate (rate at which individuals leave the population).
        tmax : float
            Maximum time for simulation.
        stepsize : float
            Time step size for the simulation.
        Notes
        -----
        The population size is dynamic and changes over time according to birth and death rates.
        """
        super().__init__(s0, i0, r0, beta, gamma, tmax, stepsize)

        # epsilon = birth rate
        # delta = death rate
        self.epsilon = epsilon
        self.delta = delta

        # population is now dynamic
        self.N = np.zeros(len(self.time))
        self.N[0] = N0

    # method override
    def run_model(self):
        
        for idx, t in enumerate(self.time):
            if idx == 0:
                self.S[0] = self.s0   # intial number of susceptible individuals     
                self.I[0] = self.i0   # intial number of infected individuals
                self.R[0] = self.r0   # intial number of recovered individuals
            else:

                dS_dt = (-self.beta * self.S[idx-1] * self.I[idx-1] / self.N[idx-1]) + (self.epsilon * self.N[idx-1]) - (self.delta * self.S[idx-1])
                dI_dt = (self.beta * self.S[idx-1] * self.I[idx-1] / self.N[idx-1]) - (self.gamma * self.I[idx-1]) - (self.delta * self.I[idx-1])
                dR_dt = (self.gamma * self.I[idx-1]) - (self.delta * self.R[idx-1])

                # new equation for dN_dt
                dN_dt = (self.epsilon - self.delta) * self.N[idx-1]

                # update equations for eulers method
                stepsize = self.stepsize
                self.S[idx] = self.S[idx-1] + dS_dt * stepsize
                self.I[idx] = self.I[idx-1] + dI_dt * stepsize
                self.R[idx] = self.R[idx-1] + dR_dt * stepsize
                self.N[idx] = self.N[idx-1] + dN_dt * stepsize
        

        return self.S, self.I, self.R, self.N, self.time
    
    # method override
    def plot(self, title='SIR-BD Model', show=False):

        # turn off auto rendering of plots if we dont want to show it
        # additional plot changes outside of this still render 
        if not show:
            plt.ioff()  

        fig, ax = plt.subplots()
        ax.plot(self.time, self.S, color='b', label='Susceptible')
        ax.plot(self.time, self.I, color='r', label='Infected')
        ax.plot(self.time, self.R, color='k', label='Recovered')  

        ax.plot(self.time, self.N, color="g", label="Population size")
        
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel('Population')
        ax.legend()
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0) 

        return fig, ax