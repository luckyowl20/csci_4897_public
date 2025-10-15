from typing import List
import numpy as np
import matplotlib.pyplot as plt

class SIS():
    def __init__(self, s0: float, i0: float, beta: float, gamma: float, tmax: float, stepsize: float):
        self.s0 = s0
        self.i0 = i0
        self.beta = beta
        self.gamma = gamma
        self.tmax = tmax
        self.stepsize = stepsize

        self.R0 = self.beta / self.gamma
        
        self.analytical = None

        self.time = np.arange(0, tmax + stepsize, stepsize)
        self.s = np.zeros(len(self.time))
        self.i = np.zeros(len(self.time))
        self.error = np.zeros(len(self.time))

    def run_model(self):
        for idx, t in enumerate(self.time):
            if idx == 0:
                # initialize to s0, i0 in time record
                self.s[0] = self.s0
                self.i[0] = self.i0

            else:
                ds_dt = (-self.beta * self.s[idx-1] * self.i[idx-1]) + (self.gamma * self.i[idx-1])
                di_dt = ( self.beta * self.s[idx-1] * self.i[idx-1]) - (self.gamma * self.i[idx-1])

                # do the timestep for forward eulers 
                self.s[idx] = self.s[idx-1] + ds_dt * self.stepsize
                self.i[idx] = self.i[idx-1] + di_dt * self.stepsize
        
        return self.s, self.i, self.time

    def analytical_i(self):
        # numpy does the entire time array at once!
        numerator = 1 - (1/self.R0)
        denominator = 1 + ((1 - 1/self.R0 - self.i0) / self.i0) * np.exp(-(self.beta - self.gamma) * self.time)
        self.analytical = numerator/denominator
        return self.analytical

    def calculate_error(self, display_result=False):
        # E(Δt) = max_t |euler(Δ) - analytical(t)|
        self.error = np.max(np.abs(self.i - self.analytical))
        if display_result: print(f"Maximum Error Over Time: {round(self.error, 4)}")
        return self.error
        

    def plot(self, title='SIS Model', show=False):
        if not show:
            plt.ioff()

        fig, ax = plt.subplots()
        # ax.plot(self.time, self.s, color='b', label='Susceptible %')
        ax.plot(self.time, self.i, color='r', label='Forward Euler')
        ax.plot(self.time, self.analytical, color='k', label='Analytical Solution', linestyle='--')
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel('Population Proportion')
        ax.legend()
        ax.set_ylim(bottom=0, top=0.5)
        ax.set_xlim(left=0)

        return fig, ax
    
