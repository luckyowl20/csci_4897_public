import numpy as np
import matplotlib.pyplot as plt

class SIRModel4Group:
    """
    4 group normalized SIR model that assumes heterogeneuos susceptibility and fully mixed contacts as per homework 3 problem 1.
    """
    def __init__(self, p, cbar, gamma, tmax, dt, s0_frac=0.999, i0_frac=0.001, r0_frac=0.0):
        """
        p: array-like of length 4; susceptibility multipliers for groups 1..4
        cbar: fully mixed contact rate (per capita per unit time)
        gamma: recovery rate (per unit time), same for all groups
        tmax: maximum simulation time
        dt: time step
        s0_frac, i0_frac, r0_frac: initial fractions within each group (sum to 1)
        """
        self.p = np.array(p, dtype=float)
        assert self.p.shape == (4,)
        self.cbar = float(cbar)
        self.gamma = float(gamma)
        self.tmax = float(tmax)
        self.dt = float(dt)
        
        # Equal-sized groups, normalize each group's population to 1 for convenience.
        self.N_groups = 4
        self.group_sizes = np.ones(4)  # each group size = 1 
        
        # Initial conditions per group (fractions)
        self.S0 = np.full(4, s0_frac, dtype=float)
        self.I0 = np.full(4, i0_frac, dtype=float)
        self.R0 = np.full(4, r0_frac, dtype=float)
        
        # time grid
        self.time = np.arange(0.0, self.tmax + self.dt, self.dt)
        
        # storage
        T = len(self.time)
        self.S = np.zeros((T, 4), dtype=float)
        self.I = np.zeros((T, 4), dtype=float)
        self.R = np.zeros((T, 4), dtype=float)
    
    def run_model(self):
        T = len(self.time)
        # set initial
        self.S[0, :] = self.S0
        self.I[0, :] = self.I0
        self.R[0, :] = self.R0
        
        for t in range(1, T):
            S_prev = self.S[t-1, :]
            I_prev = self.I[t-1, :]
            R_prev = self.R[t-1, :]
            
            # Fully mixed: force of infection on group i is p_i * cbar * I_total
            # Here I_total is the sum of infected fractions across equal-sized groups.
            # Since each group's population is normalized to 1, the total infected "mass" is sum(I_prev).
            I_total = np.sum(I_prev)
            lambda_i = self.p * self.cbar * I_total  # elementwise per group i
            
            dS = -lambda_i * S_prev
            dI = lambda_i * S_prev - self.gamma * I_prev
            dR = self.gamma * I_prev
            
            # Euler update
            self.S[t, :] = S_prev + dS * self.dt
            self.I[t, :] = I_prev + dI * self.dt
            self.R[t, :] = R_prev + dR * self.dt
        
        return self.S, self.I, self.R, self.time
    
    def prob1c_plot(self, cbar, show=False):
        title = r"Four-group SIR (fully mixed): $R_0=1.5$, $\gamma=3$, $\bar c={:.2f}$".format(cbar)
        if not show:
            plt.ioff()

        fig, ax = plt.subplots()

        # order groups by susceptibility, increasing
        order = np.argsort(self.p)

        # set alphas so darkest = highest susceptibility
        alphas = [0.2, 0.40, 0.70, 1.00]

        for rank, g in enumerate(order):
            alpha = alphas[rank]
            label = fr"$i_{{{g+1}}}(t)$ (p={self.p[g]:.0f})"
            ax.plot(self.time, self.I[:, g], alpha=alpha, label=label, color='green')

        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel('Infected fraction in group')
        ax.legend(loc='upper right', ncol=2, fontsize=9)
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)
        return fig, ax
    
    def prob1d_plots(self, show=False):
        if not show:
            plt.ioff()

        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 4.5))

        # ---- Left panel: s_i(t) ----
        order = np.argsort(self.p)  # least -> most susceptible
        alphas = [0.2, 0.40, 0.70, 1.00]

        for rank, g in enumerate(order):
            label = fr"$s_{{{g+1}}}(t)$ (p={self.p[g]:.0f})"
            ax_left.plot(self.time, self.S[:, g], alpha=alphas[rank],
                        color='green', label=label)

        ax_left.set_title(r"Susceptible fractions by group $s_i(t)$")
        ax_left.set_xlabel("Time")
        ax_left.set_ylabel("Susceptible fraction in group")
        ax_left.set_xlim(left=0)
        ax_left.set_ylim(0, 1.0)
        ax_left.legend(loc="upper right", ncol=2, fontsize=9)

        # ---- Right panel: \bar{p}(t) ----
        p_vec = self.p
        num = (self.S * p_vec).sum(axis=1)
        den = self.S.sum(axis=1)
        pbar = num / den

        ax_right.plot(self.time, pbar, color='k', lw=2.0, label=r"$\bar{p}(t)$")
        ax_right.set_title(r"Average susceptibility among susceptibles $\bar{p}(t)$")
        ax_right.set_xlabel("Time")
        ax_right.set_ylabel(r"$\bar{p}(t)$")
        ax_right.set_xlim(left=0)
        ax_right.set_ylim(bottom=0)
        ax_right.legend(loc="best")

        fig.suptitle(rf"$\gamma={self.gamma:.1f}$, $\bar c={self.cbar:.2f}$", y=1.02, fontsize=12)
        fig.tight_layout()
        return fig, (ax_left, ax_right)