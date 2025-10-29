import numpy as np
import pandas as pd
from scipy.stats import nbinom

class BranchingProcessNB:
    """
    Branching process that starts from a single infection and counts the # of secondary infections
    for each generation G. Each infection creates NB(R_0, k) additional infections.
    """
    def __init__(self, R0: float, k: int, G_max: int, max_infec: int = 1e7):
        self.R0 = R0               # mean num of 2ndary infections
        self.k = k                 # dispersion
        self.G_max = G_max         # max num of generations
        self.max_infec = max_infec # runtime bound on number of infections

        # scipy nb distribution params
        self.p = self.k / (self.k + self.R0)
        self.n = self.k

        # random seed, hard coded
        self.rng = np.random.default_rng(101)

    def _offspring_sum(self, current: int) -> int:
        """
        draw total offspring for current number of infectious individuals
        """

        if current <= 0: return 0

        return int(nbinom.rvs(n=current * self.n, p=self.p, random_state=self.rng))
    
    def simulate_step(self) -> bool:
        """
        simulate one step in the process, returns true if process goes extinct
        """

        current = 1
        for _ in range(self.G_max):
            # process died
            if current == 0: return True

            # process didnt die
            next_gen = self._offspring_sum(current)

            # process blew up
            if next_gen > self.max_infec: return False

            current = next_gen

        # if we exit the loop without hitting zero, count as non-extinct
        return current == 0

    def estimate_extinction_prob(self, n_trials: int = 100000) -> float:
        """
        monte carlo estimate of extinction prob
        """
        extinct = 0
        for _ in range(n_trials):
            if self.simulate_step():
                extinct += 1
        return extinct / n_trials
    
