# utility plotting functions
import matplotlib.pyplot as plt
from public.models.SISNormalized import SIS
import numpy as np

# homework 2, question 1d error plot
def plot_error(stepsizes):
    # run the model, find the analytical solution, calculate the error, plot per timestep
    errors = []
    for step in stepsizes:
        model = SIS(0.99, 0.01, 3, 2, 25, step)
        model.run_model()
        model.analytical_i()
        error = model.calculate_error()
        errors.append(error)

    # Plot on log-log axes
    plt.figure(figsize=(7, 5))
    plt.loglog(stepsizes, errors, 'o-', label='Error vs Step Size')
    plt.xlabel('Step Size Δt')
    plt.ylabel('Maximum Absolute Error E(Δt)')
    plt.title('Error vs Step Size (log-log)')
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.legend()
    plt.show()

# homework 2, question 2c next generation matrix eigenvalue computation
def largest_eigenvalue(C, N1, N2):
    """Compute the largest eigenvalue magnitude for a 2-group contact matrix."""
    A = np.array([
        [C[0,0], (N1/N2)*C[0,1]],
        [(N2/N1)*C[1,0], C[1,1]]
    ], dtype=float)
    return float(np.max(np.abs(np.linalg.eigvals(A))))