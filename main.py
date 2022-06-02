import matplotlib.pyplot as plt
from matplotlib import rc
import scipy, scipy.special
import numpy as np
import sympy as sp

# -- Inputs -- #
gamma = 1.4     # Specific heat ratio (given for air)
R = 287         # Gas constant (given for air)
radius = 1      # Radius of tube
xL = 0          # Position of left end of tube
xR = 1          # Position of right end of tube
p = 20          # Polynomial order of spectral method
max_n_r = 9     # Radial mode number
max_n_theta = 9 # Angular mode number
nb = p + 1      # Number of basis functions

def main():

    # Number of basis functions
    nb = p + 1
    # Create grid of collocation points. Use Chebyshev nodes
    # (the roots of the p + 1 order Chebyshev polynomial).
    num_points = nb
    x = np.polynomial.chebyshev.Chebyshev([0]*(nb) + [1], domain=[xL, xR]).roots()

    # Some function for temperature
    xs = sp.Symbol('x', real=True)
    # Options: constant vs. Gaussian bump
    # TODO
    #T0_expr = 50 * sp.exp(-200 * (xs - .1)**2) + 300
    T0_expr = 300 + .00000000001 * xs**2

    T0_func = sp.lambdify(xs, T0_expr)
    T0 = T0_func(x).reshape(-1, 1)
    dT0_dx_expr = T0_expr.diff(xs)
    dT0_dx_func = sp.lambdify(xs, dT0_dx_expr)
    dT0_dx = dT0_dx_func(x).reshape(-1, 1)

    cutoff = np.empty((max_n_r + 1, max_n_theta + 1))
    # Loop over the different radial mode numbers
    for n_r in range(max_n_r + 1):
        # Loop over the different angular mode numbers
        for n_theta in range(max_n_theta + 1):
            # Compute the cutoff frequency
            cutoff[n_r, n_theta] = compute_cutoff_frequency(x, T0, dT0_dx, n_r, n_theta, p)

    # -- Plot --  #
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    # Plot the base state temperature profile
    fig = plt.figure(figsize=(5, 5))
    plt.plot(x, T0, 'k--', linewidth=3, label='$T_0$')
    plt.xlabel('$x$', fontsize=20)
    plt.ylabel('$T_0$', fontsize=20)
    plt.tick_params(labelsize=16)
    plt.legend(loc='best', fontsize = 20)
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.savefig('T0.pdf', bbox_inches='tight')
    # Plot the cutoff frequencies for different n_r, n_theta
    fig = plt.figure(figsize=(5, 5))
    for n_theta in range(max_n_theta + 1):
        plt.plot(range(max_n_r + 1), cutoff[:, n_theta], '.-', ms=10,
                label=f'$n_\\theta = {n_theta}$')
    plt.xlabel('$n_r$', fontsize=20)
    plt.ylabel('$\\omega_\\textrm{cutoff}$ (rad/s)', fontsize=20)
    plt.tick_params(labelsize=16)
    plt.legend(loc='best', fontsize=11, ncol=2)
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.savefig('result.pdf', bbox_inches='tight')
    plt.show()


def compute_cutoff_frequency(x, T0, dT0_dx, n_r, n_theta, p):
    # -- Compute lambda -- #
    # j_prime actually has a root at x = 0 for n_theta != 1, but this is not
    # included in the Scipy function, so consider this case separately
    if n_theta != 1 and n_r == 0:
        j_prime = 0
    # This situation is not valid, since J1 does not have a root at 0
    elif n_theta == 1 and n_r == 0:
        print('J1 does not have a root at 0!')
        return np.nan
    # Otherwise, the Scipy function is used
    else:
        j_prime = scipy.special.jnp_zeros(n_theta, n_r)[-1]
    lambd = j_prime / radius

    # Matrix to be solved
    A = np.empty((nb, nb))
    BC_rows = np.empty((2, nb))
    # Basis values
    phi = np.empty_like(A)
    dphi = np.empty_like(A)
    ddphi = np.empty_like(A)
    phi_BC = np.empty((2, nb))
    dphi_BC = np.empty((2, nb))
    # Loop basis functions
    res = []
    for i in range(nb):
        coeff = np.zeros(nb)
        coeff[i] = 1
        # Get basis function and derivatives
        poly_phi = np.polynomial.chebyshev.Chebyshev(coeff, domain=[xL, xR])
        poly_dphi = poly_phi.deriv(1)
        poly_ddphi = poly_phi.deriv(2)
        # Evaluate at points
        phi[:, i] = poly_phi(x)
        dphi[:, i] = poly_dphi(x)
        ddphi[:, i] = poly_ddphi(x)
        # Evaluate at BCs
        BC_points = np.array([xL, xR])
        phi_BC[:, i] = poly_phi(BC_points)
        dphi_BC[:, i] = poly_dphi(BC_points)
    # Basis values for the RHS: same as regular basis values, but with the
    # endpoints zeroed out
    phi_RHS = phi.copy()
    phi_RHS[0] = 0
    phi_RHS[-1] = 0

    # Evaluate speed of sound squared at every point
    c0_squared = gamma * R * T0

    # Compute A
    A = (
            -c0_squared * ddphi
            - (c0_squared/T0) * dT0_dx * dphi
            + lambd**2 * c0_squared * phi
    )

    # Incorporate Neumann boundary conditions by replacing the first and last
    # rows of A
    A[0] = dphi_BC[0]
    A[-1] = dphi_BC[-1]

    # Get eigenvalues and right eigenvectors (in columns)
    eigvals, eigvecs = scipy.linalg.eig(A, phi_RHS)
    # Specifically, get the eigenvalues which have no (very little) imaginary
    # part. The imaginary values are caused by numerical error in the scheme.
    real_eigvals = np.real(eigvals[np.abs(np.imag(eigvals)) < .1])
    # Also toss out negatives, and the number 0. Then, take the square root to
    # get omega, and sort smallest to largest.
    omega = np.sort(np.sqrt(real_eigvals[real_eigvals > 0.0001]))
    # Take the first one to be the cutoff frequency
    cutoff = omega[0]
    return cutoff

if __name__ == '__main__':
    main()
