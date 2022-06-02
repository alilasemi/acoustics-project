import matplotlib.pyplot as plt
from matplotlib import rc
import scipy, scipy.special
import numpy as np
import sympy as sp

# -- Inputs -- #
gamma = 1.4         # Specific heat ratio (given for air)
R = 287             # Gas constant (given for air)
radius = 1          # Radius of tube
xL = 0              # Position of left end of tube
xR = 1              # Position of right end of tube
p = 50              # Polynomial order of spectral method
max_n_r = 5         # Radial mode number
max_n_theta = 5     # Angular mode number
nb = p + 1          # Number of basis functions
T_ambient = 300     # Temperature of ambient fluid
T_bump_max = 4000   # Maximum temperature of the peak of the Gaussian bump
n_T = 5             # Number of temperatures between 0 and T_bump_max

def main():

    # Number of basis functions
    nb = p + 1
    # Create grid of collocation points. Use Chebyshev nodes
    # (the roots of the p + 1 order Chebyshev polynomial).
    num_points = nb
    x = np.polynomial.chebyshev.Chebyshev([0]*(nb) + [1], domain=[xL, xR]).roots()

    # Possible values of the temperature peak
    T_bump = np.linspace(T_ambient, T_bump_max, n_T)
    T0_list = []
    dT0_dx_list = []
    cutoff_list = []
    min_cutoff = np.empty(n_T)
    # Loop over each temprature peak
    for i in range(n_T):
        # Gaussian bump function for temperature
        xs = sp.Symbol('x', real=True)
        T0_expr = (T_bump[i] - T_ambient + 1e-12) * sp.exp(-100 * (xs - .1)**2) + T_ambient
        # Create function and evaluate at points
        T0_func = sp.lambdify(xs, T0_expr)
        T0_list.append(T0_func(x).reshape(-1, 1))
        # Compute the derivative of the profile symbolically
        dT0_dx_expr = T0_expr.diff(xs)
        # Evaluate at points
        dT0_dx_func = sp.lambdify(xs, dT0_dx_expr)
        dT0_dx_list.append(dT0_dx_func(x).reshape(-1, 1))

        cutoff = np.empty((max_n_r + 1, max_n_theta + 1))
        # Loop over the different radial mode numbers
        for n_r in range(max_n_r + 1):
            # Loop over the different angular mode numbers
            for n_theta in range(max_n_theta + 1):
                # Compute the cutoff frequency
                cutoff[n_r, n_theta] = compute_cutoff_frequency(x, T0_list[-1],
                        dT0_dx_list[-1], n_r, n_theta, p)
        cutoff_list.append(cutoff)
        # Store the minimum cutoff frequency
        min_cutoff[i] = np.nanmin(cutoff)

    # -- Plot --  #
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    # Plot the base state temperature profile for the lowest and highest
    # temperature cases
    fig = plt.figure(figsize=(5, 5))
    for i in range(n_T):
        plt.plot(x, T0_list[i], 'k', linewidth=3, label=f'$T_\\textrm{peak} = {T_bump[i]}$ K')
    plt.xlabel('$x$', fontsize=20)
    plt.ylabel('$T_0$', fontsize=20)
    plt.tick_params(labelsize=16)
    plt.legend(loc='best', fontsize = 20)
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.savefig(f'T0.pdf', bbox_inches='tight')
    # Plot the cutoff frequencies for different n_r, n_theta for the lowest and
    # highest temperature cases
    for i in [0, n_T - 1]:
        fig = plt.figure(figsize=(5, 5))
        for n_theta in range(max_n_theta + 1):
            plt.plot(range(max_n_r + 1), cutoff_list[i][:, n_theta], '.-', ms=10,
                    label=f'$n_\\theta = {n_theta}$')
        plt.xlabel('$n_r$', fontsize=20)
        plt.ylabel('$\\omega_\\textrm{cutoff}$ (Hz)', fontsize=20)
        plt.xlim([-.2, 5.2])
        plt.ylim([0, 1300])
        plt.tick_params(labelsize=16)
        plt.legend(loc='best', fontsize=12, ncol=2)
        plt.grid(linestyle='--')
        plt.tight_layout()
        plt.savefig(f'cutoff_{int(T_bump[i])}.pdf', bbox_inches='tight')
    # Plot the minimum cutoff frequency for each temperature case
    fig = plt.figure(figsize=(5, 5))
    plt.plot(T_bump, min_cutoff, 'k', linewidth=3, label=None)
    plt.xlabel('$T_0$', fontsize=20)
    plt.ylabel('$\\omega_\\textrm{cutoff}$ (Hz)', fontsize=20)
    plt.tick_params(labelsize=16)
    #plt.legend(loc='best', fontsize = 20)
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.savefig(f'min_cutoff.pdf', bbox_inches='tight')
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
    # Convert rad/s to Hz
    return cutoff / (2 * np.pi)

if __name__ == '__main__':
    main()
