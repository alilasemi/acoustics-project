import matplotlib.pyplot as plt
from matplotlib import rc
import scipy, scipy.special
import numpy as np
import sympy as sp

def main():

    # Physics inputs
    gamma = 1.4     # Specific heat ratio (given for air)
    R = 287         # Gas constant (given for air)
    radius = 1      # Radius of tube
    n_r = 1         # Radial mode number
    n_theta = 1     # Angular mode number
    # Numerics inputs
    xL = 0
    xR = 1
    p = 200
    use_chebyshev_nodes = True

    # -- Compute lambda -- #
    # j_prime actually has a root at x = 0 for n_theta != 1, but this is not
    # included in the Scipy function, so consider this case separately
    if n_theta != 1 and n_r == 0:
        j_prime = 0
    # This situation is not valid, since J1 does not have a root at 0
    if n_theta == 1 and n_r == 0:
        print('J1 does not have a root at 0!')
    # Otherwise, the Scipy function is used
    else:
        j_prime = scipy.special.jnp_zeros(n_theta, n_r)[0]
    lambd = j_prime / radius

    # Number of basis functions
    nb = p + 1
    # Create grid of collocation points. Option between using Chebyshev nodes
    # (the roots of the p + 1 order Chebyshev polynomial) or using equispaced
    # points (not recommended).
    num_points = nb
    if use_chebyshev_nodes:
        x = np.polynomial.chebyshev.Chebyshev([0]*(nb) + [1], domain=[xL, xR]).roots()
    else:
        x = np.linspace(xL, xR, num_points)

    # Some function for temperature
    xs = sp.Symbol('x', real=True)
    # Options: constant vs. Gaussian bump
    # TODO
    T0_expr = 50 * sp.exp(-200 * (xs - .1)**2) + 300
    #T0_expr = 300 + .00000000001 * xs**2

    T0_func = sp.lambdify(xs, T0_expr)
    T0 = T0_func(x).reshape(-1, 1)
    dT0_dx_expr = T0_expr.diff(xs)
    dT0_dx_func = sp.lambdify(xs, dT0_dx_expr)
    dT0_dx = dT0_dx_func(x).reshape(-1, 1)

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
    omega2 = np.sort(eigvals)
    omega = np.sqrt(omega2)
    cut_on = omega[0]
    breakpoint()

    ## Dense output points
    #num_pts = 100
    #xd = np.linspace(xL, xR, num_pts)
    #X = np.polynomial.chebyshev.Chebyshev(xhat[:, 0], domain=[xL, xR])(xd)

    # Plot
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=False)

    fig = plt.figure(figsize=(5, 5))
    plt.plot(x, T0, 'k--', linewidth=3, label='$T_0$')
    plt.xlabel('$x$', fontsize=20)
    plt.ylabel('$T_0$', fontsize=20)
    plt.tick_params(labelsize=16)
    plt.legend(loc='best', fontsize = 20)
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.savefig('T0.pdf', bbox_inches='tight')

    fig = plt.figure(figsize=(5, 5))
    #plt.plot(xd, X, 'k', linewidth=3, label='$X$')
    #plt.plot(x, T0, 'k--', linewidth=3, label='$T_0$')
    plt.plot(np.real(omega), np.imag(omega), 'k.', ms=7, label='$T_0$')
    plt.xlabel('$x$', fontsize=20)
    plt.ylabel('$X$', fontsize=20)
    plt.tick_params(labelsize=16)
    plt.legend(loc='best', fontsize = 20)
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.savefig('result.pdf', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
