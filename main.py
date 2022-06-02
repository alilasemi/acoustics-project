import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import sympy as sp

def main():

    # Physics inputs
    lambd2 = 1 - np.pi**2
    # Numerics inputs
    xL = 0
    xR = 1
    p = 200
    use_chebyshev_nodes = True

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
    thermal_variation = False
    xs = sp.Symbol('x', real=True)
    T0_expr = .9 * sp.exp(-200 * (xs - .1)**2) + .1
    T0_func = sp.lambdify(xs, T0_expr)
    T0 = T0_func(x)
    dT0_dx_expr = T0_expr.diff(xs)
    dT0_dx_func = sp.lambdify(xs, dT0_dx_expr)
    dT0_dx = dT0_dx_func(x)

    # Matrix to be solved
    A = np.empty((nb, nb))
    b = np.zeros((nb, 1))
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

    # Compute A
    A = (
            ddphi
            + thermal_variation * (1/T0) * dT0_dx * dphi
            # TODO: Add the real terms for a. When a becomes negative (small
            # wavelengths) the wave is evanescent. This is more pronounced
            # when the temperature bump is included.
            + (1) * phi
    )

    # BCs: try Neumann left only
    A[0] = dphi_BC[0]
    b[0] = 0
    A[-1] = dphi_BC[-1]
    b[-1] = 0
    breakpoint()

    # Get eigenvalues and right eigenvectors (in columns)
    eigvals, eigvecs = np.linalg.eig(A)
    lambd = np.sqrt(np.sqrt(eigvals))
    #breakpoint()

    # Solve system
    xhat = np.linalg.inv(A) @ b

    # Dense output points
    num_pts = 100
    xd = np.linspace(xL, xR, num_pts)
    X = np.polynomial.chebyshev.Chebyshev(xhat[:, 0], domain=[xL, xR])(xd)

    # Plot
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=False)
    fig = plt.figure(figsize=(5, 5))
    plt.plot(xd, X, 'k', linewidth=3, label='$X$')
    #plt.plot(x, T0, 'k--', linewidth=3, label='$T_0$')
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
