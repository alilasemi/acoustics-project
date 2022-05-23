import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

def main():

    # Physics inputs
    a = 2
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

    # Matrix to be solved
    A = np.empty((nb, nb))
    b = np.zeros((nb, 1))
    BC_rows = np.empty((2, nb))
    # Loop basis functions
    res = []
    for i in range(nb):
        coeff = np.zeros(nb)
        coeff[i] = 1
        # Get basis function and derivatives
        phi = np.polynomial.chebyshev.Chebyshev(coeff, domain=[xL, xR])
        dphi = phi.deriv(1)
        ddphi = phi.deriv(2)
        # Get ith mode of residual
        res.append( ddphi + a**2 * phi )

        # Evaluate residual at points
        A[:, i] = res[i](x)

        # BCs: try dirichlet
        BC_rows[0,  i] = phi(x[0])
        BC_rows[-1, i] = phi(x[-1])
        b[0] = 0
        b[-1] = 1
    # Replace rows of A with the BCs
    A[0] = BC_rows[0]
    A[-1] = BC_rows[-1]

    # Solve system
    xhat = np.linalg.inv(A) @ b

    # Dense output points
    num_pts = 100
    xd = np.linspace(xL, xR, num_pts)
    X = np.polynomial.chebyshev.Chebyshev(xhat[:, 0], domain=[xL, xR])(xd)

    # Plot
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    fig = plt.figure(figsize=(5, 5))
    plt.plot(xd, X, 'k', linewidth=3, label=None)
    plt.xlabel('$x$', fontsize=20)
    plt.ylabel('$X$', fontsize=20)
    plt.tick_params(labelsize=16)
    #plt.legend(loc='center left', fontsize = 20)
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.savefig('result.pdf', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
