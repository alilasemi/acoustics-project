import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

def main():

    # Physics inputs
    a = 2 * np.pi
    # Numerics inputs
    xL = 0
    xR = 1
    p = 3

    # Number of basis functions
    nb = p + 1
    # Create grid of collocation points. There are p + 1 points to define a
    # polynomial, but two points are the end points, which have boundary
    # conditions applied to them instead.
    # There are as many points as basis functions
    num_points = nb
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
        b[0] = 1
        b[-1] = 1
    # Replace rows of A with the BCs
    A[0] = BC_rows[0]
    A[-1] = BC_rows[-1]

    # Solve system
    xhat = np.linalg.inv(A) @ b
    breakpoint()

    # Plot
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    fig = plt.figure(figsize=(7,7))
    plt.plot(x, y, 'k', linewidth=3, label='$y=x^2$')
    plt.xlabel('$x$', fontsize=30)
    plt.ylabel('$y$', fontsize=30)
    plt.tick_params(labelsize=20)
    plt.legend(loc='center left', fontsize = 20)
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.savefig('result.pdf', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
