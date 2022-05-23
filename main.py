import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

def main():

    xL = 0
    xR = 1
    p = 2

    # Create grid of collocation points. There are p + 1 points to define a
    # polynomial, but two points are the end points, which have boundary
    # conditions applied to them instead.
    num_points = (p + 1) - 2
    x = np.linspace(xL, xR, 100)

    cheb = np.polynomial.chebyshev.Chebyshev([1, 1, 1], domain=[xL, xR])
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
