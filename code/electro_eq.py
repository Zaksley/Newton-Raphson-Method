import numpy as np
import matplotlib.pyplot as ma
from numpy.polynomial.legendre import legroots, legder, legval
from math import *

from Newton import Newton_Raphson


# In the following functions, X is an (1, N) array that contains the x values of the N inner charges


# Compute the total electrostatic energy of a system of N charges located within the interval ]-1,1[ and two charges located at -1 and +1
def E(X):
    N = len(X)
    v = 0
    
    for i in range(N):
        jsum = 0
        for j in range(N):
            if (j != i):
                jsum += log(abs(X[i]-X[j]))
        v += log(abs(X[i]+1)) + log(abs(X[i]-1)) + 0.5*jsum

    return v


# Compute the gradient of E (a numpy array of shape (N,))
def grad_E(X):
    N = len(X)
    g = np.zeros(N)

    for i in range(N):
        jsum = 0
        for j in range(N):
            if (j != i):
                jsum += 1/(X[i]-X[j])
        g[i] = 1/(X[i]+1) + 1/(X[i]-1) + jsum

    return g


# Compute the Jacobian matrix of grad E
def jacobian_grad_E(X):
    N = len(X)
    J = np.zeros((N,N))

    for i in range(N):
        for j in range(N):
            if (i == j):
                ksum = 0
                for k in range(N):
                    if (k != i):
                        ksum += 1/(X[i]-X[k])**2
                J[i, j] = -1/(X[i]+1)**2 -1/(X[i]-1)**2 - ksum
            else:
                J[i, j] = 1/(X[i]-X[j])**2

    return J


# An array of 0s save for the last element which is 1
# This function is meant to be used in conjunction with Numpy's Legendre Series module to operate on single Legendre Polynomials
def P(i):
    A = [0 for j in range(i)]
    A[i-1] = 1
    return A
    

# Plot the charges' position over the real axis, along with the derivative of the (n+1)th Legendre polynomial
def plot_legendre(charges, n):
    X = np.linspace(-1,1,500)
    y = np.array([legval(x, legder(P(n+2))) for x in X])
    ma.plot(X, y, 'blue')
    ma.scatter(charges, len(charges)*[0], label="Equilibrium position", c='#ff7b00')


# Plot the difference between equilibrium positions and roots of the correspondign Legendre polynomial derivative, for charges distributions ranging from 1 to n
def plot_equilibrium_roots_error(n):
    error = [0 for i in range(n)]
    for i in range(1, n+1):
        X = np.linspace(-1, 1, i+2)[1:-1]
        solution = Newton_Raphson(grad_E, jacobian_grad_E, X, 100, 1e-9, True, False)
        roots = legroots(legder(P(i+2)))

        error[i-1] = np.linalg.norm(solution-roots)

    ma.plot(range(n), error)
    ma.xlabel("Number of charges")
    ma.yscale("log")

    
    
if __name__ == '__main__':
    nb_points = 8
    X = np.linspace(-1, 1, nb_points+2)[1:-1]
    ma.scatter(X, len(X)*[1], label="Initial distribution")
    
    solution = Newton_Raphson(grad_E, jacobian_grad_E, X, 100, 1e-9, True, False)
    ma.scatter(solution, len(solution)*[0], label="Equilibrium position found with Newton-Raphson's method")

    # First graph: initial distribution and equilibrium positions
    ma.legend()
    ma.axis([-1, 1, -1, 2])
    ma.gca().axes.get_yaxis().set_visible(False)
    ma.show()

    # Second graph: roots of the derivative of the Legendre polynomials
    plot_legendre(solution, nb_points)
    ma.axis([-1, 1, -8, 8])
    ma.show()
    
    # Third graph: error between roots and equilibrium positions
    # /!\ this takes several seconds to compute !
    plot_equilibrium_roots_error(100)
    ma.show()
