## Modules

import numpy as np
import matplotlib.pyplot as plt
import math

from newton import Newton_Raphson

### Computation of the Lagrangian points

## Force vectors

# Creates an elastic force vector
#
# @param X the point where the force is applied
# @param G The spring's base point
# @param k the spring constant
# @return The force vector
#
def elastic_force(X,G,k):
    dX = np.array([X[0] - G[0], X[1] - G[1]])
    return np.array([-k*i for i in dX])

# Creates a centrifugal force vector
#
# @param X the point where the force is applied
# @param G The center of the force
# @param k the intensity constant
# @return The force vector
#
def centrifugal_force(X,G,k):
    return np.array([k*(X[0]-G[0]), k*(X[1]-G[1])])

# Creates a gravitational force vector
#
# @param X the point where the force is applied
# @param G The object applying gravity
# @param k the intensity constant
# @return The force vector
#
def gravitational_force(X,G,k):
    d = ((X[0] - G[0])**2 + (X[1] - G[1])**2)**(3/2)
    return np.array([-k*(X[0]-G[0])/d, -k*(X[1]-G[1])/d])


## Tests

# Case : two gravitational forces with respective coefficients 1 (resp 0.01)
# originating from [0,0] (resp [1,0])

# Returns a system force (one huge and one tiny objects orbitating)
#
# @param U the position of the object
# @return The force vector applied to this object
#
def test__force(U):
    F  = gravitational_force(U, np.array([0,0]), 1)
    F += gravitational_force(U, np.array([1,0]), 0.01)
    F += centrifugal_force(U, np.array([0.01, 0]), 1)
    return F

# Returns the Jacobian matrix of the system force defined above
#
# @param U the position of the object
# @return the Jacobian matrix applied to the vector
#
def test__jacob(U):
    dF = np.zeros((2,2))
    x,y = U[0], U[1]

    dF[0,0] = 1 - ((x**2 + y**2)**(3/2) - 3*(x**2)*((x**2 + y**2)**(1/2))) / ((x**2 + y**2)**3)
    dF[0,0] -= 0.01*(((x-1)**2+y**2)**(3/2) - 3*((x-1)**2)*(((x-1)**2+y**2)**(1/2))) / (((x-1)**2 + y**2)**3)

    dF[0,1] = -(-3*x*y*((x**2 + y**2)**(1/2))) / ((x**2 + y**2)**3)
    dF[0,1] -= 0.01*(-3*(x-1)*y*(((x-1)**2+y**2)**(1/2))) / (((x-1)**2 + y**2)**3)

    dF[1,0] = (3*x*y*((x**2 + y**2)**(1/2))) / ((x**2 + y**2)**3)
    dF[1,0] += 0.01*(3*(x-1)*y*(((x-1)**2+y**2)**(1/2))) / (((x-1)**2 + y**2)**3)

    dF[1,1] = 1 - ((x**2 + y**2)**(3/2) - 3*(y**2)*((x**2 + y**2)**(1/2))) / ((x**2 + y**2)**3)
    dF[1,1] -= 0.01*(((x-1)**2+y**2)**(3/2) - 3*(y**2)*(((x-1)**2+y**2)**(1/2))) / (((x-1)**2 + y**2)**3)

    return dF

# Tries to compute the 5 Lagrangian Points
#
def test__lagrangian_points():
    # Trying different starting points in a grid, to reach the maximum of points
    Axis = [k*0.4 for k in range(-20,20)]
    Ulist = [np.array([i, j]) for i in Axis for j in Axis]
    Ufound, Points = [], []

    for i in range(len(Ulist)):
        U = Newton_Raphson(test__force, test__jacob, Ulist[i], 100000, 10**-12, True)
        Ufound.append(U)

        # Gathering close points together
        if(len(Points) == 0):
            Points.append(Ufound[i])
        new = True
        for k in range(len(Points)):
            if np.allclose(Points[k],Ufound[i], rtol=1e-2):
                new = False
        if(new and not np.allclose(Ufound[i], np.array([0,0])) and not np.allclose(Ufound[i], np.array([1,0]))):
            Points.append(Ufound[i])

    # Verifying all 5 points have been found
    assert(len(Points) == 5)

    # Verifying the points are correct
    assert(np.allclose(Points[0], np.array([-1,0]), rtol=1e-2))
    assert(np.allclose(Points[1], np.array([0.5,-math.sin(math.pi/3)]), rtol=1e-2))
    assert(np.allclose(Points[2], np.array([0.5,math.sin(math.pi/3)]), rtol=1e-2))
    assert(np.allclose(Points[3], np.array([0.86,0]), rtol=1e-2))
    assert(np.allclose(Points[4], np.array([1.16,0]), rtol=1e-2))

    # Scattering points
    # Black dots : lagrangian points
    for L in Points:
        plt.scatter(L[0],L[1], c = "black")
    # Orange diamond : massive celestial object
    plt.scatter(0, 0, c = "orange", marker = 'D')
    # Gray square : smaller celestial object
    plt.scatter(1, 0, c = "gray", marker = 's')

    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    print("Lagrangian points found :")
    return Points

# Graphs where each point in a grid leads with a color
#
def test__lagrangian_fields():
    # Trying different starting points in a grid, to reach the maximum of points
    Axis = [k*0.025 for k in range(-80,80)]
    Ulist = [np.array([i, j]) for i in Axis for j in Axis]
    Ufound, Points = [], []

    At_Fields = [[[0,0,0] for k in range(160)] for j in range(160)]

    for i in range(len(Ulist)):
        U = Newton_Raphson(test__force, test__jacob, Ulist[i], 10000, 10**-12, True)
        Ufound.append(U)

        # Gathering close points together
        if(len(Points) == 0):
            Points.append(Ufound[i])
        new = True
        for k in range(len(Points)):
            if np.allclose(Points[k],Ufound[i], rtol=1e-2):
                new = False
        if(new and not np.allclose(Ufound[i], np.array([0,0])) and not np.allclose(Ufound[i], np.array([1,0]))):
            Points.append(Ufound[i])

        # Applying a color to all starting positions relatively to their destination
        if(np.allclose(Ufound[i], Points[0])):
            c = [128,0,128]
        if(len(Points) > 1):
            if(np.allclose(Ufound[i], Points[1])):
                c = [0,255,128]
        if(len(Points) > 2):
            if(np.allclose(Ufound[i], Points[2])):
                c = [0,128,255]
        if(len(Points) > 3):
            if(np.allclose(Ufound[i], Points[3])):
                c = [255,128,0]
        if(len(Points) > 4):
            if(np.allclose(Ufound[i], Points[4])):
                c = [255,0,128]
        At_Fields[i%160][i//160] = c

        plt.imshow(At_Fields)

    # Verifying all 5 points have been found
    assert(len(Points) == 5)

    plt.xlabel("x (0,0 is at the center)")
    plt.ylabel("y")
    plt.show()

    return

# Test function to verify if test__jacob and test__force are correct
#
def test__force_n_jacob():
    U0 = np.array([1.5, 0])
    assert(np.allclose(test__force(U0), np.array([1.00565457, 0]), rtol=1e-03))
    assert(np.allclose(test__jacob(U0), np.array([[1.75259259, 0],[0, 0.6237037]])))
    return "Force and Jacobian are correct"


## Main

if __name__ == '__main__':
    print(test__force_n_jacob())

    # Use only one of the following at a time :
    print(test__lagrangian_points())
    #print(test__lagrangian_fields())