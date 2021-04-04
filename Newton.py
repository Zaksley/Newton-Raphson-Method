import matplotlib.pyplot as plt
import numpy as np
import sys

"""
1)
    f(U) + H(U) * V = 0
"""


def Newton_Raphson(f, J, U0, N, epsilon, backtrack):
    '''
        f : Function that takes a vector as input ans returns a vector.
        J : Function that returns the Jacobian Matrix of a vector.
        U0 : The initial vector. Representing the starting point.
        N : The maximum amount of iteration.
        epsilon : The precision required to stop iterations.
        backtrack : Booléen to know if we use backtracking or not 
    '''

    # === Initialization ===
    i = 0
    U = U0
    fu = f(U)
    normes = [np.linalg.norm(fu)]
    
    # Check the error case when we try to divide by 0 
    if (J(U0) == 0):
        sys.exit("Error - Division by 0")

    while i < N and normes[i] > epsilon :
        
        (V, residuals, rank, s) = np.linalg.lstsq(J(U),-fu, -1) #rcond = à quoi ? 
        
        # === Backtracking ===
        alpha = 0.5
        j = 0
        norm_fu = np.linalg.norm(fu)
        value = 0
        if (backtrack):
            value = U + V
            while np.linalg.norm(f(value)) > norm_fu and j < N: 
                V = alpha*V
                value = U + V
                j += 1

        # === Backtracking ===

        U += V
        fu = f(U)

        normes.append(norm_fu)
        i += 1

    # === Display few results ===
    print(normes)
    print(fu)

    x = range(0, len(normes), 1)
    plt.plot(x, normes)
    plt.xlabel("x")
    plt.ylabel("Normes de f(x)")
    plt.show()

    return U

# ===== Tests & interesting values ===== 

def RtoR_test1(x):
    return 5*x+2

def RtoR_test2(x):
    return x**3 - 7*(x**2) + 8*x - 3

def RtoR_test3(x):
    return 2*x**2 - 8

# Bon exemple => x = 1 et x = -8.6 : Valeur trouvée en fonction de U0
def RtoR_test4(x):
    return x**3 + 7*x**2 - 12*x + 4

def RtoR_test5(x):
    return 0.1*x**2 - 2

def RtoR_test6(x):
    return 0.05*x - 10

def RtoR_test7(x):
    return x**3 - 5*x**2

def JRtoR(x):
        # Tests 
    #return np.array([[5]])
    return np.array(3*(x**2) - 14*x + 8)
    #return np.array(4*x)
    #return np.array(3*(x**2) + 14*x - 12)
    #return  np.array(0.2*x)
    #return np.array([[0.05]])
    #return (3*x**2 -10*x)

def tests():
    U0 = np.array([[4.1]])

    res = Newton_Raphson(RtoR_test2,JRtoR,U0,15,1e-10, True)
    print(res)

# Display function 
# Used for debug
def display():
    x = range(-10, 10, 1)

    plt.plot(x, [RtoR_test2(k) for k in x])
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()
    
    
if __name__ == '__main__':
    tests()
    #display()


    
