import numpy as np

"""
1)
    f(U) + H(U) * V = 0
"""
def Newton_Raphson(f, J, U0, N, epsilon):
    '''
        No backtracking.
        f : Function that takes a vector as input ans returns a vector.
        J : Function that returns the Jacobian Matrix of a vector.
        U0 : The initial vector. Representing the starting point.
        N : The maximum amount of iteration.
        epsilon : The precision required to stop iterations.
    '''
    i = 0
    U = U0
    normes = [np.linalg.norm(f(U))]
    while i < N and normes[i] > epsilon :
        fu = f(U)
        print(J(U))
        print(fu)
        print(U)
        print("======")
        (V, residuals, rank, s) = np.linalg.lstsq(J(U),-fu) #rcond = à quoi ? 
        U = U + V
        normes.append(np.linalg.norm(fu))
        i += 1
    print(normes)
    return U


def RtoR(x):
    return 5*x+2

def JRtoR(x):
    return np.array([[5]])

def tests():
    U0 = np.array([[-20]])
    res = Newton_Raphson(RtoR,JRtoR,U0,15,1e-10)
    print(res)


    
if __name__ == '__main__':
    tests();
