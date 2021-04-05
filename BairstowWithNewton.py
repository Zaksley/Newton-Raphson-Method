import numpy as np
from Newton import Newton_Raphson

def uv_to_rs(p,u,v):
    eq = np.array([1,u,v])
    (q,rem) = np.polydiv(p,eq)
    return (rem[0], rem[1], q)

def F(p,U):
    div = np.polydiv(p,[1,U[0],U[1]])
    if len(div[1]) <= 1:
        return np.array([div[1][0], 0])
    else :
        return np.array([div[1][0], div[1][1]])

"""
    Uses the Bairstow method to calculates the roots of a polynomial.

    @param p : The polynomial expressed in an np.array.
    @param u : The initial u parameter
    @param v : The initial v parameter
    @param epsilon : The precision at wich we stop the algorithm.
    @param imax : The maximum amount of iterations.
    @returns The list of roots of the polynomial in inputs.
"""
def bairstow(p, u, v, epsilon, imax=20):
    (r,s,q) = uv_to_rs(p,u,v)
    (r1,s1,q1) = uv_to_rs(q,u,v)
    # Function that returns the jacobian of the input vector.
    J = lambda U:np.array([[U[0]*r1-s1, -r1],
                           [U[1]*r1   , -s1]])
    # Main Function to optimise.
    Fct = lambda U: F(p,U)
    # Execute Newton_Raphson on our function.
    res = Newton_Raphson(Fct,J,[u,v],25,10e-14,True)
    # Calculate the next polynomial
    next_p = np.polydiv(p,[1,res[0],res[1]])
    # If the next polynomial has more than 2 terms, it re-run bairstow with new 
    # polynomial.
    if len(next_p) > 2:
        return [res,bairstow(next_p,u,v,epsilon,imax)]
    else:
        return [res]

def tests():
    eq = np.array([1,6,9,4]) # x³ + 6x² + 9x + 4 = (x² + 2x + 1)(x + 4)
    # eq = np.array([2,5,19,16,30]) # 2x⁴ + 5x³ + 19x² + 16x + 30 = (x² + 2x + 6)(2x² + x + 5)
    # (r, s,q) = uv_to_rs(eq,1,2)
    # if r != 5 or s != 15 : 
        # print("[ERROR] In uv_to_rs. Wrong return values.")
    # else:
        # print("[SUCCESS] uv_to_rs() returns correct values.")
    res = bairstow(eq,2.5,0.5,10e-14)
    # res = qroot(eq,1.6,6.4,10e-14)
    print("final res : " + str(res))

if __name__ == '__main__':
    tests()
    # print(np.polydiv([1,6,9,4],[1,2,1]))