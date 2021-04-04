import numpy as np


def qroot(p, u, v, epsilon, imax=20):

    for i in range(imax):
        quad = np.array([1, u, v])
        
        (q1, r1) = np.polydiv(p, quad)
        d = r1[-1]
        c = 0
        if (len(r1) > 1):
            c = r1[-2]

        (q2, r2) = np.polydiv(q1, quad)
        h = r2[-1]
        g = 0
        if (len(r2) > 1):
            g = r2[-2]

        A = np.array([[-h, g], [-g*v, g*u-h]])
        X = np.array([c, d])
        div = 1./(v*g*g+h*(h-u*g))
        delta = div * np.dot(A,X)
        u -= delta[0]
        v -= delta[1]
        
        if (abs(delta[0]) < epsilon and abs(delta[1]) < epsilon):
            return (u, v)
    print("i exceeded max iterations")
    return(u, v)

P = np.array([6, 11, -33, -33, 11, 6])
u = 11/6
v = -33/6

(u2, v2) = qroot(P, u, v, 1.0e-14)
print(P)
quad = np.array([1, u2, v2])
(q, rem) = np.polydiv(P, quad)
print(rem)
