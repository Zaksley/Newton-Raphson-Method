import numpy as np

def qroot(P, b, c, epsilon):

    d = np.zeros(3) # d = X^2 + B*X + C
    d[0] = 1
    for i in range(2000):
        d[1] = b
        d[2] = c

        (q, rem) = np.polydiv(P, d)
        s = rem[-1]
        r = 0
        if (len(rem) == 2):
            r = rem[0]

        (qq, rem) = np.polydiv(P[1:], d)
        sc = -rem[-1]
        rc = 0
        if (len(rem) == 2):
            rc = rem[0]

        sb = -c*rc
        rb = sc - b*rc
        div = 1.0/(sb*rc - sc*rb)
        delb = (r*sc - s*rc)*div
        delc = (-r*sb + s*rb)*div
        b += delb
        c += delc
        if ((abs(delb) <= epsilon*abs(b) or abs(b) < 1.0e-6) and (abs(delc) <= epsilon*abs(c) or abs(c) < 1.0e-6)):
            return (b, c)

    print("ERROR : i exceeded ITMAX") 
    return (b, c)


P = np.array([1,1,1,0]) # P = X^3 + X^2 + X = X*(X^2 + X + 1)
(b, c) = qroot(P, 3, 4, 1e-6)
print("b =", b, "c =", c)

