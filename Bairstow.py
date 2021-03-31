import numpy as np

def qroot(p,n,b,c,eps,NMAX=20,ITMAX=20,TINY=1.0e-6):
    '''
        //TODO
        INTEGER n,NMAX,ITMAX
        REAL b,c,eps,p(n),TINY
        PARAMETER (NMAX=20,ITMAX=20,TINY=1.0e-6)
    '''
    d = np.zeros(3)
    q = np.zeros(NMAX)
    qq = np.zeros(NMAX)
    rem = np.zeros(NMAX)
    d[2] = 1
    for i in range(ITMAX):
        d[1] = b
        d[0] = c
        (q, rem1) = np.polydiv(p,d)
        s = rem1[0]
        r = rem1[1]
        (qq, rem2) = np.polydiv(p[1:],d)
        sc = -rem2[0]
        rc = -rem2[1]
        sb = -c*rc
        rb = sc-b*rc
        div = 1.0/(sb*rc-sc*rb)
        delb = (r*sc-s*rc)*div
        delc = (-r*sb+s*rb)*div
        b = b + delb
        c = c + delc
        if (abs(delb) <= eps*abs(b) or abs(b) < TINY)
            and ((abs(delc) <= eps*abs(c)) or abs(c) < TINY):
            return (b,c)
    print("[WARNING] : ITMAX iterations overpassed.")
    return (b,c)


def tests():
    

if __name__ == '__main__':
    tests();
