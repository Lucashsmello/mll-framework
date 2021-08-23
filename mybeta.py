import numpy as np
from math import sqrt


def mybeta(a, b, size=None):
    if(a == 0):
        if(b == 0):
            if(np.random.random() > 0.5):
                if(size is None):
                    return 1.0
                return np.ones(size)
        if(size is None):
            return 0.0
        return np.zeros(size)
    elif(b == 0):
        if(size is None):
            return 1.0
        return np.ones(size)

    a = min(1e+20,a)
    b = min(1e+20,b)
    assert(a > 0 and b > 0), "a=%f, b=%f" % (a,b)
    # limit = 0.1
    # if(a < b):
    #     if(a < limit):
    #         b = limit*b/a
    #         a = limit
    # elif(b < limit):
    #     a = limit*a/b
    #     b = limit

    return np.random.beta(a, b, size=size)


def mybeta2(d, v):
    m = sqrt(d-v) + 0.5
    t_v = (0.25-d)/v
    return np.random.beta(t_v * m, t_v * (1-m))

#m=media (0<m<1)
# th=proporcao de variancia maxima (0<th<1)


def alexBeta(m, th, size=None):
    n = (1.0/(th+1e-30) - 1)
    return mybeta(m*n, (1-m)*n, size=size)

#d=dificuldade (0<d<0.25)
# l=label dependence (0<l<0.25)


def MotherbetaM(d, l, mi0, size=None):
    th0 = 1.0 - (d+l)/(mi0*(1-mi0))
    return alexBeta(mi0, th0, size=size)

#d=dificuldade (0<d<0.25)
# l=label dependence (0<l<0.25)


def MotherbetaTh(d, l, thv, size=None):
    miv = l/(d+l)
    return alexBeta(miv, thv, size=size)

# d=dificuldade
# l=label dependence
# n=numero de valores aleatorios a serem gerados.


def mybeta3(d, l, mi0, thv, size=None, size_samemother=True):
    if(size_samemother == True):
        mi = MotherbetaM(d, l, mi0)
        thi = MotherbetaTh(d, l, thv)
        return alexBeta(mi, thi, size=size)
    mis = MotherbetaM(d, l, mi0, size=size)
    this = MotherbetaTh(d, l, thv, size=size)
    return np.array([alexBeta(mi, thi) for mi, thi in zip(mis, this)], dtype=float)
