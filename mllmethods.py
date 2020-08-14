from ProbDistribution import ProbabilityDistribution, Labelling
import numpy as np


def clr(P: ProbabilityDistribution) -> Labelling:
    n = P.getNumberOfLabels()
    Pm = P.marginal()
    Pmsum = n-sum(Pm)

    def labelscore(i):
        s = 0.0
        request1 = np.empty((2, 2), dtype=np.int)
        request2 = np.empty((2, 2), dtype=np.int)
        request1[0] = (i, 1)
        request2[0] = (i, 0)
        request1[1][1] = 0
        request2[1][1] = 1
        for k in range(n):
            if(k == i):
                continue

            request1[1][0] = k
            request2[1][0] = k
            x = P.joint(request1)
            y = P.joint(request2)
            # assert(x > 0 or y > 0), "%s,%f | %s,%f" % (str(request1), x, str(request2), y)
            if(x == y):
                s += 1/2
            else:
                s += x/(x+y)
        return s+Pm[i]

    return Labelling([1 if(labelscore(i) >= Pmsum) else 0 for i in range(n)])


def classifier_chain(P: ProbabilityDistribution):
    n = P.getNumberOfLabels()
    chain = []
    for i in range(n):
        ask = np.array([[i, 1]])
        given = np.array([(i, c) for i, c in enumerate(chain)], dtype=np.int)
        p = P.jointCond(ask, given)
        if(p > 1/2):
            chain.append(1)
        else:
            chain.append(0)

    return Labelling(chain)


def LabelPowerset(P: ProbabilityDistribution) -> Labelling:
    return P.mode()
