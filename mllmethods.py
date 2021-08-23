from ProbDistribution import ProbabilityDistribution, Labelling
import numpy as np


def br(P) -> Labelling:
    M = P.marginal()
    return Labelling(np.where(M >= 1/2, 1, 0))


def dbr(P: ProbabilityDistribution) -> Labelling:
    n = P.getNumberOfLabels()
    M = P.marginal()
    ybr = np.where(M >= 1/2, 1, 0)
    L = Labelling.zeros(n)
    for i in range(n):
        vi = ybr[i]
        ybr[i] = 0
        p0 = P.fulljoint(ybr)
        ybr[i] = 1
        p1 = P.fulljoint(ybr)
        ybr[i] = vi

        L[i] = 1 if (p1+1e-20)/(p0+p1+2e-20) >= 1/2 else 0
    return L


def _rpc_labelscore(P, i):
    n = P.getNumberOfLabels()
    s = 0.0
    request1 = np.empty((2, 2), dtype=int)
    request2 = np.empty((2, 2), dtype=int)
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
    return s


def clr(P: ProbabilityDistribution) -> Labelling:
    n = P.getNumberOfLabels()
    Pm = P.marginal()
    Pmsum = n-sum(Pm)

    return Labelling([1 if(_rpc_labelscore(P, i)+Pm[i] >= Pmsum) else 0 for i in range(n)])


def rpc(P: ProbabilityDistribution):
    n = P.getNumberOfLabels()
    scores = [_rpc_labelscore(P, i) for i in range(n)]
    temp = np.argsort(-np.array(scores))
    ranking = np.empty_like(temp)
    ranking[temp] = np.arange(n)
    return ranking+1


def classifier_chain(P: ProbabilityDistribution):
    n = P.getNumberOfLabels()
    given = np.empty((n, 2), dtype=int)
    given[:, 0] = np.arange(n)
    ask = np.array([[0, 1]])
    perm = np.random.permutation(range(n))
    for i, li in enumerate(perm):
        ask[0, 0] = li
        p = P.jointCond(ask, given[perm[:i]])
        if(p > 0.5):
            given[li, 1] = 1
        else:
            given[li, 1] = 0

    return Labelling(given[:, 1])


def classifier_chain_ranking(P: ProbabilityDistribution):
    n = P.getNumberOfLabels()
    chain = np.empty(n, dtype=int)
    probs = np.empty(n, dtype=float)
    for i in range(n):
        ask = np.array([[i, 1]])
        given = np.array([(i, c) for i, c in enumerate(chain)], dtype=int)
        p = P.jointCond(ask, given)
        probs[i] = p
        if(p > 1/2):
            chain[i] = 1
        else:
            chain[i] = 0

    temp = np.argsort(-probs)
    ranking = np.empty_like(temp)
    ranking[temp] = np.arange(n)
    return ranking+1


def LabelPowerset(P: ProbabilityDistribution) -> Labelling:
    return P.mode()
