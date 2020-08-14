import numpy as np
from typing import Iterator, Tuple, List
import itertools
import math


class Annotation:
    pass


class Labelling(Annotation):
    @staticmethod
    def iterAll(n: int):
        """
        All labellings returned here are the same. If you want one of it, you should make a copy.
        """
        for c in itertools.product([0, 1], repeat=n):
            yield Labelling(c)

    def __init__(self, vals):
        self.vals = vals

    def copy(self) -> 'Labelling':
        return Labelling(list(self.vals))

    def __getitem__(self, i):
        return self.vals[i]

    def __setitem__(self, i, value):
        self.vals[i] = value

    def __iter__(self):
        return self.vals.__iter__()

    def __next__(self):
        return self.vals.__next__()

    def __len__(self):
        return len(self.vals)

    def __str__(self):
        return str(self.vals)

    def __repr__(self):
        return str(self.vals)

    def positives(self) -> int:
        return sum(self.vals)

    @staticmethod
    def zeros(size):
        return Labelling([0]*size)

    @staticmethod
    def ones(size):
        return Labelling([1]*size)


class ProbabilityDistribution:
    def marginal(self):
        return np.array([self.joint(np.array([[i, 1]])) for i in range(self.getNumberOfLabels())])

    def joint(self, vals):
        # n = self.getNumberOfLabels()
        # m = n-len(v)
        # v2 = np.empty((m, 2), dtype=np.int)
        # v2[:, 0] = [i for i in range(n) if i not in v[:, 0]]
        s = 0.0
        # newv = np.vstack((v, v2))
        # for l in itertools.product([0, 1], repeat=m):
        #     newv[len(v):, 1] = l
        #     s += self.fulljoint(newv[newv[:, 0].argsort()][:, 1])

        for L, p in self.iterProbs():
            for lcode, lvalue in vals:
                if(L[lcode] != lvalue):
                    break
            else:
                s += p
        return s

    def fulljoint(self, v):
        raise NotImplementedError

    def jointCond(self, v, c):
        if(len(c) == 0):
            return self.joint(v)
        newv = np.vstack((v, c))
        pj = self.joint(newv)
        if(pj == 0):
            return 0
        return pj/self.joint(c)

    def iterProbs(self, threshold=0) -> Iterator[Tuple[Labelling, float]]:
        for L in Labelling.iterAll(self.getNumberOfLabels()):
            p = self.fulljoint(L.vals)
            if(p > threshold):
                yield L, p

    def getNumberOfLabels(self) -> int:
        raise NotImplementedError

    def mode(self) -> Labelling:
        raise NotImplementedError

    def getProbSize(self, i: int, size: int) -> float:
        s = 0.0
        for L, p in self.iterProbs(threshold=0):
            if(L[i] == 1 and L.positives() == size):
                s += p
        return s

    def _sum(self):
        """
        Always should be 1. This method is just for verification and debugging.
        """
        s = 0.0
        for _, p in self.iterProbs():
            s += p
        assert(s >= 1.0-1e-4 and s <= 1.0+1e-4), "Probability distribution summed up to %f" % s
        return s


def precomputeProbSize(probdist: ProbabilityDistribution):
    n = probdist.getNumberOfLabels()
    probsize = np.zeros((n, n+1), dtype=np.float)
    for L, p in probdist.iterProbs():
        s = L.positives()
        idxs = np.where(L == 1)[0]
        probsize[idxs, s] += p
    return probsize


def precomputePairProb(probdist: ProbabilityDistribution):
    n = probdist.getNumberOfLabels()
    probmatrix = np.zeros((n, n), dtype=np.float)
    marginal = np.zeros(n, dtype=np.float)
    for L, p in probdist.iterProbs():
        idxs_pos = [i for i, l in enumerate(L) if(l == 1)]
        idxs_neg = [i for i, l in enumerate(L) if(l == 0)]
        probmatrix[np.ix_(idxs_pos, idxs_neg)] += p
        marginal[idxs_pos] += p
    probmatrix[np.diag_indices_from(probmatrix)] = marginal
    return probmatrix


class CLRBadDistribution(ProbabilityDistribution):
    def __init__(self, n, m, epsilon=1e-5):
        assert(m <= n)
        self.n = n
        self.m = m
        self.epsilon = epsilon
        self.p1s = (m+1)/(2*(n+1))
        assert(epsilon*m <= 1-self.p1s)

    def marginal(self):
        n = self.n
        m = self.m
        ret = np.array([self.p1s for i in range(n)])
        a = np.full(self.m, self.epsilon)
        ret[:m] += a
        return ret

    def joint(self, v):
        """
        v: First column is label id. Second columns is the value.
        """
        m = self.m
        n = self.getNumberOfLabels()
        if(len(v) == 1):
            p = self.marginal()[v[0][0]]
            if(v[0][1] == 1):
                return p
            return 1-p

        if(len(v) == 2):
            ysum = v[:, 1].sum()
            if(ysum == 2):
                return self.p1s
            elif(ysum == 0):
                return 1-self.p1s-self.epsilon
            if(v[0][0] >= m):
                if(v[1][0] >= m):
                    return 0.0
                if(v[0][1] == 1):
                    return 0.0
            elif(v[1][0] >= m):
                if(v[1][1] == 1):
                    return 0.0

            return self.epsilon

        if(len(v) == n):
            if(len(v.shape) == 2):
                v = v[v[:, 0].argsort()][:, 1]
            return self.fulljoint(v)

        if(v[v[:, 0] >= m].sum() >= 1):
            return 0.0
        ysum = v[:m, 1].sum()
        if(ysum == 0):
            return 1-self.p1s-self.epsilon
        return self.epsilon

    def fulljoint(self, v):
        ysum = sum(v)
        if(ysum == 0):
            return 1-self.p1s-self.epsilon
        if(ysum == self.n):
            return self.p1s
        if(sum(v[:self.m]) == 1 and ysum == 1):
            return self.epsilon
        return 0.0

    def getNumberOfLabels(self) -> int:
        return self.n

    def mode(self) -> Labelling:
        maxidx = np.argmax((self.epsilon, self.p1s, 1-self.p1s-self.epsilon*self.m))
        if(maxidx == 0):
            L = Labelling.zeros(self.n)
            L[0] = 1
            return L
        if(maxidx == 1):
            return Labelling.ones(self.n)
        return Labelling.zeros(self.n)

    def iterProbs(self, threshold=0) -> Iterator[Tuple[Labelling, float]]:
        yield Labelling.ones(self.n), self.p1s
        yield Labelling.zeros(self.n), 1-self.p1s-self.epsilon*self.m
        for i in range(self.m):
            L = Labelling.zeros(self.n)
            L[i] = 1
            yield L, self.epsilon

    def print(self):
        n = self.n
        M = np.empty((n, n), dtype=np.float)
        for i in range(n):
            for j in range(n):
                if(i == j):
                    M[i, i] = self.marginal()[i]
                else:
                    M[i, j] = self.joint(np.array([(i, 1), (j, 0)]))
        print(M)

    def getProbSize(self, i, size):
        if(size == 1 and i < self.m):
            return self.epsilon
        elif(size == self.n):
            return self.fulljoint([1]*self.n)
        elif(size == 0):
            return self.fulljoint([0]*self.n)
        return 0.0


class CLRBadDistribution2(ProbabilityDistribution):
    def __init__(self, n, m, k):
        assert(m <= n)
        assert(k <= m//2)
        self.n = n
        self.m = m
        self.k = k
        self.p1s = (m+1)/(2*(n+1))# * (n - k)/n
        self.cache_combs = [math.comb(m, i) for i in range(k+1)]
        self.probsize = precomputeProbSize(self)
        self.pairprob = precomputePairProb(self)

    def fulljoint(self, v):
        n, m, k = self.n, self.m, self.k
        ysum_left = sum(v[:m])
        ysum_right = sum(v[m:])

        if(ysum_left <= k and ysum_right == 0):
            return (1-self.p1s-self.m*1e-10)/(k+1) / self.cache_combs[ysum_left]
            # return (1-self.p1s)/sum(self.cache_combs)

        # if(ysum_left >= m-k and ysum_right == n-m):
        # if(ysum_left >= m-k and ysum_right == 0):
        #     return self.p1s/(sum(self.cache_combs)+1)
        # if(ysum_left+ysum_right == n):
        #     return self.p1s/(sum(self.cache_combs)+1)
        if(ysum_left+ysum_right == n):
            return self.p1s
        if(ysum_right == 0 and ysum_left == 1):
            return 1e-10
        return 0.0

    def getNumberOfLabels(self) -> int:
        return self.n

    def mode(self):
        if(self.p1s > self.fulljoint([0]*self.n)):
            return Labelling.ones(self.n)
        return Labelling.zeros(self.n)

    def marginal(self):
        return self.pairprob.diagonal()

    def iterProbs(self, threshold=0) -> Iterator[Tuple[Labelling, float]]:
        yield Labelling.ones(self.n), self.p1s
        if(self.k == 0):
            yield Labelling.zeros(self.n), self.fulljoint([0]*self.n)
            for i in range(self.m):
                L = Labelling.zeros(self.n)
                L[i] = 1
                yield L, 1e-10
        else:
            for i in range(self.k+1):
                for c in itertools.combinations(list(range(self.m)), self.k):
                    l = [0]*self.n
                    for v in c:
                        l[v] = 1
                    yield Labelling(l), self.fulljoint(l)

    def getProbSize(self, i: int, size: int) -> float:
        return self.probsize[i, size]

    def joint(self, vals):
        if(len(vals) == 2 and vals[:, 1].sum() == 1):
            i, j = vals[:, 0]
            vi, vj = vals[:, 1]
            if(vi == 1):
                return self.pairprob[i, j]
            return self.pairprob[j, i]
        return super().joint(vals)

    def print(self):
        n = self.n
        M = np.empty((n, n), dtype=np.float)
        for i in range(n):
            for j in range(n):
                if(i == j):
                    M[i, i] = self.marginal()[i]
                else:
                    M[i, j] = self.joint(np.array([(i, 1), (j, 0)]))
        print(M)
