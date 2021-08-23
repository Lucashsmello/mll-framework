import numpy as np
from typing import Iterator, Tuple, List
import itertools
import math
from math import isclose, sqrt
from mybeta import mybeta3


class Annotation:
    pass


class Labelling(Annotation):
    @staticmethod
    def iterAll(n: int):
        # L = Labelling(None)
        for c in itertools.product([0, 1], repeat=n):
            # L.vals = c
            # yield L
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

    def where_positive(self):
        return [i for i, v in enumerate(self.vals) if v == 1]

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
        """
        Args:
            vals(np.array): matrix of two columns where each line represents a label.
        """
        s = 0.0

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
        maxL, maxp = None, -1
        for L, p in self.iterProbs(threshold=0):
            if(p > maxp):
                maxp = p
                maxL = L.copy()
        return maxL

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
        for L, p in self.iterProbs(threshold=-1):
            # print(L, "%.4f" % p)
            assert(p >= 0), "Probability value should be positive (%s,%f)" % (L, p)
            s += p
        assert(s >= 1.0-1e-4 and s <= 1.0+1e-4), "Probability distribution summed up to %f" % s
        return s

    def labelDependenceSD(self, min_level=1, max_level=-1) -> float:
        if(max_level == -1):
            max_level = self.getNumberOfLabels()
        V = np.ones((1, 2), dtype=int)
        s = 0.0
        for i in range(min_level, max_level):
            V[0][0] = i
            C = np.zeros((i, 2), dtype=int)
            C[:, 0] = np.arange(0, i)
            Pi = np.empty(2**i)
            for j, comb in enumerate(itertools.product([0, 1], repeat=i)):
                C[:, 1] = comb
                Pi[j] = self.jointCond(V, C)
            s += np.var(Pi)
        return np.sqrt(s/(max_level-min_level))

    def totalCorrelation(self, k=-1) -> float:
        M = self.marginal()
        M2 = 1-self.marginal()
        n = len(M)
        M = np.vstack((M2, M)).T
        n_range = np.arange(0, n)

        i = 0
        s = 0
        for L, p in self.iterProbs():
            if(i == k):
                break
            marg_prod = np.prod(M[L, n_range])
            s += p*np.log2(p/marg_prod)
            i += 1
        if(k >= 0):
            s *= 2**n / k
        return s

    def entropy(self, k=-1) -> float:
        n = self.getNumberOfLabels()

        i = 0
        s = 0
        for L, p in self.iterProbs():
            if(i == k):
                break
            s += p*np.log2(p)
            i += 1
        if(k >= 0):
            s *= 2**n / k
        return -s

    def print(self):
        for L, p in self.iterProbs():
            print(L, p)


class IndependentProbDist(ProbabilityDistribution):
    def __init__(self, M):
        self.M1 = M
        M0 = 1-M
        self.M = np.vstack((M0, self.M1)).T

    def joint(self, vals):
        return np.prod(self.M[vals[:, 0], vals[:, 1]])

    def fulljoint(self, v):
        return np.prod(self.M[range(len(v)), v])

    def marginal(self):
        return self.M1

    def getNumberOfLabels(self):
        return len(self.M1)

    def mode(self):
        return Labelling(self.M.argmax(axis=1))


class GenericProbDist(ProbabilityDistribution):
    def __init__(self, probs):
        self.probs = probs
        self.n = int(math.log2(len(probs)))
        self.probsize = precomputeProbSize(self)

    def fulljoint(self, labelling):
        h = 0
        for l in labelling:
            h = (h << 1) | l
        return self.probs[h]

    def getNumberOfLabels(self) -> int:
        return self.n

    @staticmethod
    def random(n: int) -> 'GenericProbDist':
        probs = np.random.random(2**n)
        probs /= probs.sum()
        return GenericProbDist(probs)

    def iterProbs(self, threshold=0) -> Iterator[Tuple[Labelling, float]]:
        for i, L in enumerate(Labelling.iterAll(self.n)):
            p = self.probs[i]
            if(p > threshold):
                yield L, p

    def getProbSize(self, i: int, size: int) -> float:
        return self.probsize[i, size]

    def joint(self, vals):
        h1 = 0
        h0 = 0
        for i, v in vals:
            x = 2**(self.n-i-1)
            if(v == 1):
                h1 += x
            else:
                h0 += x
        p = 0.0
        for i in range(2**self.n):
            if((i & h1) == h1 and (i & h0) == 0):
                p += self.probs[i]

        return p

    # def mode(self) -> Labelling:
    #     idx = np.argmax(self.probs)
    #     L = np.empty(self.n, dtype=int)
    #     for i in range(self.n):
    #         L[-i-1] = idx & 1
    #         idx = idx >> 1
    #     return Labelling(L)


class ProbDistTree(GenericProbDist):
    def __init__(self, prob_tree: List):
        self.prob_tree = prob_tree
        n = len(prob_tree)
        product_tmp = [1-prob_tree[0][0], prob_tree[0][0]]
        self.marg = np.empty(n, dtype=float)
        self.marg[-1] = prob_tree[0][0]
        for i in range(1, n):
            Pi = prob_tree[i]
            p1 = product_tmp*(1-Pi)
            p2 = product_tmp*Pi
            self.marg[-i-1] = p2.sum()
            product_tmp = np.hstack((p1, p2))

        super().__init__(product_tmp)

    def marginal(self):
        return self.marg

    @staticmethod
    def random(n, ld, dd, mu0=0.5, thv=0.5):
        """
        0<dd<0.5
        0<ld<1.0
        """
        dd = dd*dd
        ld *= sqrt(mu0*(1-mu0) - dd)
        ld = ld*ld
        assert dd+ld < mu0*(1-mu0), "%.2f+%.2f>=%.1f*(1-%.1f)!" % (dd, ld, mu0, mu0)
        # prob_tree = [np.random.random(2**i) for i in range(n)]
        prob_tree = [mybeta3(dd, ld, mu0, thv, 2**i) for i in range(n)]
        return ProbDistTree(prob_tree)


class LazyProbDistTree(ProbabilityDistribution):
    def __init__(self, callback_create_node, n):
        self.callback_create_node = callback_create_node
        self.prob_tree = [{} for _ in range(n)]
        self.prob_tree[0][''] = callback_create_node()

    def fulljoint(self, labelling):
        p = 1.0
        for i, l in enumerate(labelling):
            Pi = self.prob_tree[i]
            Lstr = ''.join([str(l) for l in labelling[:i]])
            if(l == 1):
                p *= Pi[Lstr]
            else:
                p *= 1-Pi[Lstr]
        return p

    def jointCond(self, ask, given):
        if(len(given) == 0 and ask[0][0] == 0):
            if(ask[0][1] == 1):
                return self.prob_tree[0]['']
            return 1-self.prob_tree[0]['']
        if(len(ask) == 1 and ask[0][0] == len(given) and given[:, 0].max()+1 == ask[0][0]):
            ask_vals = given[given[:, 0].argsort()][:, 1]
            Pi = self.prob_tree[len(ask_vals)]
            # j = sum([a*2**i for i, a in enumerate(ask_vals)])
            j = ''.join([str(a) for a in ask_vals])
            if(j in Pi):
                p = Pi[j]
            else:
                p = self.callback_create_node()
                Pi[j] = p
            if(ask[0][1] == 1):
                return p
            else:
                return 1-p

        return super().jointCond(ask, given)

    def getNumberOfLabels(self) -> int:
        return len(self.prob_tree)

    @staticmethod
    def random(n, ld, dd, mu0=0.5, thv=0.5):
        """
        0<dd<0.5
        0<ld<1.0
        """
        dd = dd*dd
        ld *= sqrt(mu0*(1-mu0) - dd)
        ld = ld*ld
        assert dd+ld < mu0*(1-mu0), "%.2f+%.2f>=%.1f*(1-%.1f)!" % (dd, ld, mu0, mu0)
        return LazyProbDistTree(lambda: mybeta3(dd, ld, mu0, thv), n)


class DiscartableTree(ProbabilityDistribution):
    def __init__(self, n, callback_create_node) -> None:
        self.callback_create_node = callback_create_node
        self.probs = np.empty(n, dtype=float)
        self.i = 0

    def fulljoint(self, labelling):
        p = np.prod(self.probs)
        self.probs = None
        return p

    def jointCond(self, ask, given):
        p = self.callback_create_node()
        if(p > 0.5):
            self.probs[self.i] = p
        else:
            self.probs[self.i] = 1-p
        self.i += 1
        return p

    def getNumberOfLabels(self) -> int:
        return len(self.probs)

    @staticmethod
    def random(n, ld, dd, mu0=0.5, thv=0.5):
        """
        0<dd<0.5
        0<ld<1.0
        """
        dd = dd*dd
        ld *= sqrt(mu0*(1-mu0) - dd)
        ld = ld*ld
        assert dd+ld < mu0*(1-mu0), "%.2f+%.2f>=%.1f*(1-%.1f)!" % (dd, ld, mu0, mu0)
        return DiscartableTree(n, lambda: mybeta3(dd, ld, mu0, thv))


def precomputeProbSize(probdist: ProbabilityDistribution):
    n = probdist.getNumberOfLabels()
    probsize = np.zeros((n, n+1), dtype=float)
    for L, p in probdist.iterProbs():
        idxs = L.where_positive()
        probsize[idxs, len(idxs)] += p
    return probsize


def precomputePairProb(probdist: ProbabilityDistribution):
    n = probdist.getNumberOfLabels()
    probmatrix = np.zeros((n, n), dtype=float)
    marginal = np.zeros(n, dtype=float)
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


class TwoDist(ProbabilityDistribution):
    def __init__(self, pair_probs):
        self.n = len(pair_probs)
        self.pair_probs = pair_probs
        self.marg = np.diagonal(pair_probs)
        self.coc = self._cochange()
        self.probsize = precomputeProbSize(self)

    def marginal(self):
        return self.marg

    def _cochange(self):
        M = self.marginal()
        return self.pair_probs - M.reshape(self.n, 1)*M.reshape(1, self.n)

    def joint(self, vals):
        """
        v: First column is label id. Second column is the value.
        """
        def getMarginal(i, v):
            m = self.marginal()[i]
            return m*v+(1-m)*(1-v)

        m = len(vals)
        if(m == 1):
            return getMarginal(*vals[0])

        if(m == 2):
            l1, v1 = vals[0]
            l2, v2 = vals[1]
            p1 = getMarginal(l1, v1)
            p2 = getMarginal(l2, v2)
            mu = self.coc[l1, l2]
            if(v1 == v2):
                return p1*p2+mu
            return p1*p2-mu

        U = set(range(m))
        ret = 0.0
        for c in itertools.combinations(range(m), 2):
            c1, c2 = vals[list(c)]
            ccset = U-set(c)
            ps = [getMarginal(l, v) for l, v in vals[list(ccset)]]
            ps = np.prod(ps) * self.coc[(c1[0], c2[0])]
            if(c1[1] != c2[1]):
                ret -= ps
            else:
                ret += ps

        ppp = np.prod([getMarginal(l, v) for l, v in vals])
        ret = ret+ppp
        if(abs(ret) < 1e-8):
            return 0.0
        return ret

    def fulljoint(self, vals):
        return self.joint(np.array([[i, v] for i, v in enumerate(vals)]))

    def getNumberOfLabels(self) -> int:
        return self.n

    def getProbSize(self, i: int, size: int) -> float:
        return self.probsize[i, size]


class TwoDist2(TwoDist):
    def __init__(self, coc):
        self.marg = np.diagonal(coc)**0.5
        self.coc = coc
        self.n = len(coc)
        M = self.marg
        self.pair_probs = self.coc + M.reshape(self.n, 1)*M.reshape(1, self.n)

        self.probsize = precomputeProbSize(self)


class RPCbadDistribution(ProbabilityDistribution):
    def __init__(self, n, epsilon=1e-10, kind=0):
        self.n = n
        assert(n >= 8 and n % 4 == 0)
        self.epsilon = epsilon
        self.kind = kind
        assert(kind == 0 or kind == 1)

    def fulljoint(self, vals):
        num_pos = sum(vals)
        num_pos_A = sum(vals[:self.n//4])
        num_pos_B = sum(vals[self.n//4:self.n//2])
        if(self.kind == 0):
            if(num_pos == num_pos_A == self.n//4):
                return 3/4-self.n*self.epsilon
            if(num_pos_A == 0 and num_pos == (self.n-self.n//4)):
                return 1/4
            if(num_pos == 1):
                if(num_pos_A == 1 or num_pos_B == 1):
                    return 2*self.epsilon
            return 0.0
        else:
            if(num_pos == self.n//4):
                if(num_pos_A == num_pos):
                    return 3/4-self.n*self.epsilon
                if(num_pos_B == num_pos):
                    return 1/4
                return 0.0
            if(num_pos == 1):
                if(num_pos_A == 1 or num_pos_B == 1):
                    return 2*self.epsilon
            return 0.0

    def marginal(self):
        marg = [3/4-(self.n-2)*self.epsilon]*(self.n//4)
        marg += [1/4+2*self.epsilon]*(self.n//4)
        marg += [1/4]*(self.n//2)
        return marg

    def joint(self, vals):
        n = self.n
        vals = vals[vals[:, 0].argsort()]
        m = len(vals)
        marginal = self.marginal()
        if(m == 2):
            l1, l2 = vals[:, 0]
            v1, v2 = vals[:, 1]
            vsum = v1+v2
            if(l1 < n//4):
                if(l2 < n//4):
                    if(vsum == 2):
                        return self.fulljoint([1]*(n//4)+[0]*(n-n//4))
                    if(vsum == 1):
                        return 2*self.epsilon
                    return super().joint(vals)
                if(v1 == 1 and v2 == 0):
                    return marginal[0]
                if(vsum == 2):
                    return 0.0
                if(v1 == 0 and v2 == 1):
                    return self.fulljoint([0]*(n//4)+[1]*(n-n//4))
                return 0.0 if l2 < n//2 else self.epsilon*n
            if(l1 < n//2):
                if(l2 < n//2):
                    if(vsum == 2):
                        return self.fulljoint([0]*(n//4)+[1]*(n-n//4))
                    if(vsum == 1):
                        return 2*self.epsilon
                    return super().joint(vals)
        return super().joint(vals)

    def iterProbs(self, threshold=0) -> Iterator[Tuple[Labelling, float]]:
        n = self.n
        L = Labelling([1]*(n//4) + [0]*(n-n//4))
        yield L, self.fulljoint(L)
        if(self.kind == 0):
            L = Labelling([0]*(n//4) + [1]*(n-n//4))
        else:
            L = Labelling([0]*(n//4) + [1]*(n//4)+[0]*(n//2))
        yield L, self.fulljoint(L)
        for i in range(n//2):
            L = Labelling.zeros(n)
            L[i] = 1
            yield L, 2*self.epsilon

    def getNumberOfLabels(self) -> int:
        return self.n


if __name__ == "__main__":
    n = 6
    P = GenericProbDist.random(n)
    P._sum()
    # for L, p in P.iterProbs():
    #     print(L, p)
    print(P.labelDependenceSD())
    print(P.totalCorrelation())
