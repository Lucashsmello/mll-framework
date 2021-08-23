from typing import List
import numpy as np
from ProbDistribution import Labelling, Annotation, ProbabilityDistribution, RPCbadDistribution
from more_itertools import chunked
import itertools
from numba import jit


class MLMetric:
    def __init__(self, is_loss: bool):
        self.is_loss = is_loss

    def risk(self, prediction, Pdist: ProbabilityDistribution) -> float:
        batch_size = 32
        risk = 0.0
        for lps in chunked(Pdist.iterProbs(threshold=0), batch_size):
            for l, p in lps:
                risk += self.measure(l, prediction) * p
        return risk

    def maxrisk(self, Pdist: ProbabilityDistribution):
        max_risk = 0.0
        for L in Labelling.iterAll(Pdist.getNumberOfLabels()):
            r = self.risk(L, Pdist)
            if(r > max_risk):
                max_risk = r
                max_risk_sol = L
        return (max_risk_sol, max_risk)

    def minrisk(self, Pdist: ProbabilityDistribution):
        min_risk = 128.0
        for L in Labelling.iterAll(Pdist.getNumberOfLabels()):
            r = self.risk(L, Pdist)
            if(r < min_risk):
                min_risk = r
                min_risk_sol = L
        return (min_risk_sol, min_risk)

    def regret(self, prediction, Pdist) -> float:
        r = self.risk(prediction, Pdist)
        # if(self.is_loss):
        return r - self.minrisk(Pdist)[1]
        # return self.maxrisk(Pdist)[1] - r

    def measure(self, target: Annotation, predicted: Annotation) -> float:
        pass

    def __call__(self, target: Annotation, predicted: Annotation) -> float:
        return self.measure(target, predicted)


class RankingMetric(MLMetric):
    """
    In the vector of rankings R, R[i] means the ranking of label i.
    """

    def __init__(self, is_loss: bool):
        super().__init__(is_loss)

    def minrisk(self, Pdist: ProbabilityDistribution):
        min_risk = 128.0
        n = Pdist.getNumberOfLabels()
        if(isinstance(Pdist, RPCbadDistribution)):
            for rank_template in itertools.permutations(range(3)):
                ranking = [0]*n
                k = 1
                for r in rank_template:
                    a = n//2 if r == 2 else n//4
                    for i in range(a):
                        ranking[r*(n//4)+i] = k
                        k += 1
                r = self.risk(ranking, Pdist)
                if(r < min_risk):
                    min_risk = r
                    min_risk_sol = ranking
            return (min_risk_sol, min_risk)
        for ranking in itertools.permutations(range(1, n+1)):
            r = self.risk(ranking, Pdist)
            if(r < min_risk):
                min_risk = r
                min_risk_sol = ranking
        return (min_risk_sol, min_risk)


class HammingLoss(MLMetric):
    def __init__(self):
        super().__init__(is_loss=True)

    def risk(self, prediction, Pdist: ProbabilityDistribution):
        Pm = Pdist.marginal()
        return sum([p if(pred == 0) else 1-p for pred, p in zip(prediction, Pm)])/len(Pm)

    def measure(self, target, predicted) -> float:
        s = 0
        for y1, y2 in zip(target.vals, predicted.vals):
            if(y1 != y2):
                s += 1
        return s/len(target)

    def minrisk(self, Pdist: ProbabilityDistribution):
        L = Labelling([1 if(p >= 1/2) else 0 for p in Pdist.marginal()])
        return (L, self.risk(L, Pdist))


class SubsetLoss(MLMetric):
    def __init__(self):
        super().__init__(is_loss=True)

    def risk(self, prediction, Pdist: ProbabilityDistribution):
        return 1-Pdist.fulljoint(prediction)

    def measure(self, target, predicted) -> float:
        s = 0
        for y1, y2 in zip(target.vals, predicted.vals):
            if(y1 != y2):
                s += 1
        return s/len(target)

    def minrisk(self, Pdist: ProbabilityDistribution):
        mode = Pdist.mode()
        return (mode, self.risk(mode, Pdist))


class Fmeasure(MLMetric):
    def __init__(self):
        super().__init__(is_loss=True)

    def measure(self, target, predicted) -> float:
        s = sum(target.vals)+sum(predicted.vals)
        if(s == 0):
            return 0

        a = 0
        for yt, yp in zip(target.vals, predicted.vals):
            a += yt & yp
        return 1-2*a/s

    def risk(self, prediction: Labelling,
             Pdist: ProbabilityDistribution) -> float:
        n = Pdist.getNumberOfLabels()
        k = prediction.positives()
        if(k == 0):
            return 1-Pdist.fulljoint([0]*n)
        p = 0.0
        for i in range(n):
            if(prediction[i] == 1):
                p += sum([Pdist.getProbSize(i, j)/(k+j) for j in range(1, n+1)])
        return 1-2*p

    def minrisk(self, Pdist: ProbabilityDistribution):
        n = Pdist.getNumberOfLabels()

        def predict_k(k):
            D = []
            R = range(1, n+1)
            for i in range(n):
                s1 = sum([Pdist.getProbSize(i, s)/(s+k) for s in R])
                D.append(s1)
            pos = np.argsort(D)[-k:]
            Y = [0]*n
            v = 0.0
            for p in pos:
                Y[p] = 1
                v += 2*D[p]
            return (Y, v)

        bests_Y = [([0]*n, Pdist.fulljoint([0]*n))]
        bests_Y += [predict_k(i) for i in range(1, n+1)]
        V = [Y[1] for Y in bests_Y]
        v = np.argmax(V)
        best = bests_Y[v]
        # assert(np.isclose(1-best[1], super().minrisk(Pdist)[1]))
        return Labelling(best[0]), 1-best[1]


class JaccardDistance(MLMetric):
    def __init__(self):
        super().__init__(is_loss=True)

    def measure(self, target, predicted) -> float:
        s = sum(target.vals)+sum(predicted.vals)
        if(s == 0):
            return 0

        a = 0
        for yt, yp in zip(target.vals, predicted.vals):
            a += yt & yp
        return 1-a/(s-a)


class AveragePrecision(RankingMetric):
    def __init__(self):
        super().__init__(is_loss=False)

    def measure(self, target: Labelling, predicted) -> float:
        avg_prec = 0.0
        npos = target.positives()
        if(npos == 0):
            return 1.0
        pos_idxs = target.where_positive()
        for i in pos_idxs:
            yi_rank = predicted[i]
            s = sum([1 for j in pos_idxs if predicted[j] <= yi_rank])
            avg_prec += s / yi_rank
        return avg_prec / npos


class RankingLoss(RankingMetric):
    def __init__(self, normalized=2):
        super().__init__(is_loss=True)
        self.normalized = normalized

    def measure(self, target: Labelling, predicted) -> float:
        s = 0
        for ri, yi in zip(predicted, target):
            if(yi == 0):
                continue
            for rj, yj in zip(predicted, target):
                if(yj == 1):
                    continue
                if(ri > rj):
                    s += 1
        num_pos = target.positives()
        n = len(target)
        if(self.normalized == 2):
            return s/(num_pos*(n-num_pos))
        if(self.normalized == 1):
            return 4*s/(n*n)
        return s

    def minrisk(self, Pdist: ProbabilityDistribution):
        marginal = Pdist.marginal()
        temp = np.argsort(-np.array(marginal))
        ranking = np.empty_like(temp)
        ranking[temp] = np.arange(len(temp)) + 1
        return ranking, self.risk(ranking, Pdist)


class Coverage(RankingMetric):
    def __init__(self):
        super().__init__(is_loss=True)

    def measure(self, target: Labelling, predicted) -> float:
        positives_idxs = target.where_positive()
        if(len(positives_idxs) == 0):
            return 0.0
        return max([predicted[pi] for pi in positives_idxs])  # / len(target) / target.positives()


class ReciprocalRank(RankingMetric):
    def __init__(self):
        super().__init__(is_loss=False)

    def measure(self, target: Labelling, predicted) -> float:
        positives_idxs = target.where_positive()
        return sum([1/predicted[pi] for pi in positives_idxs]) / sum(1/np.arange(1, target.positives()+1))


def toint(L):
    v = 0
    for l in L:
        v = (v << 1) | l
    return v


def hammingloss1(yt, yp):
    return np.logical_xor(yt.vals, yp.vals).mean()


def hammingloss2(yt, yp):
    s = 0
    for y1, y2 in zip(yt.vals, yp.vals):
        if(y1 != y2):
            s += 1
    return s/len(yt)


def hammingloss3(yt, yp):
    n = len(yt)
    if(n >= 23):
        return np.logical_xor(yt.vals, yp.vals).mean()
    s = 0
    for y1, y2 in zip(yt.vals, yp.vals):
        if(y1 != y2):
            s += 1
    return s/n


def hammingloss4(yt, yp):
    n = yt ^ yp
    n = (n & 0x5555555555555555) + ((n & 0xAAAAAAAAAAAAAAAA) >> 1)
    n = (n & 0x3333333333333333) + ((n & 0xCCCCCCCCCCCCCCCC) >> 2)
    n = (n & 0x0F0F0F0F0F0F0F0F) + ((n & 0xF0F0F0F0F0F0F0F0) >> 4)
    n = (n & 0x00FF00FF00FF00FF) + ((n & 0xFF00FF00FF00FF00) >> 8)
    n = (n & 0x0000FFFF0000FFFF) + ((n & 0xFFFF0000FFFF0000) >> 16)
    n = (n & 0x00000000FFFFFFFF) + ((n & 0xFFFFFFFF00000000) >> 32)  # This last & isn't strictly necessary.
    return n/123
