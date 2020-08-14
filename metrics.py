import numpy as np
from ProbDistribution import Labelling, Annotation, ProbabilityDistribution
from more_itertools import chunked


class MLMetric:
    def __init__(self, is_loss: bool):
        self.is_loss = is_loss

    def risk(self, prediction, Pdist: ProbabilityDistribution) -> float:
        batch_size = 32
        losses = np.empty(batch_size, dtype=np.float)
        risk = 0.0
        for lps in chunked(Pdist.iterProbs(threshold=0), batch_size):
            labels, probs = zip(*lps)
            for i, l in enumerate(labels):
                losses[i] = self.measure(l, prediction)
            m = len(probs)
            if(m < batch_size):
                risk += (losses[:m]*probs[:m]).sum()
            else:
                risk += (losses*probs).sum()
        return risk

    def maxrisk(self, Pdist: ProbabilityDistribution):
        batch_size = 32
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
        # print(">", self.__class__.__name__, min_risk_sol, min_risk)
        return (min_risk_sol, min_risk)

    def regret(self, prediction, Pdist) -> float:
        r = self.risk(prediction, Pdist)
        if(self.is_loss):
            return r - self.minrisk(Pdist)[1]
        return self.maxrisk(Pdist)[1] - r

    def measure(self, target: Annotation, predicted: Annotation) -> float:
        pass

    def __call__(self, target: Annotation, predicted: Annotation) -> float:
        return self.measure(target, predicted)


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
        v = np.argsort(V)[-1]
        best = bests_Y[v]
        return Labelling(best[0]), 1-best[1]


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


if __name__ == "__main__":
    from time import time
    n = 123
    l1 = Labelling(np.random.randint(2, size=n).tolist())
    l2 = Labelling(np.random.randint(2, size=n).tolist())
    b1 = toint(l1.vals)
    b2 = toint(l2.vals)

    niters = 10000
    t1 = time()
    for _ in range(niters):
        hammingloss1(l1, l2)
    t2 = time()
    for _ in range(niters):
        hammingloss2(l1, l2)
    t3 = time()
    for _ in range(niters):
        hammingloss3(l1, l2)
    t4 = time()
    for _ in range(niters):
        hammingloss4(b1, b2)
    t5 = time()
    print("numpy: %.1fms" % ((t2-t1)*1000))
    print("classic: %.1fms" % ((t3-t2)*1000))
    print("adaptative: %.1fms" % ((t4-t3)*1000))
    print("bits: %.1fms" % ((t5-t4)*1000))
