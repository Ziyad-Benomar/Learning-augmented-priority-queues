import numpy as np

# Class predictions
#------------------
def classPredictions(n,c, rankedVals=None):
    c = min(max(1,c), n+1)
    if rankedVals==None:
        rankedVals = np.arange(n)
    pRanks = np.arange(n)
    thresh = [0] + list(np.random.choice(n,c-1, replace=False)) + [n]
    thresh.sort()
    for i in range(c):
        for j in range(thresh[i], thresh[i+1]):
            pRanks[j] = np.random.randint(low=thresh[i], high=thresh[i+1])
    predictions = [(pRanks[j],rankedVals[j]) for j in range(n)]
    return predictions


# Positional predictions: decay setting
#--------------------------------------
def symBernoulli(n=1):
    ber = (np.random.rand(n) < 1/2)
    return 2*ber - 1

def decayPredictions(n,timesteps,rankedVals=None):
    if rankedVals==None:
        rankedVals = np.arange(n)
    pRanks = np.arange(n)
    toPerturb = np.random.choice(n,timesteps)
    bernoullis = symBernoulli(timesteps)
    for t in range(timesteps):
        i = toPerturb[t]
        pRanks[i] += bernoullis[t]
    predictions = [(pRanks[j],rankedVals[j]) for j in range(n)] # (prediction, true value)
    return predictions