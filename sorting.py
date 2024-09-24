import numpy as np
from skiplist import *
from heaps import *
from predictions import *


# All the functions in this file return the 


######################################################################
# Sort using Binary/Fibonacci heaps
######################################################################

# Binary heap
def countCompsBheapSort(n):
    count = 0
    for k in range(1,n):
        count += np.log2(int(np.log2(k+1))) + int(np.log2(k+1)) - 1
    return count

# Fibonacci heap
def fHeapSort(arr):
    heap = FibonacciHeap()
    for a in arr:
        heap.insert(a)
    while not heap.isEmpty():
        heap.extractMin()
    return heap


######################################################################
# Sort with online rank predictions
######################################################################

def getPredecessor(sortedArr, target):
    index = sortedArr.bisect_left(target)
    index = max(index-1,0)
    return sortedArr[index]

# Sort using Skip-List with online Rank predictions
def sortOSL(predictions):
    n = len(predictions)
    np.random.shuffle(predictions)
    osl = OnlineSL()
    for i in range(n):
        predictedRank, val = predictions[i]
        osl.insert(val, predictedRank)
    return osl.sl


######################################################################
# Sort with offline rank predictions
######################################################################

# Sort using a skip-list
def sortSL(predictions):
    n = len(predictions)
    predictions.sort(key=lambda x:x[0])
    sl = SkipList()
    source = sl.insert(predictions[0][1])
    for i in range(1,n):
        source = sl.insertES(source, predictions[i][1])
    return sl

######################################################################
# Sort with rank predictions using dirty/clean comparisons
######################################################################

# Sort given Dirty Comparisons
def sortDC(predictions): # prediction[j][1] = j for all j
    n = len(predictions)
    def dirtyCompare(i,j):
        return (predictions[i][0] - predictions[j][0])
    sl = SkipList(dcompare=dirtyCompare)
    arr = np.arange(n)
    np.random.shuffle(arr)
    for i in arr:
        dirtyPredecessor = sl.findPredecessor(i, dirty=True)
        cleanPredecessor = sl.exponentialSearch(dirtyPredecessor, i)
        sl.insertNextTo(i, cleanPredecessor)
    return sl

# Sort given "damaged" dirty comparisons: each comparison is correct w.p. 1-r
def sortDCdamaged(n,r): 
    M = np.random.rand(n,n)
    def dirtyCompare(i,j):
        if M[i,j]>r:
            return i-j
        return -(i-j)
    sl = SkipList(dcompare=dirtyCompare)
    arr = np.arange(n)
    np.random.shuffle(arr)
    for i in arr:
        dirtyPredecessor = sl.findPredecessor(i, dirty=True)
        cleanPredecessor = sl.exponentialSearch(dirtyPredecessor, i)
        sl.insertNextTo(i, cleanPredecessor)
    return sl


######################################################################
# Test sorting algorithms
######################################################################

IDtoAlgo = {
    "SL": sortSL,
    "OSL": sortOSL,
    "DC": sortDC,
    "FH": None,
    "BH": None
}

IDtoPredGen = {
    "class": classPredictions,
    "decay": decayPredictions,
    "damage": None,
    "": None,
}
    
# Test one algorithm
#-----------------------------------------------------------------------------   

def testSortAlgo(algoID, params, predGenID="", niters=30):
    countComps = np.zeros(niters)
    # Dirty Clean
    if predGenID == "damage":
        for i in range(niters):
            sl = sortDCdamaged(**params)
            countComps[i] = sl.countComps
    # Positional predictions
    else: 
        # Algos without predictions
        if algoID == "BH":
            return countCompsBheapSort(params['n'])/params['n'] , 0
        if algoID == "FH":
            arr = np.random.rand(params['n'])
            heap = fHeapSort(arr)
            return heap.countComps/params['n'], 0
        # Algos with predictions
        else:
            predGenerator = IDtoPredGen[predGenID]
            sortAlgo = IDtoAlgo[algoID]
            for i in range(niters):
                predictions = predGenerator(**params)
                sl = sortAlgo(predictions)
                countComps[i] = sl.countComps
    countComps /= params['n']
    #print(countComps)
    return countComps.mean(), countComps.std()



