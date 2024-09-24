import json
import numpy as np
from time import time
from sortedcontainers import SortedList
from sorting import *
from dijkstra import *

######################################################################
# Save Experiment Results
######################################################################

def saveToFile(mean, std, filename):
    mean2 = {}
    std2 = {}
    for key in mean:
        mean2[key] = list(mean[key].copy())
        std2[key] = list(std[key].copy())
    data = {"mean": mean2, "std": std2}
    with open("data/"+filename, 'w') as file:
        json.dump(data, file)

def uploadFromFile(filename):
    try:
        with open("data/"+filename, 'r') as file:
            data = json.load(file)
    except:
        return None, None
    mean = data["mean"]
    std = data["std"]
    for key in mean:
        mean[key] = np.array(mean[key])
        std[key] = np.array(std[key])
    return mean, std


######################################################################
# Sorting experiments
######################################################################

# Test multiple algorithms with "class" predictions
#---------------------------------------------------------------------
def testSortAlgosClass(n,algosToTest=IDtoAlgo,niters=30,m=20):
    # Class predictions
    predGenID = "class"
    cvals = [i*n//m for i in range(m+1)]
    mean = {}
    std = {}
    Ti = time()
    for algoID in algosToTest:
        ti = time()
        print("\nAlgorithm: ",algoID)
        expID = f"{algoID}_{predGenID}_{n}" #experiment ID
        mean[expID] = np.zeros(m+1, dtype="float")
        std[expID] = np.zeros(m+1, dtype="float")
        if algoID not in ["BH", "FH"]:
            for i in range(m+1):
                if i%5 == 0: print("- i = ",i,", time ",time()-ti)
                params = {'n':n, 'c':cvals[i]}
                mean[expID][i],std[expID][i] = testSortAlgo(algoID, params, predGenID, niters)
        else: # The number of comparisons used by Binary/Fibonacci heap (BH/FH) is deterministic
            meanAlgo,stdAlgo = testSortAlgo(algoID, {'n':n}, predGenID, niters)
            mean[expID] = np.ones(m+1)*meanAlgo
            std[expID] = np.ones(m+1)*stdAlgo
        print("Runtime = ", time()-ti)
    print("\nTotal runtime = ",time()-Ti)

    filename = f"data_{predGenID}_{n}.json"
    saveToFile(mean, std, filename)


# Test multiple algorithms with "decay" predictions
#--------------------------------------------------------------------- 
def testSortAlgosDecay(n,algosToTest=IDtoAlgo,niters=30,m=20):
    mean = {}
    std = {}
    predGenID = "decay"
    tsMax = int(n*np.sqrt(n))
    tsvals = np.array([i*tsMax//m for i in range(m+1)])
    Ti = time()
    for algoID in algosToTest:
        ti = time()
        print("\nAlgorithm: ",algoID,"\n- ",end='')
        expID = f"{algoID}_{predGenID}_{n}"
        mean[expID] = np.zeros(m+1, dtype="float")
        std[expID] = np.zeros(m+1, dtype="float")
        if algoID not in ["BH", "FH"]:
            for i in range(m+1):
                if i%5 == 0: print("- i = ",i,", time ",time()-ti)
                params = {'n':n, 'timesteps':tsvals[i]}
                mean[expID][i],std[expID][i] = testSortAlgo(algoID, params, predGenID, niters)
        else: # The number of comparisons used by Binary/Fibonacci heap (BH/FH) is deterministic
            meanAlgo,stdAlgo = testSortAlgo(algoID, {'n':n}, predGenID, niters)
            mean[expID] = np.ones(m+1)*meanAlgo
            std[expID] = np.ones(m+1)*stdAlgo
        print("Runtime = ", time()-ti)
    print("\nTotal runtime = ",time()-Ti)

    filename = f"data_{predGenID}_{n}.json"
    saveToFile(mean, std, filename)


# Test multiple algorithms in any prediction model
#---------------------------------------------------------------------
def testSortAlgorithms(n, predGenID, algosToTest=IDtoAlgo, niters=30,m=20):
    if predGenID == "class":
        testSortAlgosClass(n,algosToTest,niters,m)
    elif predGenID == "decay":
        testSortAlgosDecay(n,algosToTest,niters,m)
    else:
        raise ValueError("predGenID must be 'class' or 'decay'.")


######################################################################
# Dijkstra's algorithm experiments
######################################################################

# Test with class predictions
#---------------------------------------------------------------------
def testDijkstraClass(cityName, niters=50,m=20):
    graph = importCityGraph(cityName)
    n = graph.number_of_nodes()
    predGenID = "class"
    cvals = [i*n//m for i in range(m+1)]

    mean = {}
    std = {}
    Ti = time()
    for pqID in ["OSL", "DC", "BH", "FH"]:
        ti = time()
        print("\nPriority queue: ",pqID)
        expID = f"dijkstra_{pqID}_{predGenID}_{cityName}"
        mean[expID] = np.zeros(m+1, dtype="float")
        std[expID] = np.zeros(m+1, dtype="float")
        if pqID not in ["BH", "FH"]:
            for i in range(m+1):
                if i%5 == 0: print("- i = ",i,", time ",time()-ti)
                params = {'c':cvals[i]}
                mean[expID][i],std[expID][i] = testDijkstra(graph, predGenID, params, pqID, niters)
        else:
            meanAlgo,stdAlgo = testDijkstra(graph, pqID=pqID, niters=niters)
            print(meanAlgo, stdAlgo)
            mean[expID] = np.ones(m+1)*meanAlgo
            std[expID] = np.ones(m+1)*stdAlgo
        print("Runtime = ", time()-ti)
    print("\nTotal runtime = ", time()-Ti)
    filename = f"data_dijkstra_{predGenID}_{cityName}.json"
    saveToFile(mean, std, filename)


# Test with decay predictions
#---------------------------------------------------------------------
def testDijkstraDecay(cityName, niters=50,m=20):
    predGenID = "decay"
    graph = importCityGraph(cityName)
    n = graph.number_of_nodes()
    tsMax = 20*n
    tsvals = np.array([i*tsMax//m for i in range(m+1)])
    mean = {}
    std = {}

    Ti = time()
    for pqID in ["OSL", "DC", "BH", "FH"]:
        ti = time()
        print("\nPriority queue: ",pqID)
        expID = f"dijkstra_{pqID}_{predGenID}_{cityName}"
        mean[expID] = np.zeros(m+1, dtype="float")
        std[expID] = np.zeros(m+1, dtype="float")
        if pqID not in ["BH", "FH"]:
            for i in range(m+1):
                if i%5 == 0: print("- i = ",i,", time ",time()-ti)
                params = {'timesteps':tsvals[i]}
                mean[expID][i],std[expID][i] = testDijkstra(graph, predGenID, params, pqID, niters)
        else:
            meanAlgo,stdAlgo = testDijkstra(graph, pqID=pqID, niters=niters)
            print(meanAlgo, stdAlgo)
            mean[expID] = np.ones(m+1)*meanAlgo
            std[expID] = np.ones(m+1)*stdAlgo
        print("Runtime = ", time()-ti)
    print("\nTotal runtime = ", time()-Ti)
    filename = f"data_dijkstra_{predGenID}_{cityName}.json"
    saveToFile(mean, std, filename)


# Test with sorted keys
#---------------------------------------------------------------------
def testDijkstraSortedKeys(cityName, niters=50,m=20):
    graph = importCityGraph(cityName)
    n = graph.number_of_nodes()
    predGenID = "sortedKeys"
    dvals = [5000]
    m = 0
    mean = {}
    std = {}

    Ti = time()
    for pqID in ["OSL", "DC"]:
        ti = time()
        print("\nPriority queue: ",pqID)
        expID = f"dijkstra_{pqID}_{predGenID}_{cityName}"
        mean[expID] = np.zeros(m+1, dtype="float")
        std[expID] = np.zeros(m+1, dtype="float")
        if pqID not in ["BH", "FH"]:
            for i in range(m+1):
                if i%5 == 0: print("- i = ",i,", time ",time()-ti)
                params = {'d':dvals[i]}
                mean[expID][i],std[expID][i] = testDijkstra(graph, predGenID, params, pqID, niters)
        else:
            meanAlgo,stdAlgo = testDijkstra(graph, pqID=pqID, niters=30)
            mean[expID] = np.ones(m+1)*meanAlgo
            std[expID] = np.ones(m+1)*stdAlgo
        print("Runtime = ", time()-ti)
    print("\nTotal runtime = ", time()-Ti)
    print(mean)
    print(std)
    filename = f"data_dijkstra_{predGenID}_{cityName}.json"
    saveToFile(mean, std, filename)


# Test Dijkstra's algorithm in any prediction model
#---------------------------------------------------------------------
def testDijkstraAlgorithm(cityName, predGenID, niters=50, m=20):
    if predGenID == "class":
        testDijkstraClass(cityName, niters, m)
    elif predGenID == "decay":
        testDijkstraDecay(cityName, niters, m)
    elif predGenID == "sortedKeys":
        testDijkstraSortedKeys(cityName, niters, m)
    else:
        raise ValueError("predGenID must be 'class', 'decay' or 'sortedKeys.")