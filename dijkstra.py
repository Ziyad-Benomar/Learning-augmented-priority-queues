import osmnx as ox
import os
from heaps import *
from skiplist import *
from predictions import *

######################################################################
# Dijkstra's algorithm with priority queue
######################################################################

def createPQ(pqID, predictions=None, keyNode=None):
    if predictions:
        if pqID == "DC":
            def dirtyCompare(dist1, dist2):
                node1 = keyNode[dist1][-1]
                node2 = keyNode[dist2][-1]
                return predictions[node1] - predictions[node2]
            return SkipList(dcompare=dirtyCompare)
        else:
            return OnlineSL()
    elif pqID == "FH":
        return FibonacciHeap()
    else:
        return BinaryHeap()

def insertInPQ(pq, pqID, key, element=None, predGenID="class", predictions=None):

    if predictions: 
        
        # predictions is a SortedList of keys
        if predGenID=="sortedKeys": 
            predictedRank = predictions.bisect_left(key)
            pq.insert(key, predictedRank)
            
        # predictions is a list of (node, predictedRank)
        elif pqID == "DC":
            dirtyPredecessor = pq.findPredecessor(key, dirty=True)
            cleanPredecessor = pq.exponentialSearch(dirtyPredecessor, key)
            pq.insertNextTo(key, cleanPredecessor)
        else:
            pq.insert(key, predictions[element])
    else:
        pq.insert(key)

def dijkstraPQ(graph, source, predictions=None, pqID="BH", predGenID="class", graphType="city", returnAllKeys=False):
    # Dictionary to store the shortest distance to each node
    distances = {node: float('inf') for node in graph.nodes}
    distances[source] = 0
    keyNode = {0:[source]}
    allKeys = []
    pq = createPQ(pqID, predictions, keyNode)
    insertInPQ(pq, pqID, 0)
    
    
    count = 0
    while not pq.isEmpty():
        # Pop the node with the smallest distance
        current_distance = pq.extractMin()
        current_node = keyNode[current_distance].pop()
        # If the popped distance is greater than the stored distance, skip processing
        if current_distance > distances[current_node]:
            continue
        # Explore neighbors
        for neighbor, attributes in graph[current_node].items():
            count += 1
            distance = 0
            if graphType == "city":
                distance = attributes[0]["length"]
            elif graphType == "weighted":
                distance = graph[current_node][neighbor]["weight"]
            new_distance = current_distance + distance
            # If a shorter path to the neighbor is found
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                if new_distance not in keyNode:
                    keyNode[new_distance] = []
                keyNode[new_distance].append(neighbor)
                insertInPQ(pq, pqID, new_distance, neighbor, predGenID, predictions)
                allKeys.append(new_distance)
    if returnAllKeys:
        return distances, pq, allKeys
    return distances, pq




######################################################################
# Predictions
######################################################################

def getRanks(graph, source, graphType):
    distances, pq = dijkstraPQ(graph, source, graphType=graphType)
    return sorted(distances, key=distances.get), pq.countComps

def formatPredictions(predictions):
    return {p[1]: p[0] for p in predictions}
    
def getClassPredictions(rankedNodes, c):
    n = len(rankedNodes)
    cp = classPredictions(n,c, rankedNodes)
    return formatPredictions(cp)

def getDecayPredictions(rankedNodes, timesteps):
    n = len(rankedNodes)
    dp = decayPredictions(n, timesteps, rankedNodes)
    return formatPredictions(dp)

def getKeyPredictions(graph, source, graphType):
    #source = np.random.choice(list(graph.nodes()))
    getRanks(graph, source, graphType)
    distances, pq, allKeys = dijkstraPQ(graph, source, returnAllKeys=True, graphType=graphType)
    return SortedList(allKeys)

def getPredictions(predGenID, rankedNodes, params, graphType):
    if predGenID == "class":
        c = params['c']
        return getClassPredictions(rankedNodes, c)
    if predGenID == "decay":
        timesteps = params['timesteps']
        return getDecayPredictions(rankedNodes, timesteps)
    if predGenID == "sortedKeys":
        dist = params['d']
        graph = params['graph']
        source = params['source']
        return getKeyPredictions(graph, source, dist, graphType)
    return None



######################################################################
# Import city map
######################################################################

def importCityGraph(cityName): #city name lower case (?)
    filename = f"data/{cityName}.graphml"
    if os.path.exists(filename):
        graph = ox.load_graphml(filename)
    else:
        graph = ox.graph_from_place(cityName, network_type='drive')
        ox.save_graphml(graph, filename)
    return graph

def chooseRandomSource(graph, graphType="city"):
    numComps = 0
    while numComps < 100:
        source = np.random.choice(list(graph.nodes()))
        rankedNodes, numComps = getRanks(graph, source, graphType)
    return source, rankedNodes, numComps

def chooseRandomSourceInList(graph, nodes, graphType="city"):
    numComps = 0
    while numComps < 100:
        source = np.random.choice(nodes)
        rankedNodes, numComps = getRanks(graph, source, graphType)
    return source



######################################################################
# Test Dijkstra's algorithm with priority queues / predictions
######################################################################

def testDijkstra(graph, predGenID=None, params=None, pqID="OSL", niters=30, graphType="city"):
    countComps = np.zeros(niters, dtype="float")
    n = graph.number_of_nodes()
    
    #-----------------------------------
    # No predictions
    #-----------------------------------
    if pqID in ["BH", "FH"]: 
        for i in range(niters):
            source = np.random.choice(list(graph.nodes()))
            distances, pq = dijkstraPQ(graph, source, pqID=pqID, graphType=graphType)
            countComps[i] = pq.countComps

    #-----------------------------------
    # Sorted keys predictions
    #-----------------------------------
    elif predGenID == "sortedKeys":
        refSource = chooseRandomSource(graph, graphType)[0]
        predictions = getKeyPredictions(graph, refSource, graphType)
        rankedNodes = getRanks(graph, refSource, graphType)[0]
        dist = params["d"]
        for i in range(niters):
            source = refSource
            if dist > 0:
                source = chooseRandomSourceInList(graph, rankedNodes[:min(dist, n)], graphType)
            distances, pq = dijkstraPQ(graph, source, predictions, pqID="OSL", predGenID=predGenID, graphType=graphType)
            countComps[i] = pq.countComps

    #-----------------------------------
    # Class or decay predictions
    #-----------------------------------
    else:
        for i in range(niters):
            source, rankedNodes, numComps = chooseRandomSource(graph, graphType)
            params["source"] = source
            predictions = getPredictions(predGenID, rankedNodes, params, graphType)
            distances, pq = dijkstraPQ(graph, source, predictions, pqID=pqID, predGenID=predGenID, graphType=graphType)
            countComps[i] = pq.countComps
    return countComps.mean()/n, countComps.std()/n