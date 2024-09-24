import numpy as np
from time import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sortedcontainers import SortedList

######################################################################
# Helpers
######################################################################
def constructLinks(heights):
    H = np.max(heights)
    levels = [[] for l in range(H)]
    for l in range(0,H):
        level = l+1
        for x in range(len(heights)):
            if heights[x] >= level:
                levels[l].append(x)
    return levels

def showSkipList(vals, heights, u=1, r=0.5, arrow_head=0.3, scale=1):
    # u = space between two elements
    H = np.max(heights)
    heights = [H] + list(heights) + [H]
    fig = plt.figure(figsize=(scale*len(heights)*(1+u),scale*(H+r+1))) 
    plt.axis('off')
    levels = constructLinks(heights)
    # Head/tail
    #plt.fill_between([0,1],0,H,color="gold")
    xnil = (len(heights)-1) * (1+u)
    plt.fill_between([xnil,xnil+1],0,H,color="lightgrey")
    plt.text(-0.1, -0.7, "HEAD", color="black", fontsize=scale*25)
    plt.text(xnil+0.15, H/2-0.2, "NIL", color="black", fontsize=scale*25)
    # heights
    for i in range(len(heights)):
        h = heights[i]
        x = i*(1+u)
        plt.plot([x,x,x+1,x+1,x], [0,h,h,0,0], c="black")
        if i<len(heights)-1:
            for j in range(1,h):
                plt.plot([x,x+1],[j,j], c="black")
        if i>0 and i<len(heights)-1:
            plt.plot([x+0.5,x+0.5], [0,-r], c="black")
            plt.plot([x,x,x+1,x+1,x], [-r,-r-1,-r-1,-r,-r], c="black")
            plt.fill_between([x,x+1],-r-1,-r,color="gold")
            plt.text(x+0.15 + 0.15*(vals[i-1]<10), -r-1/2-0.1, vals[i-1], color="black", fontsize=scale*30)
    # arrows
    for i in range(len(levels)):
        h = i+0.5
        level = levels[i]
        for j in range(len(level)-1):
            x1 = level[j]*(1+u)+0.5
            x2 = level[j+1]*(1+u) - arrow_head
            #plt.plot([x1,x2],[h,h], c="black")
            plt.arrow(x1, h, x2-x1, 0.0, color='black', head_length=arrow_head, head_width=0.2)
            plt.scatter([x1],[h], s=scale*100, color="black")

class Node:
    
    def __init__(self, value=None, height=1):
        self.height = height
        self.value = value
        self.next = [None for h in range(height)]
        self.prev = [None for h in range(height)]
    
    def __str__(self):
        return "value = " + str(self.value) + ", height = " + str(self.height)
    
    def getNext(self, h=0):
        if h < self.height:
            return self.next[h]
        return None
    
    def getPrev(self, h=0):
        if h < self.height:
            return self.prev[h]
        return None
    
    def setNext(self, nextNode, h):
        if h < self.height:
            self.next[h] = nextNode
            if nextNode:
                nextNode.prev[h] = self
        else:
            print("non valid height")
    

def damagedCompare(val1, val2, dr=0.25):
    # inaccurate with probability damageRatio
    if np.random.rand() < dr:
        return -(val1 - val2)
    return val1 - val2




######################################################################
# Skip list
######################################################################

class SkipList:
    # The tail is None
    # The head is node with value = "head"
    def __init__(self, p=0.5, dcompare=damagedCompare):
        self.p = p
        self.head = Node(-np.inf)
        self.maxHeight = 0
        self.valueNode = {}
        self.countComps = 0
        self.countDirtyComps = 0
        self.nodes = {}
        self.dirtyCompare = dcompare
    
    def isEmpty(self):
        return (self.head.getNext() == None)
    
    def cleanCompare(self, val1, val2):
        return val1 - val2
    
    def compare(self, val1, val2, dirty=False):
        if val1 == -np.inf or val2 == -np.inf: # comparison with the head is free
            return val1 - val2
        # Dirty comparison
        if dirty:
            self.countDirtyComps += 1
            return self.dirtyCompare(val1,val2)
        # Clean comparison
        else:
            self.countComps += 1
            return self.cleanCompare(val1,val2)
    
    def sampleHeight(self):
        return np.random.geometric(0.5)
    
    def updateHeadHeight(self, newNode):
        if self.head.height >= newNode.height:
            return
        diff = newNode.height - self.head.height
        additionalPointers = [None for i in range(diff)]
        self.head.next += additionalPointers
        self.head.height = newNode.height
        self.maxHeight = newNode.height
    
    def insertNextTo(self, value, prevNode):
        # create the new node
        newHeight = self.sampleHeight()
        newNode = Node(value, newHeight)
        self.nodes[value] = newNode
        self.updateHeadHeight(newNode)
        # Add it in all the levels h<newHeight
        predecessor = prevNode
        for h in range(newHeight):
            while h >= predecessor.height:
                predecessor = predecessor.getPrev(h-1)
            successor = predecessor.getNext(h)
            predecessor.setNext(newNode, h)
            newNode.setNext(successor, h)
        return newNode
    
    def delete(self, node):
        height = node.height
        for h in range(node.height):
            predecessor = node.getPrev(h)
            successor = node.getNext(h)
            predecessor.setNext(successor,h)
        return node
    
    # Search
    #--------------------
    def findPredecessor(self, value, dirty=False):
        h = self.maxHeight
        curr = self.head
        while h >= 0:
            while curr.getNext(h) and self.compare(curr.getNext(h).value, value, dirty) <= 0:
                curr = curr.getNext(h)
            h = h-1
        return curr
    
    # Exponential search
    #--------------------
    def rightExponentialSearch(self, sourceNode, value):
        curr = sourceNode
        while curr.getNext(curr.height-1) and self.compare(curr.getNext(curr.height-1).value, value) <= 0:
            curr = curr.getNext(curr.height-1)
        h = curr.height-1
        while h >= 0:
            while curr.getNext(h) and self.compare(curr.getNext(h).value, value) <= 0:
                curr = curr.getNext(h)
            h = h-1
        return curr
    
    def leftExponentialSearch(self, sourceNode, value):
        curr = sourceNode
        while self.compare(curr.getPrev(curr.height-1).value, value) >= 0:
            curr = curr.getPrev(curr.height-1)
        h = curr.height-1
        while h >= 0:
            while self.compare(curr.getPrev(h).value, value) >= 0:
                curr = curr.getPrev(h)
            h = h-1
        return curr.getPrev()
    
    def exponentialSearch(self, sourceNode, value):
        goRight = (sourceNode.getNext() and self.compare(sourceNode.getNext().value, value) < 0)
        goLeft = (self.compare(sourceNode.value, value) > 0)
        if goRight:
            return self.rightExponentialSearch(sourceNode, value)
        if goLeft:
            return self.leftExponentialSearch(sourceNode, value)
        return sourceNode
    
    # Dirty/Clean Insertion
    #----------------------
    def dirtyCleanInsert(self, value):
        dirtyPredecessor = self.findPredecessor(value, dirty=True)
        cleanPredecessor = self.exponentialSearch(dirtyPredecessor, value)
        return self.insertNextTo(value, cleanPredecessor)
    
    # Priority Queue Operations
    #--------------------------
    def insert(self, value):
        # inserts the value and returns the new node
        predecessor = self.findPredecessor(value)
        return self.insertNextTo(value, predecessor)
    
    def insertES(self, sourceNode, value):
        # inserts the value and returns the new node
        predecessor = self.exponentialSearch(sourceNode, value)
        return self.insertNextTo(value, predecessor)
        
    
    def findMin(self):
        return self.head.getNext()
    
    def extractMin(self):
        minNode = self.findMin()
        return self.delete(minNode).value
        
    def decreaseKey(self, value, newValue):
        return
    
    def getValsHeights(self):
        vals = []
        heights = []
        curr = self.head.getNext(0)
        while curr != None:
            vals.append(curr.value)
            heights.append(curr.height)
            curr = curr.getNext(0)
        return vals, heights
    
    def show(self):
        vals, heights = self.getValsHeights()
        showSkipList(vals, heights)
    
    def __str__(self):
        vals, heights = self.getValsHeights()
        return str(vals)

def skipListSort(arr):
    sl = SkipList()
    for a in arr:
        sl.insert(a)
    arrSorted = []
    while not sl.isEmpty():
        arrSorted.append(sl.extractMin())
    return arrSorted








######################################################################
# Skip list with vEB tree
######################################################################

# We use SortedList() instead of a vEB tree implementation, 
# as it provides the required functionalities of a vEB tree

class OnlineSL:
    def __init__(self):
        self.sl = SkipList()
        self.rankVal = {}
        self.valRank = {}
        self.veb = SortedList()
        self.veb.add(-np.inf)
        self.rankVal[-np.inf] = [-np.inf]
        self.countComps = 0

    def isEmpty(self):
        return self.sl.isEmpty()
    
    def getPredecessor(self, sortedArr, target):
        index = sortedArr.bisect_left(target)
        index = max(index-1,0)
        return sortedArr[index]
        
    def insert(self,val,predictedRank=0):
        prevRank = self.getPredecessor(self.veb, predictedRank)
        prevVal = self.rankVal[prevRank][-1]
        source = self.sl.head
        if prevVal != -np.inf:
            source = self.sl.nodes[prevVal]
        source = self.sl.insertES(source, val)
        if predictedRank not in self.rankVal:
            self.veb.add(predictedRank)
            self.rankVal[predictedRank] = []
        self.rankVal[predictedRank].append(val)
        if val not in self.valRank:
            self.valRank[val] = []
        self.valRank[val].append(predictedRank)
        self.countComps = self.sl.countComps

    def extractMin(self):
        minVal = self.sl.extractMin()
        predictedRank = self.valRank[minVal][-1]
        
        self.rankVal[predictedRank].remove(minVal)
        if len(self.rankVal[predictedRank]) == 0:
            del self.rankVal[predictedRank]
            self.veb.remove(predictedRank)

        self.valRank[minVal].remove(predictedRank)
        if len(self.valRank[minVal]) == 0:
            del self.valRank[minVal]
        self.countComps = self.sl.countComps
        return minVal