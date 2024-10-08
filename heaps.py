import numpy as np
from time import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sortedcontainers import SortedList


######################################################################
# Fibonacci heap
######################################################################
# This Fibonacci heap implementation is not ours,
# It is taken from the open source github repository https://github.com/ivan-ristovic/fheap

# To allow fast deletion and concatenation, the roots of all trees are linked using a circular, 
# doubly linked list. The children of each node are also linked using such a list. For each node, 
# we maintain its number of children and whether the node is marked. Moreover, we maintain a 
# pointer to the root containing the minimum key.

class FibonacciHeap:

    # Internal Node class.
    class Node:
        def __init__(self, value):
            # We currently only support number elements.
            self.value = value
            # Pointer to the parent node.
            self.parent = None
            # Pointer to the first child in the list of children.
            self.child = None
            # Pointer to the left node.
            self.left = None
            # Pointer to the right node.
            self.right = None
            # Node degree - number of children.
            self.deg = 0
            # Is the node marked? This is needed for some of the operations.
            self.mark = False


    # Pointer to one element of the doubly-linked circular list of heap components.
    root_list =  None
    # Pointer to the node containing minimum element in the heap.
    min_node = None
    # Number of nodes in the entire heap.
    total_num_elements = 0
    countComps = 0
    
    def isEmpty(self):
        return (self.total_num_elements == 0)

    # Iterate through the node list
    def iterate(self, head = None):
        if head is None:
            head = self.root_list
        current = head
        while True:
            yield current
            if current is None:
                break
            current = current.right
            if current == head:
                break

    # Retrieving minimum node is trivial because we maintain a pointer to it.
    def find_minimum(self):
        if self.min_node is None:
            raise ValueError('Fibonacci heap is empty, minimum does not exist!')
        return self.min_node.value

    # Insert works by creating a new heap with one element and doing merge. This takes constant time, and the potential 
    # increases by one, because the number of trees increases. The amortized cost is thus still constant. 
    def insert(self, value):
        # Create a new singleton tree
        node = self.Node(value)
        node.left = node.right = node
        # Add to root list
        self.meld_into_root_list(node)
        # Update min pointer (if necessary)
        if self.min_node is not None:
            self.countComps += 1
            if self.min_node.value > node.value:
                self.min_node = node
        else:
            self.min_node = node
        self.total_num_elements += 1
        return node

    # Extracting minumum element is done in a few steps. First we take the root containing the minimum element and remove 
    # it. Its children will become roots of new trees. If the number of children was d, it takes time O(d) to process all 
    # new roots and the potential increases by d−1. Therefore, the amortized running time of this phase is O(d) = O(log n).
    def extractMin(self):
        m = self.min_node
        if m is None:
            raise ValueError('Fibonacci heap is empty, cannot extract mininum!')
        if m.child is not None:
            # Meld children into root_list
            children = [x for x in self.iterate(m.child)]
            for i in range(0, len(children)):
                self.meld_into_root_list(children[i])
                children[i].parent = None
        # Delete min node
        self.remove_from_root_list(m)
        self.total_num_elements -= 1
        # Consolidate trees so that no root has same rank
        self.consolidate()
        # Update min
        if m == m.right:
            self.min_node = None
            self.root_list = None
        else:
            self.min_node = self.find_min_node()
        return m.value

    # This operation works by taking the node, decreasing the key and if the heap property becomes violated (the new key 
    # is smaller than the key of the parent), the node is cut from its parent. If the parent is not a root, it is marked. 
    # If it has been marked already, it is cut as well and its parent is marked. We continue upwards until we reach either 
    # the root or an unmarked node. Now we set the minimum pointer to the decreased value if it is the new minimum. In the 
    # process we create some number, say k, of new trees. Each of these new trees except possibly the first one was marked 
    # originally but as a root it will become unmarked. One node can become marked. Therefore, the number of marked nodes 
    # changes by −(k − 1) + 1 = − k + 2. Combining these 2 changes, the potential changes by 2(−k + 2) + k = −k + 4. The 
    # actual time to perform the cutting was O(k), therefore (again with a sufficiently large choice of c) the amortized 
    # running time is constant.
    def decrease_key(self, node, v):
        self.countComps += 1
        if v >= node.value:
            raise ValueError("Cannot decrease key with a value greater than what it already is.")
        node.value = v
        p = node.parent
        self.countComps += 2
        if p is not None and node.value < p.value:
            self.cut(node, p)
            self.cascading_cut(p)
        if node.value < self.min_node.value:
            self.min_node = node
        return

    # Delete operation can be implemented simply by decreasing the key of the element to be deleted to minus infinity, thus 
    # turning it into the minimum of the whole heap. Then we call extract minimum to remove it. The amortized running time 
    # of this operation is O(log n).
    def delete(self, node):
        self.decrease_key(node, -1)
        self.extractMin()
        
    # Merging two heaps is implemented simply by concatenating the lists of tree roots of the two heaps. 
    # This can be done in constant time and the potential does not change, leading again to constant amortized time.
    def merge(self, fh):
        if fh.total_num_elements == 0:
            return
        self.countComps += 1
        if fh.min_node.value < self.min_node.value:
            self.min_node = fh.min_node
        self.total_num_elements += fh.total_num_elements
        last = fh.root_list.left
        fh.root_list.left = self.root_list.left
        self.root_list.left.right = fh.root_list
        self.root_list.left = last
        self.root_list.left.right = self.root_list


    ##### Helper functions #####

    def cut(self, node, parent):
        self.remove_from_child_list(parent, node)
        parent.deg -= 1
        self.meld_into_root_list(node)
        node.parent = None
        node.mark = False

    def cascading_cut(self, node):
        p = node.parent
        if p is not None:
            if p.mark is False:
                p.mark = True
            else:
                self.cut(node, p)
                self.cascading_cut(p)

    # Merge a node with the doubly linked root list by adding it to second position in the list
    def meld_into_root_list(self, node):
        if self.root_list is None:
            self.root_list = node
        else:
            node.right = self.root_list.right
            node.left = self.root_list
            self.root_list.right.left = node
            self.root_list.right = node

    # Deletes a node from the doubly linked root list.
    def remove_from_root_list(self, node):
        if self.root_list is None:
            raise ValueError('Fibonacci heap is empty, there is no node to remove!')
        if self.root_list == node:
            # Check if there's only one element in the list
            if self.root_list == self.root_list.right:
                self.root_list = None
                return
            else:
                self.root_list = node.right
        node.left.right = node.right
        node.right.left = node.left
        return

    # Removes a node from the doubly linked child list
    def remove_from_child_list(self, parent, node):
        if parent.child == parent.child.right:
            parent.child = None
        elif parent.child == node:
            parent.child = node.right
            node.right.parent = parent
        node.left.right = node.right
        node.right.left = node.left
    
    # Consolidates trees so that no root has same rank.
    def consolidate(self):
        if self.root_list is None:
            return
        ranks_mapping = [None] * self.total_num_elements
        nodes = [x for x in self.iterate(self.root_list)]
        for node in nodes:
            degree = node.deg
            while ranks_mapping[degree] != None:
                other = ranks_mapping[degree]
                self.countComps += 1
                if node.value > other.value:
                    node, other = other, node
                self.merge_nodes(node, other)
                ranks_mapping[degree] = None
                degree += 1
            ranks_mapping[degree] = node
        return

    # Links two nodes together, putting the node with greater key as child of the other node
    def merge_nodes(self, node, other):
        self.remove_from_root_list(other)
        other.left = other.right = other
        # Adding other node to child list of the frst one.
        self.merge_with_child_list(node, other)
        node.deg += 1
        other.parent = node
        other.mark = False
        return

    # Merges a node with the doubly linked child list of the root node.
    def merge_with_child_list(self, parent, node):
        if parent.child is None:
            parent.child = node
        else:
            node.right = parent.child.right
            node.left = parent.child
            parent.child.right.left = node
            parent.child.right = node

    # Iterates through whole list and finds minimum node.
    def find_min_node(self):
        if self.root_list is None:
            return None
        else:
            min = self.root_list
            for x in self.iterate(self.root_list):
                self.countComps += 1
                if x.value < min.value:
                    min = x
            return min
        
    # Prints the whole fheap.
    def print(self, head = None):
        if self.root_list is not None:
            for heap in self.iterate():
                print('---')
                self.print_tree(heap)
                print()
            print('---')
                
    # Prints the node list
    def print_tree(self, node):
        if node is None:
            return
        print(node.value, end=' ')
        if node.child is not None:
            print()
            for child in self.iterate(node.child):
                self.print_tree(child)

    # Find node in whole heap that is greater than value.
    def find_node_greater_than(self, value):
        if self.root_list is not None:
            for heap in self.iterate():
                result = self.find_child_greater_than(heap, value)
                if result != None:
                    return result
        raise ValueError(f'There is no element in the heap that is greater than {value}.')
    
    # Find node in the child list that is greater than value.
    def find_child_greater_than(self, node, value):
        if node is None:
            return None
        self.countComps += 1
        if node.value > value:
            return node
        if node.child is not None:
            for child in self.iterate(node.child):
                result = self.find_child_greater_than(child, value)
                if result is not None:
                    return result
        return None


######################################################################
# Binary heap
######################################################################
from heapq import heappush, heappop, heapify

class BinaryHeap: 
      
    # Constructor to initialize a heap 
    def __init__(self): 
        self.heap = []  
        self.n = 0
        self.countComps = 0
    
    def isEmpty(self):
        return (len(self.heap) == 0)
    
    def parent(self, i): 
        return (i-1)/2
      
    # Inserts a new key 'k' 
    def insert(self, k): 
        heappush(self.heap, k) 
        self.countComps += int(np.log2(int(np.log2(self.n+2))))
        self.n += 1          
  
    # Decrease value of key at index 'i' to new_val 
    # It is assumed that new_val is smaller than heap[i] 
    def decreaseKey(self, i, new_val): 
        self.heap[i]  = new_val  
        while(i != 0 and self.heap[self.parent(i)] > self.heap[i]): 
            # Swap heap[i] with heap[parent(i)] 
            self.heap[i] , self.heap[self.parent(i)] = ( 
            self.heap[self.parent(i)], self.heap[i]) 
              
    # Method to remove minimum element from min heap 
    def extractMin(self): 
        self.n -= 1
        self.countComps += int(np.log2(self.n+1))
        return heappop(self.heap) 
  
    # This function deletes key at index i. It first reduces 
    # value to minus infinite and then calls extractMin() 
    def deleteKey(self, i): 
        self.decreaseKey(i, float("-inf")) 
        self.extractMin() 
  
    # Get the minimum element from the heap 
    def getMin(self): 
        return self.heap[0] 

def bHeapSort(arr):
    heap = BinaryHeap()
    for a in arr:
        heap.insert(a)
    arrSorted = []
    while not heap.isEmpty():
        arrSorted.append(heap.extractMin())
    return arrSorted
