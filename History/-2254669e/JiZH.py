# max heap
import sys

class MaxHeap:
    
    # Initialize MaxHeap
    def __init__(self, maxsize):
        
        self.maxsize = maxsize
        self.size = 0
        self.Heap = [0] * (self.maxsize + 1)
        self.Heap[0] = sys.maxsize
        self.FRONT = 1
        self.printed = False
        print(f'{self.Heap[0] = }')
        print(sys.maxsize)
        
    # Function to return the position of
    # parent for the node currently
    # at pos
    def parent(self, pos):
        
        return pos // 2
    
    # Function to return the position of
    # the left child for the node currently
    # at pos
    def leftChild(self, pos):
        
        return 2 * pos
    
    # Function to return the position of
    # the right child for the node currently
    # at pos
    def rightChild(self, pos):
        
        return (2 * pos) + 1
    
    # Function that returns true if the passed
    # node is a leaf node
    def isLeaf(self, pos):
        
        if pos >= (self.size//2) and pos <= self.size:
            return True
        return False
    
    # Function to swap two nodes of the heap
    def swap(self, fpos, spos):
        
        self.Heap[fpos], self.Heap[spos] = (self.Heap[spos],
                                            self.Heap[fpos])
        
    # Function to heapify the node at pos
    def maxHeapify(self, pos):
        # To print the elements
        if (self.printed == False):
            print("Elements in array are [", end = "")
            for i in range(1, (self.size + 1)):
                print(str(self.Heap[i]), end = ", ")
            print("]")
            self.isPrinted(True)
        # If the node is a non-leaf node and smaller than any of its child
        if not self.isLeaf(pos):
            if (self.Heap[pos] < self.Heap[self.leftChild(pos)] or
                self.Heap[pos] < self.Heap[self.rightChild(pos)]):
                
                # Swap with the left child and heapify the left child
                if (self.Heap[self.leftChild(pos)] >
                    self.Heap[self.rightChild(pos)]):
                    self.swap(pos, self.leftChild(pos))
                    self.maxHeapify(self.leftChild(pos))
                    
                # Swap with the right child and heapify the right child
                else:
                    self.swap(pos, self.rightChild(pos))
                    self.maxHeapify(self.rightChild(pos))
                    
    # Function to put a node into the heap
    def put(self, theElement):
        
        if self.size >= self.maxsize:
            return
        # find place for theElement
        # currentNode starts at new leaf and moves up tree
        self.size += 1
        self.Heap[self.size] = theElement
        
        currentNode = self.size
        
        while (self.Heap[currentNode] > self.Heap[self.parent(currentNode)]):
            # cannot put theElement in Heap[currentNode]
            self.swap(currentNode, self.parent(currentNode))    # move element down
            currentNode = self.parent(currentNode)  # move to parent
            
    # Function to print the contents of the heap
    def Print(self):
        
        for i in range(1, (self.size // 2) + 1):
            print(" PARENT : " + str(self.Heap[i]) +
                  " LEFT CHILD : " + str(self.Heap[2 * i]) +
                  " RIGHT CHILD : " + str(self.Heap[2 * i + 1]))
        print("After initializing the array to heap, the elements in array are [", end = "")
        for i in range(1, (self.size + 1)):
            print(str(self.Heap[i]), end = ", ")
        print("]")

    # Function to remove and return the maximum element from the heap
    def removeMax(self):
        
        popped = self.Heap[self.FRONT]  # max element
        # reheapify
        self.Heap[self.FRONT] = self.Heap[self.size]
        self.size -= 1  # decrease the size
        self.maxHeapify(self.FRONT)
        
        return popped
    
    # Function to check if array is printed
    def isPrinted(self, printed):
        self.printed = printed
        return printed
    
    #initialize max heap to element array theHeap
    def initialize(self, heap, size):
      self.size = size
      for i in range((self.size-2)/2, 0):
        self.maxHeapify(i)


# sort the elements a[1 : a.length - 1] using the heap sort method 
def heapSort(a):
    # create a max heap of the elements
    h = MaxHeap(len(a))
    
    # extract one by one from the max heap
    for i in range(len(a) - 1, 0, -1):
        a[i] = h.removeMax(a, i+1)

# Driver Code
if __name__ == "__main__":
    
    print('The maxHeap is ')
    
    maxHeap = MaxHeap(15)
    maxHeap.put(5)
    maxHeap.put(3)
    maxHeap.put(17)
    maxHeap.put(10)
    maxHeap.put(84)
    maxHeap.put(19)
    maxHeap.put(6)
    maxHeap.put(22)
    maxHeap.put(9)

    heapSort([3,2,1])
    
    # maxHeap.Print()
    # maxHeap.isPrinted(False)
    
    # # Remove Max
    # print()
    # print("------Remove max element------")
    # print("After removing max element")
    # maxHeap. removeMax()
    # print()
    # maxHeap.Print()
    
    # print("The Max value is " + str(maxHeap. removeMax()))