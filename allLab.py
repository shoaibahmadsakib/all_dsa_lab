# first insert and delete
arr = list(map(int, input("enter and array: ").split()))
print("result is : ", arr)

print("\nInsertion Operation")
ele = int(input("Enter an element: "))
pos = int(input("Enter the position: "))
arr.insert(pos - 1, ele)
print("After insertion: ", arr)

print("\nDeletion Operation")
pos=int(input("Enter a position to delete an element: "))
arr.remove(arr[pos-1])
print("After deletion your array: ",arr)


# 2 bubble sort
def bubble(arr):
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] > arr[j]:
                temp= arr[j]
                arr[j] = arr[i]
                arr[i] = temp

                # arr[i], arr[j] = arr[j], arr[i]

    return arr


arr = list(map(int, input("Enter an array: ").split()))
print("After Bubble sort: ", bubble(arr))


# 3 search linear
def linearSearch(array):
    x = int(input("Enter the search number:"))
    n = len(array)

    for i in range(0, n):
        if (array[i] == x):
            return i
    return -1

array = list(map(int,input("Enter an array: ").split()))


result = linearSearch(array)
if (result == -1):
    print("Element not found")
else:
    print("Element found at position: ", result+1)


# 4 Binary Search in python
def binarySearch(array):
    # Repeat until the pointers low and high meet each other
    x = int(input("enter the number of array"))
    high = len(array) - 1
    low = 0
    while low <= high:

        mid = low + (high - low) // 2

        if array[mid] == x:
            return mid

        elif array[mid] < x:
            low = mid + 1

        else:
            high = mid - 1

    return -1

array = [int(x) for x in input("Enter the elements for list:").split()]


result = binarySearch(array)

if result != -1:
    print("Element is present at index " + str(result))
else:
    print("Not found")




# 5 merge sort
def mergeSort(array):
    if len(array) > 1:
        r = len(array)//2
        L = array[:r]
        M = array[r:]

        mergeSort(L)
        mergeSort(M)

        i = j = k = 0

        while i < len(L) and j < len(M):
            if L[i] < M[j]:
                array[k] = L[i]
                i += 1
            else:
                array[k] = M[j]
                j += 1
            k += 1
        while i < len(L):
            array[k] = L[i]
            i += 1
            k += 1

        while j < len(M):
            array[k] = M[j]
            j += 1
            k += 1
# Driver program
if __name__ == '__main__':
    array =  [int(x) for x in input("Enter the elements for list:").split()]
    mergeSort(array)
    print("Sorted array is: " , array)




# #### 6 selection sort
def selectionSort(nums):
    size =len(nums)
    for i in range(size):
        minpos = i

        for j in range(i,size ):
            if nums[j] < nums[minpos]:
                minpos = j

        # put min at the correct position
        temp = nums[i]
        nums[i] = nums[minpos]
        nums[minpos] = temp

nums = list(map(int , input("enter: ").split()))
selectionSort(nums)
print(nums)


# #### 7 pattern sort
def search(pat, txt):
    M = len(pat)
    N = len(txt)
    for i in range(N - M+1 ):
        j = 0
        while (j < M):
            if (txt[i + j] != pat[j]):
                break
            j += 1
        if (j == M):
            print("Pattern found at index ", i)


text = input("Enter a line of text: ")
pattern = input("Enter a Pattern: ")
search(pattern, text)



# #### 8 backtraking sort
def isSafe(mat, r, c):
    # Column check
    for i in range(len(mat)):
        if mat[i][c] == 'Q':
            return False

    # diagonal check '\'
    i, j = r, c
    while i >= 0 and j >= 0:
        if mat[i][j] == 'Q':
            return False
        i -= 1
        j -= 1

    # diagonal Check '/'
    i, j = r, c
    while i >= 0 and j < len(mat):
        if mat[i][j] == 'Q':
            return False
        i -= 1
        j += 1

    return True


def printSolve(mat):
    for r in mat:
        print(str(r).replace(",", " ").replace("\'", " "))
    print()


def NQueen(mat, r=0):
    if r == len(mat):
        printSolve(mat)
        return

    for i in range(len(mat)):
        if isSafe(mat, r, i):
            mat[r][i] = 'Q'
            NQueen(mat, r + 1)
            mat[r][i] = '-'


N = int(input("Enter N: "))
mat = [['-'] * N for x in range(N)]
NQueen(mat)




# #### 9 kruskal sort
class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = []

    def add_edge(self, u, v, w):
        self.graph.append([u, v, w])

    # Search function
    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    def apply_union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    #  Applying Kruskal algorithm
    def kruskal_algo(self):
        result = []
        i, e = 0, 0
        self.graph = sorted(self.graph, key=lambda item: item[2])
        parent = []
        rank = []
        for node in range(self.V):
            parent.append(node)
            rank.append(0)
        while e < self.V - 1:
            u, v, w = self.graph[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent, v)
            if x != y:
                e = e + 1
                result.append([u, v, w])
                self.apply_union(parent, rank, x, y)
        for u, v, weight in result:
            print("%d - %d: %d" % (u, v, weight))


g = Graph(6)
g.add_edge(0, 1, 4)
g.add_edge(0, 2, 4)
g.add_edge(1, 2, 2)
g.add_edge(1, 0, 4)
g.add_edge(2, 0, 4)
g.add_edge(2, 1, 2)
g.add_edge(2, 3, 3)
g.add_edge(2, 5, 2)
g.add_edge(2, 4, 4)
g.add_edge(3, 2, 3)
g.add_edge(3, 4, 3)
g.add_edge(4, 2, 4)
g.add_edge(4, 3, 3)
g.add_edge(5, 2, 2)
g.add_edge(5, 4, 3)
g.kruskal_algo()


# #### 10 greedy 
def printJobScheduling(arr, t):
    n = len(arr)
    for i in range(n):
        for j in range(n - 1 - i):
            if arr[j][2] < arr[j + 1][2]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    result = [False] * t
    job = ['-1'] * t
    for i in range(len(arr)):
        for j in range(min(t - 1, arr[i][1] - 1), -1, -1):
            if result[j] is False:
                result[j] = True
                job[j] = arr[i][0]
                break
    print(job)


arr = [['a', 1, 3], ['b', 3, 25], ['c', 2, 1], ['d', 1, 6], ['e', 2, 30]]
print("Following is maximum profit sequence of jobs")
printJobScheduling(arr, 3)


# #### 11 knapsack algorithm
def knapSack(W, wt, val, n):
	dp = [0 for i in range(W+1)]
	for i in range(1, n+1):
		for w in range(W, 0, -1):
			if wt[i-1] <= w:
				dp[w] = max(dp[w], dp[w-wt[i-1]]+val[i-1])

	return dp[W]


p = [15,25,13,23]
w = [2,6,12,9]
c = 50
n = 4
print(knapSack(c, w, p, n))



# #### 12 tower of honai sort
def TowerOfHanoi(n, source, destination, auxiliary):
    if n == 1:
        print("Move disk 1 from source", source, "to destination", destination)
        return
    TowerOfHanoi(n - 1, source, auxiliary, destination)
    print("Move disk", n, "from source", source, "to destination", destination)
    TowerOfHanoi(n - 1, auxiliary, destination, source)


n = int(input("enter the number: "))

TowerOfHanoi(n, 'A', 'B', 'C')



# #### 13 queue sort
class Queue:
    def __init__(self):
        self.queue = []

    def enqueue(self, item):
        self.queue.append(item)

    def dequeue(self):
        if len(self.queue) < 1:
            return None
            return self.queue.pop(0)

    def display(self):
        print(self.queue)

    def size(self):
        return len(self.queue)


q = Queue()
n = int(input("Enter the number of elements for queue : "))
print("Enter the elements :")
for i in range(n):
    q.enqueue(int(input()))
print("Queue elements:")
q.display()
q.dequeue()
print("Dequeue elements:")
q.display()







class Queue:

    def __init__(self):
        self.queue = []

    # Add an element
    def enqueue(self, item):
        self.queue.append(item)

    # Remove an element
    def dequeue(self):
        if len(self.queue) < 1:
            return None
        return self.queue.pop(0)

    # Display  the queue
    def display(self):
        print(self.queue)

    def size(self):
        return len(self.queue)


q = Queue()
n = int(input("Enter the number of elements for queue : "))
print("Enter the elements :")
for i in range(n):
    q.enqueue(int(input()))
print("Queue elements:")
q.display()

q.dequeue()

print("Dequeue elements:")
q.display()
