# BFS
# Adjacent nodes graph 
graph = {
    "S":["A","B","D"],
    "A":["C"],
    "B":["D"],
    "C":["D","G"],
    "D":["G"],    
    }
# BFS Function
def BFS(graph,start,goal):
    visited = [] # To store nodes was visited
    queue = [[start]] # queue to save paths
    
    # While loop to loop for each path
    while queue:
        path = queue.pop(0) # remove first element from queue and save in path
        node = path[-1]     # save last node for ex: path = ['S','A','D'] => node = D
        if node in visited: # check if this node visited or not
            continue
        visited.append(node)# if not add to visited
        if node == goal:    # if this node my goal then return path
            return path
        else:
            adjacent_nodes = graph.get(node,[]) # Show adjacent nodes 
            for node2 in adjacent_nodes:
                new_path = path.copy()
                new_path.append(node2)
                queue.append(new_path)

print(BFS(graph, "S", "G"))











# DFS
graph = {
    'S':['A','B','D'],
    'A':['C'],
    'B':['D'],
    'C':['D','G'],
    'D':['G'],
    }

def DFS(graph, start, goal):
    visited = []
    stack = [[start]]
    
    while stack:
        path = stack.pop()
        node = path[-1]
        
        if node in visited:
            continue
        visited.append(node)
        
        if node == goal:
            return path
        else:
            adjacent_nodes = graph.get(node,[])
            for node2 in adjacent_nodes:
                new_path = path.copy()
                new_path.append(node2)
                stack.append(new_path)

print(DFS(graph,'S','G'))

















#UCS  uniform
graph = {
    'S':[('A',2),('B',3),('D',5),],
    'A':[('C',4),],
    'B':[('D',4),],
    'C':[('D',1),('G',2),],
    'D':[('G',5),],
 
        }

def path_cost(path):
    total_cost = 0
    for (node, cost) in path:
        total_cost += cost
    return total_cost,path[-1][0]


def UCS (graph, start, goal):
    visited = []
    queue = [[(start,0)]]

    while queue:
        queue.sort(key= path_cost)
        path = queue.pop(0)
        node = path[-1][0]
        
        if node in visited:
            continue
        visited.append(node)
        
        if node == goal:
            return path
        else:
            adjacent_node = graph.get(node,[])
            for (node2, cost) in adjacent_node:
                new_path = path.copy()
                new_path.append((node2, cost))
                queue.append(new_path)
                
solution = UCS(graph, 'S', 'G')        
print("Solution: ", solution)
print("All cost: ", path_cost(solution)[0])


















# A*
gragh = {
    'S': [('A', 1), ('B', 4)],
    'A': [('B', 2), ('C', 5), ('G', 12)],
    'B': [('C', 2)],
    'C': [('G', 3)],
}
H_cost = {
    'S': 7,
    'A': 6,
    'B': 4,
    'C': 2,
    'G': 0,
}

def path_cost(path):
    G_cost = 0
    for (node, cost) in path:
        G_cost += cost
    last_node = path[-1][0]
    h_cost = H_cost[last_node]
    f_cost = h_cost+G_cost
    return f_cost, last_node


def A(gragh, start, goal):
    visited = []
    stack = [[(start, 0)]]
    while stack:
        stack.sort(key=path_cost)
        path = stack.pop(0)
        node = path[-1][0]
        if node in visited:
            continue
        visited.append(node)
        if node == goal:
            return path
        else:
            adjacent_node = gragh.get(node, [])
            for (node2, cost) in adjacent_node:
                new_path = path.copy()
                new_path.append((node2, cost))
                stack.append(new_path)


solution = A(gragh, 'S', 'G')
print('Solution: ', solution)
print('Cost of solution: ', path_cost(solution)[0])


























# greedy 
graph = {
    'S': [('A', 1), ('B', 4)],
    'A': [('B', 2), ('C', 5), ('G', 12)],
    'B': [('C', 2)],
    'C': [('G', 3)],
}

H_table = {
    'S': 7,
    'A': 6,
    'B': 4,
    'C': 2,
    'G': 0
}


def path_h_cost(path):
    last_node = path[-1][0]
    h_cost = H_table[last_node]
    return h_cost, last_node


def Greedy(graph, start, goal):
    visited = []
    queue = [[(start, 0)]]

    while queue:
        queue.sort(key=path_h_cost)
        path = queue.pop(0)
        last_node = path[-1][0]

        if last_node is visited:
            continue
        visited.append(last_node)
        if last_node == goal:
            return path
        else:
            adjecent_nodes = graph.get(last_node, [])
            for (node, cost) in adjecent_nodes:
                new_path = path.copy()
                new_path.append((node, cost))
                queue.append(new_path)


sol = Greedy(graph, 'S', 'G')
print("A Solution is: ", sol)
print(path_h_cost(sol)[0])


































#genatic
import numpy as np

def init_pop(pop_size):
    return np.random.randint(8, size=(pop_size, 8))

def calc_fitness(population):
    fitness_vals = []
    for x in population:
        penalty = 0
        for i in range(8):
            r = x[i]
            for j in range(8):
                if i == j:
                    continue
                d=abs(i-j)
                if x[j] in [r, r+d, r-d]:
                    penalty+=1
        fitness_vals.append(penalty)
    return -1 * np.array(fitness_vals)

def selection(population, fitness_vals):
    probs = fitness_vals.copy()
    probs += abs(probs.min()) + 1
    probs = probs/probs.sum()
    N = len(population)
    indices = np.arange(N)
    selected_indices = np.random.choice(indices, size=N, p=probs)
    selected_population = population[selected_indices]
    return selected_population

initial_population = init_pop(4)
print("\nInitial populations are\n", initial_population)
fitness_vals = calc_fitness(initial_population)
print("\nFitness values are\n", fitness_vals)
selected_population = selection(initial_population, fitness_vals)
print("\nSelected populations are\n",selected_population)










#***********************************************************************************************************
































#BFS

graph ={
        'S':['A','B','D'],
        'A':['C'],
        'B':['D'],
        'C':['D','G'],
        'D':['G'],
        }
def BFS (graph,start,goal):
    visited=[]
    queue=[[start]]
    while queue:
        path=queue.pop(0)
        node =path[-1]
        if node in visited:
            continue
        visited.append(node)
        if node == goal:
            return path
        else:
            adj_nodes=graph.get(node,[])
            for node2 in adj_nodes:
                new_path=path.copy()
                new_path.append(node2)
                queue.append(new_path)
                
sol=BFS(graph,'S','G')
print(sol)












#DFS:

graph ={
        'S':['A','B','D'],
        'A':['C'],
        'B':['D'],
        'C':['D','G'],
        'D':['G'],
        }

def DFS (graph,start,goal):
    visited=[]
    stack=[[start]]
    while stack:
        path=stack.pop()
        node=path[-1]
        if node in visited:
            continue
        visited.append(node)
        if node == goal:
            return path
        else:
            adj_nodes = graph.get(node,[])
            for node2 in adj_nodes:
                new_path=path.copy()
                new_path.append(node2)
                stack.append(new_path)
            
                
              
solution = DFS(graph,'S','G')
print ("solution is ",solution)









#USC:
graph ={
        'S':[('A',2),('B',3),('D',5)],
        'A':[('C',4)],
        'B':[('D',4)],
        'C':[('D',1),('G',2)],
        'D':[('G',5)],
        }

def path_cost (path):
    total_cost=0
    for (node,cost) in path :
        total_cost+=cost
    return total_cost,path[-1][0]


def UCS (graph,start,goal):
    visited=[]
    queue=[[(start,0)]]
    while queue:
        queue.sort(key=path_cost)
        path = queue.pop(0)
        node=path[-1][0]
        if node in visited:
            continue
        visited.append(node)
        if node == goal:
            return path
        else:
            adj_nodes = graph.get(node,[])
            for (node2,cost) in adj_nodes:
                new_path=path.copy()
                new_path.append((node2,cost))
                queue.append(new_path)
                
solution=UCS(graph,'S','G')
print ("solution is ",solution)
print(" and its path is ",path_cost(solution)[0])








#A*:
graph={
       'S':[('A',1),('B',4)],
       'A':[('B',2),('C',5),('G',12)],
       'B':[('C',2)],
       'C':[('G',3)],
       }
h_table ={
    'A':6,
    'S':7,
    'B':4,
    'C':2,
    'G':0,
    }

def f_cost (path):
    g_cost=0
    for (node2,cost) in path:
        g_cost+=cost
    last_node=path[-1][0]
    h_cost=h_table[last_node]
    f_cost=g_cost+h_cost
    return f_cost,last_node

def A_STAR (graph,start,goal):
    visited=[]
    queue =[[(start,0)]]
    while queue:
        queue.sort(key=f_cost)
        path=queue.pop(0)
        node=path[-1][0]
        if node in visited:
            continue
        visited.append(node)
        if node == goal:
            return path
        else:
            adj_nodes = graph.get(node,[])
            for (node2,cost) in adj_nodes:
                new_path=path.copy()
                new_path.append((node2,cost))
                queue.append(new_path)
                            
solution=A_STAR(graph,'S','G')
print("the solution is ",solution)
print ("and its f cost = ", f_cost(solution)[0])







 #genatic:

import numpy as np

def intial_pop (size_pop):
    return np.random.randint(8,size=(size_pop,8))


def cal_fitness (population):
    fitness_vals=[]
    for x in population:
        penalty=0
        for i in range(8):
            r=x[i]
            for j in range(8):
                if i==j :
                    continue
                d=abs(i-j)
                if x[j] in (r,r-d,r+d):
                    penalty+=1
        fitness_vals.append(penalty)
    return -1*np.array(fitness_vals)


def selection (population,fitness_vals):
    props=fitness_vals.copy()
    props+=abs(props.min())+1
    props=props/props.sum()
    N=len(population)
    indices=np.arange(N)
    selected_indeices=np.random.choice(indices,size=N,p=props)
    selected_population=population[selected_indeices]
    return selected_population
    

intial_population= intial_pop(4)
fitness_vals=cal_fitness(intial_population)
print(intial_population)
print(fitness_vals)
print (selection(intial_population, fitness_vals))

///////////////////////////////////////////
   greedy:

graph={
       'S':[('A',1),('B',4)],
       'A':[('B',2),('C',5),('G',12)],
       'B':[('C',2)],
       'C':[('G',3)],
       }
h_table ={
    'A':6,
    'S':7,
    'B':4,
    'C':2,
    'G':0,
    }

def h_cost (path):
    g_cost=0
    for (node2,cost) in path:
        g_cost+=cost
    last_node=path[-1][0]
    h_cost=h_table[last_node]
    return h_cost,last_node

def greedy (graph,start,goal):
    visited=[]
    queue=[[(start,0)]]
    while queue:
        queue.sort(key=h_cost)
        path=queue.pop(0)
        node=path[-1][0]
        if node in visited:
            continue
        visited.append(node)
        if node == goal:
            return path
        else:
            adj_nodes=graph.get(node,[])
            for (node2,cost) in adj_nodes:
                new_path=path.copy()
                new_path.append((node2,cost))
                queue.append(new_path)

sol=greedy(graph,'S','G')
print(sol)
print ("and h_cost = ",h_cost(sol))   