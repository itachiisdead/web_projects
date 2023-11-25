graph={
    's':['a','b','d'],
    'a':['a'],
    'b':['d'],
    'c':['d','g'],
    'd':['g']
}

def bfs(graph,start,dest):
  visited=[]
  q=[[start]]       
 
  while q:
    path= q.pop(0)
    node=path[-1]
    if path in visited:
       continue 
    visited.append(node)
    if node ==dest:
       return path
    else :
       adjecnt_nodes=graph.get(node,[])
       for node in adjecnt_nodes:
          newp=path.copy()
          newp.append(node)
          q.append(newp)


test= bfs(graph,'s','g')
print(test)



'''
while stack:
   path=stack.pop(0)
   node=path[-1]
'''





