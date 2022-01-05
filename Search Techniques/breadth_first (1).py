# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 00:27:52 2021

@author: workstation
"""

from queue import Queue
adj_list={
        'A':['B','D'],
        'B':['A','C'],
        'C':['B'],
        'D':['A','E','F'],
        'E':['D','F','G'],
        'F':['D','E','H'],
        'G':['E','H'],
        'H':['G','F']
        }
#bsf code
visited={}
level={}
parent={}
bfs_traversal_output=[]
queue=Queue()
for node in adj_list.keys():
    visited[node]=False
    parent[node]=None
    level[node]=-1
print(visited)
print(level)
print(parent)
s='A'
visited[s]=True
level[s]=0
queue.put(s)
while not queue.empty():
    u=queue.get()
    bfs_traversal_output.append(u)
    for v in adj_list[u]:
        if not visited[v]:
            visited[v]=True
            parent[v]=u
            level[v]=level[u]+1
            queue.put(v)
print(bfs_traversal_output)
#shortest distance of all nodes from the source node
#print(level)
print(level['G'])  # distance from S -> G
# the shortest path from source node to any node
v='G'
path=[]
while v is not None:
    path.append(v)
    v=parent[v]
path.reverse()
print(path)
