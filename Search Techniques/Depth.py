# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 14:58:14 2021

@author: workstation
"""

adj_list = {
            'A': ['B', 'C'],
            'B': ['D', 'E'],
            'C': ['B', 'F'],
            'D': [],
            'E': ['F'],
            'F': []
           }
color = {}
parent = {}
traverse_time = {}
dfs_traverse_output = []

for node in adj_list.keys():
    color [node] = 'white'
    parent[node] = None
    traverse_time[node] = [-1,-1]
print(color)
print(parent)
print(traverse_time)   
time = 0
def dfs(u):
    global time
    color[u] = 'gray'
    traverse_time[u][0] = time
    dfs_traverse_output.append(u)
    for q in adj_list[u]:
        if color[q] == 'white':
            parent[q] = u
            dfs(q)
    color[u] = 'black'
    traverse_time[u][1] = time
    time+=1

dfs('A') #start point

print()
print(parent)
print(color)
print(dfs_traverse_output)


for node in adj_list.keys():
    print(node,"->",traverse_time[node])