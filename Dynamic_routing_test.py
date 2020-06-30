import networkx as nx
import time
import random
import numpy as np
import matplotlib.pyplot as mp

def Dijkstra(G, start, end):
    RG = G.reverse();
    dist = {};
    previous = {}
    for v in RG.nodes():
        #都设置为无穷大
        dist[v] = float('inf')
        previous[v] = 'none'
    dist[end] = 0
    u = end
    print(dist)
    print(min(dist, key=dist.get))
    while u != start:
        #获得最小值对应的键
        u = min(dist, key=dist.get)
        distu = dist[u]
        # print(distu)
        del dist[u]
        print(RG.edges(u))
        print(RG[u])
        for u, v in RG.edges(u):
            if v in dist:
                alt = distu + RG[u][v]['weight']
                if alt < dist[v]:
                    dist[v] = alt
                    previous[v] = u
    path = (start,)
    last = start
    while last != end:
        nxt = previous[last]
        path += (nxt,)
        last = nxt
    return path

def update_weight(Graph):                   #动态更新网络的权重
    for u, v, d in Graph.edges(data=True):
        float_num = random.uniform(0.75, 1.25)
        d['weight'] = d['weight'] * float_num

def process():                              #处理时延
    process_t = random.uniform(0.5, 1.5)
    time.sleep(process_t)

NETWORK_SIZE=30                             #小世界网络生成部分
k=4
p=0.5

ws=nx.watts_strogatz_graph(NETWORK_SIZE,k,p)
ps=nx.circular_layout(ws)#布置框架
nx.draw(ws,ps,with_labels=False,node_size=30)
mp.show()
A=np.array(nx.adjacency_matrix(ws).todense())

G = nx.Graph()                              #小世界网络转出至另一个图中
for i in range(30):                         #添加边
    for j in range(i,30):
        if A[i,j] != 0:
            w = random.random()
            G.add_edge(i,j,weight = w)

start = int(input("请输入源路由节点:\n"))
end = int(input("请输入目的路由节点:\n"))
path=nx.dijkstra_path(G, source=start, target=end)
print("Start from : %d\n" % start)

while(1):
    if(path[1] == end):
        process()
        print("Next Jump is destination: %d\n"% end)
        break;
    else:
        mid = path[1]
        process()
        print("Next Jump:%d\n" % mid)
        update_weight(G)
        path = nx.dijkstra_path(G, source=mid, target=end)



# print('节点0到7的路径：', path)
# print('dijkstra方法寻找最短距离：')
# distance=nx.dijkstra_path_length(G, source=0, target=7)
# print('节点0到7的距离为：', distance)
# rs = Dijkstra(G, 0, 6)
# print(rs)

