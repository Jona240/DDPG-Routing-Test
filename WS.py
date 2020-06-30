import networkx as ne #导入建网络模型包，命名ne
import matplotlib.pyplot as mp #导入科学绘图包，命名mp
import numpy as np

#WS network graphy
NETWORK_SIZE=30
k=15
p=0.5

ws=ne.watts_strogatz_graph(NETWORK_SIZE,k,p)
ps=ne.circular_layout(ws)#布置框架
ne.draw(ws,ps,with_labels=False,node_size=30)

mp.show()
A=np.array(ne.adjacency_matrix(ws).todense())
print(A)
