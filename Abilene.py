import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

"""
画出拓扑图Abilence
"""

class Abilene:
    
    def __init__(self, O, D):
    
        self.G = nx.DiGraph()
    
        self.node_num = 11
        
        self.node_list = []
        for i in range(self.node_num):
            self.node_list.append('s'+str(i))
        
        self.nodePos = {'s0': (-122.33207,   47.60621),
                        's1': np.array([-122.03635,   37.36883]),
                        's2': np.array([-112.24368,   33.05223]),
                        's3': np.array([-110.99153,   41.74204]),
                        's4': np.array([-97.358421,   30.7499  ]),
                        's5': np.array([-98.,         38.]),
                        's6': np.array([-84.38633,    33.75374]),
                        's7': np.array([-86.163712,   39.75999 ]),
                        's8': np.array([-77.050636,   38.88924 ]),
                        's9': np.array([-87.623177,   43.88183 ]),
                        's10': np.array([-73.935242,  41.93061 ]) }
        
        self.nodeColors = [
                            'white',
                            'white',
                            'white',
                            'white',
                            'white',
                            'white',
                            'white',
                            'white',
                            'white',
                            'white',
                            'white'  ]
        
        self.node_demand = np.zeros(self.node_num).tolist()
        
        for i in range(len(O[0])):
            node = O[0][i]
            demand = O[1][i]
            
            self.nodeColors[node] = 'red'
            self.node_demand[node] = demand

        for j in range(len(D[0])):
            node = D[0][j]
            demand = D[1][j]
            
            self.nodeColors[node] = 'green'
            self.node_demand[node] = demand
    
        for i in range(self.node_num):
                self.G.add_node(self.node_list[i], pos=self.nodePos[self.node_list[i]],  desc='$'+self.node_list[i]+'$', 
                            demand=self.node_demand[i], color='grey')
    
        self.link_list = [  (0, 1), (1, 0),
                            (0, 3), (3, 0),
                            (1, 2), (2, 1),
                            (1, 3), (3, 1),
                            (2, 4), (4, 2),
                            (3, 5), (5, 3),
                            (4, 5), (5, 4),
                            (4, 6), (6, 4),
                            (5, 7), (7, 5),
                            (6, 7), (7, 6),
                            (6, 8), (8, 6),
                            (7, 9), (9, 7),
                            (8, 10), (10, 8),
                            (9, 10), (10, 9) ]
        self.link_num = len(self.link_list)
        
        self.link_failure_prob = np.ones(self.link_num) * 0.05        
        self.link_load = np.zeros(self.link_num)
        
        for i in range(self.link_num):
            self.G.add_edge(self.node_list[self.link_list[i][0]], self.node_list[self.link_list[i][1]],
                            weight=1, failure_prob = self.link_failure_prob[i],
                            load=str(self.link_load[i]), no=str(i))
            
        self.adjMatrix = np.asarray(nx.to_numpy_matrix(self.G))
        
        self.rtMatrix = np.zeros([self.node_num, self.node_num, self.link_num])
        for sNode in range(self.node_num):
            for dNode in range(self.node_num):
                if sNode != dNode:
                    path = nx.shortest_path(self.G, source='s'+str(sNode), target='s'+str(dNode))
                    for n in range(len(path) - 1):
                        link = self.link_list.index((self.node_list.index(path[n]), 
                                                     self.node_list.index(path[n+1])))
                        self.rtMatrix[sNode][dNode][link] = 1
            
    def MCF(self):
        """
        输出MCF的结果
        :return:
        """
        self.flowCost, self.flowDict = nx.network_simplex(self.G)
        print('flowCost:\n%d\n' %(self.flowCost))
        print('flowDict:')
        pprint(self.flowDict)
        
    def Plot_Network(self):
        print("画图开始")
        nx.draw_networkx_nodes(self.G, G.nodePos, node_color = self.nodeColors, edgecolors='black')
        # nx.draw_networkx_edges(self.G, self.nodePos, with_labels=True, connectionstyle='arc3, rad = 0.2')
        nx.draw_networkx_edges(self.G, self.nodePos, connectionstyle='arc3, rad = 0.2')
        
        nodeLabels = nx.get_node_attributes(self.G, 'desc')
        nx.draw_networkx_labels(self.G, self.nodePos, labels=nodeLabels, font_size=8)  
        
        edgeLabels = nx.get_edge_attributes(self.G, 'no')
        nx.draw_networkx_edge_labels(self.G, self.nodePos, 
                                     edge_labels=edgeLabels, label_pos=0.7, font_size=5)
    
        plt.savefig('Abilene_Offloading.png', format='png', dpi=200)
        print("画图结束")

if __name__ == '__main__':
    plt.close('all')
    
    O = [[2,    4],     # Origin node
         [-10,  -10]]    # demand
    
    D = [[0,     7],      # Destination node
         [10,    10]]     # supply
    
    G = Abilene(O,D)
    G.Plot_Network()
    G.MCF()
