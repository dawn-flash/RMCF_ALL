import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import itertools
import json

"""
从topozoom中获取所有的topo，同时以json的格式保存
"""
def findAllFile(base):  #遍历文件夹中的某类文件
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith('.graphml'):
                fullname = os.path.join(root, f)
                yield fullname

def draw_graph_with_pos(data):  #绘制给定坐标的拓扑图，考虑到个别点未给的情况
        G = nx.read_graphml(data)
        x=[]
        y=[]
        i=0
        for n in G.nodes( data = True):
            if 'Latitude' in n[1].keys():#读取经纬度存在的点，并保存到两个列表
#               print(n)
                x.append(n[1]['Longitude'])
                y.append(n[1]['Latitude'])
            elif x:                     #未标注经纬度的点不是第一个点的情况
                x.append(random.uniform(min(x),max(x)))
                y.append(random.uniform(min(y),max(y)))
            else:                       #是前面的点时，则随机生成
                x.append(random.uniform(-50,50))
                y.append(random.uniform(-50,50))
        #    x[i]=n[1]['Latitude']
        #    y[i]=n[1]['Longitude'] 
        coordinates = [] 
        for i in range(G.number_of_nodes()):
            coordinates.append((x[i],y[i]))
        vnode= np.array(coordinates)
        npos = dict(zip(G.nodes, vnode)) 
        pos = {} 
        pos.update(npos)#前面的x、y两个列表合并成坐标，存到pos
        plt.figure()
        nx.draw(G,pos, with_labels=True, node_size=500, node_color='red', node_shape='.')
        return G

def draw_graph_without_pos(data):#绘制未给坐标的拓扑图
    G=nx.read_graphml(data)
    pos = nx.spring_layout(G)#Layout是随机分布
    plt.figure()
    nx.draw(G,pos, with_labels=True, node_size=500, node_color='red', node_shape='.')
#    plt.show()#显示图像
    return G

def generate_all_topo():
    plt.close('all')
    print("程序开始")
    for i in findAllFile('./dataset/'):  # 遍历文件，绘制给出经纬度的拓扑
        print("开始进行第" + str(i) + "个图")
        file_path = i
        (filepath, tempfilename) = os.path.split(file_path)  # 分离文件路径名
        (filename, extension) = os.path.splitext(tempfilename)  # 分离文件类型名
        print(filename)
        g = draw_graph_with_pos(i)
        plt.savefig(filename + '.png', format='png', dpi=200)
        A = np.array(nx.adjacency_matrix(g).todense())
        np.savetxt(filename + ".txt", A)
    print("程序结束")
    pass

def generate_graph_data(graph_path=None,k=6):
    """
    生成所有的网络图的数据
    :param graph_path: 图的gml文件路径
    :return:
    """

    #获取图的名字
    (filepath, tempfilename) = os.path.split(graph_path)  # 分离文件路径名
    (filename, extension) = os.path.splitext(tempfilename)  # 分离文件类型名
    #读取graphgml文件
    G=nx.read_graphml(graph_path)
    G.name=filename
    H=G.to_directed() #转换为有向图

    if type(H)!=nx.classes.digraph.DiGraph:
        print("不是有向图",filename)
        return
    #求顶点表和边表
    node_list=list(H.nodes())
    num_nodes=H.number_of_nodes()
    edge_list=list(H.edges())
    num_edges=H.number_of_edges()
    # print("输出顶点表，边表")
    # print(node_list)
    # print(edge_list)
    # print("节点数量",num_nodes)
    # print("边数量",num_edges)

    #求邻接矩阵
    A=np.array(nx.adjacency_matrix(H).todense())
    # print("邻接矩阵")
    # print(A)

    #保存任意两点之间的k条路径
    # print("所有节点两两之间的路径")
    all_node_paths={}
    all_node_paths=all_p2p_path(H,k)
    # print(all_node_paths)

    #计算路由矩阵
    route_matrix=generate_route_matrix(node_list,edge_list,all_node_paths,k)
    route_matrix=route_matrix.tolist()


    graph_info={}
    graph_info["name"]=H.name
    graph_info["node_list"]=node_list
    graph_info["num_nodes"]=num_nodes
    graph_info["edge_list"]=edge_list
    graph_info["num_edge"]=num_edges
    graph_info["adjacency_matrix"]=A.tolist()
    graph_info["p2p_paths"]=all_node_paths
    graph_info["route_matrix"]=route_matrix
    graph_info["Kth"]=k
    file_path=os.path.join(os.getcwd(),'graph_data',filename+'.json')
    with open(file_path,'w') as fw:
        json.dump(graph_info,fw,indent=4)
    print("保存成功")
    pass

def save_all_graph():
    """
    保存所有的和图相关的数据，
    源文件路径:./dataset/dataset_1
    目的文件路径：./graph_data
    json文件的相关内容：
    graph_info["name"]=H.name  图的名字
    graph_info["node_list"]=node_list  节点表list[str,...] 一般是‘0’到'n'之间的所有数
    graph_info["num_nodes"]=num_nodes  顶点的数量
    graph_info["edge_list"]=edge_list  边表list[(str,str),...] 存放所有相连的边 如[('0','1'),('0','2'])
    graph_info["num_edge"]=num_edges   边的数量
    graph_info["adjacency_matrix"]=A    邻接矩阵 list[list]
    graph_info["p2p_paths"]=all_node_paths  所有的路径 list[(sourced,dest),...]
    graph_info["route_matrix"]=route_matrix  图的路由矩阵[path,link]
    graph_info["Kth"]=k  每个sd队之间的流的数量
    :return:
    """
    print("程序开始")
    dir_path=os.path.join(os.getcwd(),'dataset','dataset_1')
    for i in findAllFile(dir_path):  # 遍历文件，绘制给出经纬度的拓扑
        print("开始计算图"+i)
        file_path = i
        try:#有问题的图文件均不作处理
            generate_graph_data(file_path,k=6)
        except Exception:
            print("文件{0}有异常".format(i))
    print("程序结束")
    pass

def all_p2p_path(H:nx.DiGraph,k=5):
    """
    获取topo图中所有节点，两两之间的k条路径
    :param H: 网络图
    :param k: k条路径
    :return:
    """
    num_nodes=H.number_of_nodes()
    # 保存任意两点之间的k条路径
    # (1) 获取排列数$A_n^2$
    all_node_pairs = list(itertools.permutations(list(range(num_nodes)), 2))  # 存放节点对
    # print("所有的节点组合", all_node_pairs)
    all_node_paths = {}  # 存放节点之间的k条路径
    # (2) 求两两节点之间的所有paths
    for a_pair in all_node_pairs:
        a_p2p_path = p2p_path(H, str(a_pair[0]), str(a_pair[1]), k)
        if len(a_p2p_path) < k:
            for i in range(k - len(a_p2p_path)):
                a_p2p_path.append([])
        all_node_paths[str(a_pair)] = a_p2p_path  # json中的dict的key不能为元组
    # print("所有节点两两之间的路径")
    # print(all_node_paths)

    return all_node_paths

def p2p_path(graph:nx.Graph,sourceNode:str,destNode:str,K=5):
    """
    获取两点之间的所有path
    :param graph: nx中的图
    :param sourceNode: string 源节点
    :param destNode: string 目的节点
    :param K: int 最多K条路径
    :return: p2p_edge_paths list[(tuple)] 返回两点之间的所有路径
    """
    p2p_paths = list(itertools.islice(nx.shortest_simple_paths(graph, sourceNode, destNode, weight=None), K))  # 获取无重复节点的路径
    # print("节点{0}和节点{1}所有的path".format(sourceNode,destNode))
    # print(p2p_paths)
    # 将path转换为edge组合
    p2p_edge_paths = [] #所有的path
    for a_path in p2p_paths:
        p2p_a_edge_path = []  #p2p的一条path
        for i in range(len(a_path) - 1):
            a_edge = (a_path[i], a_path[i + 1])
            p2p_a_edge_path.append(a_edge)
        p2p_edge_paths.append(p2p_a_edge_path)

    return p2p_edge_paths

def generate_route_matrix(nodes:list,edges:list,nodes_path:dict,K)->np.ndarray:
    """
    将节点之间的path，转为整个网络的路由矩阵，其中两点之间最多k条path
    :param nodes: list[str]节点表
    :param edges: list[(str)]边表
    :param nodes_path: dict 路径表{'(0,1)':[[('0','2'),('2','1')],[]]}
    :param K: 两点之间最多k条路径
    :return:  整个网络的路由矩阵
    """
    node_pairs=list(itertools.permutations(list(range(len(nodes))),2))
    num_nodes_pair=len(nodes)*(len(nodes)-1)
    num_link=len(edges)
    route_matrix=np.zeros(shape=(num_nodes_pair*K,num_link))
    for p in range(num_nodes_pair):
        for k in range(K):
            for e in nodes_path[str(node_pairs[p])][k]:
                try:
                    e_index=edges.index(e)
                except Exception:
                    print("出现异常")
                else:
                    route_matrix[p*K+k][e_index]=1
    # print("输出路由矩阵")
    # print(route_matrix)
    return route_matrix

def test_generate_route_matrix():
    G = nx.DiGraph()
    # 给图添加节点
    for i in range(4):
        G.add_node(str(i))
    link = [(0, 1), (1, 0),
            (0, 2), (2, 0),
            (1, 3), (3, 1),
            (2, 3), (3, 2)]
    # 给图添加边
    for i in range(len(link)):
        G.add_edge(str((link[i][0])), str(link[i][1]), weight=1)

    print(type(G)==nx.classes.digraph.DiGraph)

    nodes_list=list(G.nodes())
    edges_list=list(G.edges())
    #所有节点之间的两两路径
    all_node_path=all_p2p_path(G)
    #求路由矩阵
    route_matrix=generate_route_matrix(nodes_list,edges_list,all_node_path,2)
    print("路由矩阵")
    print(route_matrix)

def test():
    """
    测试一些函数的功能
    :return:
    """
    # os.walk函数的功能
    for root,dirs,files in os.walk('./dataset'):
        for name in dirs:
            print(os.path.join(root,name))
        for name in files:
            print(os.path.join(root,name))
    print("测试结束")
    pass

def testksp():
    """
    测试ksp算法，kth shortpath 算法，找出两点之间最短的k条路径
    :return:
    """
    G = nx.DiGraph()
    #给图添加节点
    for i in range(11):
        G.add_node(str(i + 1))
    # link = [(1, 2), (2, 1),
    #         (1, 4), (4, 1),
    #         (2, 3), (3, 2),
    #         (2, 4), (4, 2),
    #         (3, 5), (5, 3),
    #         (4, 6), (6, 4),
    #         (5, 6), (6, 5),
    #         (5, 7), (7, 5),
    #         (6, 8), (8, 6),
    #         (7, 8), (8, 7),
    #         (7, 9), (9, 7),
    #         (8, 10), (10, 8),
    #         (9, 11), (11, 9),
    #         (10, 11), (11, 10)]

    link=[(1,2),(2,1),
          (1,3),(3,1),
          (2,4),(4,2),
          (3,4),(4,3)]
    #给图添加边
    for i in range(len(link)):
        G.add_edge(str((link[i][0])), str(link[i][1]), weight=1)

    print(list(nx.shortest_simple_paths(G,'1','4',weight=None)))
    k=5
    allpath=list(itertools.islice(nx.shortest_simple_paths(G,'1','4',weight=None),k))#获取无重复节点的路径
    print("所有的路径")
    for path in allpath:
        print(path)


if __name__ == '__main__':
    # generate_all_topo()
    # test()
    # testksp()
    # generate_all_topo()
    # generate_graph_data()
    # test()
    # test_generat_route_matrix()
    save_all_graph()

    pass













