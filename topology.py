#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/9/24 15:12
# @Author  : Lelsey
# @File    : topology.py

#topo类，保存topo的所有信息

import copy
import numpy as np
import util_tool

class NetworkGraph():

    def __init__(self,graph_adj,link_list):
        """
        初始化方法
        :param graph_adj: array 存储图的邻接矩阵
        :param link_list:  存储图的边表list[tuple]  每一个tuple（源节点，目的结点）
        """
        self.graph_adj=graph_adj
        self.link_list=link_list
        self.graph_c=copy.deepcopy(graph_adj)
        self.graph_u=copy.deepcopy(graph_adj)
        self.graph_p=copy.deepcopy(graph_adj)

        #将相关的矩阵转换为链路列表
        self.graph_c_list=util_tool.graph_cost_matrix_to_list(self.graph_adj,self.graph_c)
        self.graph_u_list=self.graph_c_list
        self.graph_p_list=self.graph_c_list

        self.num_nodes=graph_adj.shape[0]
        self.num_links=len(link_list)

    def set_graph_p(self,fail_pro):
        """
        同构概率，设置链路的失效概率矩阵
        :param fail_pro: float 链路的失效概率
        :return:
        """
        self.graph_p=self.graph_p*fail_pro            #失效概率矩阵
        self.graph_p_list=[fail_pro]*self.num_links   #失效概率列表

    def set_graph_p_isomerism(self,fail_pro_list):
        """
        异构概率，设置链路的失效概率矩阵
        :param fail_pro_list: list 每一条链路的失效概率
        :return:
        """
        k=0
        for i in range(self.graph_adj.shape[0]):
            for j in range(self.graph_adj.shape[1]):
                if self.graph_adj[i][j]==1:
                    self.graph_p[i][j]=fail_pro_list[k]
                    k+=1
        self.graph_p_list=fail_pro_list        #失效概率列表

    def set_graph_loss_isomerism(self,loss_list):
        """
        设置异构丢包列表
        :param loss_list: [list]
        :return:
        """
        self.graph_loss_list=loss_list

    def set_graph_u_isomerism(self,u_list):
        """
        异构链路
        设置所有链路的容量
        :param u_list: list 所有链路的容量
        :return:
        """
        k = 0
        for i in range(self.graph_adj.shape[0]):
            for j in range(self.graph_adj.shape[1]):
                if self.graph_adj[i][j] == 1:
                    self.graph_u[i][j] = u_list[k]
                    k += 1
        self.grap_u_list=u_list      #链路的容量列表

    def set_graph_c_isomerism(self,cost_list):
        """
        异构链路
        设置所有链路的cost
        :param cost_list: list 链路的cost列表
        :return:
        """
        k = 0
        for i in range(self.graph_adj.shape[0]):
            for j in range(self.graph_adj.shape[1]):
                if self.graph_adj[i][j] == 1:
                    self.graph_c[i][j] = cost_list[k]
                    k += 1

        self.graph_c_list=cost_list     #链路的cost列表

    def set_source_dest_nodes(self,source_nodes,dest_nodes):
        self.source_nodes = source_nodes
        self.dest_nodes = dest_nodes

    def set_demand(self,graph_deamnd):
        """
        设置所有源节点和目的结点的需求
        :param graph_deamnd: list 所有结点的需求
        :return:
        """
        # :param source_nodes: list 源节点列表 （结点编号，非下标）
        # :param dest_nodes: list  目的结点列表（结点编号，非下标）
        self.graph_demand=graph_deamnd
        self.source_demand=[graph_deamnd[s-1] for s in self.source_nodes]
        self.dest_demand=[graph_deamnd[d-1] for d in self.dest_nodes]

    def set_route(self,K,path:list):
        """
        设置路由矩阵
        :param K  int 每个SD之间的最大路径数量
        :param path:   list[list]  路径列表 [p_1,p_2,p_3...]每一个元素一条路径，由多条链路组成，有链路编号组成[[6, 3],[6, 5, 8]]
        :return:
        """
        self.path=path
        self.K=K
        flows_num=len(self.source_nodes)*len(self.dest_nodes)*K
        links_num=len(self.link_list)
        self.route=util_tool.set_route(flows_num,links_num,path)

    def set_route_matrix(self,route_list:list):
        """
        设置路由矩阵
        :param route_matrix:
        :return:
        """
        self.route=route_list

    def generate_scenes(self,fail_link_num):
        """
        产生所有的场景
        :param fail_link_num: int 最大失效链路的数量
        :return:
        """
        #scenes 所有场景矩阵
        #scenes_prob 所有场景的概率
        #scenes_list 所有场景的列表，[s_1,s_2,s_3..]每一个s由【1，0,1，0...]等组成 1表示失效，0表示未失效
        #创建场景【0,1】
        self.scenes,self.scenes_prob,self.scenes_list=util_tool.create_scenes(self.graph_adj,self.graph_p,fail_link_num,self.source_nodes,self.dest_nodes)
        self.scenes_prob=list(np.array(self.scenes_prob) / sum(self.scenes_prob)) #概率归一化
        #创建新的场景【0,1,2】
        # self.scenes,self.scenes_prob,self.scenes_list=util_tool.cerate_scenes_plus(self.graph_adj,self.graph_p_list,self.graph_loss_list,fail_link_num=2)
        # return self.scenes,self.scenes_prob,self.scenes_list


    def change_topo(self):
        """
        根据发送需求总和和接收需求总和，判断是否需要添加虚拟节点，改变topo
        :param sum_demand: 发送结点总和
        :param sum_sink: 接收结点总和
        :return:
        """
        sum_demand=sum(self.source_demand)
        sum_sink=-sum(self.dest_demand)
        if sum_demand==sum_sink:
            self.remain_default()
        elif sum_demand>sum_sink: #添加一个虚拟目的结点
            virt_node_demand=sum_sink-sum_demand
            self.add_dest_node(virt_node_demand)
        else:  #添加一个源节点
            virt_node_demand=sum_sink-sum_demand
            self.add_source_node(virt_node_demand)

    def remain_default(self):

        self.new_graph_adj=self.graph_adj
        self.new_link_list=self.link_list
        self.new_graph_c=self.graph_c
        self.new_graph_u=self.graph_u
        self.new_graph_p=self.graph_p

        self.new_graph_c_list=self.graph_c_list
        self.new_graph_p_list=self.graph_p_list
        self.new_grap_u_list=self.graph_u_list
        self.new_graph_loss_list=self.graph_loss_list

        self.new_num_nodes=self.num_nodes
        self.new_num_links=self.num_links


        self.new_source_nodes=self.source_nodes
        self.new_dest_nodes=self.dest_nodes
        self.new_graph_demand=self.graph_demand
        self.new_source_demand=self.source_demand
        self.new_dest_demand=self.dest_demand

        self.new_path=self.path
        self.new_K=self.K
        self.new_route=self.route
        #计算每一条流的cost
        self.compute_flow_cost()

        self.new_scenes=self.scenes
        self.new_scenes_prob=self.scenes_prob
        #对概率进行归一化
        self.new_scenes_prob=self.scenes_prob #概率归一化
        self.new_scenes_list=self.scenes_list

    def add_source_node(self,virt_scoure_demand):
        """
        添加一个虚拟源节点
        :param virt_scoure_demand:  虚拟源节点的需求
        :return:
        """
        #构造基本图的信息，link和邻接矩阵
        self.new_num_nodes=self.num_nodes+1
        self.vir_node_id=self.new_num_nodes
        self.new_num_links=self.num_links+len(self.dest_nodes)
        self.new_link_list=copy.deepcopy(self.link_list)
        for i in range(len(self.dest_nodes)):
            self.new_link_list.append((self.vir_node_id,self.dest_nodes[i]))

        #构造新的邻接矩阵
        self.new_graph_adj=copy.deepcopy(self.graph_adj)
        self.new_graph_adj=np.zeros(shape=(self.new_num_nodes,self.new_num_nodes))
        for i in range(len(self.new_link_list)):
            self.new_graph_adj[self.new_link_list[i][0]-1][self.new_link_list[i][1]-1]=1

        self.new_graph_u=copy.deepcopy(self.new_graph_adj)
        self.new_K=self.K

        #构造新的源节点和目的结点 和需求
        self.new_source_nodes = copy.deepcopy(self.source_nodes)
        self.new_source_nodes.append(self.vir_node_id)
        self.new_dest_nodes=self.dest_nodes
        self.new_graph_demand=copy.deepcopy(self.graph_demand)
        self.new_graph_demand.append(virt_scoure_demand)
        self.new_source_demand = [self.new_graph_demand[s - 1] for s in self.new_source_nodes]
        self.new_dest_demand=self.dest_demand

        #构造新的cost矩阵:直接添加一行【0，0】和一列【0,0】
        self.new_graph_c=copy.deepcopy(self.graph_c)
        new_row=np.zeros(self.num_nodes).reshape(1,self.num_nodes)
        new_col=np.zeros(self.new_num_nodes).reshape(1,self.new_num_nodes)
        self.new_graph_c=np.r_[self.new_graph_c,new_row]
        self.new_graph_c=np.c_[self.new_graph_c,new_col.T]
        self.new_graph_c_list=util_tool.graph_cost_matrix_to_list(self.new_graph_adj,
                                                                  self.new_graph_c)

        #构造新的概率矩阵:直接添加一行【0，0】和一列【0,0】
        self.new_graph_p=copy.deepcopy(self.graph_p)
        self.new_graph_p=np.r_[self.new_graph_p,new_row]
        self.new_graph_p=np.c_[self.new_graph_c,new_col.T]

        #构造新的场景矩阵，场景概率列表，场景列表
        # 场景矩阵 每个场景添加一行[0,0]和一列[0,0]
        # 和场景列表 每个场景list后直接添加len（dest_nodes）个0
        self.new_scenes=copy.deepcopy(self.scenes)
        self.new_scenes_list=copy.deepcopy(self.scenes_list)
        self.new_scenes_prob=copy.deepcopy(self.scenes_prob)
        # self.new_scenes_prob=list(np.array(self.scenes_prob)/sum(self.scenes_prob))    #概率归一化

        for s in range(len(self.new_scenes)):
            self.new_scenes[s]=np.r_[self.new_scenes[s],new_row]
            self.new_scenes[s]=np.c_[self.new_scenes[s],new_col.T]
            extend_list=[0]*len(self.new_dest_nodes)
            self.new_scenes_list[s].extend(extend_list)

        #构造新的路由矩阵
        self.new_path=copy.deepcopy(self.path)
        i=1
        for d in range(len(self.dest_nodes)):
            self.new_path.append([self.num_links+i])
            i+=1
            for k in range(self.K-1):
                self.new_path.append([])
        flows_num=len(self.new_source_nodes)*len(self.new_dest_nodes)*self.K
        links_num=len(self.new_link_list)
        self.new_route=util_tool.set_route(flows_num,links_num,self.new_path)
        #计算每一条流的cost
        self.compute_flow_cost()
        pass

    def add_dest_node(self,virt_dest_demand):
        """
        添加一个虚拟目的结点
        注意：添加链路的cost需要手动设置为一个常数
        :param virt_dest_demand: 虚拟目的结点的接收需求
        :return:
        """

        # 构造基本图的信息，link和邻接矩阵
        self.new_num_nodes = self.num_nodes + 1
        self.vir_node_id = self.new_num_nodes
        self.new_num_links = self.num_links + len(self.source_nodes)
        self.new_link_list = copy.deepcopy(self.link_list)
        for i in range(len(self.source_nodes)):
            self.new_link_list.append((self.source_nodes[i], self.vir_node_id))

        # 根据链路列表，构造新的邻接矩阵
        self.new_graph_adj = copy.deepcopy(self.graph_adj)
        self.new_graph_adj = np.zeros(shape=(self.new_num_nodes,self.new_num_nodes))
        for i in range(len(self.new_link_list)):
            self.new_graph_adj[self.new_link_list[i][0] - 1][self.new_link_list[i][1] - 1] = 1

        self.new_graph_u=copy.deepcopy(self.new_graph_adj)
        self.new_K=self.K

        # 构造新的源节点和目的结点 和需求
        self.new_dest_nodes = copy.deepcopy(self.dest_nodes)
        self.new_dest_nodes.append(self.vir_node_id)
        self.new_source_nodes = self.source_nodes
        self.new_graph_demand = copy.deepcopy(self.graph_demand)
        self.new_graph_demand.append(virt_dest_demand)
        self.new_dest_demand = [self.new_graph_demand[d - 1] for d in self.new_dest_nodes]
        self.new_source_demand = self.source_demand

        # 构造新的cost矩阵
        self.new_graph_c = copy.deepcopy(self.graph_c)
        new_row = np.zeros(self.num_nodes).reshape(1, self.num_nodes)
        new_col=np.zeros(self.new_num_nodes)
        for s in range(len(self.new_source_nodes)):
            new_col[self.new_source_nodes[s]-1]=1  #设置cost=1
        new_col=new_col.reshape(1,self.new_num_nodes)
        # new_col = np.zeros(self.new_num_nodes).reshape(1, self.new_num_nodes)
        self.new_graph_c = np.r_[self.new_graph_c, new_row]
        self.new_graph_c = np.c_[self.new_graph_c, new_col.T]
        self.new_graph_c_list = util_tool.graph_cost_matrix_to_list(self.new_graph_adj,
                                                                    self.new_graph_c)

        # 构造新的概率矩阵
        self.new_graph_p = copy.deepcopy(self.graph_p)
        new_row = np.zeros(self.num_nodes).reshape(1, self.num_nodes)
        new_col = np.zeros(self.new_num_nodes)
        for s in range(len(self.new_source_nodes)):
            new_col[self.new_source_nodes[s]-1]=1  #设置概率=1
        new_col=new_col.reshape(1,self.new_num_nodes)
        self.new_graph_p = np.r_[self.new_graph_p, new_row]
        self.new_graph_p = np.c_[self.new_graph_c, new_col.T]

        # 构造新的场景矩阵，每个矩阵添加一行[0,0...]和一列【0,1,1】，1对应source节点
        # 和场景列表,
        # 场景概率列表
        self.new_scenes = copy.deepcopy(self.scenes)
        self.new_scenes_list = copy.deepcopy(self.scenes_list)
        self.new_scenes_prob = copy.deepcopy(self.scenes_prob)
        # self.new_scenes_prob = list(np.array(self.scenes_prob) / sum(self.scenes_prob))  # 概率归一化

        new_row = np.zeros(self.num_nodes).reshape(1, self.num_nodes)
        new_col = np.zeros(self.new_num_nodes)
        for s in range(len(self.new_source_nodes)):
            new_col[self.new_source_nodes[s] - 1] = 1  # 设置概率=1
        new_col = new_col.reshape(1, self.new_num_nodes)
        for s in range(len(self.new_scenes)):
            self.new_scenes[s] = np.r_[self.new_scenes[s], new_row]
            self.new_scenes[s] = np.c_[self.new_scenes[s], new_col.T]
        #将场景矩阵转换为场景列表
        self.new_scenes_list=util_tool.scenes_to_scenes_list(self.new_graph_adj,self.new_scenes)

        # (暂时不用这种方法)构造新的路由矩阵
        # self.new_path = copy.deepcopy(self.path)
        # #将新的链路插入路由矩阵中
        # for s in range(len(self.new_source_nodes),0,-1):
        #     insert_index=s*len(self.dest_nodes)*self.K
        #     self.new_path.insert(insert_index,[self.new_num_links + i])
        #     for k in range(self.K - 1):
        #         self.new_path.insert(insert_index+k+1,[])
        self.path_3 = [[6, 3],
                       [6, 5, 9],
                       [7, 13, 16, 9],

                       [7, 13, 18],
                       [7, 14, 20],
                       [6, 5, 11, 18],

                       [7, 14, 21],
                       [7, 13, 18, 23, 21],
                       [6, 5, 11, 17, 14, 21],

                       [8],
                       [],
                       [],

                       [12, 6, 3],
                       [12, 6, 5, 9],
                       [13, 16, 9],

                       [13, 18],
                       [14, 20],
                       [14, 21, 26, 30, 27],

                       [14, 21],
                       [13, 18, 23, 21],
                       [13, 18, 24, 28, 29],

                       [15],
                       [],
                       [],

                       [30, 27, 22, 16, 9],
                       [30, 27, 22, 16, 10, 3],
                       [29, 25, 19, 12, 6, 3],

                       [30, 27],
                       [29, 25, 20],
                       [29, 25, 19, 13, 18],

                       [29],
                       [30, 27, 23, 21],
                       [30, 27, 22, 17, 14, 21],

                       [31],
                       [],
                       []
                       ]
        #使用直接给定path的方法，使用路径3，直接给定路径然后计算

        self.new_path=self.path_3
        flows_num = len(self.new_source_nodes) * len(self.new_dest_nodes) * self.K
        links_num = len(self.new_link_list)
        self.new_route = util_tool.set_route(flows_num, links_num, self.new_path)

        #计算每一条流的cost
        self.compute_flow_cost()

    def compute_flow_cost(self):
        """
        根据路由矩阵和链路cost列表，计算每一条流的cost
        :return:
        """
        self.new_graph_c_list
        self.new_route
        self.new_flow_cost=self.new_route.dot(self.new_graph_c_list)
        #array转为list
        self.new_flow_cost=self.new_flow_cost.tolist()

