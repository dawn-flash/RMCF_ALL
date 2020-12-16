#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/24 15:12
# @Author  : Lelsey
# @File    : offload_strategy.py
"""
实现所有的offloading策略，其中包括MCF，RMCF，Random，MAXMIN，ECMP
"""
import numpy as np
import json
import os
import util_tool
import math
import networkx as nx
import pulp
from pprint import pprint
from data_save import DataSave
import sys
from decimal import Decimal
import time
from itertools import combinations
import copy
import networkx as nx
from sympy import *
import scipy
import cvxpy
from topology import NetworkGraph



def rmcf_lp_loss(graph_adj,graph_c, graph_u, graph_p,graph_demand,source_nodes,dest_nodes,scenes,scenes_list,
                 scenes_prob,A,K,alpha=0.9, C=0.23, capacity_flag=True):
    """
        基于链路流的算法，目标函数=cost+cvar
        复现论文Polynomial-time identification of robust network flows under uncertain arc failures中的例子：figure2，C
        中的RMCF_LP 算法求解问题，不确定下性下带cvar约束的鲁棒线性规划问题，主要是求带风险的资源分配下的最小cost，
        使用python的PULP包进行线性规划问题的求解

        :param graph_adj array  图的邻接矩阵
        :param graph_c:  array 图的cost矩阵
        :param graph_u: array 图的链路容量矩阵
        :param graph_p: array 链路失效概率矩阵
        :param graph_demand: list 图中所有结点的需求
        :param source_nodes: list  发送结点列表 存储发送结点的编号（不是结点下标）
        :param dest_nodes: list 目的结点列表  存储接收结点编号（不是结点下标）
        :param scenes: list[array] 所有场景 每个场景是一个矩阵，所有场景是一个列表
        :param scenes_prob: list  每一个场景的概率
        :param A: ndarray 路由矩阵
        :param K: int 两点之间的路径路径数量
        :param alpha: float 可用性概率
        :param C: float cvar的约束
        :param capacity_flag 是否为链路capacity设置上限
        :return: （解状态，objective（cost），资源分配矩阵,zeta(var)）
        """

    # :paramgraph_u: list（tuple） 图的边表[(0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (2, 1), (2, 3), (2, 4), (3, 4), (3, 5), (4, 3), (4, 5)
    # route=np.array([[0,1,0,0,0,0,0,0,0,0,0,0],
    #                 [1,0,0,1,0,0,0,0,0,0,0,0],
    #                 [0,1,0,0,0,0,0,1,0,0,0,0],
    #                 [0,1,0,0,0,0,0,1,0,0,0,1],
    #                 [0,0,1,0,0,0,0,0,0,0,0,0],
    #                 [0,0,0,1,0,0,0,0,0,0,0,0],
    #                 [0,0,0,0,1,0,0,0,0,0,0,0],
    #                 [0,0,0,1,0,0,0,0,0,1,0,0],
    #                 ])
    route=A
    #将cost矩阵转换为cost列表
    graph_cost_list=util_tool.graph_cost_matrix_to_list(graph_adj,graph_c)
    S=np.array(scenes).shape[0]  #场景数
    source_demand=[graph_demand[i-1] for i in source_nodes] #源节点demand
    dest_demand=[graph_demand[i-1] for i in dest_nodes]  #目的结点demand
    source_num = len(source_nodes)
    dest_num=len(dest_nodes)
    num_nodes = len(graph_demand)  # 图中所有结点的数量


    flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]  # 将矩阵或者多维列表扁平化

    # 定义最小化线性规划问题
    problem = pulp.LpProblem("rmcf_lp", sense=pulp.LpMinimize)
    # 定义线性规划中的变量
    if capacity_flag:  # 设置上限
        flow = [[[pulp.LpVariable(f'f{i}{j}_{k}', lowBound=0) for k in range(K)] for j in range(dest_num)] for i in
                range(source_num)]
    else:  # 不设置上限
        flow = [[[pulp.LpVariable(f'f{i}{j}_{k}', lowBound=0) for k in range(K)] for j in range(dest_num)] for i in
                range(source_num)]
        # flows_matrix = [[pulp.LpVariable(f'fm{s}{d}', lowBound=0) for d in range(Dest_num)] for s in range(Source_num)]
        if source_num==3 and dest_num==4:
            problem+=(flow[0][3][1]==0)
            problem += (flow[0][3][2] == 0)
            problem += (flow[1][3][1] == 0)
            problem += (flow[1][3][2] == 0)
            problem += (flow[2][3][1] == 0)
            problem += (flow[2][3][2] == 0)
        if source_num==4 and dest_num==3:
            problem+=(flow[3][0][1]==0)
            problem+=(flow[3][0][2]==0)
            problem+=(flow[3][1][1]==0)
            problem+=(flow[3][1][2]==0)
            problem+=(flow[3][2][1]==0)
            problem+=(flow[3][2][2]==0)


    zeta = pulp.LpVariable('zeta',lowBound=0)
    t_s = [pulp.LpVariable(f't_{s}', lowBound=0) for s in range(S)]

    # 目标函数
    # problem += pulp.lpDot(np.dot(np.array(flatten(flow)),route).flatten(), graph_cost_list)
    #新的目标函数
    # problem+=pulp.lpDot(np.dot(np.array(flatten(flow)),route).flatten(), graph_cost_list)+(zeta + ((1 / (1 - alpha)) * pulp.lpDot(t_s, scenes_prob)))
    problem+=np.dot(np.dot(np.array(flatten(flow)),route).flatten(), graph_cost_list)+(zeta + ((1 / (1 - alpha)) *np.dot(t_s, scenes_prob)))

    # 约束条件1：结点demand约束
    #对每一行进行约束
    for s in range(source_num):
        problem+=(pulp.lpSum(flow[s])==source_demand[s])
    #对每一列进行约束
    for d in range(dest_num):
        problem += (pulp.lpSum(flatten([[flow[s][d][k] for k in range(K)] for s in range(source_num)])) == -dest_demand[d])

    # 约束条件2,：cvar约束
    # problem += (zeta + ((1 / (S * (1 - alpha))) * pulp.lpSum(t_s)) <= (C * pulp.lpSum(flatten(flow))))   #每个场景的概率都是1/S
    # problem += (zeta + ((1 / ((1 - alpha))) * pulp.lpDot(t_s, scenes_prob)) <= (C * pulp.lpSum(flatten(flow)))) #场景的概率不一样
    # (C*pulp.lpSum(flatten(x_a)))

    # 约束条件3，场景loss约束
    s_count=0
    for s in range(S):
        new_route=copy.deepcopy(route)
        failure_flow=[]
        for i in range(len(scenes_list[s])):
            if scenes_list[s][i]==1:
                for j in range(new_route.shape[0]):
                    if new_route[j][i]==1:
                        # failure_flow.append((j,i))  #可能有错
                        failure_flow.append(j)
        # print("S{0}".format(s)+"failure flow",failure_flow)
        failure_flow=list(set(failure_flow))
        if len(failure_flow)>0:
            # print("场景",scenes_list[s])
            s_count+=1
            problem += (pulp.lpSum([flatten(flow)[i] for i in failure_flow]) - zeta <= t_s[s])
        else:
            # print("删减的场景",scenes_list[s])
            problem+=(0-zeta<=t_s[s])
    print("场景约束的数量",s_count)
        # 更新loss方程
        # problem += (pulp.lpDot(flatten(x_a), (scenes[s] * (1 - graph_p) / np.sum(scenes[s])).flatten()))
    # 线性规划问题求解
    problem.solve()
    print("解的情况", pulp.LpStatus[problem.status])  # 判断

    result = {"objective": pulp.value(problem.objective),
              "flow": [[[pulp.value(flow[i][j][k]) for k in range(K)] for j in range(dest_num)] for i in range(source_num)],  #多对多场景下
              "t_s": [pulp.value(t_s[i]) for i in range(S)],
              "zeta": pulp.value(zeta)}

    # print(problem) #输出所有问题约束
    print("objective", result["objective"])
    print("RMCF: x_a")
    print("输出分配矩阵")
    pprint(result["flow"])
    print("输出具体的分配方案")
    result_flow = result["flow"]
    for i in range(source_num):
        for j in range(dest_num):
            # print("{0}->{1}flow_{2}".format(souce_nodes[i]+1,dest_nodes[i]+1,j+1)) #单对单场景下
            print("{0}->{1}flow".format(source_nodes[i] , dest_nodes[j] ))  # 多对多场景下
            pprint(result_flow[i][j])
    print("流的总量", sum(flatten(result["flow"])))
    print("t_s", result["t_s"])
    print("zeta", result["zeta"])

    #计算任务分配比例
    allot_ratio=np.zeros(shape=(source_num,dest_num))
    for s in range(source_num):
        sum_s=np.sum(result["flow"][s])
        for d in range(dest_num):
            allot_ratio[s,d]=sum(result["flow"][s][d])/sum_s
    print("分配比例是：")
    print(allot_ratio)
    return pulp.LpStatus[problem.status], result["objective"], result["flow"], result["zeta"]

def rmcf_lp_loss_plus(graph_adj,graph_c, graph_u, graph_p,graph_loss_list,graph_demand,source_nodes,dest_nodes,scenes,scenes_list,
                 scenes_prob,flow_cost,A,K,gamma,alpha=0.9, capacity_flag=True):
    """
        基于链路流的算法，目标函数=(1-g)cost+g*cvar
        中的RMCF_LP 算法求解问题，不确定性下带cvar约束的鲁棒线性规划问题，主要是求带风险的资源分配下的最小cost，
        使用python的PULP包进行线性规划问题的求解
        #目标函数=cost+cvar
        cost:表示为网络正常时任务分配的总cost
        cvar：表示为网络出现异常时，如果链路断裂或者链路丢包时，需要重传丢失数据花费的cost
        #网络链路的状态为[0,1,2],即：链路正常，链路失效，链路丢包

        :param graph_adj array  图的邻接矩阵
        :param graph_c:  array 图的cost矩阵
        :param graph_u: array 图的链路容量矩阵
        :param graph_p: array 链路失效概率矩阵
        :param graph_loss_list list 图的丢包率列表 存储每一条链路的丢包率
        :param graph_demand: list 图中所有结点的需求
        :param source_nodes: list  发送结点列表 存储发送结点的编号（不是结点下标）
        :param dest_nodes: list 目的结点列表  存储接收结点编号（不是结点下标）
        :param scenes: list[array] 所有场景 每个场景是一个矩阵，所有场景是一个列表
        :param scenes_prob: list  每一个场景的概率
        :param flow_cost: list  每一条流，单位任务量的cost
        :param A: ndarray 路由矩阵
        :param K: int 两点之间的路径路径数量
        :param gamma cost和cvar之间的比重
        :param alpha: float 可用性概率
        :param C: float cvar的约束
        :param capacity_flag 是否为链路capacity设置上限
        :return: （解状态，objective（cost），资源分配矩阵,zeta(var)）
        """

    # :paramgraph_u: list（tuple） 图的边表[(0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (2, 1), (2, 3), (2, 4), (3, 4), (3, 5), (4, 3), (4, 5)
    # route=np.array([[0,1,0,0,0,0,0,0,0,0,0,0],
    #                 [1,0,0,1,0,0,0,0,0,0,0,0],
    #                 [0,1,0,0,0,0,0,1,0,0,0,0],
    #                 [0,1,0,0,0,0,0,1,0,0,0,1],
    #                 [0,0,1,0,0,0,0,0,0,0,0,0],
    #                 [0,0,0,1,0,0,0,0,0,0,0,0],
    #                 [0,0,0,0,1,0,0,0,0,0,0,0],
    #                 [0,0,0,1,0,0,0,0,0,1,0,0],
    #                 ])
    route=A
    #将cost矩阵转换为cost列表
    graph_cost_list=util_tool.graph_cost_matrix_to_list(graph_adj,graph_c)
    S=np.array(scenes).shape[0]  #场景数
    source_demand=[graph_demand[i-1] for i in source_nodes] #源节点demand
    dest_demand=[graph_demand[i-1] for i in dest_nodes]  #目的结点demand
    source_num = len(source_nodes)
    dest_num=len(dest_nodes)
    num_nodes = len(graph_demand)  # 图中所有结点的数量


    flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]  # 将矩阵或者多维列表扁平化

    # 定义最小化线性规划问题
    problem = pulp.LpProblem("rmcf_lp", sense=pulp.LpMinimize)
    # 定义线性规划中的变量
    if capacity_flag:  # 设置上限
        flow = [[[pulp.LpVariable(f'f{i}{j}_{k}', lowBound=0) for k in range(K)] for j in range(dest_num)] for i in
                range(source_num)]
    else:  # 不设置上限
        flow = [[[pulp.LpVariable(f'f{i}{j}_{k}', lowBound=0) for k in range(K)] for j in range(dest_num)] for i in
                range(source_num)]
        # flows_matrix = [[pulp.LpVariable(f'fm{s}{d}', lowBound=0) for d in range(Dest_num)] for s in range(Source_num)]
        if source_num==3 and dest_num==4:
            problem+=(flow[0][3][1]==0)
            problem += (flow[0][3][2] == 0)
            problem += (flow[1][3][1] == 0)
            problem += (flow[1][3][2] == 0)
            problem += (flow[2][3][1] == 0)
            problem += (flow[2][3][2] == 0)
        if source_num==4 and dest_num==3:
            problem+=(flow[3][0][1]==0)
            problem+=(flow[3][0][2]==0)
            problem+=(flow[3][1][1]==0)
            problem+=(flow[3][1][2]==0)
            problem+=(flow[3][2][1]==0)
            problem+=(flow[3][2][2]==0)


    zeta = pulp.LpVariable('zeta',lowBound=0)
    t_s = [pulp.LpVariable(f't_{s}', lowBound=0) for s in range(S)]
    cost=pulp.LpVariable('cost',lowBound=0)
    cvar=pulp.LpVariable('cvar',lowBound=0)

    # 目标函数
    # problem += pulp.lpDot(np.dot(np.array(flatten(flow)),route).flatten(), graph_cost_list)
    #新的目标函数
    # problem+=pulp.lpDot(np.dot(np.array(flatten(flow)),route).flatten(), graph_cost_list)+(zeta + ((1 / (1 - alpha)) * pulp.lpDot(t_s, scenes_prob)))
    # phi=0.5
    problem+=(1-gamma)*cost+gamma*cvar
    problem+=(cost==np.dot(np.dot(np.array(flatten(flow)),route).flatten(), graph_cost_list))
    problem+=(cvar==(zeta + ((1 / (1 - alpha)) *np.dot(t_s, scenes_prob))))

    # 约束条件1：结点demand约束
    #对每一行进行约束
    for s in range(source_num):
        problem+=(pulp.lpSum(flow[s])==source_demand[s])
    #对每一列进行约束
    for d in range(dest_num):
        problem += (pulp.lpSum(flatten([[flow[s][d][k] for k in range(K)] for s in range(source_num)])) == -dest_demand[d])

    # 约束条件2,：cvar约束
    # problem += (zeta + ((1 / (S * (1 - alpha))) * pulp.lpSum(t_s)) <= (C * pulp.lpSum(flatten(flow))))   #每个场景的概率都是1/S
    # problem += (zeta + ((1 / ((1 - alpha))) * pulp.lpDot(t_s, scenes_prob)) <= (C * pulp.lpSum(flatten(flow)))) #场景的概率不一样
    # (C*pulp.lpSum(flatten(x_a)))

    # 约束条件3，场景loss约束
    s_count=0
    flow_list=flatten(flow)  #flow 矩阵扁平化
    for s in range(S):
        new_route=copy.deepcopy(route)
        all_flow_loss_cost=[]          #存储所有流的loss
        for f in range(new_route.shape[0]):
            a_flow_sus=flow_list[f] #存储一条流成功传输部分
            for l in range(new_route.shape[1]):
                if new_route[f][l]==1:
                    if scenes_list[s][l]==1: #如果链路中存在失效链路，成功传输量为0
                        a_flow_sus=0
                        break
                    else: #如果链路中存在丢包的情况，成功传输量为 总量*（1-丢失百分比）
                        a_flow_sus=a_flow_sus*(1-graph_loss_list[l])

            a_flow_loss=flow_list[f]-a_flow_sus #计算每一条flow的loss
            a_flow_loss_cost=a_flow_loss*flow_cost[f] #计算每一条flow的的cost
            all_flow_loss_cost.append(a_flow_loss_cost)
        problem+=(pulp.lpSum(all_flow_loss_cost)-zeta<=t_s[s])

    print("场景约束的数量",s_count)
    # 更新loss方程
    # problem += (pulp.lpDot(flatten(x_a), (scenes[s] * (1 - graph_p) / np.sum(scenes[s])).flatten()))
    # 线性规划问题求解
    problem.solve()
    print("解的情况", pulp.LpStatus[problem.status])  # 判断

    result = {"objective": pulp.value(problem.objective),
              "flow": [[[pulp.value(flow[i][j][k]) for k in range(K)] for j in range(dest_num)] for i in range(source_num)],  #多对多场景下
              "t_s": [pulp.value(t_s[i]) for i in range(S)],
              "zeta": pulp.value(zeta),
              "cost":pulp.value(cost),
              "cvar":pulp.value(cvar)}

    # print(problem) #输出所有问题约束
    print("objective", result["objective"])
    print("RMCF: x_a")
    print("输出分配矩阵")
    pprint(result["flow"])
    print("输出具体的分配方案")
    result_flow = result["flow"]
    for i in range(source_num):
        for j in range(dest_num):
            # print("{0}->{1}flow_{2}".format(souce_nodes[i]+1,dest_nodes[i]+1,j+1)) #单对单场景下
            print("{0}->{1}flow".format(source_nodes[i] , dest_nodes[j] ))  # 多对多场景下
            pprint(result_flow[i][j])
    print("流的总量", sum(flatten(result["flow"])))
    print("t_s", result["t_s"])
    print("zeta", result["zeta"])
    print("cost",result["cost"])
    print("cvar",result["cvar"])

    #计算任务分配比例
    allot_ratio=np.zeros(shape=(source_num,dest_num))
    for s in range(source_num):
        sum_s=np.sum(result["flow"][s])
        for d in range(dest_num):
            allot_ratio[s,d]=sum(result["flow"][s][d])/sum_s
    print("分配比例是：")
    print(allot_ratio)
    return pulp.LpStatus[problem.status], result["objective"], result["flow"], result["zeta"],result["cost"],result["cvar"]

def rmcf_lp_loss_plus_test(old_allot,graph_adj,graph_c, graph_u, graph_p,graph_loss_list,graph_demand,source_nodes,dest_nodes,scenes,scenes_list,
                 scenes_prob,flow_cost,A,K,gamma,alpha=0.9, capacity_flag=True):
    """
        rmcf_plus算法用于测试的版本
        :param old_allot: array 测试的分配方案
        :param graph_adj array  图的邻接矩阵
        :param graph_c:  array 图的cost矩阵
        :param graph_u: array 图的链路容量矩阵
        :param graph_p: array 链路失效概率矩阵
        :param graph_loss_list list 图的丢包率列表 存储每一条链路的丢包率
        :param graph_demand: list 图中所有结点的需求
        :param source_nodes: list  发送结点列表 存储发送结点的编号（不是结点下标）
        :param dest_nodes: list 目的结点列表  存储接收结点编号（不是结点下标）
        :param scenes: list[array] 所有场景 每个场景是一个矩阵，所有场景是一个列表
        :param scenes_prob: list  每一个场景的概率
        :param flow_cost: list  每一条流，单位任务量的cost
        :param A: ndarray 路由矩阵
        :param K: int 两点之间的路径路径数量
        :param gamma cost和cvar之间的比重
        :param alpha: float 可用性概率
        :param C: float cvar的约束
        :param capacity_flag 是否为链路capacity设置上限
        :return: （解状态，objective（cost），资源分配矩阵,zeta(var)）
        """

    # :paramgraph_u: list（tuple） 图的边表[(0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (2, 1), (2, 3), (2, 4), (3, 4), (3, 5), (4, 3), (4, 5)
    # route=np.array([[0,1,0,0,0,0,0,0,0,0,0,0],
    #                 [1,0,0,1,0,0,0,0,0,0,0,0],
    #                 [0,1,0,0,0,0,0,1,0,0,0,0],
    #                 [0,1,0,0,0,0,0,1,0,0,0,1],
    #                 [0,0,1,0,0,0,0,0,0,0,0,0],
    #                 [0,0,0,1,0,0,0,0,0,0,0,0],
    #                 [0,0,0,0,1,0,0,0,0,0,0,0],
    #                 [0,0,0,1,0,0,0,0,0,1,0,0],
    #                 ])
    route=A
    #将cost矩阵转换为cost列表
    graph_cost_list=util_tool.graph_cost_matrix_to_list(graph_adj,graph_c)
    S=np.array(scenes).shape[0]  #场景数
    source_demand=[graph_demand[i-1] for i in source_nodes] #源节点demand
    dest_demand=[graph_demand[i-1] for i in dest_nodes]  #目的结点demand
    source_num = len(source_nodes)
    dest_num=len(dest_nodes)
    num_nodes = len(graph_demand)  # 图中所有结点的数量


    flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]  # 将矩阵或者多维列表扁平化

    # 定义最小化线性规划问题
    problem = pulp.LpProblem("rmcf_lp", sense=pulp.LpMinimize)
    # 定义线性规划中的变量
    if capacity_flag:  # 设置上限
        flow = [[[pulp.LpVariable(f'f{i}{j}_{k}', lowBound=0) for k in range(K)] for j in range(dest_num)] for i in
                range(source_num)]
    else:  # 不设置上限
        flow = [[[pulp.LpVariable(f'f{i}{j}_{k}', lowBound=0) for k in range(K)] for j in range(dest_num)] for i in
                range(source_num)]
        # flows_matrix = [[pulp.LpVariable(f'fm{s}{d}', lowBound=0) for d in range(Dest_num)] for s in range(Source_num)]
        if source_num==3 and dest_num==4:
            problem+=(flow[0][3][1]==0)
            problem += (flow[0][3][2] == 0)
            problem += (flow[1][3][1] == 0)
            problem += (flow[1][3][2] == 0)
            problem += (flow[2][3][1] == 0)
            problem += (flow[2][3][2] == 0)
        if source_num==4 and dest_num==3:
            problem+=(flow[3][0][1]==0)
            problem+=(flow[3][0][2]==0)
            problem+=(flow[3][1][1]==0)
            problem+=(flow[3][1][2]==0)
            problem+=(flow[3][2][1]==0)
            problem+=(flow[3][2][2]==0)


    zeta = pulp.LpVariable('zeta',lowBound=0)
    t_s = [pulp.LpVariable(f't_{s}', lowBound=0) for s in range(S)]
    cost=pulp.LpVariable('cost',lowBound=0)
    cvar=pulp.LpVariable('cvar',lowBound=0)

    # 目标函数
    # problem += pulp.lpDot(np.dot(np.array(flatten(flow)),route).flatten(), graph_cost_list)
    #新的目标函数
    # problem+=pulp.lpDot(np.dot(np.array(flatten(flow)),route).flatten(), graph_cost_list)+(zeta + ((1 / (1 - alpha)) * pulp.lpDot(t_s, scenes_prob)))
    # phi=0.5
    problem+=(1-gamma)*cost+gamma*cvar
    problem+=(cost==np.dot(np.dot(np.array(flatten(flow)),route).flatten(), graph_cost_list))
    problem+=(cvar==(zeta + ((1 / (1 - alpha)) *np.dot(t_s, scenes_prob))))

    # 定义flow的相应值,赋初值
    problem += (flow[0][0][0] == old_allot[0])
    problem += (flow[0][0][1] == old_allot[1])
    problem += (flow[0][0][2] == old_allot[2])

    # 约束条件1：结点demand约束
    #对每一行进行约束
    for s in range(source_num):
        problem+=(pulp.lpSum(flow[s])==source_demand[s])
    #对每一列进行约束
    for d in range(dest_num):
        problem += (pulp.lpSum(flatten([[flow[s][d][k] for k in range(K)] for s in range(source_num)])) == -dest_demand[d])

    # 约束条件2,：cvar约束
    # problem += (zeta + ((1 / (S * (1 - alpha))) * pulp.lpSum(t_s)) <= (C * pulp.lpSum(flatten(flow))))   #每个场景的概率都是1/S
    # problem += (zeta + ((1 / ((1 - alpha))) * pulp.lpDot(t_s, scenes_prob)) <= (C * pulp.lpSum(flatten(flow)))) #场景的概率不一样
    # (C*pulp.lpSum(flatten(x_a)))

    # 约束条件3，场景loss约束
    s_count=0
    flow_list=flatten(flow)  #flow 矩阵扁平化
    for s in range(S):
        new_route=copy.deepcopy(route)
        all_flow_loss_cost=[]          #存储所有流的loss
        for f in range(new_route.shape[0]):
            a_flow_sus=flow_list[f] #存储一条流成功传输部分
            for l in range(new_route.shape[1]):
                if new_route[f][l]==1:
                    if scenes_list[s][l]==1: #如果链路中存在失效链路，成功传输量为0
                        a_flow_sus=0
                        break
                    else: #如果链路中存在丢包的情况，成功传输量为 总量*（1-丢失百分比）
                        a_flow_sus=a_flow_sus*(1-graph_loss_list[l])

            a_flow_loss=flow_list[f]-a_flow_sus #计算每一条flow的loss
            a_flow_loss_cost=a_flow_loss*flow_cost[f] #计算每一条flow的的cost
            all_flow_loss_cost.append(a_flow_loss_cost)
        problem+=(pulp.lpSum(all_flow_loss_cost)-zeta<=t_s[s])

    print("场景约束的数量",s_count)
    # 更新loss方程
    # problem += (pulp.lpDot(flatten(x_a), (scenes[s] * (1 - graph_p) / np.sum(scenes[s])).flatten()))
    # 线性规划问题求解
    problem.solve()
    print("解的情况", pulp.LpStatus[problem.status])  # 判断

    result = {"objective": pulp.value(problem.objective),
              "flow": [[[pulp.value(flow[i][j][k]) for k in range(K)] for j in range(dest_num)] for i in range(source_num)],  #多对多场景下
              "t_s": [pulp.value(t_s[i]) for i in range(S)],
              "zeta": pulp.value(zeta),
              "cost":pulp.value(cost),
              "cvar":pulp.value(cvar)}

    # print(problem) #输出所有问题约束
    print("objective", result["objective"])
    print("RMCF: x_a")
    print("输出分配矩阵")
    pprint(result["flow"])
    print("输出具体的分配方案")
    result_flow = result["flow"]
    for i in range(source_num):
        for j in range(dest_num):
            # print("{0}->{1}flow_{2}".format(souce_nodes[i]+1,dest_nodes[i]+1,j+1)) #单对单场景下
            print("{0}->{1}flow".format(source_nodes[i] , dest_nodes[j] ))  # 多对多场景下
            pprint(result_flow[i][j])
    print("流的总量", sum(flatten(result["flow"])))
    print("t_s", result["t_s"])
    print("zeta", result["zeta"])
    print("cost",result["cost"])
    print("cvar",result["cvar"])

    #计算任务分配比例
    allot_ratio=np.zeros(shape=(source_num,dest_num))
    for s in range(source_num):
        sum_s=np.sum(result["flow"][s])
        for d in range(dest_num):
            allot_ratio[s,d]=sum(result["flow"][s][d])/sum_s
    print("分配比例是：")
    print(allot_ratio)
    return pulp.LpStatus[problem.status], result["objective"], result["flow"], result["zeta"],result["cost"],result["cvar"]
def mcf_lp_flow(graph_adj,graph_c,graph_u,graph_demand,A,K,source_nodes,dest_nodes,capacity_flag=False):
    num_nodes = len(graph_demand)  # 图中所有结点的数量
    flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]  # 将矩阵或者多维列表扁平化

    route = A
    # 将cost矩阵转换为cost列表
    graph_cost_list = util_tool.graph_cost_matrix_to_list(graph_adj,graph_c)
    source_demand = [graph_demand[i - 1] for i in source_nodes]  # 源节点demand
    dest_demand = [graph_demand[i - 1] for i in dest_nodes]  # 目的结点demand
    source_num = len(source_nodes)
    dest_num = len(dest_nodes)
    # 定义最小化线性规划问题
    problem = pulp.LpProblem("rmcf_lp_flow", sense=pulp.LpMinimize)
    # 定义线性规划中的变量
    if capacity_flag:  # 设置上限
        flow = [[[pulp.LpVariable(f'f{i}{j}_{k}', lowBound=0) for k in range(K)] for j in range(dest_num)] for i in
                range(source_num)]
    else:  # 不设置上限
        flow = [[[pulp.LpVariable(f'f{i}{j}_{k}', lowBound=0) for k in range(K)] for j in range(dest_num)] for i in
                range(source_num)]
        if source_num==3 and dest_num==4:
            problem+=(flow[0][3][1]==0)
            problem += (flow[0][3][2] == 0)
            problem += (flow[1][3][1] == 0)
            problem += (flow[1][3][2] == 0)
            problem += (flow[2][3][1] == 0)
            problem += (flow[2][3][2] == 0)
        if source_num==4 and dest_num==3:
            problem+=(flow[3][0][1]==0)
            problem+=(flow[3][0][2]==0)
            problem+=(flow[3][1][1]==0)
            problem+=(flow[3][1][2]==0)
            problem+=(flow[3][2][1]==0)
            problem+=(flow[3][2][2]==0)

    # 目标函数
    problem += pulp.lpDot(np.dot(np.array(flatten(flow)),route).flatten(), graph_cost_list)

    # 约束条件1：结点demand约束
    for s in range(source_num):
        problem += (pulp.lpSum(flow[s]) == source_demand[s])
    # 对每一列进行约束
    for d in range(dest_num):
        problem += (pulp.lpSum(flatten([[flow[s][d][k] for k in range(K)] for s in range(source_num)])) == -dest_demand[d])


    # 线性规划问题求解
    problem.solve()
    print("解的情况", pulp.LpStatus[problem.status])  # 判断

    result = {"objective": pulp.value(problem.objective),
              "flow": [[[pulp.value(flow[i][j][k]) for k in range(K)] for j in range(dest_num)] for i in range(source_num)],  #多对多场景下
              }

    # print(problem) #输出所有问题约束
    print("objective", result["objective"])
    print("MCF: flow")
    result_flow = result["flow"]
    for i in range(source_num):
        for j in range(dest_num):
            # print("{0}->{1}flow_{2}".format(souce_nodes[i]+1,dest_nodes[i]+1,j+1)) #单对单场景下
            print("{0}->{1}flow".format(source_nodes[i], dest_nodes[j]))  # 多对多场景下
            pprint(result_flow[i][j])
    # 计算任务分配比例
    allot_ratio = np.zeros(shape=(source_num, dest_num))
    for s in range(source_num):
        sum_s = np.sum(result["flow"][s])
        for d in range(dest_num):
            allot_ratio[s, d] = sum(result["flow"][s][d]) / sum_s
    print("分配比例是：")
    print(allot_ratio)
    # pprint(result["x_a"])
    # print("t_s", result["t_s"])
    # print("zeta", result["zeta"])

    return pulp.LpStatus[problem.status], result["objective"], result["flow"]
    pass


def allot_list_rmcf_to_ecmp(allot_list_rmcf,source_nodes,dest_nodes):
    """
    根据rmcf产生的分配方案，将分配方案转换为ECMP算法，即SD总量不变，所有的路径均分资源量
    根据美国骨干网特例所得到的结果，其中3个源节点，3个目的结点
    :param allot_list_rmcf: list rmcf的分配方案
    :param source_nodes:   list  源节点列表
    :param dest_nodes:  list 目的结点列表
    :return: allot_list_ecmp(ecmp的算法分配结果)
    """
    allot_list_ecmp=[]
    if len(source_nodes)==3 and len(dest_nodes)==3:
        for i in range(3):
            row_list=[]
            for j in range(3):
                col_list=[]
                sum_SD=sum(allot_list_rmcf[i][j])
                a_path=sum_SD/3
                for k in range(3):
                    col_list.append(a_path)
                row_list.append(col_list)
            allot_list_ecmp.append(row_list)

    if len(source_nodes)==4 and len(dest_nodes)==3:
        for i in range(4):
            if i!=3:
                row_list=[]
                for j in range(3):
                    col_list=[]
                    sum_SD=sum(allot_list_rmcf[i][j])
                    a_path=sum_SD/3
                    for k in range(3):
                        col_list.append(a_path)
                    row_list.append(col_list)
                allot_list_ecmp.append(row_list)
            else:
                vir_node=copy.deepcopy(allot_list_rmcf[i])
                allot_list_ecmp.append(vir_node)
    if len(source_nodes)==3 and len(dest_nodes)==4:
        for i in range(3):
            row_list=[]
            for j in range(4):
                if j!=3:
                    col_list=[]
                    sum_SD=sum(allot_list_rmcf[i][j])
                    a_path=sum_SD/3
                    for k in range(3):
                        col_list.append(a_path)
                    row_list.append(col_list)
                else:
                    col_list=copy.deepcopy(allot_list_rmcf[i][j])
                    row_list.append(col_list)
            allot_list_ecmp.append(row_list)

    # print("ecmp算法分配")
    # pprint(allot_list_ecmp)
    return allot_list_ecmp


def test_cvxpy():
    """
    测试cvxpy包的使用
    :return:
    """
    #计算(x-y)^2的最小值
    # x=cvxpy.Variable()
    # y=cvxpy.Variable()
    # constraints=[x+y==1,x-y>=1]
    # obj1=cvxpy.Minimize(cvxpy.square(x-y))
    # prob1=cvxpy.Problem(obj1,constraints)
    # prob1.solve()

    # print("非线性规划1")
    # print("status:", prob1.status)
    # print("optimal:", prob1.value)
    # print("optimal var:", x.value, y.value)

    #计算（x)^2的最小值
    print("线性规划1")
    x = cvxpy.Variable(name='x')
    y=cvxpy.Variable(name='y')

    pow_exper=cvxpy.power(x,3)
    obj2=cvxpy.Minimize(pow_exper)
    constraints2=[x<=-2]
    constraints2.append(x>=0)
    prob2=cvxpy.Problem(obj2,constraints2)
    prob2.solve()
    print("status:",prob2.status)
    print("optimal:",prob2.value)
    print("optimal var:",x.value)
    #验证是否符合DCP 凸线性规划条件
    print("cvxpy.Minimize(cvxpy.square(x))", cvxpy.Minimize(cvxpy.square(x)).is_dcp())
    prob3=cvxpy.Problem(cvxpy.Minimize(cvxpy.power(x,3)),[x>=-1])
    print("cvxpy.Minimize(cvxpy.power(x,3))",cvxpy.Minimize(cvxpy.power(x,3)).is_dcp())
    print("prob3",prob3.is_dcp())
    print("cvxpy.Minimize(cvxpy.sqrt(x))",cvxpy.Minimize(cvxpy.sqrt(x)).is_dcp())
    print("cvxpy.Minimize(cvxpy.log(x))",cvxpy.Minimize(cvxpy.log(x)).is_dcp())
    print("cvxpy.Minimize(x*y)",cvxpy.Minimize(x*y).is_dcp())
    print("cvxpy.Minimize(cvxpy.log(x*y))",cvxpy.Minimize(cvxpy.log(x*y)).is_dcp())
    print("cvxpy.Minimize(cvxpy.log(x)",cvxpy.Minimize(cvxpy.log(x)).is_dcp())
    print("cvxpy.Maximize(cvxpy.log(x)",cvxpy.Maximize(cvxpy.log(x)).is_dcp())


    pass


def test_demo(graph_adj,graph_c, graph_u, graph_p,graph_demand,source_nodes,dest_nodes,scenes,scenes_list,
                 scenes_prob,A,K,alpha=0.9, C=0.23, capacity_flag=True):
    route_T = A.T
    route=A
    #使用cvxpy测试一个demo
    # 将cost矩阵转换为cost列表
    graph_cost_list = util_tool.graph_cost_matrix_to_list(graph_adj, graph_c)
    graph_cost_list=np.array(graph_cost_list)
    scenes_prob=np.array(scenes_prob)
    S = np.array(scenes).shape[0]  # 场景数
    source_demand = [graph_demand[i - 1] for i in source_nodes]  # 源节点demand
    dest_demand = [graph_demand[i - 1] for i in dest_nodes]  # 目的结点demand
    num_nodes = len(graph_demand)  # 图中所有结点的数量

    #设置变量
    flow=cvxpy.Variable(2,name='f')
    zeta=cvxpy.Variable(name='zeta')
    t_s=cvxpy.Variable(S,name='t_s')

    variable_constrain=[]
    for i in range(2):
        variable_constrain.append(flow[i]>=0)
    variable_constrain.append(zeta>=0)
    for i in range(S):
        variable_constrain.append(t_s>=0)

    #目标函数
    cost=cvxpy.sum(route_T@flow@graph_cost_list)
    cvar=zeta+(1/(1-alpha))*cvxpy.sum(t_s*scenes_prob)
    obj=cvxpy.Minimize(cvxpy.log(cost*cvar))

    #demand约束
    demand_constrain=[cvxpy.sum(flow)==source_demand[0]]

    #场景loss约束
    loss_constrain=[]
    for s in range(S):
        new_route = copy.deepcopy(route)
        failure_flow = []
        for i in range(len(scenes_list[s])):
            if scenes_list[s][i] == 1:
                for j in range(new_route.shape[0]):
                    if new_route[j][i] == 1:
                        # failure_flow.append((j,i))  #可能有错
                        failure_flow.append(j)
        # print("S{0}".format(s)+"failure flow",failure_flow)
        failure_flow = list(set(failure_flow))
        if len(failure_flow) > 0:
            # print("场景",scenes_list[s])
            loss_constrain.append(cvxpy.sum([flow[i] for i in failure_flow])-zeta<=t_s[s])
        else:
            # print("删减的场景",scenes_list[s])
            loss_constrain.append(0 - zeta <= t_s[s])

    prob=cvxpy.Problem(obj,demand_constrain+loss_constrain+variable_constrain)
    prob.solve()
    print("status:", prob.status)
    print("optimal:", prob.value)
    print("flow:", flow.value)
    print("t_s",t_s.value)
    print("zeta",zeta.value)

    pass


def MaxMin(route,source_node:list,dest_node:list,demand:list,K):
    """
    最大最小流算法
    :param route: ndarray 路由矩阵【path，link】
    :param source_node: 源节点list，存放源节点的编号
    :param dest_node: 目的节点list，存放目的节点编号
    :param demand: 所有节点的需求列表
    :param K: 每个sd之间的最大路径数量
    :return:
    """

    #获取源节点和目的节点的需求
    source_demand=[]
    dest_demand=[]
    for i in range(len(source_node)):
        source_demand.append(demand[source_node[i]-1])
    for i in range(len(dest_node)):
        dest_demand.append(-demand[dest_node[i]-1])

    #设置源节点和目的节点的剩余需求
    real_dest_remain_demand=copy.copy(dest_demand)
    real_source_demand=copy.copy(source_demand)
    allot_S_D=np.zeros(shape=(len(source_node),len(dest_node)))
    for s in range(len(source_node)):
        while (not math.isclose(real_source_demand[s],0,abs_tol=1e-6)):
            print("{0}分配".format(s))
            #进行每轮的计算资源分配
            s_all_demand=real_source_demand[s]
            dest_remain_list=util_tool.remain_dest_node(real_dest_remain_demand)
            s_a_demand=s_all_demand/len(dest_remain_list)
            #对还需要进行分配的剩余节点进行分配
            for d in dest_remain_list:
                if(real_dest_remain_demand[d]>0):
                    if(real_dest_remain_demand[d]>=s_a_demand):
                        real_dest_remain_demand[d]-=s_a_demand
                        real_source_demand[s]-=s_a_demand
                        allot_S_D[s][d]+=s_a_demand
                    else:
                        real_allot=real_dest_remain_demand[d]
                        real_source_demand[s]-=real_allot
                        real_dest_remain_demand[d]=0
                        allot_S_D[s][d]+=real_allot
    #输出s_d的分配方案
    for s in range(len(source_node)):
        print(s,end=" ")
        for d in range(len(dest_node)):
            print(allot_S_D[s][d],end=" ")
        print()

    new_allot_S_D=np.zeros(shape=(len(source_node),len(dest_node),K))
    for s in range(len(source_node)):
        for d in range(len(dest_node)):
            flow=(s*len(dest_node)+d)*K
            #求有效路径数量，和每条流的分配
            valid_path_num=util_tool.valid_path(route[flow:flow+K])
            allor_a_flow=allot_S_D[s][d]/valid_path_num
            for i in range(valid_path_num):#有效流的赋值
                new_allot_S_D[s][d][i]=allor_a_flow

    # 输出s_d的每一条流的分配方案
    for s in range(len(source_node)):
        for d in range(len(dest_node)):
            print("源节点{0},目的节点{1}".format(source_node[s],dest_node[d]), end=" ")
            for i in range(K):
                print(new_allot_S_D[s][d][i],end=" ")
            print()
    return new_allot_S_D

def test_Max_Min():
    route=np.zeros(shape=(12,10))
    for i in range(12):
        route[i][1]=1
    route[2][1]=0
    route[8][1]=0
    route[11][1]=0

    source_node=[1,2]
    dest_node=[3,4]
    demand=[5,10,-5,-10]
    K=3
    MaxMin(route,source_node,dest_node,demand,K)

def test_rmcf_lp_loss_plus():

    graph1,all_graph_demand=create_path_3()
    # all_alpha = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    all_alpha = [round(i * 0.01 + 0.2, 2) for i in range(80)]
    experment_time=1
    gamma=1
    # parameter = {
    #     "fail_num": fail_num,
    #     "alpha": all_alpha,
    #     "sum_sink": sum_sink,
    #     "sum_demand": sum_demand,
    #     "C": C,
    #     "expermnet_time": experment_time
    # }

    # rmcf算法结果统计
    all_exper_obj_result_rmcf = []  # mcf，每一个概率对应的平均目标值
    all_exper_sus_num_rmcf = []  # mcf，每一个概率下实验成功的次数
    all_availability_rmcf = []  # mcf，每一个概率下的概率期望均值
    all_var_rmcf = []
    all_var_rmcf_experment = []
    all_cost_rmcf = []
    all_expect_loss_rmcf = []
    all_allot_rmcf = []
    all_cvar_rmcf = []
    all_pulp_cost=[]
    all_pulp_cvar=[]


    for a_alpha in range(len(all_alpha)):
        a_sus_obj_rmcf = []
        a_failure_obj_rmcf = []
        a_availability_rmcf = []
        a_var_rmcf = []
        a_var_rmcf_experment = []
        a_expect_loss_rmcf = []
        a_allot_rmcf = []
        a_cvar_rmcf = []
        a_pulp_cost=[]
        a_pulp_cvar=[]


        for e in range(experment_time):
            # 设置demand
            graph1.set_demand(all_graph_demand[e])
            # graph1.set_demand([-10, 0, 10, 0, 10, 0, 0, -10, -10, 0, 10])
            # 判断是否需要增加结点
            graph1.change_topo()

            # rmcf_plus算法
            result_rmcf, obj_rmcf, allot_list_rmcf, zeta,pulp_cost,pulp_cvar = rmcf_lp_loss_plus(graph1.new_graph_adj,
                                                                         graph1.new_graph_c,
                                                                         graph1.new_graph_u,
                                                                         graph1.new_graph_p,
                                                                         graph1.graph_loss_list,
                                                                         graph1.new_graph_demand,
                                                                         graph1.new_source_nodes,
                                                                         graph1.new_dest_nodes,
                                                                         graph1.new_scenes,
                                                                         graph1.new_scenes_list,
                                                                         graph1.new_scenes_prob,
                                                                         graph1.new_flow_cost,
                                                                         graph1.new_route, graph1.new_K,
                                                                         gamma,
                                                                         all_alpha[a_alpha],
                                                                         capacity_flag=False)



            print("rmcf------------------start")
            # 计算可用性期望rmcf：新的计算方式
            temp_availability_rmcf, temp_var_rmcf, temp_cvar_rmcf = util_tool.new_compute_scenes_availability(
                allot_list_rmcf,
                graph1.new_route,
                graph1.new_scenes_list,
                graph1.new_scenes_prob,
                graph1.new_flow_cost,
                graph1.new_graph_loss_list,
                zeta, all_alpha[a_alpha])

            print("gamma", gamma)
            print("可用性",all_alpha[a_alpha])
            print("分配方式", allot_list_rmcf)
            print("优化的目标值",obj_rmcf)
            print("实验的_cost", pulp_cost)
            print("实验的var",zeta)
            print("计算的var", temp_var_rmcf)
            print("实验的_cvar",pulp_cvar)
            print("计算的cvar",temp_cvar_rmcf)

            # print("rmcf期望", temp_expect_loss_rmcf)
            a_availability_rmcf.append(temp_availability_rmcf)
            a_var_rmcf.append(temp_var_rmcf)
            a_var_rmcf_experment.append(zeta)
            a_sus_obj_rmcf.append(obj_rmcf)
            # a_expect_loss_rmcf.append(temp_expect_loss_rmcf)
            a_cvar_rmcf.append(temp_cvar_rmcf)
            a_pulp_cost.append(pulp_cost)
            a_pulp_cvar.append(pulp_cvar)

            a_allot_rmcf.append(allot_list_rmcf)
            print("rmcf------------------------end")

        # 计算平均目标值-rmcf
        # mean_obj_rmcf = np.mean(a_sus_obj_rmcf)
        all_exper_obj_result_rmcf.append(a_sus_obj_rmcf)
        all_exper_sus_num_rmcf.append(len(a_sus_obj_rmcf))
        # 计算pf概率下的可用性-rmcf
        all_availability_rmcf.append(a_availability_rmcf)
        all_var_rmcf.append(a_var_rmcf)
        all_var_rmcf_experment.append(a_var_rmcf_experment)
        # all_cost_rmcf=[all_var_rmcf[i]/sum_demand for i in range(len(all_var_rmcf))]
        all_allot_rmcf.append(a_allot_rmcf)
        all_cvar_rmcf.append(a_cvar_rmcf)
        all_pulp_cost.append(a_pulp_cost)
        all_pulp_cvar.append(a_pulp_cvar)


        print("所有算法的var")
        print("rmcf", a_var_rmcf)

    mean_obj_rmcf = [np.mean(all_exper_obj_result_rmcf[i]) for i in range(len(all_exper_obj_result_rmcf))]
    mean_availability_rmcf = [np.mean(all_availability_rmcf[i]) for i in range(len(all_availability_rmcf))]
    mean_var_rmcf = [np.mean(all_var_rmcf[i]) for i in range(len(all_var_rmcf))]
    mean_var_rmcf_experment = [np.mean(all_var_rmcf_experment[i]) for i in range(len(all_var_rmcf_experment))]
    # mean_var_demand_rmcf = [mean_var_rmcf[i] / sum_demand for i in range(len(mean_var_rmcf))]
    mean_expect_loss_rmcf = [np.mean(all_expect_loss_rmcf[i]) for i in range(len(all_expect_loss_rmcf))]

    print("链路概率同构场景")
    print("链路的可用性", list(map(lambda x: str(int(x * 100)) + '%', all_alpha)))
    print("RMCF的所有结果:")
    print("所有分配方案：")
    for i in range(len(all_alpha)):
        print("可用性{0},分配方案{1}".format(all_alpha[i],all_allot_rmcf[i]))
    print("所有试验目标值结果")
    pprint(mean_obj_rmcf)
    print("gamma",gamma)
    print("所有概率的真实var", mean_var_rmcf)
    print("所有概率的实验var", mean_var_rmcf_experment)
    print("所有pulp中的cost",all_pulp_cost)
    print("所有pulp中的cvar",all_pulp_cvar)
    print("所有的alpha的cvar")
    pprint(all_cvar_rmcf)
    print("每个试验成功的次数", all_exper_sus_num_rmcf)
    print("所有的概率可用性", mean_availability_rmcf)

def create_path_2():
    """
    测试两条路径的情况
    :return: graph ,all_graph_demand  图类和所有实验的demand
    """
    # 图邻接矩阵
    ad_matrix = np.zeros(shape=(4, 4))
    link = [(1, 2), (1, 3),
            (2, 1), (2, 4),
            (3, 1), (3, 4),
            (4, 2), (4, 3)
            ]
    for i in range(len(link)):
        ad_matrix[link[i][0] - 1][link[i][1] - 1] = 1
    source_nodes_1 = [1]
    dest_nodes_1 = [4]
    num_nodes = 4
    K = 2  # 每一个sd流中路径的数量
    fail_num = 4
    # all_alpha = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    path_1 = [[1, 4],
              [2, 6]
              ]
    experment_time = 1
    sum_sink = 20
    sum_demand = 20
    all_graph_demand = util_tool.random_graph_demand(experment_time, num_nodes, sum_demand, sum_sink,
                                                     source_nodes_1,
                                                     dest_nodes_1)
    print(all_graph_demand)
    # 初始化拓扑
    graph1 = NetworkGraph(ad_matrix, link)
    # 设置概率矩阵和丢包率
    # fail_list = [0.1]*8
    # fail_list = [0.3, 0.1, 0.1, 0.3, 0.1, 0.1, 0.1, 0.1]#l1和l4增加失效概率
    fail_list = [0.1, 0.2, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1]  # l2和l6增加失效概率
    loss_list = [0] * 8
    graph1.set_graph_p_isomerism(fail_list)
    graph1.set_graph_loss_isomerism(loss_list)

    # 设置源节点和目的结点
    graph1.set_source_dest_nodes(source_nodes_1, dest_nodes_1)
    # 设置路由
    graph1.set_route(K, path_1)
    # 设置链路cost
    # cost_list = [1, 1, 1, 1, 1, 1, 1, 1]
    cost_list = [1, 1, 1, 1, 1, 1, 1, 1]  # l1和l4cost比较高
    graph1.set_graph_c_isomerism(cost_list)

    # 生成所有场景
    graph1.generate_scenes(fail_num)
    sum_scenes_prob = sum(graph1.scenes_prob)

    return graph1,all_graph_demand

def create_path_3():
    """
        测试三条路径的情况：SD流之间有三条路径
        :return: graph ,all_graph_demand  图类和所有实验的demand
        """
    # 图邻接矩阵
    ad_matrix = np.zeros(shape=(5, 5))
    link = [(1, 2), (1, 3),(1,4),
            (2, 1), (2, 5),
            (3, 1), (3, 5),
            (4, 1), (4, 5),
            (5,2),(5,3),(5,4)
            ]
    for i in range(len(link)):
        ad_matrix[link[i][0] - 1][link[i][1] - 1] = 1
    source_nodes_1 = [1]
    dest_nodes_1 = [5]
    num_nodes = 5
    K = 3  # 每一个sd流中路径的数量
    fail_num = 4
    # all_alpha = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    path_1 = [[1, 5],
              [2, 7],
              [3, 9]
              ]
    experment_time = 1
    sum_sink = 30
    sum_demand = 30
    all_graph_demand = util_tool.random_graph_demand(experment_time, num_nodes, sum_demand, sum_sink,
                                                     source_nodes_1,
                                                     dest_nodes_1)
    print(all_graph_demand)
    # 初始化拓扑
    graph1 = NetworkGraph(ad_matrix, link)
    # 设置概率矩阵,路径l1(1,5),l2(2,7),l3(3,9)
    fail_list=[0.1]*12
    fail_list[0]=0.3
    fail_list[4]=0.3
    fail_list[1]=0.2
    fail_list[6]=0.2

    # fail_list = [0.1, 0.2, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1]  # l2和l6增加失效概率
    #设置丢包率
    loss_list = [0] * 12
    graph1.set_graph_p_isomerism(fail_list)
    graph1.set_graph_loss_isomerism(loss_list)

    # 设置源节点和目的结点
    graph1.set_source_dest_nodes(source_nodes_1, dest_nodes_1)
    # 设置路由
    graph1.set_route(K, path_1)
    # 设置链路cost

    cost_list = [1]*12 #
    graph1.set_graph_c_isomerism(cost_list)

    # 生成所有场景
    graph1.generate_scenes(fail_num)
    sum_scenes_prob = sum(graph1.scenes_prob)

    return graph1, all_graph_demand

def test_allot_strategy():
    graph1,all_graph_demand=create_path_3()
    # 设置demand
    graph1.set_demand(all_graph_demand[0])
    a_alpha=0.4
    gamma=0.5
    # graph1.set_demand([-10, 0, 10, 0, 10, 0, 0, -10, -10, 0, 10])
    # 判断是否需要增加结点
    graph1.change_topo()
    all_old_allot=[[30,0,0],
                   [20,10,0],
                   [20,5,5],
                   [10,10,10],
                   [0, 20, 10],
                   [0,15,15],
                   [0,30,0],
                   [0,0,30]]
    all_result=[]
    all_alpha = [round(i * 0.01 + 0.2, 2) for i in range(80)]
    # all_alpha=[0.2]
    for a_alpha in all_alpha:
        a_result=[]
        for old_allot in all_old_allot:
            print("*************************************")
            print("可用性",a_alpha,"  old_allot",old_allot)
            # rmcf_plus算法
            result_rmcf, obj_rmcf, allot_list_rmcf, zeta, pulp_cost, pulp_cvar = rmcf_lp_loss_plus_test(old_allot,
                                                                                                        graph1.new_graph_adj,
                                                                                                   graph1.new_graph_c,
                                                                                                   graph1.new_graph_u,
                                                                                                   graph1.new_graph_p,
                                                                                                   graph1.graph_loss_list,
                                                                                                   graph1.new_graph_demand,
                                                                                                   graph1.new_source_nodes,
                                                                                                   graph1.new_dest_nodes,
                                                                                                   graph1.new_scenes,
                                                                                                   graph1.new_scenes_list,
                                                                                                   graph1.new_scenes_prob,
                                                                                                   graph1.new_flow_cost,
                                                                                                   graph1.new_route,
                                                                                                   graph1.new_K,
                                                                                                   gamma,
                                                                                                   a_alpha,
                                                                                                   capacity_flag=False)
            a_result.append([zeta,pulp_cvar])
        all_result.append(a_result)
    print("可用性",a_alpha)
    print("分配方案:",all_old_allot)
    for i in range(len(all_alpha)):
        print("可用性",all_alpha[i],end=" ")
        for j in range(len(all_old_allot)):
            print(all_result[i][j],end=" ")
        print()

    #写入表格
    output = open('path_321cost_111.xls', 'w', encoding='gbk')
    output.write('分配方案\t'+str([30, 0, 0])+'\t\t'+str([20, 10, 0])+'\t\t'+
                 str([20, 5, 5])+'\t\t'+str([10, 10, 10])+'\t\t'+
                 str([0, 20, 10])+'\t\t'+str([0, 15, 15])+'\t\t'+
                 str([0, 30, 0])+'\t\t'+str([0, 0, 30])+'\n')
    output.write('可用性\t'+'var\tcvar\t'*7+'var\tcvar\n')
    for i in range(len(all_result)):
        output.write(str(all_alpha[i]))
        output.write('\t')
        for j in range(len(all_result[i])):
            output.write(str(all_result[i][j][0]))  # write函数不能写int类型的参数，所以使用str()转化
            output.write('\t')  # 相当于Tab一下，换一个单元格
            output.write(str(all_result[i][j][1]))  # write函数不能写int类型的参数，所以使用str()转化
            output.write('\t')  # 相当于Tab一下，换一个单元格
        output.write('\n')  # 写完一行立马换行
    output.close()
    print("写入文件结束")

if __name__ == '__main__':
    # test_cvxpy()
    # test_Max_Min()
    # test_rmcf_lp_loss_plus()
    test_allot_strategy()






