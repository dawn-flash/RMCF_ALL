#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/9/24 15:14
# @Author  : Lelsey
# @File    : experiment.py

#实验类，进行所有的实验
import offload_strategy
from topology import NetworkGraph
import numpy as np
import util_tool
from pprint import pprint
import os
import sys
import time
from data_save import DataSave
import json


def experiment_prob_availability():
    """
    测试链路失效概率，对不同的算法影响
    :return:
    """
    # 图邻接矩阵
    ad_matrix = np.zeros(shape=(11, 11))
    link = [(1, 2), (2, 1),
            (1, 4), (4, 1),
            (2, 3), (3, 2),
            (2, 4), (4, 2),
            (3, 5), (5, 3),
            (4, 6), (6, 4),
            (5, 6), (6, 5),
            (5, 7), (7, 5),
            (6, 8), (8, 6),
            (7, 8), (8, 7),
            (7, 9), (9, 7),
            (8, 10), (10, 8),
            (9, 11), (11, 9),
            (10, 11), (11, 10)]
    for i in range(len(link)):
        ad_matrix[link[i][0] - 1][link[i][1] - 1] = 1
    source_nodes_1 = [3, 5, 11]
    dest_nodes_1 = [1, 8, 9]
    num_nodes=11
    K = 3  # 每一个sd流中路径的数量
    fail_num = 2
    alpha = 0.9
    C = 0.3
    path_1 = [[6, 3],
              [6, 5, 8],
              [7, 12, 14, 8],

              [7, 12, 16],
              [7, 13, 18],
              [6, 5, 10, 16],

              [7, 13, 19],
              [7, 12, 16, 21, 19],
              [6, 5, 10, 15, 13, 19],

              [11, 6, 3],
              [11, 6, 5, 8],
              [12, 14, 8],

              [12, 16],
              [13, 16],
              [13, 19, 24, 28, 25],

              [13, 19],
              [12, 16, 21, 19],
              [12, 16, 22, 26, 27],

              [28, 25, 20, 14, 8],
              [28, 25, 20, 14, 9, 3],
              [27, 23, 17, 11, 6, 3],

              [28, 25],
              [27, 23, 18],
              [27, 23, 17, 12, 16],

              [27],
              [28, 25, 21, 19],
              [28, 25, 20, 15, 13, 19]
              ]

    experment_time =20
    # experment_time=1
    sum_sink = 30
    sum_demand = 30

    probability_edge=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10]
    # probability_edge=[0.01]
    # probability_edge=[0.01,0.03,0.05,0.07,0.09,0.11]
    # probability_edge=[0.05]

    parameter={
        "fail_num":fail_num,
        "alpha":alpha,
        "sum_sink":sum_sink,
        "sum_demand":sum_demand,
        "C":C,
        "probability_edge":probability_edge,
        "expermnet_time":experment_time
    }
    #mcf算法，结果统计
    all_exper_obj_result_mcf = []  # mcf，每一个概率对应的平均目标值
    all_exper_sus_num_mcf = []  # mcf，每一个概率下实验成功的次数
    all_availability_mcf = []  # mcf，每一个概率下的概率期望均值
    all_var_mcf=[]
    all_cost_mcf=[]
    all_expect_loss_mcf=[]

    #rmcf算法结果统计
    all_exper_obj_result_rmcf = []  # mcf，每一个概率对应的平均目标值
    all_exper_sus_num_rmcf = []  # mcf，每一个概率下实验成功的次数
    all_availability_rmcf = []  # mcf，每一个概率下的概率期望均值
    all_var_rmcf=[]
    all_cost_rmcf=[]
    all_expect_loss_rmcf=[]

    #ecmp算法结果统计
    all_exper_obj_result_ecmp=[]
    all_availability_ecmp = []
    all_var_ecmp = []
    all_result=[]

    # 获取需求矩阵
    all_graph_demand = util_tool.random_graph_demand(experment_time, num_nodes, sum_demand, sum_sink,
                                                     source_nodes_1,
                                                     dest_nodes_1)
    print("输出所有的需求种类")
    pprint(all_graph_demand)
    for p in range(len(probability_edge)):
        print("链路失效概率", probability_edge[p])
        a_sus_obj_mcf = []
        a_failure_obj_mcf = []
        a_availability_mcf = []
        a_var_mcf=[]
        a_expect_loss_mcf=[]

        a_sus_obj_rmcf = []
        a_failure_obj_rmcf = []
        a_availability_rmcf = []
        a_var_rmcf=[]
        a_expect_loss_rmcf=[]

        a_sus_obj_ecmp = []
        a_availability_ecmp = []
        a_var_ecmp = []

        # 初始化拓扑
        graph1 = NetworkGraph(ad_matrix, link)
        # 设置概率矩阵
        graph1.set_graph_p(probability_edge[p])
        print(all_graph_demand)
        # 设置源节点和目的结点
        graph1.set_source_dest_nodes(source_nodes_1, dest_nodes_1)
        # 设置路由
        graph1.set_route(K, path_1)
        # 生成所有场景
        graph1.generate_scenes(fail_num)
        sum_scenes_prob=sum(graph1.scenes_prob)

        for e in range(experment_time):
            # 设置demand
            graph1.set_demand(all_graph_demand[e])
            # graph1.set_demand([-10, 0, 10, 0, 10, 0, 0, -10, -10, 0, 10])
            # 判断是否需要增加结点
            graph1.change_topo(sum_demand, sum_sink)
            # rmcf算法
            result_rmcf, obj_rmcf, allot_list_rmcf, zeta = offload_strategy.rmcf_lp_loss(graph1.new_graph_adj,
                                                                                         graph1.new_graph_c,
                                                                                         graph1.new_graph_u,
                                                                                         graph1.new_graph_p,
                                                                                         graph1.new_graph_demand,
                                                                                         graph1.new_source_nodes,
                                                                                         graph1.new_dest_nodes,
                                                                                         graph1.new_scenes,
                                                                                         graph1.new_scenes_list,
                                                                                         graph1.new_scenes_prob,
                                                                                         graph1.new_route, graph1.new_K,
                                                                                         alpha, C,
                                                                                         capacity_flag=False)
            #mcf算法
            result_mcf, obj_mcf, allot_list_mcf = offload_strategy.mcf_lp_flow(graph1.new_graph_adj, graph1.new_graph_c,
                                                                               graph1.new_graph_u,
                                                                               graph1.new_graph_demand,
                                                                               graph1.new_route, graph1.new_K,
                                                                               graph1.new_source_nodes,
                                                                               graph1.new_dest_nodes,
                                                                               capacity_flag=False)

            # ECMP算法
            allot_list_ecmp = util_tool.allot_list_rmcf_to_ecmp(allot_list_rmcf, graph1.new_source_nodes,
                                                                graph1.new_dest_nodes)
            if result_rmcf == "Optimal":
                # 计算可用性期望rmcf
                temp_availability_rmcf,temp_var_rmcf= util_tool.compute_scenes_availability(allot_list_rmcf, graph1.new_route,
                                                                               graph1.new_scenes_list,
                                                                               graph1.new_scenes_prob,
                                                                               zeta,alpha)
                temp_expect_loss_rmcf=util_tool.expect_loss(allot_list_rmcf,graph1.new_route,graph1.new_scenes_list,graph1.new_scenes_prob)
                # temp_availvbility_rmcf+=1-sum_scenes_prob
                a_availability_rmcf.append(temp_availability_rmcf)
                a_var_rmcf.append(temp_var_rmcf)
                a_sus_obj_rmcf.append(obj_rmcf)
                a_expect_loss_rmcf.append(temp_expect_loss_rmcf)

                # 计算ecmp算法的可用性和cost
                temp_availability_ecmp, temp_var_ecmp = util_tool.compute_scenes_availability(allot_list_ecmp,
                                                                                              graph1.new_route,
                                                                                              graph1.new_scenes_list,
                                                                                              graph1.new_scenes_prob,
                                                                                              zeta, alpha)

                temp_cost_ecmp=util_tool.compute_cost_ecmp(allot_list_ecmp,graph1.new_path,graph1.new_graph_adj,
                                                           graph1.new_graph_c)
                a_availability_ecmp.append(temp_availability_ecmp)
                a_var_ecmp.append(temp_var_ecmp)
                a_sus_obj_ecmp.append(temp_cost_ecmp)

                #计算mcf的可用性，在rmcf成功有解时，才计算mcf
                if result_mcf == "Optimal":
                    a_sus_obj_mcf.append(obj_mcf)
                    # 计算可用性期望 mcf
                    temp_availability_mcf, temp_var_mcf = util_tool.compute_scenes_availability(allot_list_mcf,
                                                                                                graph1.new_route,
                                                                                                graph1.new_scenes_list,
                                                                                                graph1.new_scenes_prob,
                                                                                                zeta,alpha)
                    temp_expect_loss_mcf=util_tool.expect_loss(allot_list_mcf,graph1.new_route,graph1.new_scenes_list,graph1.new_scenes_prob)
                    # temp_availvbility_mcf+=1-sum_scenes_prob
                    a_availability_mcf.append(temp_availability_mcf)
                    a_var_mcf.append(temp_var_mcf)
                    a_expect_loss_mcf.append(temp_expect_loss_mcf)
                else:
                    a_failure_obj_mcf.append(obj_mcf)

            else:
                a_failure_obj_rmcf.append(obj_rmcf)
            #直接计算mcf的相关值
            # if result_mcf == "Optimal":
            #     a_sus_obj_mcf.append(obj_mcf)
            #     # 计算可用性期望 mcf
            #     temp_availvbility_mcf,temp_var_mcf = util_tool.compute_scenes_availability(allot_list_mcf, graph1.new_route,
            #                                                                   graph1.new_scenes_list,
            #                                                                   graph1.new_scenes_prob,
            #                                                                   zeta)
            #     # temp_availvbility_mcf+=1-sum_scenes_prob
            #     a_availability_mcf.append(temp_availvbility_mcf)
            #     a_var_mcf.append(temp_var_mcf)
            # else:
            #     a_failure_obj_mcf.append(obj_mcf)
        all_result.append(a_sus_obj_mcf)
        # 计算平均目标值-rmcf
        # mean_obj_rmcf = np.mean(a_sus_obj_rmcf)
        all_exper_obj_result_rmcf.append(a_sus_obj_rmcf)
        all_exper_sus_num_rmcf.append(len(a_sus_obj_rmcf))
        # 计算pf概率下的可用性-rmcf
        all_availability_rmcf.append(a_availability_rmcf)
        all_var_rmcf.append(a_var_rmcf)
        # all_cost_rmcf=[all_var_rmcf[i]/sum_demand for i in range(len(all_var_rmcf))]
        all_expect_loss_rmcf.append(a_expect_loss_rmcf)

        # 计算平均目标值-mcf
        # mean_obj_mcf = np.mean(a_sus_obj_mcf)
        all_exper_obj_result_mcf.append(a_sus_obj_mcf)
        all_exper_sus_num_mcf.append(len(a_sus_obj_mcf))
        # 计算pf概率下的可用性-mcf
        all_availability_mcf.append(a_availability_mcf)
        all_var_mcf.append(a_var_mcf)
        # all_cost_mcf=[all_var_mcf[i]/sum_demand for i in range(len(all_var_mcf))]
        all_expect_loss_mcf.append(a_expect_loss_mcf)

        # 计算ecmp算法的可用性
        all_exper_obj_result_ecmp.append(a_sus_obj_ecmp)
        all_availability_ecmp.append(a_availability_ecmp)
        all_var_ecmp.append(a_var_ecmp)

        print("所有算法的var")
        print("rmcf", a_var_rmcf)
        print("mcf", a_var_mcf)
        print("ecmp", a_var_ecmp)


    mean_obj_rmcf=[np.mean(all_exper_obj_result_rmcf[i]) for i in range(len(all_exper_obj_result_rmcf))]
    mean_availability_rmcf=[np.mean(all_availability_rmcf[i]) for i in range(len(all_availability_rmcf))]
    mean_var_rmcf=[np.mean(all_var_rmcf[i]) for i in range(len(all_var_rmcf))]
    mean_var_demand_rmcf=[mean_var_rmcf[i]/sum_demand for i in range(len(mean_var_rmcf))]
    mean_expect_loss_rmcf=[np.mean(all_expect_loss_rmcf[i]) for i in range(len(all_expect_loss_rmcf))]

    print("链路概率同构场景")
    print("链路的失效概率", probability_edge)
    print("RMCF的所有结果:")
    print("所有试验目标值结果",mean_obj_rmcf )
    print("每个试验成功的次数", all_exper_sus_num_rmcf)
    print("所有的概率可用性",mean_availability_rmcf )
    print("所有概率的var",mean_var_rmcf)
    print("所有的var/demand",mean_var_demand_rmcf)
    print("所有概率的loss期望",mean_expect_loss_rmcf)

    mean_obj_mcf = [np.mean(all_exper_obj_result_mcf[i]) for i in range(len(all_exper_obj_result_mcf))]
    mean_availability_mcf = [np.mean(all_availability_mcf[i]) for i in range(len(all_availability_mcf))]
    mean_var_mcf = [np.mean(all_var_mcf[i]) for i in range(len(all_var_mcf))]
    mean_var_demand_mcf = [mean_var_mcf[i] / sum_demand for i in range(len(mean_var_mcf))]
    mean_expect_loss_mcf = [np.mean(all_expect_loss_mcf[i]) for i in range(len(all_expect_loss_mcf))]
    print("MCF的所有结果:")
    print("所有试验目标值结果", mean_obj_mcf)
    print("每个试验成功的次数", all_exper_sus_num_mcf)
    print("所有的概率可用性", mean_availability_mcf)
    print("所有概率的var",mean_var_mcf)
    print("所有的var/deamnd",mean_var_demand_mcf)
    print("所有概率的loss期望",mean_expect_loss_mcf)

    #保存所有ecmp算法的数据
    mean_obj_ecmp = [np.mean(all_exper_obj_result_ecmp[i]) for i in range(len(all_exper_obj_result_ecmp))]
    mean_availability_ecmp = [np.mean(all_availability_ecmp[i]) for i in range(len(all_availability_ecmp))]
    mean_var_ecmp = [np.mean(all_var_ecmp[i]) for i in range(len(all_var_ecmp))]
    mean_var_demand_ecmp=[mean_var_ecmp[i] / sum_demand for i in range(len(mean_var_ecmp))]
    print("ECMP")
    print("所有实验的目标值",mean_obj_ecmp)
    print("所有需求的可用性", mean_availability_ecmp)
    print("所有需求的var", mean_var_ecmp)
    print("所有的var/demand",mean_var_demand_ecmp)

    print(np.array(all_result))
    data_dict1 = {
        "parameter":parameter,
        "probability_edge": probability_edge,
         "all_availvbility_rmcf": all_availability_rmcf,
         "all_availability_mcf": all_availability_mcf,
         "all_availability_ecmp": all_availability_ecmp,
         "all_cost_rmcf": all_exper_obj_result_rmcf,
         "all_cost_mcf": all_exper_obj_result_mcf,
        "all_cost_ecmp":all_exper_obj_result_ecmp,
        "mean_var_demand_rmcf":mean_var_demand_rmcf,
        "mean_var_demand_mcf":mean_var_demand_mcf,
        "mean_var_demand_ecmp":mean_var_demand_ecmp
             }
    # 保存文件
    ds = DataSave(sys._getframe().f_code.co_name)
    ds.sava_all_data(data_dict1)
    print("运行完毕")

def experiment_demand_availability():
    """
    测试需求规模，对不同的算法的影响
    :return:
    """
    # 图邻接矩阵
    ad_matrix = np.zeros(shape=(11, 11))
    link = [(1, 2), (2, 1),
            (1, 4), (4, 1),
            (2, 3), (3, 2),
            (2, 4), (4, 2),
            (3, 5), (5, 3),
            (4, 6), (6, 4),
            (5, 6), (6, 5),
            (5, 7), (7, 5),
            (6, 8), (8, 6),
            (7, 8), (8, 7),
            (7, 9), (9, 7),
            (8, 10), (10, 8),
            (9, 11), (11, 9),
            (10, 11), (11, 10)]
    for i in range(len(link)):
        ad_matrix[link[i][0] - 1][link[i][1] - 1] = 1
    source_nodes_1 = [3, 5, 11]
    dest_nodes_1 = [1, 8, 9]
    num_nodes = 11
    K = 3  # 每一个sd流中路径的数量
    fail_num = 2
    alpha = 0.9
    C = 0.4
    path_1 = [[6, 3],
              [6, 5, 8],
              [7, 12, 14, 8],

              [7, 12, 16],
              [7, 13, 18],
              [6, 5, 10, 16],

              [7, 13, 19],
              [7, 12, 16, 21, 19],
              [6, 5, 10, 15, 13, 19],

              [11, 6, 3],
              [11, 6, 5, 8],
              [12, 14, 8],

              [12, 16],
              [13, 16],
              [13, 19, 24, 28, 25],

              [13, 19],
              [12, 16, 21, 19],
              [12, 16, 22, 26, 27],

              [28, 25, 20, 14, 8],
              [28, 25, 20, 14, 9, 3],
              [27, 23, 17, 11, 6, 3],

              [28, 25],
              [27, 23, 18],
              [27, 23, 17, 12, 16],

              [27],
              [28, 25, 21, 19],
              [28, 25, 20, 15, 13, 19]
              ]

    experment_time = 60
    sum_sink = 30
    sum_demand = 30

    # demand_ratio= [0.4,0.5, 0.6,0.7,0.8,0.9,1.0,1.1, 1.2,1.3]
    # C_ratio=[0.2,0.22,0.24,0.26,0.28,0.30,0.32,0.34,0.36,0.38]
    C_list=[0.1,0.12,0.14,0.16,0.2,0.28,0.28,0.4,0.43,0.45]
    demand_ratio= [0.4,0.5, 0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3]
    # C_list=[0.1]
    # demand_ratio=[0.4]
    # C_list=[0.4]
    # demand_ratio=[1.1]

    real_sum_demand=[int(demand_ratio[i]*sum_demand) for i in range(len(demand_ratio))]
    probability_edge=[0.03]
    parameter={
        "fail_num":fail_num,
        "alpha":alpha,
        "sum_sink":sum_sink,
        "sum_dmeand":sum_demand,
        "C_list":C_list,
        "demand_ratio":demand_ratio,
        "probability_edge":probability_edge,
        "experment_time":experment_time
    }
    all_exper_obj_result_mcf = []  # mcf，每一个概率对应的平均目标值
    all_exper_sus_num_mcf = []  # mcf，每一个概率下实验成功的次数
    all_availability_mcf = []  # mcf，每一个概率下的概率期望均值
    all_var_mcf = []
    all_expect_loss_mcf = []

    all_exper_obj_result_rmcf = []  # mcf，每一个概率对应的平均目标值
    all_exper_sus_num_rmcf = []  # mcf，每一个概率下实验成功的次数
    all_availability_rmcf = []  # mcf，每一个概率下的概率期望均值
    all_var_rmcf = []
    all_expect_loss_rmcf = []
    all_result = []

    all_exper_obj_result_ecmp=[]
    all_availability_ecmp=[]
    all_var_ecmp=[]

    for d in range(len(demand_ratio)):
        # 获取需求矩阵
        all_graph_demand = util_tool.random_graph_demand(experment_time, num_nodes, real_sum_demand[d], sum_sink,
                                                         source_nodes_1,
                                                         dest_nodes_1)
        # print("输出需求矩阵")
        # for i in range(len(all_graph_demand)):
        #     print(all_graph_demand[i])
        a_sus_obj_mcf = []
        a_failure_obj_mcf = []
        a_availability_mcf = []
        a_var_mcf = []
        a_expect_loss_mcf = []

        a_sus_obj_rmcf = []
        a_failure_obj_rmcf = []
        a_availability_rmcf = []
        a_var_rmcf = []
        a_expect_loss_rmcf = []

        a_sus_obj_ecmp=[]
        a_availability_ecmp=[]
        a_var_ecmp=[]

        # 初始化拓扑
        graph1 = NetworkGraph(ad_matrix, link)
        # 设置概率矩阵
        graph1.set_graph_p(probability_edge[0])
        # 设置源节点和目的结点
        graph1.set_source_dest_nodes(source_nodes_1, dest_nodes_1)
        # 设置路由
        graph1.set_route(K, path_1)
        # 生成所有场景
        graph1.generate_scenes(fail_num)
        sum_scenes_prob = sum(graph1.scenes_prob)

        for e in range(experment_time):
            # 设置demand
            graph1.set_demand(all_graph_demand[e])
            # graph1.set_demand([-10, 0, 10, 0, 10, 0, 0, -10, -10, 0, 10])
            # 判断是否需要增加结点
            graph1.change_topo(real_sum_demand[d], sum_sink)
            print("实际的需求分配",graph1.new_graph_demand)
            # rmcf算法
            result_rmcf, obj_rmcf, allot_list_rmcf, zeta = offload_strategy.rmcf_lp_loss(graph1.new_graph_adj,
                                                                                         graph1.new_graph_c,
                                                                                         graph1.new_graph_u,
                                                                                         graph1.new_graph_p,
                                                                                         graph1.new_graph_demand,
                                                                                         graph1.new_source_nodes,
                                                                                         graph1.new_dest_nodes,
                                                                                         graph1.new_scenes,
                                                                                         graph1.new_scenes_list,
                                                                                         graph1.new_scenes_prob,
                                                                                         graph1.new_route, graph1.new_K,
                                                                                         alpha, C_list[d],
                                                                                         capacity_flag=False)

            print("实际的需求分配", graph1.new_graph_demand)
            #MCF算法
            result_mcf, obj_mcf, allot_list_mcf = offload_strategy.mcf_lp_flow(graph1.new_graph_adj, graph1.new_graph_c,
                                                                               graph1.new_graph_u,
                                                                               graph1.new_graph_demand,
                                                                               graph1.new_route, graph1.new_K,
                                                                               graph1.new_source_nodes,
                                                                               graph1.new_dest_nodes,
                                                                               capacity_flag=False)


            print("实际的需求分配", graph1.new_graph_demand)
            #ECMP算法
            allot_list_ecmp = util_tool.allot_list_rmcf_to_ecmp(allot_list_rmcf, graph1.new_source_nodes, graph1.new_dest_nodes)

            #rmcf保存数据
            if result_rmcf == "Optimal":
                print("*************************************mcf_start")
                print("*************************************mcf_end")
                # 计算可用性期望rmcf
                temp_availability_rmcf, temp_var_rmcf = util_tool.compute_scenes_availability(allot_list_rmcf,
                                                                                              graph1.new_route,
                                                                                              graph1.new_scenes_list,
                                                                                              graph1.new_scenes_prob,
                                                                                              zeta,alpha)
                temp_expect_loss_rmcf = util_tool.expect_loss(allot_list_rmcf, graph1.new_route, graph1.new_scenes_list,
                                                              graph1.new_scenes_prob)
                a_availability_rmcf.append(temp_availability_rmcf)
                a_var_rmcf.append(temp_var_rmcf)
                a_sus_obj_rmcf.append(obj_rmcf)
                a_expect_loss_rmcf.append(temp_expect_loss_rmcf)


                # 计算可用性期望 mcf
                temp_availability_mcf, temp_var_mcf = util_tool.compute_scenes_availability(allot_list_mcf,
                                                                                            graph1.new_route,
                                                                                            graph1.new_scenes_list,
                                                                                            graph1.new_scenes_prob,
                                                                                            zeta,alpha)
                temp_expect_loss_mcf = util_tool.expect_loss(allot_list_mcf, graph1.new_route, graph1.new_scenes_list,
                                                             graph1.new_scenes_prob)
                # temp_availvbility_mcf+=1-sum_scenes_prob
                a_availability_mcf.append(temp_availability_mcf)
                a_var_mcf.append(temp_var_mcf)
                a_sus_obj_mcf.append(obj_mcf)
                a_expect_loss_mcf.append(temp_expect_loss_mcf)

                #计算ecmp算法的可用性
                temp_availability_ecmp,temp_var_ecmp=util_tool.compute_scenes_availability(allot_list_ecmp,graph1.new_route,
                                                                                           graph1.new_scenes_list,
                                                                                           graph1.new_scenes_prob,
                                                                                           zeta,alpha)
                temp_cost_ecmp = util_tool.compute_cost_ecmp(allot_list_ecmp, graph1.new_path, graph1.new_graph_adj,
                                                             graph1.new_graph_c)
                a_availability_ecmp.append(temp_availability_ecmp)
                a_var_ecmp.append(temp_var_ecmp)
                a_sus_obj_ecmp.append(temp_cost_ecmp)
            else:
                a_failure_obj_rmcf.append(obj_rmcf)
                a_failure_obj_mcf.append(obj_mcf)
            #mcf保存数据
            # if result_mcf == "Optimal":
            #     a_sus_obj_mcf.append(obj_mcf)
            #     # 计算可用性期望 mcf
            #     temp_availvbility_mcf, temp_var_mcf = util_tool.compute_scenes_availability(allot_list_mcf,
            #                                                                                 graph1.new_route,
            #                                                                                 graph1.new_scenes_list,
            #                                                                                 graph1.new_scenes_prob,
            #                                                                                 zeta,alpha)
            #     # temp_availvbility_mcf+=1-sum_scenes_prob
            #     a_availability_mcf.append(temp_availvbility_mcf)
            #     a_var_mcf.append(temp_var_mcf)
            # else:
            #     a_failure_obj_mcf.append(obj_mcf)
            # all_result.append(a_sus_obj_mcf)
        # 计算平均目标值-rmcf
        # mean_obj_rmcf = np.mean(a_sus_obj_rmcf)
        all_exper_obj_result_rmcf.append(a_sus_obj_rmcf)
        all_exper_sus_num_rmcf.append(len(a_sus_obj_rmcf))
        # 计算demand下的可用性-rmcf
        all_availability_rmcf.append(a_availability_rmcf)
        all_var_rmcf.append(a_var_rmcf)
        all_expect_loss_rmcf.append(a_expect_loss_rmcf)

        # 计算平均目标值-mcf
        # mean_obj_mcf = np.mean(a_sus_obj_mcf)
        all_exper_obj_result_mcf.append(a_sus_obj_mcf)
        all_exper_sus_num_mcf.append(len(a_sus_obj_mcf))
        # 计算demand下的可用性-mcf
        all_availability_mcf.append(a_availability_mcf)
        all_var_mcf.append(a_var_mcf)
        all_expect_loss_mcf.append(a_expect_loss_mcf)

        #计算ecmp算法的可用性
        all_exper_obj_result_ecmp.append(a_sus_obj_ecmp)
        all_availability_ecmp.append(a_availability_ecmp)
        all_var_ecmp.append(a_var_ecmp)

        print("所有算法的var")
        print("rmcf",a_var_rmcf)
        print("mcf",a_var_mcf)
        print("ecmp",a_var_ecmp)

    #保存所有rmcf算法的数据
    # cost_mcf=[all_var_mcf[i]/real_sum_demand[i] for i in range(len(real_sum_demand))]
    # cost_rmcf=[all_var_rmcf[i]/real_sum_demand[i] for i in range(len(real_sum_demand))]
    mean_obj_rmcf = [np.mean(all_exper_obj_result_rmcf[i]) for i in range(len(all_exper_obj_result_rmcf))]
    mean_availability_rmcf = [np.mean(all_availability_rmcf[i]) for i in range(len(all_availability_rmcf))]
    mean_var_rmcf = [np.mean(all_var_rmcf[i]) for i in range(len(all_var_rmcf))]
    mean_var_demand_rmcf=[mean_var_rmcf[i]/real_sum_demand[i] for i in range(len(real_sum_demand))]
    mean_expect_loss_rmcf = [np.mean(all_expect_loss_rmcf[i]) for i in range(len(all_expect_loss_rmcf))]
    print("需求逐渐增大")
    print("链路的失效概率", probability_edge)
    print("RMCF的所有结果:")
    print("所有试验目标值结果", mean_obj_rmcf)
    print("每个试验成功的次数", all_exper_sus_num_rmcf)
    print("所有的可用性", mean_availability_rmcf)

    print("所有需求的mean var", mean_var_rmcf)
    print("所有的var/demand",mean_var_demand_rmcf)
    print("所有概率的loss期望", mean_expect_loss_rmcf)

    #保存所有的mcf算法的数据
    mean_obj_mcf = [np.mean(all_exper_obj_result_mcf[i]) for i in range(len(all_exper_obj_result_mcf))]
    mean_availability_mcf = [np.mean(all_availability_mcf[i]) for i in range(len(all_availability_mcf))]
    mean_var_mcf = [np.mean(all_var_mcf[i]) for i in range(len(all_var_mcf))]
    mean_var_demand_mcf = [mean_var_mcf[i] / real_sum_demand[i] for i in range(len(real_sum_demand))]
    mean_expect_loss_mcf = [np.mean(all_expect_loss_mcf[i]) for i in range(len(all_expect_loss_mcf))]
    print("MCF的所有结果:")
    print("所有试验目标值结果", mean_obj_mcf)
    print("每个试验成功的次数", all_exper_sus_num_mcf)
    print("所有的需求可用性", mean_availability_mcf)
    print("所有需求的mean_var", mean_var_mcf)
    print("平均的 var/demand",mean_var_demand_mcf)
    print("所有概率的loss期望", mean_expect_loss_mcf)

    #保存所有ecmp算法的数据
    mean_obj_ecmp = [np.mean(all_exper_obj_result_ecmp[i]) for i in range(len(all_exper_obj_result_ecmp))]
    mean_availability_ecmp = [np.mean(all_availability_ecmp[i]) for i in range(len(all_availability_ecmp))]
    mean_var_ecmp = [np.mean(all_var_ecmp[i]) for i in range(len(all_var_ecmp))]
    mean_var_demand_ecmp = [mean_var_ecmp[i] / real_sum_demand[i] for i in range(len(real_sum_demand))]
    print("ECMP")
    print("所有实验的目标值",mean_obj_ecmp)
    print("所有需求的可用性",mean_availability_ecmp)
    print("所有需求的var",mean_var_ecmp)
    print("平均的var/demand",mean_var_demand_ecmp)

    print("开始画图")
    data_dict1 = {
        "parameter":parameter,
        "demand_ratio": demand_ratio,
         "all_availability_rmcf": all_availability_rmcf,
         "all_availability_mcf": all_availability_mcf,
         "all_availability_ecmp":all_availability_ecmp,
         "all_cost_rmcf": all_exper_obj_result_rmcf,
         "all_cost_mcf": all_exper_obj_result_mcf,
        "all_cost_ecmp":all_exper_obj_result_ecmp,
        "all_var_rmcf":all_var_rmcf,
        "all_var_mcf":all_var_mcf,
        "all_var_ecmp":all_var_ecmp,
        "mean_var_demand_rmcf": mean_var_demand_rmcf,
        "mean_var_demand_mcf": mean_var_demand_mcf,
        "mean_var_demand_ecmp": mean_var_demand_ecmp
             }

    ds = DataSave(sys._getframe().f_code.co_name)
    ds.sava_all_data(data_dict1)

def demo_alpha_influence():
    """
    设置一个简单图，四个结点，对var和cvar进行分析
    测试alpha对不同算法的影响
    :return:
    """
    # 图邻接矩阵
    ad_matrix = np.zeros(shape=(4, 4))
    link = [(1, 2),(1, 3),
            (2, 1),(2, 4),
            (3, 1),(3, 4),
            (4, 2), (4, 3)
            ]
    for i in range(len(link)):
        ad_matrix[link[i][0] - 1][link[i][1] - 1] = 1
    source_nodes_1 = [1]
    dest_nodes_1 = [4]
    num_nodes = 4
    K = 2  # 每一个sd流中路径的数量
    fail_num = 2
    # alpha = 0.9
    #['0%', '5%', '10%', '15%', '20%', '25%', '30%', '35%', '40%', '45%', '50%',
    # '55%', '60%', '65%', '70%', '75%', '80%', '85%', '90%', '95%']
    # all_alpha=[0.8]
    # all_alpha=[round(0.0+0.05*x,2) for x in range(0,20)]
    # all_alpha=[round(0.6+0.02*x,2) for x in range(0,20)]
    all_alpha=[0.7,0.75,0.8,0.85,0.9,0.95]
    C = 0.45
    gamma=0.5
    path_1 = [[1, 4],
              [2, 6]
              ]
    experment_time = 1
    sum_sink = 20
    sum_demand = 20

    # probability_edge = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    probability_edge = [0.1]
    # probability_edge=[0.01,0.03,0.05,0.07,0.09,0.11]
    # probability_edge=[0.05]

    parameter = {
        "fail_num": fail_num,
        "alpha": all_alpha,
        "sum_sink": sum_sink,
        "sum_demand": sum_demand,
        "C": C,
        "probability_edge": probability_edge,
        "expermnet_time": experment_time
    }
    # mcf算法，结果统计
    all_exper_obj_result_mcf = []  # mcf，每一个概率对应的平均目标值
    all_exper_sus_num_mcf = []  # mcf，每一个概率下实验成功的次数
    all_availability_mcf = []  # mcf，每一个概率下的概率期望均值
    all_var_mcf = []
    all_cost_mcf = []
    all_expect_loss_mcf = []
    all_cvar_mcf=[]
    all_allot_mcf=[]

    # rmcf算法结果统计
    all_exper_obj_result_rmcf = []  # mcf，每一个概率对应的平均目标值
    all_exper_sus_num_rmcf = []  # mcf，每一个概率下实验成功的次数
    all_availability_rmcf = []  # mcf，每一个概率下的概率期望均值
    all_var_rmcf = []
    all_var_rmcf_experment=[]
    all_cost_rmcf = []
    all_expect_loss_rmcf = []
    all_allot_rmcf=[]
    all_cvar_rmcf=[]

    # ecmp算法结果统计
    all_exper_obj_result_ecmp = []
    all_availability_ecmp = []
    all_var_ecmp = []
    all_result = []

    # 获取需求矩阵
    all_graph_demand = util_tool.random_graph_demand(experment_time, num_nodes, sum_demand, sum_sink,
                                                     source_nodes_1,
                                                     dest_nodes_1)

    pprint(all_graph_demand)
    for a_alpha in range(len(all_alpha)):
        print("链路失效概率", probability_edge[0])
        a_sus_obj_mcf = []
        a_failure_obj_mcf = []
        a_availability_mcf = []
        a_var_mcf = []
        a_expect_loss_mcf = []
        a_var_mcf=[]
        a_cvar_mcf=[]
        a_allot_mcf=[]



        a_sus_obj_rmcf = []
        a_failure_obj_rmcf = []
        a_availability_rmcf = []
        a_var_rmcf = []
        a_var_rmcf_experment=[]
        a_expect_loss_rmcf = []
        a_allot_rmcf = []
        a_cvar_rmcf=[]

        a_sus_obj_ecmp = []
        a_availability_ecmp = []
        a_var_ecmp = []

        # 初始化拓扑
        graph1 = NetworkGraph(ad_matrix, link)
        # 设置概率矩阵
        #（1）同构矩阵
        # graph1.set_graph_p(probability_edge[0])
        # fail_list=[0.02]*8
        # fail_list[0]=0.03
        # fail_list[3]=0.03
        #（2）异构矩阵
        # fail_list=[0.1,0.2,0.1,0.1,0.2,0.2,0.1,0.2]
        fail_list=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
        loss_list=[0.05]*8
        graph1.set_graph_p_isomerism(fail_list)
        graph1.set_graph_loss_isomerism(loss_list)
        print(all_graph_demand)
        # 设置源节点和目的结点
        graph1.set_source_dest_nodes(source_nodes_1, dest_nodes_1)
        # 设置路由
        graph1.set_route(K, path_1)
        #设置链路cost
        # cost_list=[1,1.1,1,1,1.1,1.1,1,1.1]
        cost_list=[1.3,1,1.3,1.3,1,1,1.3,1]
        graph1.set_graph_c_isomerism(cost_list)

        # 生成所有场景
        graph1.generate_scenes(fail_num)
        sum_scenes_prob = sum(graph1.scenes_prob)

        for e in range(experment_time):
            # 设置demand
            graph1.set_demand(all_graph_demand[e])
            # graph1.set_demand([-10, 0, 10, 0, 10, 0, 0, -10, -10, 0, 10])
            # 判断是否需要增加结点
            graph1.change_topo(sum_demand, sum_sink)
            # rmcf算法
            # result_rmcf, obj_rmcf, allot_list_rmcf, zeta = offload_strategy.rmcf_lp_loss(graph1.new_graph_adj,
            #                                                                              graph1.new_graph_c,
            #                                                                              graph1.new_graph_u,
            #                                                                              graph1.new_graph_p,
            #                                                                              graph1.new_graph_demand,
            #                                                                              graph1.new_source_nodes,
            #                                                                              graph1.new_dest_nodes,
            #                                                                              graph1.new_scenes,
            #                                                                              graph1.new_scenes_list,
            #                                                                              graph1.new_scenes_prob,
            #                                                                              graph1.new_route, graph1.new_K,
            #                                                                              all_alpha[a_alpha], C,
            #                                                                              capacity_flag=False)

            #rmcf_plus算法
            result_rmcf, obj_rmcf, allot_list_rmcf, zeta = offload_strategy.rmcf_lp_loss(graph1.new_graph_adj,
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
                                                                                         all_alpha[a_alpha], C,
                                                                                         capacity_flag=False)
            # mcf算法
            result_mcf, obj_mcf, allot_list_mcf = offload_strategy.mcf_lp_flow(graph1.new_graph_adj, graph1.new_graph_c,
                                                                               graph1.new_graph_u,
                                                                               graph1.new_graph_demand,
                                                                               graph1.new_route, graph1.new_K,
                                                                               graph1.new_source_nodes,
                                                                               graph1.new_dest_nodes,
                                                                               capacity_flag=False)

            # ECMP算法:固定的ecmp算法
            # allot_list_ecmp = util_tool.allot_list_rmcf_to_ecmp(allot_list_rmcf, graph1.new_source_nodes,
            #             #                                                     graph1.new_dest_nodes)
            # 新的ECMP算法
            allot_list_ecmp = util_tool.ecmp(allot_list_rmcf, graph1.new_source_nodes,
                                             graph1.new_dest_nodes, graph1.new_K)

            print("rmcf------------------start")
            # 计算可用性期望rmcf
            # temp_availability_rmcf, temp_var_rmcf, temp_cvar_rmcf = util_tool.compute_scenes_availability(
            #     allot_list_rmcf,
            #     graph1.new_route,
            #     graph1.new_scenes_list,
            #     graph1.new_scenes_prob,
            #     zeta, all_alpha[a_alpha])

            # 计算可用性期望rmcf：新的计算方式
            temp_availability_rmcf, temp_var_rmcf, temp_cvar_rmcf = util_tool.compute_scenes_availability(
                allot_list_rmcf,
                graph1.new_route,
                graph1.new_scenes_list,
                graph1.new_scenes_prob,
                graph1.new_flow_cost,
                graph1.new_graph_loss_list,
                zeta, all_alpha[a_alpha])
            # temp_expect_loss_rmcf = util_tool.expect_loss(allot_list_rmcf, graph1.new_route, graph1.new_scenes_list,
            #                                               graph1.new_scenes_prob)
            # temp_availvbility_rmcf+=1-sum_scenes_prob

            print("分配方式", allot_list_rmcf)
            print("var", temp_var_rmcf)
            # print("rmcf期望", temp_expect_loss_rmcf)
            a_availability_rmcf.append(temp_availability_rmcf)
            a_var_rmcf.append(temp_var_rmcf)
            a_var_rmcf_experment.append(zeta)
            a_sus_obj_rmcf.append(obj_rmcf)
            # a_expect_loss_rmcf.append(temp_expect_loss_rmcf)
            a_cvar_rmcf.append(temp_cvar_rmcf)

            a_allot_rmcf.append(allot_list_rmcf)
            print("rmcf------------------------end")

            print("ecmp--------------------start")

            # 计算ecmp算法的可用性和cost
            # temp_availability_ecmp, temp_var_ecmp, temp_cvar_ecmp = util_tool.compute_scenes_availability(
            #     allot_list_ecmp,
            #     graph1.new_route,
            #     graph1.new_scenes_list,
            #     graph1.new_scenes_prob,
            #     zeta, all_alpha[a_alpha])

            temp_availability_ecmp, temp_var_ecmp, temp_cvar_ecmp = util_tool.compute_scenes_availability(
                allot_list_ecmp,
                graph1.new_route,
                graph1.new_scenes_list,
                graph1.new_scenes_prob,
                graph1.new_flow_cost,
                graph1.new_graph_loss_list,
                zeta, all_alpha[a_alpha])

            # temp_cost_ecmp = util_tool.compute_cost_ecmp(allot_list_ecmp, graph1.new_path, graph1.new_graph_adj,
            #                                              graph1.new_graph_c)
            a_availability_ecmp.append(temp_availability_ecmp)
            a_var_ecmp.append(temp_var_ecmp)
            # a_sus_obj_ecmp.append(temp_cost_ecmp)
            print("ecmp----------------------------end")

            # 计算mcf的可用性，在rmcf成功有解时，才计算mcf
            if result_mcf == "Optimal":
                print("mcf------------------start")
                a_sus_obj_mcf.append(obj_mcf)
                # 计算可用性期望 mcf
                # temp_availability_mcf, temp_var_mcf, temp_cvar_mcf = util_tool.compute_scenes_availability(
                #     allot_list_mcf,
                #     graph1.new_route,
                #     graph1.new_scenes_list,
                #     graph1.new_scenes_prob,
                #     zeta, all_alpha[a_alpha])
                # temp_expect_loss_mcf = util_tool.expect_loss(allot_list_mcf, graph1.new_route,
                #                                              graph1.new_scenes_list, graph1.new_scenes_prob)

                temp_availability_mcf, temp_var_mcf, temp_cvar_mcf = util_tool.compute_scenes_availability(
                    allot_list_mcf,
                    graph1.new_route,
                    graph1.new_scenes_list,
                    graph1.new_scenes_prob,
                    graph1.new_flow_cost,
                    graph1.new_graph_loss_list,
                    zeta, all_alpha[a_alpha])

                print("分配方式", allot_list_mcf)
                print("var", temp_var_mcf)
                # print("mcf期望", temp_expect_loss_mcf)
                # temp_availvbility_mcf+=1-sum_scenes_prob
                a_availability_mcf.append(temp_availability_mcf)
                a_var_mcf.append(temp_var_mcf)
                # a_expect_loss_mcf.append(temp_expect_loss_mcf)
                a_cvar_mcf.append(temp_cvar_mcf)
                a_allot_mcf.append(allot_list_mcf)
                print("mcf----------------------end")
            else:
                a_failure_obj_mcf.append(obj_mcf)


        all_result.append(a_sus_obj_mcf)
        # 计算平均目标值-rmcf
        # mean_obj_rmcf = np.mean(a_sus_obj_rmcf)
        all_exper_obj_result_rmcf.append(a_sus_obj_rmcf)
        all_exper_sus_num_rmcf.append(len(a_sus_obj_rmcf))
        # 计算pf概率下的可用性-rmcf
        all_availability_rmcf.append(a_availability_rmcf)
        all_var_rmcf.append(a_var_rmcf)
        all_var_rmcf_experment.append(a_var_rmcf_experment)
        # all_cost_rmcf=[all_var_rmcf[i]/sum_demand for i in range(len(all_var_rmcf))]
        all_expect_loss_rmcf.append(a_expect_loss_rmcf)
        all_allot_rmcf.append(a_allot_rmcf)
        all_cvar_rmcf.append(a_cvar_rmcf)


        # 计算平均目标值-mcf
        # mean_obj_mcf = np.mean(a_sus_obj_mcf)
        all_exper_obj_result_mcf.append(a_sus_obj_mcf)
        all_exper_sus_num_mcf.append(len(a_sus_obj_mcf))
        # 计算pf概率下的可用性-mcf
        all_availability_mcf.append(a_availability_mcf)
        all_var_mcf.append(a_var_mcf)
        all_cvar_mcf.append(a_cvar_mcf)
        all_allot_mcf.append(a_allot_mcf)

        # all_cost_mcf=[all_var_mcf[i]/sum_demand for i in range(len(all_var_mcf))]
        all_expect_loss_mcf.append(a_expect_loss_mcf)

        # 计算ecmp算法的可用性
        all_exper_obj_result_ecmp.append(a_sus_obj_ecmp)
        all_availability_ecmp.append(a_availability_ecmp)
        all_var_ecmp.append(a_var_ecmp)

        print("所有算法的var")
        print("rmcf", a_var_rmcf)
        print("mcf", a_var_mcf)
        print("ecmp", a_var_ecmp)

    mean_obj_rmcf = [np.mean(all_exper_obj_result_rmcf[i]) for i in range(len(all_exper_obj_result_rmcf))]
    mean_availability_rmcf = [np.mean(all_availability_rmcf[i]) for i in range(len(all_availability_rmcf))]
    mean_var_rmcf = [np.mean(all_var_rmcf[i]) for i in range(len(all_var_rmcf))]
    mean_var_rmcf_experment = [np.mean(all_var_rmcf_experment[i]) for i in range(len(all_var_rmcf_experment))]
    mean_var_demand_rmcf = [mean_var_rmcf[i] / sum_demand for i in range(len(mean_var_rmcf))]
    mean_expect_loss_rmcf = [np.mean(all_expect_loss_rmcf[i]) for i in range(len(all_expect_loss_rmcf))]

    print("链路概率同构场景")
    print("链路的可用性", list(map(lambda x: str(int(x * 100)) + '%', all_alpha)))
    print("RMCF的所有结果:")
    print("所有分配方案：")
    pprint(all_allot_rmcf)
    print("所有试验目标值结果")
    # print(mean_obj_rmcf)
    print(mean_obj_rmcf[0:10])
    print(mean_obj_rmcf[10:20])
    print("每个试验成功的次数", all_exper_sus_num_rmcf)
    print("所有的概率可用性", mean_availability_rmcf)
    print("所有概率的真实var", mean_var_rmcf)
    print("所有概率的实验var",mean_var_rmcf_experment)
    print("所有的alpha的cvar")
    # print(all_cvar_rmcf)
    print(all_cvar_rmcf[0:10])
    print(all_cvar_rmcf[10:20])
    print("所有的var/demand", mean_var_demand_rmcf)
    print("所有概率的loss期望", mean_expect_loss_rmcf)

    mean_obj_mcf = [np.mean(all_exper_obj_result_mcf[i]) for i in range(len(all_exper_obj_result_mcf))]
    mean_availability_mcf = [np.mean(all_availability_mcf[i]) for i in range(len(all_availability_mcf))]
    mean_var_mcf = [np.mean(all_var_mcf[i]) for i in range(len(all_var_mcf))]
    mean_var_demand_mcf = [mean_var_mcf[i] / sum_demand for i in range(len(mean_var_mcf))]
    mean_expect_loss_mcf = [np.mean(all_expect_loss_mcf[i]) for i in range(len(all_expect_loss_mcf))]
    print("MCF的所有结果:")
    print("所有试验目标值结果", mean_obj_mcf)
    print("每个试验成功的次数", all_exper_sus_num_mcf)
    print("所有的概率可用性", mean_availability_mcf)
    print("所有分配方案：")
    pprint(all_allot_mcf)
    print("所有概率的var", mean_var_mcf)
    print("所有概率的cvar",all_cvar_mcf)
    print("所有的var/deamnd", mean_var_demand_mcf)
    print("所有概率的loss期望", mean_expect_loss_mcf)

    # 画图
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # plt.figure(1)
    # x = all_alpha
    # y_rmcf = all_cvar_rmcf
    # y_mcf=all_cvar_mcf
    # plt.plot(x, y_rmcf, 'ro-', label="rmcf_cvar")
    # plt.plot(x, y_mcf, 'cs-', label="mcf")
    # plt.xlabel("alpha")
    # plt.ylabel("计算的cvar")
    # plt.legend()
    # x_ticks=list(map(lambda x:str(int(x*100))+'%',x[0:-1:2]))
    # # x_ticks=list(map(lambda x:str(int(x*100))+'%',x))
    # plt.xticks(x[0:-1:2],x_ticks)
    # plt.title(u"同构场景下的cvar随着可用性变化图")  # u加上中文
    # plt.show()
    #
    # plt.figure(2)
    # x = all_alpha
    # y_rmcf = np.array(all_allot_rmcf).flatten().reshape(20,2)
    # y_rmcf=np.std(y_rmcf,axis=1)
    # y_mcf = np.array(all_allot_mcf).flatten().reshape(20,2)
    # y_mcf=np.std(y_mcf,axis=1)
    # plt.plot(x, y_rmcf, 'ro-', label="rmcf_cvar")
    # plt.plot(x, y_mcf, 'cs-', label="mcf")
    # plt.xlabel("alpha")
    # plt.ylabel("分配方案的标准差")
    # plt.legend()
    # x_ticks = list(map(lambda x: str(int(x * 100)) + '%', x[0:-1:2]))
    # plt.xticks(x[0:-1:2], x_ticks)
    # plt.title(u"同构场景下的分配方案标准差随着可用性变化图")  # u加上中文
    # plt.show()


def test_cvxpy_demo():
    """
      设置一个简单图，四个结点，对var和cvar进行分析
      测试alpha对不同算法的影响
      :return:
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
    fail_num = 2
    # alpha = 0.9
    # ['0%', '5%', '10%', '15%', '20%', '25%', '30%', '35%', '40%', '45%', '50%',
    # '55%', '60%', '65%', '70%', '75%', '80%', '85%', '90%', '95%']
    all_alpha=[0.8]
    # all_alpha=[round(0.0+0.05*x,2) for x in range(0,20)]
    # all_alpha = [round(0.6 + 0.02 * x, 2) for x in range(0, 20)]
    # all_alpha=[round(0.8+0.01*x,2) for x in range(0,20)]
    C = 0.45
    path_1 = [[1, 4],
              [2, 6]
              ]
    experment_time = 1
    sum_sink = 20
    sum_demand = 20

    # probability_edge = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    probability_edge = [0.1]
    # probability_edge=[0.01,0.03,0.05,0.07,0.09,0.11]
    # probability_edge=[0.05]

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

    # 获取需求矩阵
    all_graph_demand = util_tool.random_graph_demand(experment_time, num_nodes, sum_demand, sum_sink,
                                                     source_nodes_1,
                                                     dest_nodes_1)
    pprint(all_graph_demand)
    for a_alpha in range(len(all_alpha)):
        print("链路失效概率", probability_edge[0])


        a_sus_obj_rmcf = []
        a_failure_obj_rmcf = []
        a_availability_rmcf = []
        a_var_rmcf = []
        a_var_rmcf_experment = []
        a_expect_loss_rmcf = []
        a_allot_rmcf = []
        a_cvar_rmcf = []

        # 初始化拓扑
        graph1 = NetworkGraph(ad_matrix, link)
        # 设置概率矩阵
        # （1）同构矩阵
        # graph1.set_graph_p(probability_edge[0])
        # fail_list=[0.02]*8
        # fail_list[0]=0.03
        # fail_list[3]=0.03
        # （2）异构矩阵
        # fail_list=[0.1,0.2,0.1,0.1,0.2,0.2,0.1,0.2]
        fail_list = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        graph1.set_graph_p_isomerism(fail_list)
        print(all_graph_demand)
        # 设置源节点和目的结点
        graph1.set_source_dest_nodes(source_nodes_1, dest_nodes_1)
        # 设置路由
        graph1.set_route(K, path_1)
        # 设置链路cost
        # cost_list=[1,1.1,1,1,1.1,1.1,1,1.1]
        cost_list = [1.3, 1, 1.3, 1.3, 1, 1, 1.3, 1]
        graph1.set_graph_c_isomerism(cost_list)

        # 生成所有场景
        graph1.generate_scenes(fail_num)
        sum_scenes_prob = sum(graph1.scenes_prob)

        for e in range(experment_time):
            # 设置demand
            graph1.set_demand(all_graph_demand[e])
            # graph1.set_demand([-10, 0, 10, 0, 10, 0, 0, -10, -10, 0, 10])
            # 判断是否需要增加结点
            graph1.change_topo(sum_demand, sum_sink)
            # rmcf算法
            # result_rmcf, obj_rmcf, allot_list_rmcf, zeta = offload_strategy.rmcf_lp_loss(graph1.new_graph_adj,
            #                                                                              graph1.new_graph_c,
            #                                                                              graph1.new_graph_u,
            #                                                                              graph1.new_graph_p,
            #                                                                              graph1.new_graph_demand,
            #                                                                              graph1.new_source_nodes,
            #                                                                              graph1.new_dest_nodes,
            #                                                                              graph1.new_scenes,
            #                                                                              graph1.new_scenes_list,
            #                                                                              graph1.new_scenes_prob,
            #                                                                              graph1.new_route, graph1.new_K,
            #                                                                              all_alpha[a_alpha], C,
            #                                                                              capacity_flag=False)
            offload_strategy.test_demo(graph1.new_graph_adj,
                                          graph1.new_graph_c,
                                          graph1.new_graph_u,
                                          graph1.new_graph_p,
                                          graph1.new_graph_demand,
                                          graph1.new_source_nodes,
                                          graph1.new_dest_nodes,
                                          graph1.new_scenes,
                                          graph1.new_scenes_list,
                                          graph1.new_scenes_prob,
                                          graph1.new_route, graph1.new_K,
                                          all_alpha[a_alpha], C,
                                          capacity_flag=False)

            # print("rmcf------------------start")
            # # 计算可用性期望rmcf
            # temp_availability_rmcf, temp_var_rmcf, temp_cvar_rmcf = util_tool.compute_scenes_availability(
            #     allot_list_rmcf,
            #     graph1.new_route,
            #     graph1.new_scenes_list,
            #     graph1.new_scenes_prob,
            #     zeta, all_alpha[a_alpha])
            # temp_expect_loss_rmcf = util_tool.expect_loss(allot_list_rmcf, graph1.new_route, graph1.new_scenes_list,
            #                                               graph1.new_scenes_prob)
            # # temp_availvbility_rmcf+=1-sum_scenes_prob
            # print("分配方式", allot_list_rmcf)
            # print("var", temp_var_rmcf)
            # print("rmcf期望", temp_expect_loss_rmcf)
            # a_availability_rmcf.append(temp_availability_rmcf)
            # a_var_rmcf.append(temp_var_rmcf)
            # a_var_rmcf_experment.append(zeta)
            # a_sus_obj_rmcf.append(obj_rmcf)
            # a_expect_loss_rmcf.append(temp_expect_loss_rmcf)
            # a_cvar_rmcf.append(temp_cvar_rmcf)
            #
            # a_allot_rmcf.append(allot_list_rmcf)
            # print("rmcf------------------------end")


        # # 计算平均目标值-rmcf
        # # mean_obj_rmcf = np.mean(a_sus_obj_rmcf)
        # all_exper_obj_result_rmcf.append(a_sus_obj_rmcf)
        # all_exper_sus_num_rmcf.append(len(a_sus_obj_rmcf))
        # # 计算pf概率下的可用性-rmcf
        # all_availability_rmcf.append(a_availability_rmcf)
        # all_var_rmcf.append(a_var_rmcf)
        # all_var_rmcf_experment.append(a_var_rmcf_experment)
        # # all_cost_rmcf=[all_var_rmcf[i]/sum_demand for i in range(len(all_var_rmcf))]
        # all_expect_loss_rmcf.append(a_expect_loss_rmcf)
        # all_allot_rmcf.append(a_allot_rmcf)
        # all_cvar_rmcf.append(a_cvar_rmcf)




    # mean_obj_rmcf = [np.mean(all_exper_obj_result_rmcf[i]) for i in range(len(all_exper_obj_result_rmcf))]
    # mean_availability_rmcf = [np.mean(all_availability_rmcf[i]) for i in range(len(all_availability_rmcf))]
    # mean_var_rmcf = [np.mean(all_var_rmcf[i]) for i in range(len(all_var_rmcf))]
    # mean_var_rmcf_experment = [np.mean(all_var_rmcf_experment[i]) for i in range(len(all_var_rmcf_experment))]
    # mean_var_demand_rmcf = [mean_var_rmcf[i] / sum_demand for i in range(len(mean_var_rmcf))]
    # mean_expect_loss_rmcf = [np.mean(all_expect_loss_rmcf[i]) for i in range(len(all_expect_loss_rmcf))]

    # print("链路概率同构场景")
    # print("链路的可用性", list(map(lambda x: str(int(x * 100)) + '%', all_alpha)))
    # print("RMCF的所有结果:")
    # print("所有分配方案：")
    # pprint(all_allot_rmcf)
    # print("所有试验目标值结果")
    # # print(mean_obj_rmcf)
    # print(mean_obj_rmcf[0:10])
    # print(mean_obj_rmcf[10:20])
    # print("每个试验成功的次数", all_exper_sus_num_rmcf)
    # print("所有的概率可用性", mean_availability_rmcf)
    # print("所有概率的真实var", mean_var_rmcf)
    # print("所有概率的实验var", mean_var_rmcf_experment)
    # print("所有的alpha的cvar")
    # # print(all_cvar_rmcf)
    # print(all_cvar_rmcf[0:10])
    # print(all_cvar_rmcf[10:20])
    # print("所有的var/demand", mean_var_demand_rmcf)
    # print("所有概率的loss期望", mean_expect_loss_rmcf)

def experiment_prob_availability_demo():
    """
    测试链路失效概率，对不同的算法影响
    :return:
    """
    # 图邻接矩阵
    ad_matrix = np.zeros(shape=(11, 11))
    link = [(1, 2), (2, 1),
            (1, 4), (4, 1),
            (2, 3), (3, 2),
            (2, 4), (4, 2),
            (3, 5), (5, 3),
            (4, 6), (6, 4),
            (5, 6), (6, 5),
            (5, 7), (7, 5),
            (6, 8), (8, 6),
            (7, 8), (8, 7),
            (7, 9), (9, 7),
            (8, 10), (10, 8),
            (9, 11), (11, 9),
            (10, 11), (11, 10)]
    for i in range(len(link)):
        ad_matrix[link[i][0] - 1][link[i][1] - 1] = 1
    source_nodes_1 = [3, 5, 11]
    dest_nodes_1 = [1, 8, 9]
    num_nodes=11
    K = 3  # 每一个sd流中路径的数量
    fail_num = 2
    alpha = 0.9
    C = 0.3
    path_1 = [[6, 3],
              [6, 5, 8],
              [7, 12, 14, 8],

              [7, 12, 16],
              [7, 13, 18],
              [6, 5, 10, 16],

              [7, 13, 19],
              [7, 12, 16, 21, 19],
              [6, 5, 10, 15, 13, 19],

              [11, 6, 3],
              [11, 6, 5, 8],
              [12, 14, 8],

              [12, 16],
              [13, 16],
              [13, 19, 24, 28, 25],

              [13, 19],
              [12, 16, 21, 19],
              [12, 16, 22, 26, 27],

              [28, 25, 20, 14, 8],
              [28, 25, 20, 14, 9, 3],
              [27, 23, 17, 11, 6, 3],

              [28, 25],
              [27, 23, 18],
              [27, 23, 17, 12, 16],

              [27],
              [28, 25, 21, 19],
              [28, 25, 20, 15, 13, 19]
              ]

    experment_time =20
    # experment_time=1
    sum_sink = 30
    sum_demand = 30

    probability_edge=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10]
    # probability_edge=[0.01]
    # probability_edge=[0.01,0.03,0.05,0.07,0.09,0.11]
    # probability_edge=[0.05]

    parameter={
        "fail_num":fail_num,
        "alpha":alpha,
        "sum_sink":sum_sink,
        "sum_demand":sum_demand,
        "C":C,
        "probability_edge":probability_edge,
        "expermnet_time":experment_time
    }
    #mcf算法，结果统计
    all_exper_obj_result_mcf = []  # mcf，每一个概率对应的平均目标值
    all_exper_sus_num_mcf = []  # mcf，每一个概率下实验成功的次数
    all_availability_mcf = []  # mcf，每一个概率下的概率期望均值
    all_var_mcf=[]
    all_cost_mcf=[]
    all_expect_loss_mcf=[]

    #rmcf算法结果统计
    all_exper_obj_result_rmcf = []  # mcf，每一个概率对应的平均目标值
    all_exper_sus_num_rmcf = []  # mcf，每一个概率下实验成功的次数
    all_availability_rmcf = []  # mcf，每一个概率下的概率期望均值
    all_var_rmcf=[]
    all_cost_rmcf=[]
    all_expect_loss_rmcf=[]

    #ecmp算法结果统计
    all_exper_obj_result_ecmp=[]
    all_availability_ecmp = []
    all_var_ecmp = []
    all_result=[]

    # 获取需求矩阵
    all_graph_demand = util_tool.random_graph_demand(experment_time, num_nodes, sum_demand, sum_sink,
                                                     source_nodes_1,
                                                     dest_nodes_1)
    pprint(all_graph_demand)
    for p in range(len(probability_edge)):
        print("链路失效概率", probability_edge[p])
        a_sus_obj_mcf = []
        a_failure_obj_mcf = []
        a_availability_mcf = []
        a_var_mcf=[]
        a_expect_loss_mcf=[]

        a_sus_obj_rmcf = []
        a_failure_obj_rmcf = []
        a_availability_rmcf = []
        a_var_rmcf=[]
        a_expect_loss_rmcf=[]

        a_sus_obj_ecmp = []
        a_availability_ecmp = []
        a_var_ecmp = []

        # 初始化拓扑
        graph1 = NetworkGraph(ad_matrix, link)
        # 设置概率矩阵
        graph1.set_graph_p(probability_edge[p])
        print(all_graph_demand)
        # 设置源节点和目的结点
        graph1.set_source_dest_nodes(source_nodes_1, dest_nodes_1)
        # 设置路由
        graph1.set_route(K, path_1)
        # 生成所有场景
        graph1.generate_scenes(fail_num)
        sum_scenes_prob=sum(graph1.scenes_prob)

        for e in range(experment_time):
            # 设置demand
            graph1.set_demand(all_graph_demand[e])
            # graph1.set_demand([-10, 0, 10, 0, 10, 0, 0, -10, -10, 0, 10])
            # 判断是否需要增加结点
            graph1.change_topo(sum_demand, sum_sink)
            # rmcf算法
            result_rmcf, obj_rmcf, allot_list_rmcf, zeta = offload_strategy.rmcf_lp_loss(graph1.new_graph_adj,
                                                                                         graph1.new_graph_c,
                                                                                         graph1.new_graph_u,
                                                                                         graph1.new_graph_p,
                                                                                         graph1.new_graph_demand,
                                                                                         graph1.new_source_nodes,
                                                                                         graph1.new_dest_nodes,
                                                                                         graph1.new_scenes,
                                                                                         graph1.new_scenes_list,
                                                                                         graph1.new_scenes_prob,
                                                                                         graph1.new_route, graph1.new_K,
                                                                                         alpha, C,
                                                                                         capacity_flag=False)
            #mcf算法
            result_mcf, obj_mcf, allot_list_mcf = offload_strategy.mcf_lp_flow(graph1.new_graph_adj, graph1.new_graph_c,
                                                                               graph1.new_graph_u,
                                                                               graph1.new_graph_demand,
                                                                               graph1.new_route, graph1.new_K,
                                                                               graph1.new_source_nodes,
                                                                               graph1.new_dest_nodes,
                                                                               capacity_flag=False)

            # ECMP算法
            allot_list_ecmp = util_tool.allot_list_rmcf_to_ecmp(allot_list_rmcf, graph1.new_source_nodes,
                                                                graph1.new_dest_nodes)
            if result_rmcf == "Optimal":
                # 计算可用性期望rmcf
                temp_availability_rmcf,temp_var_rmcf= util_tool.compute_scenes_availability(allot_list_rmcf, graph1.new_route,
                                                                               graph1.new_scenes_list,
                                                                               graph1.new_scenes_prob,
                                                                               zeta,alpha)
                temp_expect_loss_rmcf=util_tool.expect_loss(allot_list_rmcf,graph1.new_route,graph1.new_scenes_list,graph1.new_scenes_prob)
                # temp_availvbility_rmcf+=1-sum_scenes_prob
                a_availability_rmcf.append(temp_availability_rmcf)
                a_var_rmcf.append(temp_var_rmcf)
                a_sus_obj_rmcf.append(obj_rmcf)
                a_expect_loss_rmcf.append(temp_expect_loss_rmcf)

                # 计算ecmp算法的可用性和cost
                temp_availability_ecmp, temp_var_ecmp = util_tool.compute_scenes_availability(allot_list_ecmp,
                                                                                              graph1.new_route,
                                                                                              graph1.new_scenes_list,
                                                                                              graph1.new_scenes_prob,
                                                                                              zeta, alpha)

                temp_cost_ecmp=util_tool.compute_cost_ecmp(allot_list_ecmp,graph1.new_path,graph1.new_graph_adj,
                                                           graph1.new_graph_c)
                a_availability_ecmp.append(temp_availability_ecmp)
                a_var_ecmp.append(temp_var_ecmp)
                a_sus_obj_ecmp.append(temp_cost_ecmp)

                #计算mcf的可用性，在rmcf成功有解时，才计算mcf
                if result_mcf == "Optimal":
                    a_sus_obj_mcf.append(obj_mcf)
                    # 计算可用性期望 mcf
                    temp_availability_mcf, temp_var_mcf = util_tool.compute_scenes_availability(allot_list_mcf,
                                                                                                graph1.new_route,
                                                                                                graph1.new_scenes_list,
                                                                                                graph1.new_scenes_prob,
                                                                                                zeta,alpha)
                    temp_expect_loss_mcf=util_tool.expect_loss(allot_list_mcf,graph1.new_route,graph1.new_scenes_list,graph1.new_scenes_prob)
                    # temp_availvbility_mcf+=1-sum_scenes_prob
                    a_availability_mcf.append(temp_availability_mcf)
                    a_var_mcf.append(temp_var_mcf)
                    a_expect_loss_mcf.append(temp_expect_loss_mcf)
                else:
                    a_failure_obj_mcf.append(obj_mcf)

            else:
                a_failure_obj_rmcf.append(obj_rmcf)
            #直接计算mcf的相关值
            # if result_mcf == "Optimal":
            #     a_sus_obj_mcf.append(obj_mcf)
            #     # 计算可用性期望 mcf
            #     temp_availvbility_mcf,temp_var_mcf = util_tool.compute_scenes_availability(allot_list_mcf, graph1.new_route,
            #                                                                   graph1.new_scenes_list,
            #                                                                   graph1.new_scenes_prob,
            #                                                                   zeta)
            #     # temp_availvbility_mcf+=1-sum_scenes_prob
            #     a_availability_mcf.append(temp_availvbility_mcf)
            #     a_var_mcf.append(temp_var_mcf)
            # else:
            #     a_failure_obj_mcf.append(obj_mcf)
        all_result.append(a_sus_obj_mcf)
        # 计算平均目标值-rmcf
        # mean_obj_rmcf = np.mean(a_sus_obj_rmcf)
        all_exper_obj_result_rmcf.append(a_sus_obj_rmcf)
        all_exper_sus_num_rmcf.append(len(a_sus_obj_rmcf))
        # 计算pf概率下的可用性-rmcf
        all_availability_rmcf.append(a_availability_rmcf)
        all_var_rmcf.append(a_var_rmcf)
        # all_cost_rmcf=[all_var_rmcf[i]/sum_demand for i in range(len(all_var_rmcf))]
        all_expect_loss_rmcf.append(a_expect_loss_rmcf)

        # 计算平均目标值-mcf
        # mean_obj_mcf = np.mean(a_sus_obj_mcf)
        all_exper_obj_result_mcf.append(a_sus_obj_mcf)
        all_exper_sus_num_mcf.append(len(a_sus_obj_mcf))
        # 计算pf概率下的可用性-mcf
        all_availability_mcf.append(a_availability_mcf)
        all_var_mcf.append(a_var_mcf)
        # all_cost_mcf=[all_var_mcf[i]/sum_demand for i in range(len(all_var_mcf))]
        all_expect_loss_mcf.append(a_expect_loss_mcf)

        # 计算ecmp算法的可用性
        all_exper_obj_result_ecmp.append(a_sus_obj_ecmp)
        all_availability_ecmp.append(a_availability_ecmp)
        all_var_ecmp.append(a_var_ecmp)

        print("所有算法的var")
        print("rmcf", a_var_rmcf)
        print("mcf", a_var_mcf)
        print("ecmp", a_var_ecmp)


    mean_obj_rmcf=[np.mean(all_exper_obj_result_rmcf[i]) for i in range(len(all_exper_obj_result_rmcf))]
    mean_availability_rmcf=[np.mean(all_availability_rmcf[i]) for i in range(len(all_availability_rmcf))]
    mean_var_rmcf=[np.mean(all_var_rmcf[i]) for i in range(len(all_var_rmcf))]
    mean_var_demand_rmcf=[mean_var_rmcf[i]/sum_demand for i in range(len(mean_var_rmcf))]
    mean_expect_loss_rmcf=[np.mean(all_expect_loss_rmcf[i]) for i in range(len(all_expect_loss_rmcf))]

    print("链路概率同构场景")
    print("链路的失效概率", probability_edge)
    print("RMCF的所有结果:")
    print("所有试验目标值结果",mean_obj_rmcf )
    print("每个试验成功的次数", all_exper_sus_num_rmcf)
    print("所有的概率可用性",mean_availability_rmcf )
    print("所有概率的var",mean_var_rmcf)
    print("所有的var/demand",mean_var_demand_rmcf)
    print("所有概率的loss期望",mean_expect_loss_rmcf)

    mean_obj_mcf = [np.mean(all_exper_obj_result_mcf[i]) for i in range(len(all_exper_obj_result_mcf))]
    mean_availability_mcf = [np.mean(all_availability_mcf[i]) for i in range(len(all_availability_mcf))]
    mean_var_mcf = [np.mean(all_var_mcf[i]) for i in range(len(all_var_mcf))]
    mean_var_demand_mcf = [mean_var_mcf[i] / sum_demand for i in range(len(mean_var_mcf))]
    mean_expect_loss_mcf = [np.mean(all_expect_loss_mcf[i]) for i in range(len(all_expect_loss_mcf))]
    print("MCF的所有结果:")
    print("所有试验目标值结果", mean_obj_mcf)
    print("每个试验成功的次数", all_exper_sus_num_mcf)
    print("所有的概率可用性", mean_availability_mcf)
    print("所有概率的var",mean_var_mcf)
    print("所有的var/deamnd",mean_var_demand_mcf)
    print("所有概率的loss期望",mean_expect_loss_mcf)

    #保存所有ecmp算法的数据
    mean_obj_ecmp = [np.mean(all_exper_obj_result_ecmp[i]) for i in range(len(all_exper_obj_result_ecmp))]
    mean_availability_ecmp = [np.mean(all_availability_ecmp[i]) for i in range(len(all_availability_ecmp))]
    mean_var_ecmp = [np.mean(all_var_ecmp[i]) for i in range(len(all_var_ecmp))]
    mean_var_demand_ecmp=[mean_var_ecmp[i] / sum_demand for i in range(len(mean_var_ecmp))]
    print("ECMP")
    print("所有实验的目标值",mean_obj_ecmp)
    print("所有需求的可用性", mean_availability_ecmp)
    print("所有需求的var", mean_var_ecmp)
    print("所有的var/demand",mean_var_demand_ecmp)

    print(np.array(all_result))
    data_dict1 = {
        "parameter":parameter,
        "probability_edge": probability_edge,
         "all_availvbility_rmcf": all_availability_rmcf,
         "all_availability_mcf": all_availability_mcf,
         "all_availability_ecmp": all_availability_ecmp,
         "all_cost_rmcf": all_exper_obj_result_rmcf,
         "all_cost_mcf": all_exper_obj_result_mcf,
        "all_cost_ecmp":all_exper_obj_result_ecmp,
        "mean_var_demand_rmcf":mean_var_demand_rmcf,
        "mean_var_demand_mcf":mean_var_demand_mcf,
        "mean_var_demand_ecmp":mean_var_demand_ecmp
             }
    # 保存文件
    ds = DataSave(sys._getframe().f_code.co_name)
    ds.sava_all_data(data_dict1)
    print("运行完毕")

def experiment_gamma_influence():
    """
    实验，调和参数gamma对MCF_Cvar算法的影响
    拓扑：美国骨干网 node：11  edge：15
    :return:
    """
    #图的相关信息
    ad_matrix=np.zeros(shape=(11,11))
    link = [(1, 2), (2, 1),
            (1, 4), (4, 1),
            (2, 3), (3, 2),
            (2, 4), (4, 2),
            (3, 5), (5, 3),
            (4, 6), (6, 4),
            (5, 6), (6, 5),
            (5, 7), (7, 5),
            (6, 8), (8, 6),
            (7, 8), (8, 7),
            (7, 9), (9, 7),
            (8, 10), (10, 8),
            (9, 11), (11, 9),
            (10, 11), (11, 10)]
    for i in range(len(link)):
        ad_matrix[link[i][0] - 1][link[i][1] - 1] = 1
    source_nodes_1 = [3, 5, 11]
    dest_nodes_1 = [1, 8, 9]
    num_nodes = 11
    K = 3  # 每一个sd流中路径的数量
    fail_num = 2
    alpha = 0.95
    C = 0.4
    all_gamma=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    path_1 = [[6, 3],
              [6, 5, 8],
              [7, 12, 14, 8],

              [7, 12, 16],
              [7, 13, 18],
              [6, 5, 10, 16],

              [7, 13, 19],
              [7, 12, 16, 21, 19],
              [6, 5, 10, 15, 13, 19],

              [11, 6, 3],
              [11, 6, 5, 8],
              [12, 14, 8],

              [12, 16],
              [13, 16],
              [13, 19, 24, 28, 25],

              [13, 19],
              [12, 16, 21, 19],
              [12, 16, 22, 26, 27],

              [28, 25, 20, 14, 8],
              [28, 25, 20, 14, 9, 3],
              [27, 23, 17, 11, 6, 3],

              [28, 25],
              [27, 23, 18],
              [27, 23, 17, 12, 16],

              [27],
              [28, 25, 21, 19],
              [28, 25, 20, 15, 13, 19]
              ]
    experment_time=1
    sum_sink=30
    sum_demand=30
    parameter = {
        "fail_num": fail_num,
        "alpha": alpha,
        "sum_sink": sum_sink,
        "sum_demand": sum_demand,
        "C": C,
        "expermnet_time": experment_time
    }
    # rmcf算法结果统计
    all_exper_obj_result_rmcf = []  # rmcf，每一个参数对应的平均目标值
    all_exper_sus_num_rmcf = []  # rmcf，每一个参数下实验成功的次数
    all_availability_rmcf = []  # rmcf，每一个参数下的平均可用性
    all_var_compute_rmcf = []   #rmcf 最终计算的var
    all_var_experiment_rmcf = [] #rmcf  实验结果所得的var
    all_cost_experiment_rmcf = [] #RMCF 实验中算法得出的cost
    all_cvar_experiment_rmcf = [] #rmcf 实验中算法的出的cvar
    all_cvar_compute_rmcf=[]   #rmcf  最后计算所得的cvar
    all_allot_rmcf=[]   #所有分配方案

    # 获取需求矩阵
    all_graph_demand = util_tool.random_graph_demand(experment_time, num_nodes, sum_demand, sum_sink,
                                                     source_nodes_1,
                                                     dest_nodes_1)
    print("所有的需求列表")
    pprint(all_graph_demand)
    for a_gamma in all_gamma:
        print("可用性",a_gamma)
        a_exper_obj_result_rmcf = []  # rmcf，每一个参数对应的平均目标值
        a_exper_sus_num_rmcf = 0  # rmcf，每一个参数下实验成功的次数
        a_availability_rmcf = []  # rmcf，每一个参数下的平均可用性
        a_var_compute_rmcf = []  # rmcf 最终计算的var
        a_var_experiment_rmcf = []  # rmcf  实验结果所得的var
        a_cost_experiment_rmcf = []  # RMCF 实验中算法得出的cost
        a_cvar_experiment_rmcf = []  # rmcf 实验中算法的出的cvar
        a_cvar_compute_rmcf = []  # rmcf  最后计算所得的cvar
        a_allot_rmcf=[]   #rmcf  单个实验的所有分配方案

        graph1=NetworkGraph(ad_matrix,link)
        fail_list=[0.1]*28
        loss_list=[0.05]*28
        graph1.set_graph_p_isomerism(fail_list)
        graph1.set_graph_loss_isomerism(loss_list)
        print(all_graph_demand)
        # 设置源节点和目的结点
        graph1.set_source_dest_nodes(source_nodes_1, dest_nodes_1)
        # 设置路由
        graph1.set_route(K, path_1)
        # 设置链路cost
        cost_list = [1]*28 #l1和l4cost比较高
        graph1.set_graph_c_isomerism(cost_list)

        # 生成所有场景
        graph1.generate_scenes(fail_num)
        sum_scenes_prob = sum(graph1.scenes_prob)
        for e in range(experment_time):
            print("第{0}次实验".format(e))
            graph1.set_demand(all_graph_demand[e])
            # 判断是否需要增加结点
            graph1.change_topo(sum_demand, sum_sink)
            # rmcf_plus算法
            result_rmcf, obj_rmcf, allot_list_rmcf, zeta, cost_exper_rmcf, cvar_exper_rmcf = offload_strategy.rmcf_lp_loss_plus(graph1.new_graph_adj,
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
                                                                                                   a_gamma,
                                                                                                   alpha,
                                                                                                   C,
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
                zeta, alpha)
            print("分配方式", allot_list_rmcf)
            print("var", temp_var_rmcf)
            print("gamma", a_gamma)
            print("cost_exper_rmcf", cost_exper_rmcf)
            print("cvar_exper_rmcf", cvar_exper_rmcf)

            a_exper_obj_result_rmcf.append(obj_rmcf)
            a_exper_sus_num_rmcf+=1
            a_availability_rmcf.append(temp_availability_rmcf)
            a_var_compute_rmcf.append(temp_var_rmcf)
            a_var_experiment_rmcf.append(zeta)
            a_cost_experiment_rmcf.append(cost_exper_rmcf)
            a_cvar_experiment_rmcf.append(cvar_exper_rmcf)
            a_cvar_compute_rmcf.append(temp_cvar_rmcf)
            a_allot_rmcf.append(allot_list_rmcf)

            print("rmcf__________________________________end")

        # rmcf算法结果统计
        all_exper_obj_result_rmcf.append(a_exper_obj_result_rmcf)  # rmcf，每一个参数对应的平均目标值
        all_exper_sus_num_rmcf.append(a_exper_sus_num_rmcf)  # rmcf，每一个参数下实验成功的次数
        all_availability_rmcf.append(a_availability_rmcf)  # rmcf，每一个参数下的平均可用性
        all_var_compute_rmcf.append(a_var_compute_rmcf)  # rmcf 最终计算的var
        all_var_experiment_rmcf.append(a_var_experiment_rmcf)  # rmcf  实验结果所得的var
        all_cost_experiment_rmcf.append(a_cost_experiment_rmcf)  # RMCF 实验中算法得出的cost
        all_cvar_experiment_rmcf.append(a_cvar_experiment_rmcf)  # rmcf 实验中算法的出的cvar
        all_cvar_compute_rmcf.append(a_cvar_compute_rmcf)  # rmcf  最后计算所得的cvar
        all_allot_rmcf.append(a_allot_rmcf)

    mean_obj_rmcf = [np.mean(all_exper_obj_result_rmcf[i]) for i in range(len(all_exper_obj_result_rmcf))]
    mean_availability_rmcf = [np.mean(all_availability_rmcf[i]) for i in range(len(all_availability_rmcf))]
    mean_var_compute_rmcf=[np.mean(all_var_compute_rmcf[i]) for i in range(len(all_var_compute_rmcf))]
    mean_var_exper_rmcf=[np.mean(all_var_experiment_rmcf[i]) for i in range(len(all_var_experiment_rmcf))]
    mean_cost_exper_rmcf=[all_cost_experiment_rmcf[i] for i in range(len(all_cost_experiment_rmcf))]
    mean_cvar_exper_rmcf=[all_cvar_experiment_rmcf[i] for i in range(len(all_cvar_experiment_rmcf))]
    mean_cvar_compute_rmcf=[all_cvar_compute_rmcf[i] for i in range(len(all_cvar_compute_rmcf))]

    print("链路概率同构场景")
    print("链路的可用性", list(map(lambda x: str(int(x * 100)) + '%', [alpha])))
    print("RMCF的所有结果:")
    print("所有分配方案：")
    # pprint(all_allot_rmcf)
    print("所有试验目标值结果",mean_obj_rmcf)
    print("每个试验成功的次数", all_exper_sus_num_rmcf)
    print("gamma", all_gamma)
    print("所有的概率可用性", mean_availability_rmcf)
    print("所有实验中的cost", mean_cost_exper_rmcf)
    print("所有实验中的cvar", mean_cvar_exper_rmcf)
    print("所有真实的cvar",mean_cvar_compute_rmcf)
    print("所有实验var", mean_var_exper_rmcf)
    print("所有真实var", mean_var_compute_rmcf)


if __name__ == '__main__':
    # experiment_prob_availability_demo()
    # experiment_prob_availability()
    # demo_alpha_influence()
    # test_cvxpy_demo()
    experiment_gamma_influence()
