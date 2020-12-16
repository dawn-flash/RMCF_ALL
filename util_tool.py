#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/9/24 15:13
# @Author  : Lelsey
# @File    : util_tool.py

#工具类

import numpy as np
import copy
import os
import json
import random
from pprint import pprint
import matplotlib.pyplot as plt
from scipy import stats
import itertools
import math

def Floyd(dis):
    # min (Dis(i,j) , Dis(i,k) + Dis(k,j) )
    nums_vertex = len(dis[0])
    for k in range(nums_vertex):
        for i in range(nums_vertex):
            for j in range(nums_vertex):
                if dis[i][j] > dis[i][k] + dis[k][j]:
                    dis[i][j] = dis[i][k] + dis[k][j]
    return dis
def generate_random_number(sum_demand,count):
    """
    随机生成随机数
    :param sum_demand: 随机数的总和
    :param count: 随机数的数量
    :return:
    """
    #只生成一个随机数时直接返回这个值即可
    if count==1:
        result=[sum_demand]
        return result

    #通过生成随机间隔从而计算区间最后得到所有的随机数
    # split_num=np.random.randint(0,sum_demand,size=count-1)   #(0,sum)之间的随机数  随机数可能重复
    split_num=random.sample(range(1,sum_demand),count-1)       #[1,sum-1]之间随机数  保证随机数不重复 且随机数不会在边界上
    split_num.sort()
    # print(split_num)
    result=[]
    for i in range(len(split_num)):
        if i==0:
            temp=split_num[i]-0
            result.append(temp)
        else:
            temp=split_num[i]-split_num[i-1]
            result.append(temp)
    temp=sum_demand-split_num[-1]
    result.append(temp)
    # print(result)
    return result
def test_generate_random():
    # sum_demand=10
    # count=3
    # split_num = random.sample(range(1, sum_demand), count - 1)  #产生随机数
    # print(split_num)
    generate_random_number(10,3)
    pass
def createDemand(num_node,task_sum,num_demand):
    """
    产生一系列的需求数组
    :param num_node: 结点的数量
    :param task_sum: 任务的总量
    :param num_demand: 需求的数量
    :return: 需求向量
    """
    all_task_allot = []
    for i in range(num_demand):
        a_task_allot=[]
        a_task_allot = generate_random_number(task_sum,num_node)
        all_task_allot.append(a_task_allot)
    print("all_task_allot")
    print("num_node{0},task_num{1},num_demand{2}".format(num_node,task_sum,num_demand))
    # for i in all_task_allot:
    #     print(i)

    return all_task_allot
def saveDemand():
    num_node=10
    task_sum=10
    num_demand=10

    all_task_allot=createDemand(num_node,task_sum,num_demand)
    json_dict={}
    json_dict["all_task_allot"]=np.array(all_task_allot).tolist()

    filepath=os.path.join(os.getcwd(),"data","demand.json")
    with open(filepath,"w")as fw:
        json.dump(json_dict,fw,indent=4)
    print("demand写入文件完毕",filepath)


def createScenarios(num_nodes,cutoff):
    """
    #喜欢建所有的场景，和计算所有场景的概率
    :param num_nodes: 结点的数量
    :return: 所有的场景，所有场景概率
    """
    all_scenarios = []      #所有的场景
    all_scenarios_prob=[]   #所有场景的概率
    scenerios(num_nodes,[],all_scenarios)
    probilities=weibull(num_nodes) #威布尔分布产生的所有概率
    # print("probiities",probilities)
    #为所有的场景赋概率值
    for i in range(len(all_scenarios)):
        a_scenarios_prob=1
        for j in range(len(all_scenarios[i])):
            if all_scenarios[i][j]==0:
                a_scenarios_prob*=probilities[j]
            else:
                a_scenarios_prob*=1-probilities[j]
        all_scenarios_prob.append(a_scenarios_prob)
    # print(all_scenarios_prob)

    #对所有的场景排序概率从大到小
    all_scenarios_index_prob=[]
    for i in range(len(all_scenarios_prob)):
        all_scenarios_index_prob.append((i,all_scenarios_prob[i]))
    all_scenarios_index_prob=sorted(all_scenarios_index_prob,key=lambda x:x[1],reverse=True)

    #保存概率较大的前cutoff场景,保存到all_scenarios_cutoff【【场景，概率】...】
    all_scenarios_cutoff=[]
    all_scenarios_index_prob_cutoff=all_scenarios_index_prob[0:cutoff]
    for i in range(cutoff):
        a_scenario_prob=[]
        a_scenario_prob.append(all_scenarios[all_scenarios_index_prob_cutoff[i][0]])
        a_scenario_prob.append(all_scenarios_index_prob_cutoff[i][1])
        all_scenarios_cutoff.append(a_scenario_prob)
    # print(all_scenarios_index_prob_cutoff)

    return all_scenarios_cutoff


def saveScenarios():
    # 建立所有场景和所有场景的概率
    all_scenarios_probs = []
    experment_time=50
    num_nodes=10
    cutoff=100
    data = {}
    data["all_scenarios_probs"]={}
    for i in range(experment_time):
        scenarios_prons= createScenarios(num_nodes,cutoff)
        all_scenarios_probs .append(scenarios_prons)
        data["all_scenarios_probs"][str(i)]=scenarios_prons
    #写入文件
    print("写入文件")
    file_path=os.path.join(os.getcwd(),"data","scenarios.json")
    with open(file_path,"w")as fw:
        json.dump(data,fw,indent=4)
    print("写入文件完毕",file_path)

def scenerios(num,a_sce:list,all_sce):
    """
    递归实现所有的场景
    :param num: 结点数量
    :param a_sce: 单个场景向量
    :param all_sce: 所有场景向量
    :return:
    """
    if num==0:
        all_sce.append(a_sce)
        # a_sce.pop()
        return
    a_sce.append(0)
    scenerios(num-1,copy.deepcopy(a_sce),all_sce)
    a_sce.pop()
    a_sce.append(1)
    scenerios(num-1,copy.deepcopy(a_sce),all_sce)
    a_sce.pop()

def weibull(num,shape=0.8,scale=0.001):
    """
    #利用shape形状参数和scale范围参数参数 残生威布尔分布
    :param num: 数量
    :param shape:  形状参数
    :param scale:   范围参数
    :return:
    """
    return scale*np.random.weibull(shape,num)


def calcu_availability(demand,allot_strategy,scenarios):
    """
    计算分配后的可用性；\sigma(场景概率*利用率）
    :param demand: 任务需求
    :param allot_strategy: 任务分配矩阵
    :param scenarios: 所有场景和对应概率
    :return: 任务分配下的可用性
    """
    #获取所有的场景
    scenarios_10=[]
    scenarios_prob=[]
    for i in range(len(scenarios)):
        scenarios_10.append(scenarios[i][0])
        scenarios_prob.append(scenarios[i][1])

    satisfiability_result=np.dot(np.array(allot_strategy),np.array(scenarios_10).T)  #计算（d,s）矩阵每一列向量是场景下总实现需求

    #计算需求场景矩阵中每一个需求完成的百分比
    satisfiability_result_percent=np.zeros(shape=satisfiability_result.shape)
    for i in range(satisfiability_result.shape[1]):
        for d in range(len(demand)):
            if demand[d]==0:
                satisfiability_result_percent[d][i]=1
            else:
                satisfiability_result_percent[d][i]=satisfiability_result[d][i]/demand[d]

    #计算所有场景的可用性，存在avaibility_list列表中
    satisfiability_result_percent_sumcol=np.sum(satisfiability_result_percent,axis=0)
    avaibility_list=np.array(scenarios_prob)/len(demand)*satisfiability_result_percent_sumcol
    #所有场景加权和
    avaibilty_val=np.sum(avaibility_list)
    return avaibilty_val


def calcu_availability_travar(demand,allot_strategy,scenarios):
    """
    teavar算法中对所有场景的可用性的算法，\sigma（场景概率）,其中的场景是所有需求都同时满足的场景
    :param demand:
    :param allot_strategy:
    :param scenarios:
    :return:
    """
    scenarios_list=[]
    scenarios_prob=[]
    for i in range(len(scenarios)):
        scenarios_list.append(scenarios[i][0])
        scenarios_prob.append(scenarios[i][1])
    satisfiability_result = np.dot(np.array(allot_strategy), np.array(scenarios_list).T)  # 计算（d,s）矩阵每一列向量是场景下总实现需求

    satisfiability_result_percent = np.zeros(shape=satisfiability_result.shape)
    for i in range(satisfiability_result.shape[1]):
        for d in range(len(demand)):
            if demand[d] == 0:
                satisfiability_result_percent[d][i] = 1
            else:
                satisfiability_result_percent[d][i] = satisfiability_result[d][i] / demand[d]

    satisfiability_scenarios_list=np.sum(satisfiability_result_percent,axis=0)/len(demand)

    finally_availabillity=0
    for i in range(len(satisfiability_scenarios_list)):
        if satisfiability_scenarios_list[i]==1:
            finally_availabillity+=scenarios_prob[i]

    return finally_availabillity

def judge_allot_rationality(demand,allot_strategy,capacity):
    """根据需求，分配策略和结点的容量判断分配是否合理"""
    #计算所有需求满足总量和比例
    satisfied_demands=[]

    for  i in range(len(allot_strategy)):
        satisfied_demand_a=sum(allot_strategy[i])
        satisfied_demands.append(satisfied_demand_a)
    print("需求满足的量",satisfied_demands)

    proportion_demands=[]
    for i in range(len(demand)):
        if demand[i]==0:
            proportion_demands.append(0)
        else:
            proportion_demands.append(satisfied_demands[i]/demand[i])
    print("需求满足的比例",proportion_demands)

    real_node_capacities=np.sum(np.array(allot_strategy),axis=0).tolist()
    proportion_node_capacities=[real_node_capacities[i]/capacity[i] for i in range(len(capacity))]
    print("结点分配的量",real_node_capacities)
    print("结点分配的比例",proportion_node_capacities)
    pass


def pre_teavar(demands,capaciies,R):
    """
    对teavar算法的输入进行预处理，返回预处理后真实的需求和剩余容量,已经分配的容量，和分配选择
    对结点进行预分配后，真实需求和剩余容量和已经分配的容量都会变化
    分配集合的所有元素必须加1
    :param demands:
    :param capaciies:
    :param R:
    :return:
    """
    real_demands=[]
    real_remain_capacities=[]
    real_alloted_capacities=[]
    for i in range(len(demands)):
        a_demand=demands[i]-capaciies[i]
        if a_demand<=0:
            real_demands.append(0)
            a_capacity=-a_demand
            real_remain_capacities.append(a_capacity)
            real_alloted_capacities.append(demands[i])
        else:
            real_demands.append(a_demand)
            real_remain_capacities.append(0)
            real_alloted_capacities.append(capaciies[i])
    # print(real_demands)
    # print(real_remain_capacities)
    # print(real_alloted_capacities)

    #分配矩阵的所有元素加1，结点是从1开始
    teavar_R=copy.deepcopy(R)
    for i in range(len(R)):
        for j in range(len(R[i])):
            teavar_R[i][j]=R[i][j]+1

    # print(teavar_R)
    return real_demands,real_remain_capacities,real_alloted_capacities,teavar_R
def delete_demand_0(teavar_demand,teavar_flows,teavar_Tf):
    """删除需求为0的情况，以免teavar算法失效"""
    new_teavar_demand=copy.deepcopy(teavar_demand)
    new_teavar_flows=copy.deepcopy(teavar_flows)
    new_teavar_Tf=copy.deepcopy(teavar_Tf)
    for i in range(len(teavar_demand)-1,-1,-1):
        if new_teavar_demand[i]==0:
            new_teavar_demand.pop(i)
            new_teavar_flows.pop(i)
            new_teavar_Tf.pop(i)
    print(new_teavar_demand)
    print(new_teavar_flows)
    print(new_teavar_Tf)
    return new_teavar_demand,new_teavar_flows,new_teavar_Tf
def test_delete():
    """函数测试"""
    a=[1,2,3,0,4]
    b=[1,2,3,4,5]
    c=[[1,2],[3,4],[2,3],[4,5],[0,4]]
    delete_demand_0(a,b,c)

def post_travar(alpha,teavar_allot,new_Tf,new_demand,new_flows,real_alloted_capacities):
    """
    对travar算法得出的分配方案进行后置处理
    :param alpha:
    :param teavar_allot: teavar算法计算出的分配策略
    :param new_Tf: 简化后的分配集合
    :param new_demand: 简化后的需求列表
    :param new_flows:  简化后的需求编号
    :param real_alloted_capacities:  已经分配的资源列表
    :return:
    """
    all_allot=np.zeros((len(real_alloted_capacities),len(real_alloted_capacities)))
    for d in range(1,len(real_alloted_capacities)+1):
        if d in new_flows:
            d_index=new_flows.index(d)
            sum_demand=sum(teavar_allot[d_index])
            for v in range(1,len(real_alloted_capacities)+1):
                if v in new_Tf[d_index]:
                    v_index=new_Tf[d_index].index(v)
                    if sum_demand==0:
                        all_allot[d-1,v-1]=0
                    else:
                        all_allot[d-1,v-1]=new_demand[d_index]*(1-alpha)*teavar_allot[d_index][v_index]/sum_demand
        else:
            for v in range(1,len(real_alloted_capacities)+1):
                all_allot[d-1,v-1]=0
    #分配方案加上原有已分配的资源
    for d in range(len(real_alloted_capacities)):
        all_allot[d,d]+=real_alloted_capacities[d]
    print(all_allot)

    return  all_allot.tolist()
def test_post_teavar():
    alpha=0
    new_flows = [2,3,4,5]
    teavar_allot=[[1,1],[1,1],[1,1],[1,1]]
    new_Tf=[[1,2],[2,3],[3,4],[1,5]]
    new_demand=[1,2,2,2]
    real_allot_capacities=[0,0,0,0,0]
    allot=post_travar(alpha,teavar_allot,new_Tf,new_demand,new_flows,real_allot_capacities)
    print(allot)
# def post_travar(alpha,Tf,teavar_allot,demand,real_alloted_capacities):
#     """
#
#     :param alpha:
#     :param Tf:
#     :param teavar_allot:
#     :param demand:
#     :param real_alloted_capacities:
#     :return:
#     """
#     all_allot=[]
#     for d in range(len(Tf)):
#         a_allot=[]
#         sum_demand=sum(teavar_allot[d])
#         for v in range(1,len(real_alloted_capacities)+1):
#             if v in Tf[d]:
#                 v_index=Tf[d].index(v)
#                 a_allot.append(demand[d]*(1-alpha)*teavar_allot[d][v_index]/sum_demand)
#             else:
#                 a_allot.append(0)
#         all_allot.append(a_allot)
#
#     for d in range(len(real_alloted_capacities)):
#         all_allot[d][d]+=real_alloted_capacities[d]
#     #输出最后的分配结果
#     # for i in range(len(all_allot)):
#     #     print(all_allot[i])
#     return all_allot


# def test_post_reavar():
#     alpha=0.2
#     demand = [2, 1, 3, 3, 1]
#     Tf=[[2,3] ,[3,4] ,[1,2] ,[1,5], [2,4]]
#     teavar_allot=[[0,2],[0,1],[1,2],[1,2],[0,1]]
#     rea_c=[0,0,0,0,0]
#     post_travar(alpha,Tf,teavar_allot,demand,rea_c)


def computer_availability(allot_matrix,var,scenes,S,scenes_prob):
    """
    基于链路的RMCF 算法相关计算方法
    计算算法的可用性
    方法：（1）对所有的场景按照场景概率排序，
    （2）根据var的概率密度图，对loss小于var的场景概率进行累加操作，求出alpha。
    :param allot_matrix: list 分配矩阵
    :param var:  float 条件风险值
    :param scenes:  list 所有场景
    :param S:  int 场景的数量
    :param scenes_prob:  list 所有场景的list
    :return:  可用性alpha
    """

    #计算所有场景的loss
    scenes_loss=[]
    for s in range(S):
        temp_loss=np.sum(np.array(allot_matrix)*np.array((scenes)[s]))
        scenes_loss.append(temp_loss)

    #根据loss对所有场景和场景概率进行排序
    scenes_loss_sorted = []
    scenes_prob_sorted = []
    temp_loss_prob=list(copy.deepcopy(scenes_prob))
    for i in range(S):
        s_index=scenes_loss.index(min(scenes_loss))
        scenes_loss_sorted.append(scenes_loss[s_index])
        scenes_prob_sorted.append(scenes_prob[s_index])
        scenes_loss.pop(s_index)
        temp_loss_prob.pop(s_index)

    #根据var进行alpha的计算
    alpha=0
    for s in range(S):
        s_loss=scenes_loss_sorted[s]
        if s_loss>var:
            break
        alpha+=scenes_prob_sorted[s]

    return alpha

def expect_loss(allot_flows,route:np.ndarray,scenes_list,scenes_prob):
    """
    基于flow的算法的相关计算方法，场景，链路状态【0,1】正常，拥塞
    计算期望的loss
    :param allot_flows:
    :param route:
    :param scenes_list:
    :param scenes_prob:
    :return:
    """
    flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]  # 将矩阵或者多维列表扁平化
    s_count = 0
    S = len(scenes_list)
    all_loss = []  # 所有场景的loss
    for s in range(S):
        new_route = copy.deepcopy(route)
        failure_flow = []
        for i in range(len(scenes_list[s])):
            if scenes_list[s][i] == 1:
                for j in range(new_route.shape[0]):
                    if new_route[j][i] == 1:
                        # failure_flow.append((j, i))  #可能有错
                        failure_flow.append(j)
        # print("S{0}".format(s)+"failure flow",failure_flow)
        failure_flow = list(set(failure_flow))
        if len(failure_flow) > 0:
            # print("场景",scenes_list[s])
            s_count += 1
            a_loss = sum([flatten(allot_flows)[i] for i in failure_flow])
            all_loss.append(a_loss)
        else:
            # print("删减的场景",scenes_list[s])
            a_loss = 0
            all_loss.append(a_loss)

    # 根据loss对所有场景和场景概率进行排序
    scenes_loss_sorted = []
    scenes_prob_sorted = []
    temp_all_loss = copy.deepcopy(all_loss)
    temp_loss_prob = list(copy.deepcopy(scenes_prob))
    for i in range(S):
        s_index = temp_all_loss.index(min(temp_all_loss))
        scenes_loss_sorted.append(temp_all_loss[s_index])
        scenes_prob_sorted.append(temp_loss_prob[s_index])
        temp_all_loss.pop(s_index)
        temp_loss_prob.pop(s_index)

    statistic_loss_sorted=sorted(list(set(scenes_loss_sorted)))
    statistic_prob_sorted=[0]*len(statistic_loss_sorted)
    for s in range(S):
        s_index=statistic_loss_sorted.index(scenes_loss_sorted[s])
        statistic_prob_sorted[s_index]+=scenes_prob_sorted[s]
    print("统计的loss和概率")
    print(statistic_loss_sorted)
    print(statistic_prob_sorted)
    #画直方图
    # plot_histtograph(all_loss,scenes_prob)

    result_expect_loss=np.dot(np.array(all_loss),np.array(scenes_prob))

    return result_expect_loss


def compute_scenes_availability(allot_flows,route:np.ndarray,scenes_list,scenes_prob,var,expect_alpha):
    """
    基于flow的算法计算，链路状态有【0,1】正常，拥塞
    计算当前分配方案的可用性
    和当前分配方案的var
    :param allot_flows: list 分配方案
    :param route: array  路由矩阵
    :param scenes_list:   list 场景列表
    :param scenes_prob:   list  场景矩阵
    :param var: float  风险值
    :return: alpha 可用性
    """
    #计算所有场景的loss

    flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]  # 将矩阵或者多维列表扁平化
    s_count = 0
    S=len(scenes_list)
    all_loss=[]  #所有场景的loss
    for s in range(S):
        new_route = copy.deepcopy(route)
        failure_flow = []
        for i in range(len(scenes_list[s])):
            if scenes_list[s][i] == 1:
                for j in range(new_route.shape[0]):
                    if new_route[j][i] == 1:
                        # failure_flow.append((j, i))  #可能有错
                        failure_flow.append(j)
        # print("S{0}".format(s)+"failure flow",failure_flow)
        failure_flow = list(set(failure_flow))
        if len(failure_flow) > 0:
            # print("场景",scenes_list[s])
            s_count += 1
            # print(failure_flow)
            a_loss=sum([flatten(allot_flows)[i] for i in failure_flow])
            all_loss.append(a_loss)
        else:
            # print("删减的场景",scenes_list[s])
            a_loss=0
            all_loss.append(a_loss)

    #根据loss对所有场景和场景概率进行排序
    scenes_loss_sorted=[]
    scenes_prob_sorted=[]
    temp_all_loss=copy.deepcopy(all_loss)
    temp_loss_prob = list(copy.deepcopy(scenes_prob))
    for i in range(S):
        s_index = temp_all_loss.index(min(temp_all_loss))
        scenes_loss_sorted.append(temp_all_loss[s_index])
        scenes_prob_sorted.append(temp_loss_prob[s_index])
        temp_all_loss.pop(s_index)
        temp_loss_prob.pop(s_index)

    # 画直方图
    # plot_histtograph(scenes_loss_sorted,scenes_prob_sorted)
    #根据var进行alpha的计算
    alpha = 0
    for s in range(S):
        s_loss = scenes_loss_sorted[s]
        if s_loss > var:
            break
        alpha += scenes_prob_sorted[s]
    print("总概率：", sum(scenes_prob))
    print("可用性",alpha)

    # 根据alpha进行var的计算
    zeta = 0
    totol_alpha = 0
    sum_scenes_prob = sum(scenes_prob)
    for s in range(S):
        totol_alpha += scenes_prob_sorted[s]
        if totol_alpha > sum_scenes_prob*expect_alpha:   #当前概率和的一个百分比
        # if totol_alpha > alpha:   #直接使用计算的可用性
        # if totol_alpha > expect_alpha:   #直接设置为一个值
            break
        zeta = scenes_loss_sorted[s]

    #根据alpha进行CVAR的计算
    zeta=0
    totol_alpha=0   #小于var的概率
    totol_prob=0    #大于var的概率
    cvar_loss=0
    for s in range(S):
        totol_alpha+=scenes_prob_sorted[s]
        if totol_alpha>=expect_alpha:
            cvar_loss+=scenes_loss_sorted[s]*scenes_prob_sorted[s]
            totol_prob+=scenes_prob_sorted[s]
    cvar=cvar_loss/totol_prob
    print("cvar",cvar)
    # , cvar
    return alpha,zeta

def new_compute_scenes_availability(allot_flows,route:np.ndarray,scenes_list,scenes_prob,
                                    flow_cost,graph_loss_list,var,expect_alpha):
    """
    目标函数为联合cost和cvar:链路正常时会有丢包的情况
    基于flow的算法计算，链路状态有【0,1]正常，拥塞
    计算当前分配方案的可用性
    和当前分配方案的var
    :param allot_flows: list [S,D,K] 分配方案
    :param route: array  路由矩阵
    :param scenes_list:   list 场景列表
    :param scenes_prob:   list  场景矩阵
    :param flow_cost: list[f] 存放每一条流，单位任务量的cost
    :param graph_loss_list list[l] 存放每一条链路的丢包率
    :param var: float  风险值
    :return: alpha 可用性
    """
    #计算所有场景的loss

    flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]  # 将矩阵或者多维列表扁平化
    s_count = 0
    S=len(scenes_list)
    all_loss=[]  #所有场景的loss
    flow_list=flatten(allot_flows)#存放每一条流分配的资源量
    for s in range(S):
        new_route=copy.deepcopy(route)
        all_flow_loss_cost=[]
        for f in range(new_route.shape[0]):
            a_flow_sus=flow_list[f]
            for l in range(new_route.shape[1]):
                if new_route[f][l]==1:
                    if scenes_list[s][l]==1:
                        a_flow_sus=0
                        break
                    # if scenes_list[s][l]==2:
                    else:
                        a_flow_sus=a_flow_sus*(1-graph_loss_list[l])
            a_flow_loss=flow_list[f]-a_flow_sus
            a_flow_loss_cost=a_flow_loss*flow_cost[f]
            all_flow_loss_cost.append(a_flow_loss_cost)
        s_loss=sum(all_flow_loss_cost)
        all_loss.append(s_loss)

    #根据loss从小到大
    # 对所有场景和场景概率进行排序
    scenes_loss_sorted=[]
    scenes_prob_sorted=[]
    temp_all_loss=copy.deepcopy(all_loss)
    temp_loss_prob = list(copy.deepcopy(scenes_prob))
    for i in range(S):
        s_index = temp_all_loss.index(min(temp_all_loss))
        scenes_loss_sorted.append(temp_all_loss[s_index])
        scenes_prob_sorted.append(temp_loss_prob[s_index])
        temp_all_loss.pop(s_index)
        temp_loss_prob.pop(s_index)

    # 画直方图
    # plot_histtograph(scenes_loss_sorted,scenes_prob_sorted)
    #根据var进行alpha的计算
    alpha = 0
    for s in range(S):
        s_loss = scenes_loss_sorted[s]
        if s_loss > var:
            break
        alpha += scenes_prob_sorted[s]
    print("总概率：", sum(scenes_prob))
    print("计算可用性",alpha)

    # 根据alpha进行var的计算
    ture_zeta = 0
    totol_alpha = 0
    sum_scenes_prob = sum(scenes_prob)
    true_alpha=sum_scenes_prob*expect_alpha
    for s in range(S):
        totol_alpha += scenes_prob_sorted[s]
        if totol_alpha > true_alpha:   #当前概率和的一个百分比
        # if totol_alpha > alpha:   #直接使用计算的可用性
        # if totol_alpha > expect_alpha:   #直接设置为一个值
            break
        ture_zeta = scenes_loss_sorted[s]

    #根据alpha进行CVAR的计算
    zeta=0
    totol_alpha=0   #小于var的概率
    totol_prob=0    #大于var的概率
    cvar_loss=0
    for s in range(S):
        totol_alpha+=scenes_prob_sorted[s]
        if totol_alpha>=true_alpha:
            cvar_loss+=scenes_loss_sorted[s]*scenes_prob_sorted[s]
            totol_prob+=scenes_prob_sorted[s]
    cvar=cvar_loss/totol_prob
    print("cvar",cvar)
    # , cvar
    return alpha,ture_zeta, cvar

def compute_scenes_var(allot_flows,route:np.ndarray,scenes_list,scenes_prob,alpha):
    """
    基于flow的RMCF算法，目标函数最小化cost
    根据alpha计算var
    :param allot_flows:
    :param route:
    :param scenes_list:
    :param scenes_prob:
    :param alpha:
    :return:
    """
    flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]  # 将矩阵或者多维列表扁平化

    s_count = 0
    S = len(scenes_list)
    all_loss = []  # 所有场景的loss
    for s in range(S):
        new_route = copy.deepcopy(route)
        failure_flow = []
        for i in range(len(scenes_list[s])):
            if scenes_list[s][i] == 1:
                for j in range(new_route.shape[0]):
                    if new_route[j][i] == 1:
                        failure_flow.append((j, i))
        # print("S{0}".format(s)+"failure flow",failure_flow)
        failure_flow = list(set(failure_flow))
        if len(failure_flow) > 0:
            # print("场景",scenes_list[s])
            s_count += 1
            a_loss = sum([flatten(allot_flows)[i[0]] for i in failure_flow])
            all_loss.append(a_loss)
        else:
            # print("删减的场景",scenes_list[s])
            a_loss = 0
            all_loss.append(a_loss)
    # 根据loss对所有场景和场景概率进行排序
    scenes_loss_sorted = []
    scenes_prob_sorted = []
    temp_all_loss = copy.deepcopy(all_loss)
    temp_loss_prob = list(copy.deepcopy(scenes_prob))
    for i in range(S):
        s_index = temp_all_loss.index(min(temp_all_loss))
        scenes_loss_sorted.append(temp_all_loss[s_index])
        scenes_prob_sorted.append(temp_loss_prob[s_index])
        temp_all_loss.pop(s_index)
        temp_loss_prob.pop(s_index)

    #根据alpha进行var的计算
    zeta=0
    totol_alpha=0

    sum_scenes_prob=sum(scenes_prob_sorted)
    for s in range(S):
        totol_alpha+=scenes_prob_sorted[s]
        zeta = scenes_loss_sorted[s]
        if totol_alpha>alpha*sum_scenes_prob:  #是否需要乘还需要讨论
            break

    return zeta

def plot_histtograph(all_loss,scenes_prob):
    print("画图开始")
    bins=plt.hist(all_loss,bins=100,weights=scenes_prob,facecolor="blue",edgecolor="black",alpha=0.7)
    # print("bins",list(bins[0]))
    # print("patch",list(bins[1]))
    # print("patch",patch)
    plt.xlabel("loss")
    plt.ylabel("frequency")
    plt.show()
    print("画图结束")


# 对于图的相关操作
def graph_cost_matrix_to_list(graph_adj:np.ndarray,graph_cost:np.ndarray):
    """
    将图的cost矩阵转换为cost列表
    :param graph_cost: array 图的cost矩阵
    :return:  图的cost列表
    """
    graph_cost_list=[]
    for i in range(graph_adj.shape[0]):
        for j in range(graph_adj.shape[1]):
            if graph_adj[i][j]>0:
                graph_cost_list.append(graph_cost[i][j])
    return graph_cost_list

def graph_p_matrix_to_list(graph_adj:np.ndarray,graph_p:np.ndarray):
    """
    将图的概率矩阵转换为变的概率列表
    :param graph_adj: array  图的邻接矩阵
    :param graph_p:   array  图的概率矩阵
    :return:
    """
    graph_p_list = []
    for i in range(graph_adj.shape[0]):
        for j in range(graph_adj.shape[1]):
            if graph_adj[i][j] > 0:
                graph_p_list.append(graph_p[i][j])
    return graph_p_list

def set_route(flows_num,links_num,path:list):
    """
    创建路由矩阵array[path,link] 第一维是路径，第二维度是链路
    :param flows_num:  int 流的数量
    :param links_num:  int 链路的数量
    :param path:   list[list]  路径列表 每一个元素是一个path，有链路编号组成[[6, 3],[6, 5, 8]]
    :return: 路由矩阵
    """
    route=np.zeros(shape=(flows_num,links_num))
    # 为路由矩阵赋值
    for p in range(len(path)):
        if len(path[p])>0:
            for pi in path[p]:
                route[p, pi - 1] = 1
    return  route

def scenes_to_scenes_list(graph_adj:np.ndarray,scenes:list):
    """
    根据图的邻接矩阵将所有的场景矩阵转换为场景list
    :param graph_adj: 邻接矩阵
    :param scenes: list[array]所有的场景矩阵
    :return:  所有场景的list
    """
    all_scenes_list=[]
    for s in range(len(scenes)):
        a_scenes_list=[]
        for i in range(graph_adj.shape[0]):
            for j in range(graph_adj.shape[1]):
                if graph_adj[i][j]==1:
                    if scenes[s][i][j]==1:
                        a_scenes_list.append(1)
                    else:
                        a_scenes_list.append(0)
        all_scenes_list.append(a_scenes_list)
    return all_scenes_list

def random_graph_demand(num_demand,graph_num_node,sum_demand,sum_sink,source_nodes,dest_nodes):
    """
    为拓扑图产生总量一定的随机需求
    :param num_demand:  int 需求的组数
    :param graph_num_node: int 结点的数量
    :param sum_demand:  int 发送需求的总和
    :param sum_sink:   int 接收需求的总和
    :param source_nodes:  list 源节点列表
    :param dest_nodes:  list 目的结点列表
    :return:  num_demand 组需求
    """
    random.seed(1)
    all_graph_demand=[]
    for i in range(num_demand):
        source_demand = generate_random_number(sum_demand, len(source_nodes))
        dest_sink = generate_random_number(sum_sink, len(dest_nodes))
        graph_demand = np.zeros(graph_num_node)
        for i in range(len(source_nodes)):
            graph_demand[source_nodes[i]-1]=source_demand[i]
        for i in range(len(dest_nodes)):
            graph_demand[dest_nodes[i]-1]=-dest_sink[i]
        graph_demand=list(graph_demand)
        all_graph_demand.append(graph_demand)

    # pprint(all_graph_demand)
    return all_graph_demand

def random_graph_demand_test():
    graph_num_node=11
    sum_demand=50
    sum_sink=50
    source_nodes=[3, 5, 11]
    dest_nodes=[1, 8, 9]
    num=10
    random_graph_demand(num,graph_num_node, sum_demand, sum_sink, source_nodes, dest_nodes)

def add_source_node_scenes(scenes,scenes_list,scenes_prob,dest_nodes,num_nodes):
    new_scenes = copy.deepcopy(scenes)
    new_scenes_list = copy.deepcopy(scenes_list)
    new_scenes_prob = copy.deepcopy(scenes_prob)
    new_num_nodes=num_nodes+1
    new_row = np.zeros(num_nodes).reshape(1, num_nodes)
    new_col = np.zeros(new_num_nodes).reshape(1, new_num_nodes)
    for s in range(len(new_scenes)):
        new_scenes[s] = np.r_[new_scenes[s], new_row]
        new_scenes[s] = np.c_[new_scenes[s], new_col.T]
        extend_list = [0] * len(dest_nodes)
        new_scenes_list[s].extend(extend_list)

    return new_scenes,new_scenes_list,new_scenes_prob

def add_dest_node_scenes(scenes,scenes_list,scenes_prob,source_nodes,num_nodes,graph_adj):
    new_scenes = copy.deepcopy(scenes)
    new_scenes_list = copy.deepcopy(scenes_list)
    new_scenes_prob = copy.deepcopy(scenes_prob)

    new_row = np.zeros(num_nodes).reshape(1, num_nodes)
    new_num_nodes=num_nodes+1
    new_col = np.zeros(new_num_nodes)
    for s in range(len(source_nodes)):
        new_col[source_nodes[s] - 1] = 1  # 设置概率=1
    new_col = new_col.reshape(1, new_num_nodes)
    for s in range(len(new_scenes)):
        new_scenes[s] = np.r_[new_scenes[s], new_row]
        new_scenes[s] = np.c_[new_scenes[s], new_col.T]
    # 将场景矩阵转换为场景列表
    snew_scenes_list = scenes_to_scenes_list(graph_adj, new_scenes)


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

def ecmp(allot_list_rmcf,source_nodes,dest_nodes,K,virtual_node=None):
    """
    根据rmcf产生的分配方案，将分配方案转换为ECMP算法，即SD总量不变，所有的路径均分资源量
    一般结点分配
    :param allot_list_rmcf: list rmcf的分配方案
    :param source_nodes:   list  源节点列表
    :param dest_nodes:  list 目的结点列表
    :param K： int 路径数量
    :param virtual_node 虚拟结点判断是否添加源节点或者目的结点，或者不添加结点
    :return: allot_list_ecmp(ecmp的算法分配结果)
    """
    allot_list_ecmp=[]
    len_source=len(source_nodes)
    len_dest=len(dest_nodes)
    for i in range(len_source):
        row_list=[]
        for j in range(len_dest):
            col_list=[]
            sum_SD=sum(allot_list_rmcf[i][j])
            a_path=sum_SD/K
            for k in range(K):
                col_list.append(a_path)
            row_list.append(col_list)
        allot_list_ecmp.append(row_list)

    # print("ecmp算法分配")
    # pprint(allot_list_ecmp)
    return allot_list_ecmp


def compute_cost_ecmp(allot_list_ecmp,path_link,graph_adj,graph_cost):
    flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]  # 将多维列表扁平化
    graph_cost_list = graph_cost_matrix_to_list(graph_adj, graph_cost)

    allot_list=flatten(allot_list_ecmp)
    all_cost=0
    for f in range(len(allot_list)):
        f_cost=0
        if len(path_link)>0:
            for l in range(len(path_link[f])):
                link_index=path_link[f][l]-1
                f_cost+=allot_list[f]*graph_cost_list[link_index]
        all_cost+=f_cost

    # print("所有的cost",all_cost)
    return  all_cost

    pass

def allot_list_to_allot_ratio(allot_list):
    """
    将分配list转换成比例list
    :param allot_list:  list(3,3,3) 分配list
    :return:
    """
    # pprint(allot_list)
    ratio_list=[]
    for s in range(len(allot_list)):
        sum_source=np.sum(np.array(allot_list[s]))
        s_list=[]
        for d in range(len(allot_list[s])):
            d_list=[]
            for k in range(len(allot_list[s][d])):
                temp_sdk=allot_list[s][d][k]/sum_source
                d_list.append(temp_sdk)
            s_list.append(d_list)
        ratio_list.append(s_list)

    # print("分配比例")
    # pprint(ratio_list)
    return ratio_list
def ratio_test():
    rmcf=[[[10.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [5.0, 2.7601494, 0.0], [2.2398506, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [2.2398506, 0.0, 0.0], [7.7601494, 0.0, 0.0]]]
    mcf=[[[10.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]]
    ecmp=[[[3.3333333333333335, 3.3333333333333335, 3.3333333333333335],
          [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0],
          [2.5867164666666667, 2.5867164666666667, 2.5867164666666667],
          [0.7466168666666667, 0.7466168666666667, 0.7466168666666667]],
         [[0.0, 0.0, 0.0],
          [0.7466168666666667, 0.7466168666666667, 0.7466168666666667],
          [2.5867164666666667, 2.5867164666666667, 2.5867164666666667]]]
    print("rmcf")
    allot_list_to_allot_ratio(rmcf)
    print("mcf")
    allot_list_to_allot_ratio(mcf)
    print("ecmp")
    allot_list_to_allot_ratio(ecmp)

def compute_path_prob(path_link,link_prob):
    """
    计算链路的失效概率，仅限于所有链路概率相同情况
    :param path_link:  list[list] 路径中链路的列表
    :param link_prob:  float 链路失效概率
    :return:
    """
    all_path_prob=[]
    for p in range(len(path_link)):
        link_num=len(path_link[p])
        p_coin=stats.binom.pmf(np.arange(1,link_num+1),link_num,link_prob) #(成功列表，实验次数，成功概率)

        a_path_prob =sum(p_coin[0:2])
        all_path_prob.append(a_path_prob)

    print("所有路径的概率")
    all_path_prob=[round(all_path_prob[i],2) for i in range(len(all_path_prob))]

    for i in range(3):
        print(all_path_prob[i*9:i*9+9])
    # print(all_path_prob)
def compute_path_prob_test():
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
    link_prob=0.03
    compute_path_prob(path_1,link_prob)

def binomial_distribution():
    """
    测试伯努利分布
    :return:
    """
    n=4   #实验次数
    k=np.arange(0,n+1)  #所有情况
    p=0.02     #成功概率
    pcoin=stats.binom.pmf(k,n,p)
    print(pcoin)

    p_sum=sum(pcoin)
    print("成功的概率和",p_sum)


def compute_link_state_prob(graph_adj,graph_p_fail,graph_p_loss):
    """
    计算每一条链路的各种状态的概率
    【0,1,2】【链路正常且不丢包，链路失效，链路正常但丢包】
    :param graph_adj: array 图的邻接矩阵
    :param graph_p_fail: array 图的邻接矩阵失效概率
    :param graph_p_loss: array 图的邻接矩阵 丢包概率
    :return:  链路状态概率列表[(s0,s1,s2)...]
    """
    prob_list=[] #存放所有链路的状态概率
    prob_index=[] #存放所有链路在邻接矩阵中的下标
    for i in range(graph_adj.shape[0]):
        for j in range(graph_adj.shape[1]):
            if graph_adj[i][j]==1:
                s_0=(1-graph_p_fail[i][j])*(1-graph_p_loss[i][j])
                s_1=graph_p_fail[i][j]
                s_2=(1-graph_p_fail[i][j])*graph_p_loss[i][j]
                temp_link_state = (s_0,s_1,s_2)
                prob_list.append(temp_link_state)
                prob_index.append((i,j))

    return prob_list,prob_index

def compute_link_loss_value(graph_adj,graph_loss):
    pass

def compute_scenes_fail_num(scene:tuple):
    """
    计算单个场景中失效链路的数量
    :param scene: tuple 场景元组
    :return: 失效链路的个数
    """
    fail_num=0
    for i in range(len(scene)):
        if scene[i]!=0:
            fail_num+=1

    return fail_num


#**********************************************
#************项目迁移后新的工具函数****************
#**********************************************
def delete_scenes_1(graph_p:np.ndarray,failure_num):
    """
    根据概率矩阵和失效链路的个数，筛选出所有的场景，该场景符合所有的链路失效数量<failure_num
    :param graph_p: 图的概率矩阵
    :param failure_num: 失效链路的最大个数
    :return: 所有的场景 和所有场景的概率
    """
    #将概率矩阵中所有有效概率放入概率列表中
    prob_list=[]
    prob_index=[]
    for i in range(graph_p.shape[0]):
        for j in range(graph_p.shape[1]):
            if graph_p[i][j]!=0:
                prob_list.append(graph_p[i][j])
                prob_index.append((i,j))
    node_num=graph_p.shape[0]

    #存放所有组合的排列(普通方法)
    all_zuhe=[]
    all_zuhe.append([0.0]*len(prob_list)) #存放c(n,0)
    all_index=[i for i in range(len(prob_list))]
    for i in range(1,failure_num+1):
        temp_list=list(itertools.combinations(all_index,i))
        i_scenes_list=[]
        for a_tuple in range(len(temp_list)):
            a_scenes=np.zeros(len(prob_list))
            for k in temp_list[a_tuple]:
                a_scenes[k]=1
            i_scenes_list.append(list(a_scenes))
        all_zuhe.extend(i_scenes_list)

    #存放所有组合的排列(使用包）
    # all_zuhe = list(itertools.product([0, 1], repeat=len(prob_list)))  #list[tuple]

    #将场景列表变为场景矩阵
    all_scenes=[] #所有的场景
    all_scenes_prob=[]  #所有场景的概率
    for s in range(np.array(all_zuhe).shape[0]):
        a_scenes=np.zeros(shape=(node_num,node_num))
        for i in range(np.array(all_zuhe).shape[1]):
            if all_zuhe[s][i]==1:
                a_scenes[prob_index[i][0],prob_index[i][1]]=1
        all_scenes.append(a_scenes)
        a_scenes_prob=1
        for i in range(np.array(all_zuhe).shape[1]):
            if all_zuhe[s][i]==0:
                a_scenes_prob*=1-prob_list[i]
            else:
                a_scenes_prob*=prob_list[i]
        all_scenes_prob.append(a_scenes_prob)
    # pprint(all_scenes)
    # pprint(all_scenes_prob)
    # print(sum(all_scenes_prob))

    return all_scenes,all_scenes_prob,all_zuhe

def create_scenes(graph_adj,graph_p,failure_num,source_nodes,dest_nodes):
    """
    生成所有符合要求的场景，场景都是固定场景
    链路的状态只有两种[0,1]，0：链路正常  1：链路失效
    :param graph_adj 图的邻接矩阵
    :param graph_p:  图的概率矩阵
    :param failure_num:  失效链路的数量
    :param source_nodes:  源节点列表
    :param dest_nodes:   目的结点列表
    :return:  所有场景的矩阵；所有场景概率list;所有的场景list[list](每一个场景是一个list)
    """

    #根据失效链路数量删减场景
    all_scenes,all_scenes_prob,all_scenes_list=delete_scenes_1(graph_p,failure_num)
    # all_scenes,all_scenes_prob,all_scenes_list=delete_scenes_k_2(graph_p,failure_num)    #k=2场景下删减，对指定的链路操作
    print("生成的所有场景")
    print(len(all_scenes_prob))
    # pprint(all_scenes)
    # pprint(all_scenes_prob)
    return all_scenes,all_scenes_prob,all_scenes_list

    # #根据连通性删减场景
    # new_all_scenes,new_all_scenes_prob,new_all_scenes_list=delete_scenes_2(graph_adj,all_scenes,all_scenes_list,all_scenes_prob,source_nodes,dest_nodes)
    # print("删减后的所有场景")
    # # print(new_all_scenes)
    # # print(new_all_scenes_prob)
    # print(len(new_all_scenes_prob))
    # return new_all_scenes,new_all_scenes_prob,new_all_scenes_list
def test_create_scenes():
     graph_adj = np.array([[0, 1, 1, 0, 0, 0],
                         [0, 0, 1, 1, 1, 0],
                         [0, 1, 0, 1, 1, 0],
                         [0, 0, 0, 0, 1, 1],
                         [0, 0, 0, 1, 0, 1],
                         [0, 0, 0, 0, 0, 0]])

     graph_p=graph_adj*0.1
     source_nodes=[0,1]
     dest_nodes=[2,3,4,5]
     create_scenes(graph_adj,graph_p,2,source_nodes,dest_nodes)
     # print(connectivity(graph_adj,1,2))

def cerate_scenes_plus(graph_adj, graph_p_fail, graph_p_loss, failure_num=2):
    """
    生成所有符合要求的场景,所有的链路有三个状态【0,1,2】
    0:链路正常 1：链路失效  2：链路丢包
    :param graph_adj:  array 图的邻接矩阵
    :param graph_p_fail: array 图的链路失效概率矩阵
    :param graph_p_loss: array 图的链路丢包概率矩阵
    :param failure_num:   int 失效链路的数量
    :return: 所有场景 ，所有场景的概率，所有组合
    """

    # （1）将链路失效矩阵和链路丢包矩阵合并，生成链路状态为【0,1,2】链路状态概率列表
    prob_list, prob_index = compute_link_state_prob(graph_adj, graph_p_fail, graph_p_loss)
    node_num = graph_adj.shape[0]
    edge_num = graph_adj.sum()

    # （2）求出所有场景的场景列表:使用包，无法使用，复杂度太高
    # all_zuhe = list(itertools.product([0, 1, 2], repeat=edge_num))  #list[tuple]
    # #对所有的场景进行根据需求进行删减 逆序删减
    # for s in range(len(all_zuhe)-1,-1,-1):
    #     s_fail=compute_scenes_fail_num(all_zuhe[s])
    #     if s_fail>failure_num:
    #         all_zuhe.pop(s)

    #（2）存放所有组合的排列(普通方法)
    all_zuhe = []
    all_zuhe.append([0.0] * len(prob_list))  # 存放c(n,0)
    all_index = [i for i in range(len(prob_list))]
    for i in range(1, failure_num + 1):
        #求出有i条链路故障，其中丢包或者拥塞的所有可能，一共2^i次方种情况
        #求在n个位置上所有可能出现问题的下标组合
        temp_list = list(itertools.combinations(all_index, i)) #[[list]],有问题链路的下标组合
        loss_congest_list=list(itertools.product([1,2],repeat=i))#i条链路失效或者拥塞，的所有可能[[list]],1,和2的组合
        i_scenes_list = []
        for t in range(len(temp_list)):#对每一个可能的情况，求所有丢包或者拥塞情况的下标组合
            for a_loss_congest in loss_congest_list:
                a_scenes = np.zeros(len(prob_list))
                for k in range(len(temp_list[t])):
                    a_scenes[temp_list[t][k]] = a_loss_congest[k]
                i_scenes_list.append(list(a_scenes))
        all_zuhe.extend(i_scenes_list)


    # （3）求出所有场景的概率，并将场景列表转换为场景矩阵
    all_scenes = []
    all_scenes_prob = []
    for s in range(len(all_zuhe)):
        print(s)
        a_scenes = np.zeros(shape=(node_num, node_num))
        for i in range(len(all_zuhe[s])):
            a_scenes[prob_index[i][0], prob_index[i][1]] = all_zuhe[s][i]
        all_scenes.append(a_scenes)

        a_scenes_prob = 1
        for i in range(len(all_zuhe[s])):
            a_scenes_prob *= prob_list[i][int(all_zuhe[s][i])]
        all_scenes_prob.append(a_scenes_prob)

    print("所有的场景")
    # pprint(all_scenes)
    print("所有场景的概率")
    # pprint(all_scenes_prob)
    print("所有场景的排列组合")
    print(all_zuhe)
    return all_scenes, all_scenes_prob, all_zuhe

def test_create_scenes_plus():
    graph_adj=np.array([[0,1,1,0],
                        [1,0,0,1],
                        [1,0,0,1],
                        [0,1,1,0]])

    graph_p_fail=copy.deepcopy(graph_adj*0.1)
    graph_p_loss=copy.deepcopy(graph_adj*0.1)
    cerate_scenes_plus(graph_adj, graph_p_fail, graph_p_loss, failure_num=2)

def remain_dest_node(dest_remain_demand):
    """
    返回目标点中，还有剩余空间的点个数
    :param dest_remain_demand: list
    :return: 返回需要分配的节点下标列表
    """
    num=[]
    for i in range(len(dest_remain_demand)):
        if(not math.isclose(0,dest_remain_demand[i],abs_tol=1e-6)):
            num.append(i)
    return num

def valid_path(route:np.ndarray):
    """
    求S_D的有效路径数，为了满足k条路径，会使用空路径补充，因此从新求
    :param route: ndarray  [path][link]S_D之间的所有路径矩阵
    :return:  有效路径数量
    """
    num=0
    for i in range(route.shape[0]):
        if(route[i].sum()>0):
            num+=1
    return num


if __name__ == '__main__':
    # print(generate_random_number(10,1))
    # createDemand(10,10,10)
    # a,b=createScenarios(10,50)
    # print(a)
    # print(b)
    # saveScenarios()
    # saveDemand()
    # d=[1,2,3,4,5,0,2]
    # c=[2,2,2,2,2,2,2]
    # r=[[1,2],[1,2],[3,4],[4,5],[5,6],[1,4],[3,4]]
    # pre_teavar(d,c,r)
    # test_post_reavar()
    # test_generate_random()
    # test_delete()
    # test_post_teavar()
    # random_graph_demand_test()
    # ratio_test()
    # binomial_distribution()
    # compute_path_prob_test()
    test_create_scenes_plus()

    pass