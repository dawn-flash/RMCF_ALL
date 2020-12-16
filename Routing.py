from math import*
import numpy as np
import os

"""
求出所有topo的路由矩阵
"""

_=float('inf')      #无穷大
def findAllFile(base):  #遍历文件夹中的某类文件
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith('.txt'):
                fullname = os.path.join(root, f)
                yield fullname


def readmat(data):
    a = np.loadtxt(data, dtype=np.int)
    num=a.shape[0]
    b=np.zeros((a.shape[0],a.shape[0]))
    for i in range(a.shape[0]):             #将读入的a转化成方便计算的矩阵b(在b中区分了0和无穷大)
        for j in range(a.shape[0]):
            if i==j:
                b[i][j]=0
            elif a[i][j]==0:
                b[i][j]=_
            else:
                b[i][j]=a[i][j]
    distance_0=b
    prior = [[0 for i in range(1,num+1)] for j in range(1,num+1)]   #初始化一个空矩阵用来存放所经过的节点
    for i in range(num):
        for j in range(num):
            prior[i][j]=j
    # print(prior)
    for m in range(num):      #i相当于是中间点
        for i in range(num):      #j相当于是起始点
            for j in range(num):      #m相当于是终点
                if (distance_0[i][j] > distance_0[i][m]+distance_0[m][j]):
                    distance_0[i][j] = distance_0[i][m]+distance_0[m][j]
                    prior[i][j]=prior[i][m]
                else:
                    continue
    # print("v_1到v_11的最短路程为：",distance_0[0][10])      #打印v_1到v_11距离
    # print("v_%d"%p,"到v_%d"%q,"的最短路程为：",distance_0[p-1][q-1])       #打印任意两点距离
    #创建一个（num x num）x（num x num）阶的矩阵，行是0-0至num-num(链路)
    #列是0-0至num-num（点对，即OD对）
    A=np.zeros((num**2,num**2))
    for p in range(num):####################在这个循环里赋值
        for q in range(num):
            if prior[p][q]==q:
                # print('v_%d'%p,'到v_%d'%q,'的路径为：v_%d'%(p)+'->'+'v_%d'%(q),":","路程为：",distance_0[p][q])
                A[p*num+q,p*num+q]=1;
            else:
                x=prior[p][q]
                mid=[x]      #创建一个空列表装载路径中间点
                while (prior[x][q]!=q):
                    x=prior[x][q]
                    mid.append(x)
            
                # print('v_%d'% p,'到v_%d'%q,'的路径为:v_%d'%p,end="")
                j=p
                for i in mid:
                    A[j*num+i,p*num+q]=1
                    j=i
                    # print('->'+'v_%d'%i,end="")
                    
                # print('->'+'v_%d'%(q),":","路程为:",distance_0[p][q])
                A[i*num+q,p*num+q]=1
    #由于求得的矩阵太大，去掉没必要的行和列
    k=0
    l=0
    for i in range(num):#删冗余行
        for j in range(num):
            if a[i][j]==0:
                A=np.delete(A,i*num+j-k,axis=0)
                k=k+1
    for i in range(num):#删冗余列
        j=i*num+i
        A=np.delete(A,j-l,axis=1)
        l=l+1
    return A
        

if __name__ == '__main__':
    for i in findAllFile('./Adjacency/'):#遍历邻接矩阵文件（txt格式）
        file_path = i
        (filepath, tempfilename) = os.path.split(file_path)#分离文件路径名
        (filename, extension) = os.path.splitext(tempfilename)#分离文件类型名
        print(filename)
        a=readmat(i)
        np.savetxt(filename+".txt", a, fmt="%d", delimiter=",")
        
        
        
        
        
        
        
        
        


