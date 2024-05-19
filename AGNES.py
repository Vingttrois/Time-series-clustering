# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 15:16:15 2021

@author: FanW
"""
import numpy as np
import random
import gc


def AGNES(Vector, Method):
    'AGNES算法'
    if Method not in ['DTW','DTW_GPU','Euclidean','Euclidean_UE']:
        print('unsupported Method')
        return
    #初始化簇，将样本初始化为单点簇
    Sample_Num = len(Vector)
    ini_Cluster = [[i] for i in range(0, Sample_Num)]
    Cluster_Record = [] #记录簇变化的矩阵----
    Cluster_Record.append(ini_Cluster) #附加第一个簇分类
    Dis_Record = [] #记录距离矩阵变化的矩阵----
    #-------------------------Dendrogram---------------------------------------
    #Dendrogram用于生成聚类图
    Cluster_Record_Dendrogram = [] #按照dendrogram格式准备的簇记录矩阵
    Cluster_Record_Dendrogram.append(ini_Cluster)
    Distance_Record_Dendrogram = [] #按照dendrogram格式准备的距离记录矩阵
    Count_Record_Dendrogram = [] #按照dendrogram格式准备的个数记录矩阵
    Code_Record_Dendrogram = [] #按照dendrogram格式准备的簇编码记录矩阵
    #计算第一个距离矩阵---------------------------------------------------------
    Dis_temp = DisMatrix(Vector, Method)
    Dis_Record.append(Dis_temp)
    #合并簇-计算距离矩阵-合并簇--循环  
    for i in range(0, Sample_Num - 1):
        #找到距离最近的两个元素ele1和ele2
        #首先将对角线元素替换
        Dis_Inf = Dis_Record[0].copy() #从Record中取出距离矩阵
        Dis_Range = len(Dis_Inf)
        Dis_Range_list = np.arange(0, Dis_Range)
        Dis_Inf[Dis_Range_list, Dis_Range_list] = np.inf
        #找最小元素所在位置，##暂未考虑可能存在多个最小，并不影响结果，每一轮只合并一个
        ele1 = np.where(Dis_Inf==np.min(Dis_Inf))[0][0] #所在行数组第一个元素
        ele2 = np.where(Dis_Inf==np.min(Dis_Inf))[1][0] #所在列数组第一个元素 
        #合并两个簇
        Cluster_add = Cluster_Record[i][ele1] + Cluster_Record[i][ele2]
        Mark_max = max(ele1, ele2)
        Mark_min = min(ele1, ele2)
        Cluster_New = Cluster_Record[i].copy()
        del Cluster_New[Mark_max]
        del Cluster_New[Mark_min]
        #-------------------------Dendrogram---start---------------------------
        #记录合并簇的距离-Dendrogram格式
        Distance_Record_Dendrogram.append(np.min(Dis_Inf))
        #记录合并的簇-Dendrogram格式
        Code_temp = Cluster_Record_Dendrogram[i].copy()
        Code_Record_Dendrogram.append([min(Code_temp[Mark_min], Code_temp[Mark_max]),\
                                        max(Code_temp[Mark_min], Code_temp[Mark_max])])
        del Code_temp[Mark_max]
        Code_temp[Mark_min] = [Sample_Num + i]
        Cluster_Record_Dendrogram.append(Code_temp)
        Count_Record_Dendrogram.append(len(Cluster_add))
        #-------------------------Dendrogram---end-----------------------------
        #得到新的簇
        Cluster_New.insert(Mark_min, Cluster_add)
        #将新簇加入Record中
        Cluster_Record.append(Cluster_New)
        #计算新的距离矩阵-------------------------------------------------------
        #1复制距离矩阵
        Dis_Temp = Dis_Record[0].copy()
        #2重新计算距离矩阵中新簇与其他簇（以待合并的近簇为基础）的距离
        for k in range(0, Dis_Range):
            Dis_Temp[Mark_min][k] = (Dis_Temp[Mark_min][k] + Dis_Temp[Mark_max][k])/2.
            Dis_Temp[k][Mark_min] = Dis_Temp[Mark_min][k] 
            if k == Mark_min:
                Dis_Temp[k][Mark_min] = 0.
        #3删除距离矩阵中的待合并的远簇所在的行与列
        Dis_Temp_delete = np.delete(Dis_Temp, Mark_max, axis = 0)
        Dis_New = np.delete(Dis_Temp_delete, Mark_max, axis = 1)
        del(Dis_Record[-1])
        Dis_Record.append(Dis_New)

    Dis_Record_Output = Dis_temp
    del Dis_Record
    gc.collect()
    #生成Dendrogram要求的格式
    Mergings = []
    for i in range(0, Sample_Num - 1):
        Mergings.append([Code_Record_Dendrogram[i][0][0],\
                          Code_Record_Dendrogram[i][1][0],\
                          round(Distance_Record_Dendrogram[i], 2),\
                          Count_Record_Dendrogram[i]])
    del Cluster_Record_Dendrogram
    del Distance_Record_Dendrogram
    del Count_Record_Dendrogram
    del Code_Record_Dendrogram
    gc.collect()
    
    return Cluster_Record, Dis_Record_Output, Mergings