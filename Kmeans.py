# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 15:16:15 2021

@author: FanW
"""
import numpy as np
import random


def Kmeans(Vector, K, Method):
    'Kmeans++算法'
    Sample_Num = len(Vector)
    select_list = range(0, Sample_Num)
    #随机选取初始簇中心
    ini_list = random.sample(select_list, 1)
    #计算第一距离矩阵
    Dis_temp = DisMatrix(Vector, Method)
    for i in range(1, K):
        temp_list = []
        for j in range(0, Sample_Num):
            temp_v = 0
            for k in range(0, len(ini_list)):
                temp_v = temp_v + Dis_temp[ini_list[k]][j]
            temp_list.append(temp_v)        
        temp_c = np.argmax(temp_list)
        ini_list.append(temp_c)
    #生成初始簇
    ini_cluster = [[] for i in range(0, K)]
    for i in range(0, K):
        ini_cluster[i].append(ini_list[i])
    #分配至各簇中
    for i_temp in range(0, Sample_Num):
        temp_d = []
        for j_temp in range(0, K):
            temp_d.append(Dis_temp[i_temp][ini_list[j_temp]])
        mark = 0
        for i in range(0, len(ini_list)):
            if i_temp in ini_cluster[i]:
                mark = 1
        if mark == 0:
            ini_cluster[np.argmin(temp_d)].append(i_temp)
    #迭代重分配簇元素
    #新的簇中心
    c_list = []
    for i in range(0, K):
        temp_c = np.zeros(len(Vector[0]))
        for j in range(0, len(ini_cluster[i])):
            temp_c = temp_c + Vector[ini_cluster[i][j]]
        c_list.append(temp_c/len(ini_cluster[i]))
    #分配元素
    temp_cluster = [[] for i in range(0, K)]
    for i_temp in range(0, Sample_Num):
        temp_d = []
        for j_temp in range(0, K):
            if Method == 'DTW' or 'DTW_GPU':
                temp_d.append(DTW(Vector[i_temp], c_list[j_temp]))
            elif Method == 'Euclidean':
                temp_d.append(Euclidean(Vector[i_temp], c_list[j_temp]))
            elif Method == 'Euclidean_UE': 
                temp_d.append(Euclidean_UE(Vector[i_temp], c_list[j_temp]))
        temp_cluster[np.argmin(temp_d)].append(i_temp)
    while temp_cluster != ini_cluster:
        ini_cluster =  temp_cluster
        c_list = []
        for i in range(0, K):
            temp_c = np.zeros(len(Vector[0]))
            for j in range(0, len(ini_cluster[i])):
                temp_c = temp_c + Vector[ini_cluster[i][j]]
            c_list.append(temp_c/len(ini_cluster[i]))
        #分配元素
        temp_cluster = [[] for i in range(0, K)]
        for i_temp in range(0, Sample_Num):
            temp_d = []
            for j_temp in range(0, K):
                if Method == 'DTW' or 'DTW_GPU':
                    temp_d.append(DTW(Vector[i_temp], c_list[j_temp]))
                elif Method == 'Euclidean':
                    temp_d.append(Euclidean(Vector[i_temp], c_list[j_temp]))
                elif Method == 'Euclidean_UE': 
                    temp_d.append(Euclidean_UE(Vector[i_temp], c_list[j_temp]))
            temp_cluster[np.argmin(temp_d)].append(i_temp)
    return temp_cluster