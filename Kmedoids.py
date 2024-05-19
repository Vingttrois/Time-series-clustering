# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 15:16:15 2021

@author: FanW
"""
import numpy as np
import random



def Kmedoids(Vector, K, Method):
    'Kmedoids++算法'
    Sample_Num = len(Vector)
    select_list = range(0, Sample_Num)
    Len_array = []
    for i in range(0, Sample_Num):
        Len_array.append(len(Vector[i]))
    if Method not in ['DTW','DTW_GPU','Euclidean','Euclidean_UE']:
        print('unsupported Method')
        return
    if Method == 'Euclidean' and len(list(set(Len_array))) > 1:
        print('Euclidean distance does not support vectors with unequal lengths')
        return 
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
    c_list = [] #新的簇中心
    for i in range(0, K):
        dis_mat_temp = np.zeros([len(ini_cluster[i]),len(ini_cluster[i])])
        for i_temp in range(0, len(ini_cluster[i])):
            for j_temp in range(i_temp, len(ini_cluster[i])):
                if Method == 'DTW' or 'DTW_GPU':
                    dis_mat_temp[i_temp][j_temp] = DTW(np.array(Vector[ini_cluster[i][i_temp]]),\
                            np.array(Vector[ini_cluster[i][j_temp]]))
                elif Method == 'Euclidean':
                    dis_mat_temp[i_temp][j_temp] = Euclidean(np.array(Vector[ini_cluster[i][i_temp]]),\
                            np.array(Vector[ini_cluster[i][j_temp]]))
                elif Method == 'Euclidean_UE': 
                    dis_mat_temp[i_temp][j_temp] = Euclidean_UE(np.array(Vector[ini_cluster[i][i_temp]]),\
                            np.array(Vector[ini_cluster[i][j_temp]]))
        for i_temp in range(0, len(ini_cluster[i])):
            for j_temp in range(i_temp, len(ini_cluster[i])):
                dis_mat_temp[j_temp][i_temp] = dis_mat_temp[i_temp][j_temp]
        sum_list = []
        for j in range(0, dis_mat_temp.shape[0]):
            sum_list.append(sum(dis_mat_temp[j]))    
        c_list.append(Vector[ini_cluster[i][np.argmin(sum_list)]])
    #分配元素
    temp_cluster = [[] for i in range(0, K)]
    for i_temp in range(0, Sample_Num):
        temp_d = []
        for j_temp in range(0, K):
            if Method == 'DTW' or 'DTW_GPU':
                temp_d.append(DTW(np.array(Vector[i_temp]), np.array(c_list[j_temp])))
            elif Method == 'Euclidean':
                temp_d.append(Euclidean(np.array(Vector[i_temp]), np.array(c_list[j_temp])))
            elif Method == 'Euclidean_UE': 
                temp_d.append(Euclidean_UE(np.array(Vector[i_temp]), np.array(c_list[j_temp])))  
        temp_cluster[np.argmin(temp_d)].append(i_temp)
    while temp_cluster != ini_cluster:
        ini_cluster = temp_cluster
        c_list = []
        for i in range(0, K):
            dis_mat_temp = np.zeros([len(ini_cluster[i]),len(ini_cluster[i])])
            for i_temp in range(0, len(ini_cluster[i])):
                for j_temp in range(i_temp, len(ini_cluster[i])):
                    if Method == 'DTW' or 'DTW_GPU':
                        dis_mat_temp[i_temp][j_temp] = DTW(np.array(Vector[ini_cluster[i][i_temp]]),\
                                                                   np.array(Vector[ini_cluster[i][j_temp]]))
                    elif Method == 'Euclidean':
                        dis_mat_temp[i_temp][j_temp] = Euclidean(np.array(Vector[ini_cluster[i][i_temp]]),\
                                                                         np.array(Vector[ini_cluster[i][j_temp]]))
                    elif Method == 'Euclidean_UE': 
                        dis_mat_temp[i_temp][j_temp] = Euclidean_UE(np.array(Vector[ini_cluster[i][i_temp]]),\
                                                                            np.array(Vector[ini_cluster[i][j_temp]]))
            for i_temp in range(0, len(ini_cluster[i])):
                for j_temp in range(i_temp, len(ini_cluster[i])):
                    dis_mat_temp[j_temp][i_temp] = dis_mat_temp[i_temp][j_temp]
            
            sum_list = []
            for j in range(0, dis_mat_temp.shape[0]):
                sum_list.append(sum(dis_mat_temp[j]))
            c_list.append(Vector[ini_cluster[i][np.argmin(sum_list)]])
        #分配元素
        temp_cluster = [[] for i in range(0, K)]
        for i_temp in range(0, Sample_Num):
            temp_d = []
            for j_temp in range(0, K):
                if Method == 'DTW' or 'DTW_GPU':
                    temp_d.append(DTW(np.array(Vector[i_temp]), np.array(c_list[j_temp])))
                elif Method == 'Euclidean':
                    temp_d.append(Euclidean(np.array(Vector[i_temp]), np.array(c_list[j_temp])))
                elif Method == 'Euclidean_UE': 
                    temp_d.append(Euclidean_UE(np.array(Vector[i_temp]), np.array(c_list[j_temp])))  
            temp_cluster[np.argmin(temp_d)].append(i_temp)
    return temp_cluster
