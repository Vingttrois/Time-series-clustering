# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 09:56:41 2021

@author: FanW
"""

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram #绘图



a = [2,1,5,1]
b = [2,1,4,1]
c = [1,2,3,4]
d = [1,2,3,5]
e = [6,1,2,3]
f = [6,1,2,3]
g = [1,1,1,1]
h = [6,1,2,3]
 
Vector = []
Vector.append(a)
Vector.append(b)
Vector.append(c)
Vector.append(d)
Vector.append(e)
Vector.append(f)
Vector.append(g)
Vector.append(h)

K = 3
Method = 'DTW'
# Method = 'DTW_GPU'
# Method = 'Euclidean'
# Method = 'Euclidean_UE'



# Kmedoids
temp_cluster = Kmedoids(Vector, K, Method)
print(temp_cluster)
# Kmeans
temp_cluster = Kmeans(Vector, K, Method)
# AGNES
Cluster_Record, Dis_Record_Output, Mergings = fanw_ml.AGNES(Vector, Method)
print(temp_cluster)


plt.figure(figsize=(12,4))
plt.rc('font',family='Times New Roman',size=14)
fontcn = {'family': 'STSong','size':'12'} 
dendrogram(Mergings, leaf_rotation=45, leaf_font_size=10)
plt.xticks(size=13)
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.98, top=0.98)