import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn import datasets
import pandas as pd
import seaborn as sns
import warnings


data = pd.read_csv('tea.csv')

# 为了方便仅选取2个简单的特征
datas = data[['hcq','hch']]
#打印出前五行查看数据情况
print(datas.head(5))

#查看数据描述
print(datas.describe())
# 画出数据分布直方图
#datas.hist(bins=50,figsize=(20,15))
plt.title('Data Constribution')
plt.hist(datas['hcq'],bins=50)
plt.hist(datas['hch'],bins=50)

plt.legend(["hcq","hch"])
plt.show()

#转化为array类型
score=datas.values
scare=score
scan=score

#分类拟合
estimator0=KMeans(n_clusters=3)
estimator1=DBSCAN(eps=1.09,min_samples=2)
estimator2=AgglomerativeClustering(affinity='euclidean',linkage='ward',n_clusters=3)
estimator0.fit(score)
estimator1.fit(scare)
estimator2.fit(scan)
label_pred0=estimator0.labels_
label_pred1=estimator1.labels_
label_pred2=estimator2.labels_

#绘图
x0=score[label_pred0==0]
x1=score[label_pred0==1]
x2=score[label_pred0==2]
x3=scare[label_pred1==0]
x4=scare[label_pred1==1]
x5=scare[label_pred1==2]
x6=scan[label_pred2==0]
x7=scan[label_pred2==1]
x8=scan[label_pred2==2]
print(x3)
print(x4)
print(x5)
plt.title('K-MEANS')
plt.scatter(x0[:,0],x0[:,1],c="red",marker='o',label='label0')
plt.scatter(x1[:,0],x1[:,1],c="green",marker='o',label='label1')
plt.scatter(x2[:,0],x2[:,1],c="blue",marker='o',label='label2')
plt.show()

plt.title('DBSCAN')
plt.scatter(x3[:,0],x3[:,1],c="red",marker='*',label='label0')
plt.scatter(x4[:,0],x4[:,1],c="green",marker='*',label='label1')
plt.scatter(x5[:,0],x5[:,1],c="blue",marker='*',label='label2')
plt.show()

plt.title('AGENS')
plt.scatter(x6[:,0],x6[:,1],c="red",marker='+',label='label0')
plt.scatter(x7[:,0],x7[:,1],c="green",marker='+',label='label1')
plt.scatter(x8[:,0],x8[:,1],c="blue",marker='+',label='label2')
plt.show()
