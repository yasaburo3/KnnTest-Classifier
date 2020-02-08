from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np


#生成数据
centers = [[-2,2],[2,2],[0,4]]      #三个簇中心
X, y = make_blobs(n_samples=60, centers=centers,
                random_state=0, cluster_std=0.60)
'''
make_blobs生成样本
n_samples   待生成的样本的总数(default=100)
n_features  每个样本的特征数 (default=2)
centers     要生成的样本中心（类别）数，或者是确定的中心点。(default=3)
cluster_std 每个类别的方差(default=1.0)
X   array of shape [n_samples, n_features]         
    生成的样本数据集。
y   array of shape [n_samples]
    样本数据集的标签。
'''
#使用 matplotlib 库把生成的点画出来
plt.figure(figsize=(16,10), dpi=144)#dpi表示窗口的分辨率
c = np.array(centers)
'''
#画出样本
plt.scatter(X[:,0], X[:,1], c=y, s=100, cmap='cool')
#画出中心点
plt.scatter(c[:,0], c[:,1], s=100, marker='^',c='orange')
plt.savefig=('knn_centers.png')
plt.show()
'''

#模型训练
k = 5
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(X,y)
"""
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=None, n_neighbors=5, p=2,
           weights='uniform')
"""
#进行预测
X_sample = np.array([[0, 2]])
y_sample = clf.predict(X_sample)
neighbors = clf.kneighbors(X_sample, return_distance=False)

#画出示意图
plt.figure(figsize=(16,10),dpi=144)
c = np.array(centers)
#画出样本
plt.scatter(X[:,0], X[:,1], c=y, s=100, cmap='cool')
#画出中心点
plt.scatter(c[:,0], c[:,1], s=100, marker='^',c='orange')
#画出待预测的点
plt.scatter(X_sample[0][0], X_sample[0][1], marker="x",
            s=100, cmap='cool')
#预测点与距离最近的5个点连线
for i in neighbors[0]:
    plt.plot([X[i][0], X_sample[0][0]], [X[i][1], X_sample[0][1]],
            'k--', linewidth=0.6)
plt.savefig('knn_predict.png')
plt.show()
