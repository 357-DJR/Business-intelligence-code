import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, mutual_info_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# 读取CSV数据文件，确保文件路径正确
df = pd.read_csv(r'E:\桌面\train_without_noise.csv')

# 将分类数据转换为数值数据
label_encoders = {}
for column in df.columns:
    if df[column].dtype == np.object:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# 分离特征和标签
X = df.drop('house_category', axis=1).values
y = df['house_category'].values

# 处理缺失值
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# 特征缩放
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# 使用PCA进行降维
pca = PCA(n_components=5, svd_solver='randomized')
X_pca = pca.fit_transform(X_scaled)

# 设置随机种子
np.random.seed(42)

# 初始化性能指标列表
silhouette_scores = []
davies_bouldin_scores = []
mutual_info_scores = []

# 5折交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 运行3次实验
for experiment in range(3):
    fold_silhouette_scores = []
    fold_davies_bouldin_scores = []
    fold_mutual_info_scores = []
    fold_count = 0
    total_folds = skf.get_n_splits(X_pca, y)

    # 随机打乱数据集
    X_pca, y = shuffle(X_pca, y, random_state=experiment)

    for train_index, test_index in skf.split(X_pca, y):
        fold_count += 1
        X_train, X_test = X_pca[train_index], X_pca[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # DBSCAN聚类
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        labels_train = dbscan.fit_predict(X_train)
        labels_test = dbscan.fit_predict(X_test)

        # 计算性能指标
        if len(set(labels_train)) > 1:  # 确保有足够的聚类
            fold_silhouette_scores.append(silhouette_score(X_train, labels_train))
            fold_davies_bouldin_scores.append(davies_bouldin_score(X_train, labels_train))
            fold_mutual_info_scores.append(mutual_info_score(y_train, labels_train))

        # 在最后一折进行TSNE可视化
        if fold_count == total_folds:
            tsne = TSNE(n_components=2, init='pca', random_state=experiment, method='barnes_hut', learning_rate=200.0, perplexity=30)
            X_train_tsne = tsne.fit_transform(X_train)
            X_test_tsne = tsne.fit_transform(X_test)

            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=labels_train, cmap='viridis', s=5)
            plt.title(f'Train Set - Experiment {experiment + 1}')
            plt.subplot(1, 2, 2)
            plt.scatter(X_test_tsne[:, 0], X_test_tsne[:, 1], c=labels_test, cmap='viridis', s=5)
            plt.title(f'Test Set - Experiment {experiment + 1}')
            plt.show()

    # 计算每个实验的平均性能指标
    avg_silhouette = np.mean(fold_silhouette_scores) if fold_silhouette_scores else 0
    avg_davies_bouldin = np.mean(fold_davies_bouldin_scores) if fold_davies_bouldin_scores else 0
    avg_mutual_info = np.mean(fold_mutual_info_scores) if fold_mutual_info_scores else 0

    # 存储每个实验的平均性能指标
    silhouette_scores.append(avg_silhouette)
    davies_bouldin_scores.append(avg_davies_bouldin)
    mutual_info_scores.append(avg_mutual_info)

    # 打印每个实验的平均性能指标
    print(f'Experiment {experiment + 1} - Average Silhouette Score: {avg_silhouette:.4f}, Average Davies-Bouldin Index: {avg_davies_bouldin:.4f}, Average Mutual Information: {avg_mutual_info:.4f}')

# 输出每次实验的平均性能指标
print(f'Overall Average Silhouette Scores: {silhouette_scores}')
print(f'Overall Average Davies-Bouldin Index: {davies_bouldin_scores}')
print(f'Overall Average Mutual Information: {mutual_info_scores}')