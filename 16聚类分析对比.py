import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 定义性能指标数据
data = {
    'Algorithm': ['K-Means', 'Agglomerative', 'DBSCAN'],
    'Average Silhouette Score': [0.3776, 0.3832, 0.1679],
    'Average Davies-Bouldin Index': [1.1517, 1.0497, 0.9889],
    'Average Mutual Information': [0.4548, 0.3780, 0.8314],
    'Silhouette Score Std': [0.0055, 0.0001, 0.0052],
    'Davies-Bouldin Index Std': [0.0133, 0.0021, 0.0189],
    'Mutual Information Std': [0.0015, 0.0003, 0.0042]
}

# 创建DataFrame
df = pd.DataFrame(data)

# 输出汇总表
print("Performance Metrics Summary:")
print(df)

# 设置图表大小
plt.figure(figsize=(12, 8))

# 绘制条形图
bar_width = 0.2
index = np.arange(len(df['Algorithm']))

# 绘制每个指标的条形图
plt.bar(index, df['Average Silhouette Score'], bar_width, label='Average Silhouette Score', yerr=df['Silhouette Score Std'], capsize=5, color='b')
plt.bar(index + bar_width, df['Average Davies-Bouldin Index'], bar_width, label='Average Davies-Bouldin Index', yerr=df['Davies-Bouldin Index Std'], capsize=5, color='r')
plt.bar(index + 2 * bar_width, df['Average Mutual Information'], bar_width, label='Average Mutual Information', yerr=df['Mutual Information Std'], capsize=5, color='g')

# 添加标签和标题
plt.xlabel('Algorithm', fontsize=14)
plt.ylabel('Performance Metrics', fontsize=14)
plt.title('Comparison of Clustering Algorithms', fontsize=16)
plt.xticks(index + bar_width, df['Algorithm'], fontsize=12)
plt.legend(fontsize=12)

# 添加数值标签
for i in index:
    plt.text(i, df['Average Silhouette Score'][i] + 0.01, f"{df['Average Silhouette Score'][i]:.4f}", ha='center', va='bottom')
    plt.text(i + bar_width, df['Average Davies-Bouldin Index'][i] + 0.05, f"{df['Average Davies-Bouldin Index'][i]:.4f}", ha='center', va='bottom')
    plt.text(i + 2 * bar_width, df['Average Mutual Information'][i] + 0.01, f"{df['Average Mutual Information'][i]:.4f}", ha='center', va='bottom')

# 显示图表
plt.tight_layout()
plt.show()