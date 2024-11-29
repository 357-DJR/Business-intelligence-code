import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Apriori算法的实验结果
apriori_results = {
    'Support': [
        [0.1797, 0.1793, 0.1811, 0.1785, 0.1788],
        [0.1788, 0.1802, 0.1793, 0.1804, 0.1786],
        [0.1808, 0.1809, 0.1785, 0.1792, 0.1778]
    ],
    'Confidence': [
        [0.4857, 0.4853, 0.4872, 0.4830, 0.4853],
        [0.4849, 0.4861, 0.4851, 0.4865, 0.4838],
        [0.4884, 0.4872, 0.4821, 0.4855, 0.4830]
    ],
    'Lift': [
        [1.0252, 1.0254, 1.0206, 1.0233, 1.0303],
        [1.0277, 1.0228, 1.0264, 1.0225, 1.0249],
        [1.0255, 1.0236, 1.0210, 1.0268, 1.0270]
    ]
}

# FP-Growth算法的实验结果
fpgrowth_results = {
    'Support': [
        [0.1442, 0.1437, 0.1451, 0.1430, 0.1437],
        [0.1434, 0.1444, 0.1440, 0.1446, 0.1431],
        [0.1449, 0.1454, 0.1432, 0.1436, 0.1425]
    ],
    'Confidence': [
        [0.4820, 0.4811, 0.4820, 0.4793, 0.4828],
        [0.4812, 0.4817, 0.4823, 0.4818, 0.4801],
        [0.4832, 0.4837, 0.4792, 0.4813, 0.4796]
    ],
    'Lift': [
        [1.1080, 1.1072, 1.0988, 1.1069, 1.1159],
        [1.1106, 1.1037, 1.1109, 1.1024, 1.1084],
        [1.1042, 1.1052, 1.1061, 1.1086, 1.1116]
    ]
}

# 计算平均值和标准差
def calculate_stats(results):
    stats_data = []
    for algorithm, metrics in results.items():
        for metric, experiments in metrics.items():
            mean = np.mean(experiments)
            std = np.std(experiments)
            stats_data.append({
                'Algorithm': algorithm,
                'Metric': metric,
                'Mean': mean,
                'Std': std
            })
    return pd.DataFrame(stats_data)

# 创建汇总表
stats_df = calculate_stats({'Apriori': apriori_results, 'FP-Growth': fpgrowth_results})

# 打印汇总表
print(stats_df)

# 可视化
plt.figure(figsize=(12, 6))

# 绘制条形图
for i, metric in enumerate(['Support', 'Confidence', 'Lift']):
    subset = stats_df[stats_df['Metric'] == metric]
    plt.subplot(1, 3, i+1)
    sns.barplot(x='Algorithm', y='Mean', data=subset, yerr=subset['Std'], capsize=10)
    plt.title(f'{metric} - Mean and Standard Deviation')
    plt.xlabel('Algorithm')
    plt.ylabel('Value')
    plt.ylim(0, 1)  # 根据实际数据范围调整

plt.tight_layout()
plt.show()