import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 定义性能指标数据
data = {
    'Algorithm': ['MLR', 'LGBM', 'XGB', 'MLR', 'LGBM', 'XGB'],
    'Dataset': ['Training Set', 'Training Set', 'Training Set', 'Test Set', 'Test Set', 'Test Set'],
    'MAE': [2048921.9178, 1533087.5904, 1398310.7366, 2041100.0737, 1624565.9744, 1629455.4464],
    'RMSE': [3450390.7812, 2643687.3463, 2276675.9714, 3298615.5078, 2928056.0768, 3197336.4574],
    'MAPE': [0.5510, 0.4727, 0.4425, 0.5588, 0.4669, 0.4725]
}

# 创建DataFrame
df = pd.DataFrame(data)

# 计算平均值和标准差
summary_df = df.groupby(['Algorithm']).agg({
    'MAE': ['mean', 'std'],
    'RMSE': ['mean', 'std'],
    'MAPE': ['mean', 'std']
}).reset_index()

# 重命名列
summary_df.columns = ['Algorithm', 'MAE_Mean', 'MAE_Std', 'RMSE_Mean', 'RMSE_Std', 'MAPE_Mean', 'MAPE_Std']

# 打印汇总表
print(summary_df)

# 绘制条形图
fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

# 定义指标名称
metrics = ['MAE', 'RMSE', 'MAPE']

# 绘制每个指标的条形图和标准差
for i, metric in enumerate(metrics):
    sns.barplot(x='Algorithm', y=f'{metric}_Mean', data=summary_df, ax=axs[i], palette='viridis')
    axs[i].set_title(f'{metric} by Algorithm')
    axs[i].set_ylabel('Mean Value')
    axs[i].set_xlabel('Algorithm')
    axs[i].set_xticklabels(summary_df['Algorithm'])
    for j, row in summary_df.iterrows():
        axs[i].errorbar(row['Algorithm'], row[f'{metric}_Mean'], yerr=row[f'{metric}_Std'], fmt='none', ecolor='black', capsize=5)

# 调整布局
plt.tight_layout()
plt.show()