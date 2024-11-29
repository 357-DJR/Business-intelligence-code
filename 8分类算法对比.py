import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 算法名称
algorithms = ['Decision Tree', 'KNN', 'Random Forest']

# 存储每个算法的指标平均值和标准差
results = {
    'Accuracy': {
        'Decision Tree': [0.919698477245198, 0.919733085308877, 0.9196314241218204],
        'KNN': [0.9336844609794083, 0.9334811386052951, 0.9317766914691124],
        'Random Forest': [0.9194800138432255, 0.9196854992213186, 0.9197049662571379]
    },
    'F1 Score': {
        'Decision Tree': [0.8776415405796157, 0.8743172398374096, 0.8772946637616537],
        'KNN': [0.9183405574333408, 0.9182959682830095, 0.9171502134921722],
        'Random Forest': [0.8776270819304466, 0.8770113522864078, 0.8768731271499736]
    },
    'Recall': {
        'Decision Tree': [0.8858019632597726, 0.8844283645332054, 0.8855894149698524],
        'KNN': [0.9336844609794083, 0.9334811386052951, 0.9317766914691124],
        'Random Forest': [0.8856313401069122, 0.8854627002635376, 0.8854200114934322]
    },
    'Precision': {
        'Decision Tree': [0.9070051989744252, 0.908770062082699, 0.906833904393380],
        'KNN': [0.9257568853184821, 0.9253983864666091, 0.922699717049734],
        'Random Forest': [0.9049233755833453, 0.9069264982951781, 0.9071955937890952]
    }
}

# 计算每个算法的平均值和标准差
summary = pd.DataFrame()
for metric, alg_results in results.items():
    summary[metric] = [np.mean(values) for values in alg_results.values()]
    summary[f'{metric} Std'] = [np.std(values) for values in alg_results.values()]

# 为汇总表添加算法名称
summary.index = algorithms

# 打印汇总表
print(summary)

# 绘制条形图
fig, ax = plt.subplots(figsize=(14, 8))

# 定义指标标签
labels = ['Accuracy', 'F1 Score', 'Recall', 'Precision']

# 定义条形的宽度和位置
bar_width = 0.2
index = np.arange(len(labels))

# 绘制每个算法的条形
for i, alg in enumerate(algorithms):
    ax.bar(index + i * bar_width, summary.loc[alg, labels], bar_width, label=alg)

# 添加图例
ax.legend()

# 添加标签和标题
ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_title('Performance Metrics Comparison')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(labels)

# 显示平均值和标准差
for i, label in enumerate(labels):
    for j, alg in enumerate(algorithms):
        ax.text(i + j * bar_width, summary.loc[alg, label] + 0.01, f"Avg: {summary.loc[alg, label]:.4f}\nStd: {summary.loc[alg, f'{label} Std']:.4f}", ha='center', va='bottom')

# 显示图表
plt.tight_layout()
plt.show()