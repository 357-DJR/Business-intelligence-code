import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
from sklearn.model_selection import KFold

# 加载数据
df = pd.read_csv(r'E:\桌面\train_without_noise.csv')

# 数据预处理
# 假设您已经处理了缺失值和分类数据
# 将价格分箱，以便更好地进行关联分析
df['price_doc_bin'] = pd.qcut(df['price_doc'] / 1000000, q=5, labels=False)

# 创建事务数据集
transactions = []
for i in range(len(df)):
    transaction = []
    if df['school_education_centers_raion'].iloc[i] > 0:
        transaction.append('school_education_centers_raion')
    if df['healthcare_centers_raion'].iloc[i] > 0:
        transaction.append('healthcare_centers_raion')
    transaction.append(f'price_doc_bin_{df["price_doc_bin"].iloc[i]}')
    transactions.append(transaction)

# 使用TransactionEncoder转换数据
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_te = pd.DataFrame(te_ary, columns=te.columns_)

# 设置5折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 存储每次实验的结果
results = []
rules_list = []

# 进行3次实验
for experiment in range(3):
    print(f"Experiment {experiment + 1}")
    kf = KFold(n_splits=5, shuffle=True, random_state=experiment)  # 每次实验使用不同的随机种子
    fold_results = []
    for train_index, test_index in kf.split(df_te):
        # 训练集和测试集
        train_data = df_te.iloc[train_index]
        test_data = df_te.iloc[test_index]
        
        # 使用FP-Growth算法找到频繁项集
        frequent_itemsets = fpgrowth(train_data, min_support=0.01, use_colnames=True)
        
        # 找到关联规则
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
        
        # 评估测试集
        test_rules = association_rules(fpgrowth(test_data, min_support=0.01, use_colnames=True), 
                                      metric="confidence", min_threshold=0.1)
        
        # 计算测试集上的指标
        test_support = test_rules['support'].mean()
        test_confidence = test_rules['confidence'].mean()
        test_lift = test_rules['lift'].mean()
        
        fold_results.append({
            'support': test_support,
            'confidence': test_confidence,
            'lift': test_lift
        })
        
        # 保存关联规则
        rules_list.append(rules)
    
    results.append(fold_results)
    print("Results for this experiment:")
    for i, result in enumerate(fold_results):
        print(f"  Fold {i + 1}: Support={result['support']:.4f}, Confidence={result['confidence']:.4f}, Lift={result['lift']:.4f}")

# 可视化关联规则
# 使用最后一次实验的最后一折的规则进行可视化
rules = rules_list[-1]
plt.figure(figsize=(12, 6))

# 绘制支持度
plt.subplot(1, 3, 1)
plt.bar(rules['antecedents'].astype(str) + ' -> ' + rules['consequents'].astype(str), rules['support'])
plt.title('Support of Rules')

# 绘制置信度
plt.subplot(1, 3, 2)
plt.bar(rules['antecedents'].astype(str) + ' -> ' + rules['consequents'].astype(str), rules['confidence'])
plt.title('Confidence of Rules')

# 绘制提升度
plt.subplot(1, 3, 3)
plt.bar(rules['antecedents'].astype(str) + ' -> ' + rules['consequents'].astype(str), rules['lift'])
plt.title('Lift of Rules')

# 手动调整子图参数
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)

# 显示图表
plt.show()