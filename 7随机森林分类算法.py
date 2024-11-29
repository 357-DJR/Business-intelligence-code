import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import chardet

# 尝试自动检测文件编码
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    result = chardet.detect(raw_data)
    return result['encoding']

# 读取数据集，自动检测编码
file_path = r"E:\桌面\train_without_noise3.csv"
encoding = detect_encoding(file_path)
data = pd.read_csv(file_path, encoding=encoding)

# 提取自变量和因变量
X = data[['price_doc']]
y = data[['raion_popul', 'green_zone_part', 'school_education_centers_raion', 'healthcare_centers_raion']]

# 对各列按照设定的标准进行分类转化
thresholds = {
    'raion_popul': 80000,
    'green_zone_part': 0.3,
    'school_education_centers_raion': 7,
    'healthcare_centers_raion': 2
}

for col, threshold in thresholds.items():
    y[col] = y[col].astype(float)
    y.loc[y[col] > threshold, col] = 1
    y.loc[y[col] <= threshold, col] = 0

# 检查和处理y中的NaN值
y.fillna(y.mean(), inplace=True)

# 检查和处理X中的NaN值
X.fillna(X.mean(), inplace=True)

# 检查和处理无穷值
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.mean(), inplace=True)

# 进行多次实验，这里设置为3次，可根据需求修改
num_experiments = 3
accuracy_train_list = []
f1_train_list = []
recall_train_list = []
precision_train_list = []
accuracy_test_list = []
f1_test_list = []
recall_test_list = []
precision_test_list = []

for experiment in range(num_experiments):
    kf = KFold(n_splits=5, shuffle=True, random_state=experiment)
    accuracy_train = []
    f1_train = []
    recall_train = []
    precision_train = []
    accuracy_test = []
    f1_test = []
    recall_test = []
    precision_test = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # 构建随机森林分类模型
        model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=experiment)
        model.fit(X_train, y_train)

        # 在训练集上进行预测
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # 分别计算每个输出列的评估指标
        num_outputs = y_train.shape[1]
        accuracy_train_per_column = []
        f1_train_per_column = []
        recall_train_per_column = []
        precision_train_per_column = []

        for i in range(num_outputs):
            accuracy_train_per_column.append(accuracy_score(y_train.iloc[:, i], y_train_pred[:, i]))
            f1_train_per_column.append(f1_score(y_train.iloc[:, i], y_train_pred[:, i], average='macro'))
            recall_train_per_column.append(recall_score(y_train.iloc[:, i], y_train_pred[:, i], average='macro'))
            precision_train_per_column.append(precision_score(y_train.iloc[:, i], y_train_pred[:, i], average='macro'))

        average_accuracy_train = sum(accuracy_train_per_column) / num_outputs
        average_f1_train = sum(f1_train_per_column) / num_outputs
        average_recall_train = sum(recall_train_per_column) / num_outputs
        average_precision_train = sum(precision_train_per_column) / num_outputs

        accuracy_train.append(average_accuracy_train)
        f1_train.append(average_f1_train)
        recall_train.append(average_recall_train)
        precision_train.append(average_precision_train)

        accuracy_test_per_column = []
        f1_test_per_column = []
        recall_test_per_column = []
        precision_test_per_column = []

        for i in range(num_outputs):
            accuracy_test_per_column.append(accuracy_score(y_test.iloc[:, i], y_test_pred[:, i]))
            f1_test_per_column.append(f1_score(y_test.iloc[:, i], y_test_pred[:, i], average='macro'))
            recall_test_per_column.append(recall_score(y_test.iloc[:, i], y_test_pred[:, i], average='macro'))
            precision_test_per_column.append(precision_score(y_test.iloc[:, i], y_test_pred[:, i], average='macro'))

        average_accuracy_test = sum(accuracy_test_per_column) / num_outputs
        average_f1_test = sum(f1_test_per_column) / num_outputs
        average_recall_test = sum(recall_test_per_column) / num_outputs
        average_precision_test = sum(precision_test_per_column) / num_outputs

        accuracy_test.append(average_accuracy_test)
        f1_test.append(average_f1_test)
        recall_test.append(average_recall_test)
        precision_test.append(average_precision_test)

    # 计算平均指标
    average_accuracy_train = sum(accuracy_train) / len(accuracy_train)
    average_f1_train = sum(f1_train) / len(f1_train)
    average_recall_train = sum(recall_train) / len(recall_train)
    average_precision_train = sum(precision_train) / len(precision_train)
    average_accuracy_test = sum(accuracy_test) / len(accuracy_test)
    average_f1_test = sum(f1_test) / len(f1_test)
    average_recall_test = sum(recall_test) / len(recall_test)
    average_precision_test = sum(precision_test) / len(precision_test)

    accuracy_train_list.append(average_accuracy_train)
    f1_train_list.append(average_f1_train)
    recall_train_list.append(average_recall_train)
    precision_train_list.append(average_precision_train)
    accuracy_test_list.append(average_accuracy_test)
    f1_test_list.append(average_f1_test)
    recall_test_list.append(average_recall_test)
    precision_test_list.append(average_precision_test)

# 打印实验结果
for i in range(num_experiments):
    print(f"实验{i + 1}：")
    print(f"平均训练集准确率: {accuracy_train_list[i]}")
    print(f"平均训练集F-度量值: {f1_train_list[i]}")
    print(f"平均训练集召回率: {recall_train_list[i]}")
    print(f"平均训练集精确率: {precision_train_list[i]}")
    print(f"平均测试集准确率: {accuracy_test_list[i]}")
    print(f"平均测试集F-度量值: {f1_test_list[i]}")
    print(f"平均测试集召回率: {recall_test_list[i]}")
    print(f"平均测试集精确率: {precision_test_list[i]}")

# 可视化随机森林中的8个决策树模型结构
def visualize_trees(estimators, feature_names, class_names, num_trees=8):
    plt.figure(figsize=(20, 20))
    for i, estimator in enumerate(estimators[:num_trees]):
        plt.subplot(2, 4, i + 1)  # 2行4列的子图
        plot_tree(estimator, filled=True, feature_names=feature_names, class_names=class_names)
        plt.title(f"Tree {i+1}")
    plt.tight_layout()
    plt.show()

# 获取最后一次实验的模型用于可视化
last_model = None
if len(accuracy_train_list) > 0:
    last_model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=num_experiments - 1)
    last_model.fit(X, y)

if last_model is not None:
    # 可视化前8棵树
    visualize_trees(last_model.estimators_, X.columns, ['0', '1'])
else:
    print("没有成功完成实验，无法进行决策树可视化。")