import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.tree import export_graphviz
import graphviz
import matplotlib.pyplot as plt
import chardet


# 尝试自动检测文件编码
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    result = chardet.detect(raw_data)
    return result['encoding']


# 读取数据集，自动检测编码
file_path =r"E:\桌面\train_without_noise3.csv"
encoding = detect_encoding(file_path)
data = pd.read_csv(file_path, encoding=encoding)

# 提取自变量和因变量
X = data[['price_doc']]
y = data[['raion_popul', 'green_zone_part','school_education_centers_raion', 'healthcare_centers_raion']]

# 对各列按照设定的标准进行分类转化（使用.loc解决SettingWithCopyWarning问题）
threshold_raion_popul = 80000
y.loc[:, 'raion_popul'] = y['raion_popul'].astype(int)
y.loc[y['raion_popul'] > threshold_raion_popul, 'raion_popul'] = 1
y.loc[y['raion_popul'] <= threshold_raion_popul, 'raion_popul'] = 0

threshold_green_zone_part = 0.3
y.loc[:, 'green_zone_part'] = y['green_zone_part'].astype(float)
y.loc[y['green_zone_part'] > threshold_green_zone_part, 'green_zone_part'] = 1
y.loc[y['green_zone_part'] <= threshold_green_zone_part, 'green_zone_part'] = 0

threshold_school_education_centers_raion = 7
y.loc[:, 'school_education_centers_raion'] = y['school_education_centers_raion'].astype(int)
y.loc[y['school_education_centers_raion'] > threshold_school_education_centers_raion, 'school_education_centers_raion'] = 1
y.loc[y['school_education_centers_raion'] <= threshold_school_education_centers_raion, 'school_education_centers_raion'] = 0

threshold_healthcare_centers_raion = 2
y.loc[:, 'healthcare_centers_raion'] = y['healthcare_centers_raion'].astype(int)
y.loc[y['healthcare_centers_raion'] > threshold_healthcare_centers_raion, 'healthcare_centers_raion'] = 1
y.loc[y['healthcare_centers_raion'] <= threshold_healthcare_centers_raion, 'healthcare_centers_raion'] = 0


# 检查X的数据类型以及每列的数据类型
print("X的整体数据类型:", type(X))
for col in X.columns:
    print(f"{col}列的数据类型:", type(X[col].iloc[0]))


# 检查和处理y中的NaN值
y = y.apply(lambda col: col.fillna(col.mean()))

# 检查和处理X中的NaN值
if isinstance(X, pd.DataFrame):
    X = X.apply(lambda col: col.fillna(col.mean()))
else:
    print("X的数据类型不是DataFrame，无法按预期处理NaN值，请检查数据！")

# 检查和处理无穷值
X.replace([np.inf, -np.inf], np.nan, inplace=True)
if isinstance(X, pd.DataFrame):
    X = X.apply(lambda col: col.fillna(col.mean()))
else:
    print("X的数据类型不是DataFrame，无法按预期处理NaN值，请检查数据！")


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

        # 构建决策树分类模型，设置最大深度为4
        model = DecisionTreeClassifier(max_depth=4)
        model.fit(X_train, y_train)

        # 在训练集上进行预测
        y_train_pred = model.predict(X_train)

        # 确保y_train和y_train_pred为DataFrame类型，解决AttributeError问题
        if not isinstance(y_train, pd.DataFrame):
            y_train = pd.DataFrame(y_train)
        if not isinstance(y_train_pred, pd.DataFrame):
            y_train_pred = pd.DataFrame(y_train_pred)

        # 分别计算每个输出列的评估指标
        num_outputs = y_train.shape[1]
        accuracy_train_per_column = []
        f1_train_per_column = []
        recall_train_per_column = []
        precision_train_per_column = []

        for i in range(num_outputs):
            accuracy_train_per_column.append(accuracy_score(y_train.iloc[:, i], y_train_pred.iloc[:, i]))
            f1_train_per_column.append(f1_score(y_train.iloc[:, i], y_train_pred.iloc[:, i], average='macro'))
            recall_train_per_column.append(recall_score(y_train.iloc[:, i], y_train_pred.iloc[:, i], average='macro'))
            precision_train_per_column.append(precision_score(y_train.iloc[:, i], y_train_pred.iloc[:, i], average='macro'))

        average_accuracy_train = sum(accuracy_train_per_column) / num_outputs
        average_f1_train = sum(f1_train_per_column) / num_outputs
        average_recall_train = sum(recall_train_per_column) / num_outputs
        average_precision_train = sum(precision_train_per_column) / num_outputs

        accuracy_train.append(average_accuracy_train)
        f1_train.append(average_f1_train)
        recall_train.append(average_recall_train)
        precision_train.append(average_precision_train)

        # 在测试集上进行预测
        y_test_pred = model.predict(X_test)

        # 确保y_test和y_test_pred为DataFrame类型，解决AttributeError问题
        if not isinstance(y_test, pd.DataFrame):
            y_test = pd.DataFrame(y_test)
        if not isinstance(y_test_pred, pd.DataFrame):
            y_test_pred = pd.DataFrame(y_test_pred)

        # 分别计算每个输出列的评估指标
        accuracy_test_per_column = []
        f1_test_per_column = []
        recall_test_per_column = []
        precision_test_per_column = []

        for i in range(num_outputs):
            accuracy_test_per_column.append(accuracy_score(y_test.iloc[:, i], y_test_pred.iloc[:, i]))
            f1_test_per_column.append(f1_score(y_test.iloc[:, i], y_test_pred.iloc[:, i], average='macro'))
            recall_test_per_column.append(recall_score(y_test.iloc[:, i], y_test_pred.iloc[:, i], average='macro'))
            precision_test_per_column.append(precision_score(y_test.iloc[:, i], y_test_pred.iloc[:, i], average='macro'))

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


# 可视化决策树模型结构
def visualize_tree(model):
    dot_data = export_graphviz(model, out_file=None,
                               feature_names=X.columns,
                               class_names=['0', '1'],
                               filled=True, rounded=True,
                               special_characters=True)
    dot_data = graphviz.Source(dot_data)
    dot_data.render("decision_tree", format="png")
    plt.imshow(plt.imread("decision_tree.png"))
    plt.axis('off')
    plt.show()

# 获取最后一次实验的模型用于可视化
last_model = None
if len(accuracy_train_list) > 0:
    last_model = DecisionTreeClassifier(max_depth=4)  # 设置最大深度为4
    last_model.fit(X, y)

if last_model is not None:
    visualize_tree(last_model)
else:
    print("没有成功完成实验，无法进行决策树可视化。")