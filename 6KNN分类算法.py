import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
import chardet
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimSun']


# 尝试自动检测文件编码
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    result = chardet.detect(raw_data)
    return result['encoding']


# 读取数据集并进行数据预处理（包括编码检测、缺失值处理、数据类型转换等）
def read_and_preprocess_data(file_path):
    encoding = detect_encoding(file_path)
    data = pd.read_csv(file_path, encoding=encoding)

    # 提取自变量和因变量（这里假设已经知道具体的列名，可根据实际情况修改）
    X = data[['price_doc']]
    y = data[['raion_popul', 'green_zone_part','school_education_centers_raion', 'healthcare_centers_raion']]

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

    return X, y


# 对数据进行分类转化（根据设定的阈值将连续型数据转化为离散型数据）
def categorize_data(y):
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

    return y


# 使用K近邻分类模型进行多次实验并计算评估指标
def run_knn_experiments(X, y, num_experiments=3):
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

            # 构建K近邻分类模型，设置n_neighbors为5（可根据实际情况调整）
            model = KNeighborsClassifier(n_neighbors=5)
            model.fit(X_train, y_train)

            # 在训练集上进行预测
            y_train_pred = model.predict(X_train)

            # 确保y_train和y_train_pred为DataFrame类型
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

            # 确保y_test和y_test_pred为DataFrame类型
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

    return accuracy_train_list, f1_train_list, recall_train_list, precision_train_list, accuracy_test_list, f1_test_list, recall_test_list, precision_test_list


# 可视化模型性能（通过绘制训练集和测试集准确率的折线图）
def visualize_model_performance(accuracy_train_list, accuracy_test_list):
    plt.plot(range(1, len(accuracy_train_list) + 1), accuracy_train_list, label='训练集准确率')
    plt.plot(range(1, len(accuracy_test_list) + 1), accuracy_test_list, label='测试集准确率')
    plt.xlabel('实验次数')
    plt.ylabel('准确率')
    plt.title('KNN分类模型准确率对比')
    plt.legend()
    plt.show()


# 可视化多个评估指标的折线图
def visualize_multiple_metrics(accuracy_train_list, accuracy_test_list, f1_train_list, f1_test_list, recall_train_list, recall_test_list, precision_train_list, precision_test_list):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(accuracy_train_list) + 1), accuracy_train_list, label='训练集准确率')
    plt.plot(range(1, len(accuracy_test_list) + 1), accuracy_test_list, label='测试集准确率')
    plt.xlabel('实验次数')
    plt.ylabel('指标值')
    plt.title('准确率对比')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(range(1, len(f1_train_list) + 1), f1_train_list, label='训练集F-度量值')
    plt.plot(range(1, len(f1_test_list) + 1), f1_test_list, label='测试集F-度量值')
    plt.xlabel('实验次数')
    plt.ylabel('指标值')
    plt.title('F-度量值对比')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(range(1, len(recall_train_list) + 1), recall_train_list, label='训练集召回率')
    plt.plot(range(1, len(recall_test_list) + 1), recall_test_list, label='测试集召回率')
    plt.xlabel('实验次数')
    plt.ylabel('指标值')
    plt.title('召回率对比')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(range(1, len(precision_train_list) + 1), precision_train_list, label='训练集精确率')
    plt.plot(range(1, len(precision_test_list) + 1), precision_test_list, label='测试集精确率')
    plt.xlabel('实验次数')
    plt.ylabel('指标值')
    plt.title('精确率对比')
    plt.legend()

    plt.tight_layout()
    plt.show()


# 可视化雷达图展示多个指标综合情况
def visualize_radar_chart(accuracy_train_list, accuracy_test_list, f1_train_list, f1_test_list, recall_train_list, recall_test_list, precision_train_list, precision_test_list):
    labels = ['准确率', 'F-度量值', '召回率', '精确率']
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    train_metrics = [
        np.mean(accuracy_train_list),
        np.mean(f1_train_list),
        np.mean(recall_train_list),
        np.mean(precision_train_list)
    ]
    train_metrics += train_metrics[:1]

    test_metrics = [
        np.mean(accuracy_test_list),
        np.mean(f1_test_list),
        np.mean(recall_test_list),
        np.mean(precision_test_list)
    ]
    test_metrics += test_metrics[:1]

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(1, 1, 1, polar=True)

    ax.plot(angles, train_metrics, label='训练集', marker='o')
    ax.fill(angles, train_metrics, alpha=0.25)

    ax.plot(angles, test_metrics, label='测试集', marker='x')
    ax.fill(angles, test_metrics, alpha=0.25)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], labels)
    plt.yticks(np.arange(0, 1.1, 0.1))

    plt.title('KNN分类模型各项指标综合对比')
    plt.legend(loc='upper right')
    plt.show()


if __name__ == "__main__":
    file_path = r"E:\桌面\train_without_noise3.csv"

    # 读取并预处理数据
    X, y = read_and_preprocess_data(file_path)

    # 对因变量进行分类转化
    y = categorize_data(y)

    # 运行K近邻实验并获取评估指标列表
    accuracy_train_list, f1_train_list, recall_train_list, precision_train_list, accuracy_test_list, f1_test_list, recall_test_list, precision_test_list = run_knn_experiments(X, y)

    # 打印实验结果
    num_experiments = len(accuracy_train_list)
    for i in range(num_experiments):
        print(f"实验{i + 1}：")
        print(f"平均训练集准确率: {accuracy_train_list[i]}")
        print(f"平均训练集F-度量值: {f1_train_list[i]}")
        print(f"平均训练集召回率: {accuracy_train_list[i]}")
        print(f"平均训练集精确率: {precision_train_list[i]}")
        print(f"平均测试集准确率: {accuracy_test_list[i]}")
        print(f"平均测试集F-度量值: {f1_test_list[i]}")
        print(f"平均测试集召回率: {accuracy_test_list[i]}")
        print(f"平行测试集精确率: {precision_test_list[i]}")

    # 可视化模型性能
    visualize_model_performance(accuracy_train_list, accuracy_test_list)
    visualize_multiple_metrics(accuracy_train_list, accuracy_test_list, f1_train_list, f1_test_list, recall_train_list, recall_test_list, precision_train_list, precision_test_list)
    visualize_radar_chart(accuracy_train_list, accuracy_test_list, f1_train_list, f1_test_list, recall_train_list, recall_test_list, precision_train_list, precision_test_list)