import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import warnings

warnings.filterwarnings("ignore")  # 忽略一些可能出现的警告信息

def perform_association_analysis():
    # 读取数据集
    try:
        data = pd.read_csv(r'E:\桌面\train_without_noise.csv')
    except FileNotFoundError:
        print("文件不存在，请检查文件路径是否正确。")
        return

    # 提取特征和目标变量
    X = data[['preschool_quota','school_quota']]
    y = data['price_doc']

    # 5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    for train_index, test_index in kf.split(X):
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]

        # 训练模型，并添加错误处理
        try:
            model = LinearRegression()
            model.fit(X_train, y_train)
        except Exception as e:
            print(f"模型训练出错: {e}")
            continue

        # 在测试集上进行预测
        try:
            y_pred = model.predict(X_test)
        except Exception as e:
            print(f"预测出错: {e}")
            continue

        # 计算性能指标
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append({'MSE': mse, 'R2': r2})

    # 检查并处理数据框中的问题
    results_df = pd.DataFrame(results)
    if 'MSE' in results_df.columns:
        mean_mse = results_df['MSE'].mean()
        std_mse = results_df['MSE'].std()
    else:
        print("数据框中不存在'MSE'列，无法计算平均MSE。")
        return

    mean_r2 = results_df['R2'].mean()
    std_r2 = results_df['R2'].std()

    # 制作汇总表
    summary = {'Metric': ['MSE', 'R2'],
               'Average': [mean_mse, mean_r2],
               'Standard Deviation': [std_mse, std_r2]}
    summary_df = pd.DataFrame(summary)

    # 打印汇总结果
    print("关联分析结果汇总：")
    print(summary_df)

if __name__ == "__main__":
    perform_association_analysis()