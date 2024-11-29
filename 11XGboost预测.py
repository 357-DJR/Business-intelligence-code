import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据集
data = pd.read_csv(r'E:\桌面\train_without_noise3.csv')

# 数据探索和预处理
features = ['full_sq', 'life_sq', 'floor', 'sub_area', 'raion_popul', 'green_zone_part', 'indust_part', 'children_preschool', 'preschool_quota', 'school_quota', 'hospital_beds_raion', 'healthcare_centers_raion', 'sport_objects_raion', 'shopping_centers_raion']
X = data[features]
y = data['price_doc']

# 调整特征范围
X = X.select_dtypes(include=[np.number])
X.loc[:, 'full_sq'] = X['full_sq'].clip(20, 130)
X.loc[:, 'life_sq'] = X['life_sq'].clip(10, 100)
# 处理缺失值
X.fillna(X.mean(), inplace=True)

# 存储结果
results = []

# 进行三次实验
for i in range(3):
    # 划分训练集和测试集，每次实验设置不同的随机数种子
    random_state_value = np.random.randint(0, 1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state_value)

    # 创建XGBoost数据集
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # 设置XGBoost参数，这里也可以根据需要调整随机数种子的设置方式
    params = {
        'objective':'reg:squarederror',
        'eval_metric': ['mae', 'rmse'],
        'max_depth': 6,
        'eta': 0.1,
        'gamma': 0,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'lambda': 1,
        'alpha': 0,
        'seed': np.random.randint(0, 1000)
    }

    # 训练模型
    model = xgb.train(params, dtrain, num_boost_round=100)

    # 预测
    y_train_pred = model.predict(dtrain)
    y_test_pred = model.predict(dtest)

    # 计算性能指标
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test))

    # 存储结果
    results.append({
        'Train MAE': train_mae,
        'Train RMSE': train_rmse,
        'Train MAPE': train_mape,
        'Test MAE': test_mae,
        'Test RMSE': test_rmse,
        'Test MAPE': test_mape
    })

    # 打印每次实验的结果
    print(f'Experiment {i + 1}:')
    print(f'Training Set - MAE: {train_mae}, RMSE: {train_rmse}, MAPE: {train_mape}%')
    print(f'Test Set - MAE: {test_mae}, RMSE: {test_rmse}, MAPE: {test_mape}%')
    print('-----------------------------------')

# 计算所有实验的平均性能指标
avg_results = {
    'Train MAE': np.mean([result['Train MAE'] for result in results]),
    'Train RMSE': np.mean([result['Train RMSE'] for result in results]),
    'Train MAPE': np.mean([result['Train MAPE'] for result in results]),
    'Test MAE': np.mean([result['Test MAE'] for result in results]),
    'Test RMSE': np.mean([result['Test RMSE'] for result in results]),
    'Test MAPE': np.mean([result['Test MAPE'] for result in results])
}

# 打印平均性能指标
print('Average Performance Metrics:')
print(f'Training Set - MAE: {avg_results["Train MAE"]}, RMSE: {avg_results["Train RMSE"]}, MAPE: {avg_results["Train MAPE"]}')
print(f'Test Set - MAE: {avg_results["Test MAE"]}, RMSE: {avg_results["Test RMSE"]}, MAPE: {avg_results["Test MAPE"]}')

# 可视化性能指标
metrics = ['MAE', 'RMSE', 'MAPE']

fig, axs = plt.subplots(3, 1, figsize=(10, 15))

for i, metric in enumerate(metrics):
    train_values = [result['Train ' + metric] for result in results]
    test_values = [result['Test ' + metric] for result in results]

    axs[i].plot(['Train', 'Test'], [train_values[0], test_values[0]], marker='o', label='Experiment 1')
    axs[i].plot(['Train', 'Test'], [train_values[1], test_values[1]], marker='s', label='Experiment 2')
    axs[i].plot(['Train', 'Test'], [train_values[2], test_values[2]], marker='^', label='Experiment 3')

    axs[i].set_title(f'{metric} over 3 Experiments')
    axs[i].set_ylabel(metric)
    axs[i].set_xlabel('Dataset')
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()
plt.show()

# 绘制训练集和测试集误差分布图
plt.figure(figsize=(12, 6))

# 训练集误差分布图
plt.subplot(1, 2, 1)
sns.histplot(y_train - y_train_pred, kde=True)
plt.title('Train Set Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')

# 测试集误差分布图
plt.subplot(1, 2, 2)
sns.histplot(y_test - y_test_pred, kde=True)
plt.title('Test Set Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# 绘制训练集和测试集实际值与预测值的散点图
plt.figure(figsize=(12, 6))

# 训练集实际值与预测值的散点图
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2)
plt.title('Train Set Actual vs Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')

# 测试集实际值与预测值的散点图
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Test Set Actual vs Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')

plt.tight_layout()
plt.show()