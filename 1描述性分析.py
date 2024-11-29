import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# 加载数据集，指定编码格式为ISO-8859-1
df = pd.read_csv(r'E:\桌面\train_without_noise.csv', encoding='ISO-8859-1')


# 显示数据集的基本信息
print(df.info())

# 计算描述性统计
print(df.describe())

# 检查缺失值
print(df.isnull().sum())

# 可视化部分

# 直方图：展示healthcare_centers_raion和school_education_centers_raion的分布
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(df['healthcare_centers_raion'], bins=20, color='blue', alpha=0.7)
plt.title('Distribution of Healthcare Centers (healthcare_centers_raion)')
plt.xlabel('Number of Healthcare Centers')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(df['school_education_centers_raion'], bins=20, color='green', alpha=0.7)
plt.title('Distribution of School Education Centers (school_education_centers_raion)')
plt.xlabel('Number of School Education Centers')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# 散点图：展示school_education_centers_raion和price_doc的关系
plt.figure(figsize=(10, 6))
sns.scatterplot(x='school_education_centers_raion', y='price_doc', data=df)
plt.title('Scatter Plot of School Education Centers vs Price')
plt.xlabel('Number of School Education Centers (school_education_centers_raion)')
plt.ylabel('Price (price_doc)')
plt.show()

# 条形图：展示house_category的分布
plt.figure(figsize=(12, 6))
sns.countplot(x='house_category', data=df)
plt.title('Distribution of House Categories')
plt.xlabel('House Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# 相关性热图
plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Heatmap')
plt.show()