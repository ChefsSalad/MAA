import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 读取CSV文件（注意替换文件路径）
df = pd.read_csv('./processed_原油_day.csv', encoding='utf-8')

# 2. 截取从第31行开始（注意：Python中索引从0开始，所以第31行对应索引30），且只取第二列之后的数据
df_sub = df.iloc[30:, 1:]
print("截取后的数据预览：")
print(df_sub.head())

# 3. 查看截取后数据的统计信息（可选）
print("\n截取后数据统计信息：")
print(df_sub.describe())

# 4. 计算截取后数据的相关性矩阵
corr_matrix = df_sub.corr()
print("\n相关性矩阵：")
print(corr_matrix)

# 3. 可视化相关性矩阵（热图）
plt.figure(figsize=(20, 16), dpi=500)
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5)
plt.title("各列相关性热图")
plt.show()
