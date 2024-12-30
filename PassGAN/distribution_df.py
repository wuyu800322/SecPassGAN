import re
import numpy as np

# 读取 scores.txt 文件
with open('scores.txt', 'r', encoding='utf-8') as file:
    data = file.read()

# 使用正则表达式提取所有的数值
numbers = re.findall(r"\[(\d+\.\d+)\]", data)
numbers = list(map(float, numbers))

# 定义分布区间
bins = [0, 0.5, 1, 1.5, 2, float('inf')]
labels = ['0~0.5', '0.5~1', '1~1.5', '1.5~2', '2.0~4']

# 使用 numpy 来统计每个区间的频率
counts, _ = np.histogram(numbers, bins=bins)

# 打印分布结果
distribution = dict(zip(labels, counts))

# 显示分布结果
import pandas as pd
distribution_df = pd.DataFrame(list(distribution.items()), columns=['Range', 'Count'])

# 输出分布情况
print(distribution_df)
