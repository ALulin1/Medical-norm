# Patent Pending: [RU2024170315-001D] - Dynamic Norm Induction in Multi-Agent Healthcare Systems
# Use of this technology requires explicit authorization from patent holders.
# Copyright [2024] [CHAO LI/ITMO]
# Licensed under the Apache License, Version 2.0 (the "License");
# Medical use requires compliance with ethical guidelines in ETHICAL_GUIDELINES.md.

#%%
import pandas as pd
import os

# 定义文件夹路径
folder_path = './by-year'

# 初始化一个空的DataFrame来存储结果
all_years_data = pd.DataFrame()

# 遍历文件夹中的每个CSV文件
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        # 获取年份
        year = file_name.split('_')[0]
        
        # 读取CSV文件
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)
        
        # 打印每年文件内容
        print(f"\n{year} Shannon Diversity Indices:")
        print(df)

        # 将数据添加到汇总DataFrame中
        df['Year'] = year  # 添加年份列
        all_years_data = pd.concat([all_years_data, df], ignore_index=True)

#%%
import pandas as pd
import os
import matplotlib.pyplot as plt

# 定义文件夹路径
folder_path = './by-year'

# 初始化一个空的DataFrame来存储结果
all_years_data = pd.DataFrame()

# 遍历文件夹中的每个CSV文件
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        # 获取年份
        year = file_name.split('_')[0]
        
        # 读取CSV文件
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)
        
        # 将数据添加到汇总DataFrame中
        df['Year'] = year  # 添加年份列
        all_years_data = pd.concat([all_years_data, df], ignore_index=True)

# 确保'Year'列是数值类型
all_years_data['Year'] = all_years_data['Year'].astype(int)

# 获取所有唯一的医生名
doctor_names = all_years_data['Doctor Name'].unique()

# 设置图像大小
plt.figure(figsize=(10, 6))

# 为每个医生绘制折线图
for doctor in doctor_names:
    # 筛选出该医生的所有年份数据
    doctor_data = all_years_data[all_years_data['Doctor Name'] == doctor]
    
    # 按年份排序
    doctor_data = doctor_data.sort_values('Year')
    
    # 特别加粗“Невролог - Петручик Ольга Викторовна”的线条
    if doctor == 'Невролог - Петручик Ольга Викторовна':
        plt.plot(doctor_data['Year'], doctor_data['Shannon Diversity Index'], marker='o', label=doctor, linewidth=5, color='red')
    else:
        plt.plot(doctor_data['Year'], doctor_data['Shannon Diversity Index'], marker='o', label=doctor, linewidth=1)

# # 添加图例
# plt.legend(loc='upper right', fontsize='small', bbox_to_anchor=(1.15, 1))


plt.xlabel('Year', fontsize=30)
plt.ylabel('Shannon Diversity Index', fontsize=25)
plt.xticks([2016, 2017, 2018, 2019, 2020],fontsize=20)  # 确保年份按顺序显示
plt.yticks(fontsize=25)
plt.grid(True)

# 保存图像
# Save the plot as a PNG file with a white background, high resolution, and tight bounding box
plt.savefig('shannon_diversity_all_doctors.png', bbox_inches='tight', dpi=300, facecolor='white')

# 展示图像
plt.show()


#%%
from matplotlib.ticker import FixedLocator
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# 定义文件夹路径
folder_path = './by-year'

# 初始化 DataFrame
all_years_data = pd.DataFrame()

# 读取文件数据
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        year = file_name.split('_')[0]
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)
        df['Year'] = int(year)
        df['Date'] = pd.to_datetime(df['Year'], format='%Y') + pd.offsets.YearEnd(0)
        df['Doctor Name'] = df['Doctor Name'].str.strip()
        all_years_data = pd.concat([all_years_data, df], ignore_index=True)

# 获取医生名
doctor_names = all_years_data['Doctor Name'].unique()

plt.figure(figsize=(12, 8))

# 绘制香农多样性指数
for doctor in doctor_names:
    doctor_data = all_years_data[all_years_data['Doctor Name'] == doctor].sort_values('Date')
    if doctor == 'Невролог - Петручик Ольга Викторовна':
        plt.plot(doctor_data['Date'], doctor_data['Shannon Diversity Index'], marker='o', linewidth=5, color='red')
    else:
        plt.plot(doctor_data['Date'], doctor_data['Shannon Diversity Index'], marker='o', linewidth=1)

# 定义工作坊和教学活动的日期（年, 月, 日）
workshops = {
    2017: [(2017, 4, 20), (2017, 12, 7)],
    2018: [(2018, 3, 16), (2018, 10, 12)],
    2019: [(2019, 4, 25), (2019, 10, 10)],
    2020: [(2020, 2, 6), (2020, 10, 15)]
}

teaching_activities = {
    2016: [(2016, 12, 23)],
    2017: [(2017, 6, 15), (2017, 11, 23)],
    2018: [(2018, 6, 28), (2018, 12, 7)],
    2019: [(2019, 6, 6), (2019, 12, 12)],
    2020: [(2020, 6, 26), (2020, 12, 10)]
}

# 在 Y 轴上方绘制事件点
y_event_offset = 2.7  # 设置点位于 Y 轴上方的固定位置

# 使用不同的符号来区分工作坊和教学活动
for dates in workshops.values():
    for date in dates:
        date_obj = datetime(*date)
        plt.scatter(date_obj, y_event_offset, color='green', marker='*', s=200, label='Workshop')

for dates in teaching_activities.values():
    for date in dates:
        date_obj = datetime(*date)
        plt.scatter(date_obj, y_event_offset - 0.1, color='blue', marker='X', s=200, label='Teaching Activity')

# 设置横轴格式，使标签垂直显示
ax = plt.gca()
plt.style.use('default')
ax.xaxis.set_major_locator(FixedLocator(mdates.date2num([datetime(year, 12, 31) for year in range(2016, 2021)])))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=0)  # 将横轴标签设置为不旋转，保持垂直显示

plt.ylabel('Shannon Diversity Index', fontsize=28)
plt.xticks(fontsize=16.5)
plt.yticks(fontsize=26)
plt.ylim([0.5, y_event_offset + 0.5])  # 扩展Y轴范围，以容纳事件点
plt.grid(True)

# 如果有多个相同标签只显示一次
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper left',fontsize=16)

plt.show()

#%%
import pandas as pd

# 读取 CSV 文件
data = pd.read_csv('output_by_year_v2/experiment_results_20_agents_50_runs_2016_to_2016.csv')

# 提取 steps 列
convergence_steps = data['Convergence Step']

# 计算平均值、最大值和最小值
average_steps = convergence_steps.mean()
max_step = convergence_steps.max()
min_step = convergence_steps.min()

# 输出结果
print(f'Average Convergence Steps: {average_steps}')
print(f'Highest Convergence Steps: {max_step}')
print(f'Lowest Convergence Steps: {min_step}')
#%%
import pandas as pd
import matplotlib.pyplot as plt


def calculate_and_plot_convergence_steps(file_info):
    """
    计算每个指定文件的平均收敛步数并生成柱状图。
    
    :param file_info: 一个包含文件路径和对应标签的列表
                      格式: [('文件路径', '标签')]
    """
    # 存储每个文件的平均收敛步数
    avg_steps = []
    labels = []

    # 逐个处理文件
    for file_path, label in file_info:
        # 读取 CSV 文件
        data = pd.read_csv(file_path)

        # 提取 'Convergence Step' 列并计算平均值
        convergence_steps = data['Convergence Step']
        average_steps = convergence_steps.mean()

        # 将结果添加到列表中
        avg_steps.append(average_steps)
        labels.append(label)

        # 输出每个文件的平均值
        print(f'{label} - Average Convergence Steps: {average_steps}')

    # 生成柱状图
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, avg_steps, color='blue')

    # 在每个柱子上标注数值
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', ha='center', va='bottom', fontsize=14)

    # 设置坐标轴标签和刻度字体大小
    plt.xlabel('Year Range', fontsize=28)
    plt.ylabel('Average Convergence Steps', fontsize=28)
    plt.grid(axis='y')  # 仅显示y轴的网格线
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    # 显示图形
    plt.tight_layout()
    plt.show()

# 使用示例：计算16-17，18-19，19-20三个文件的平均收敛步数
file_info = [
    ('output_by_year/experiment_results_20_agents_100_runs_2016_to_2017.csv', '2016-2017'),
    ('output_by_year/experiment_results_20_agents_100_runs_2017_to_2018_V3.csv', '2017-2018'),
    ('output_by_year/experiment_results_20_agents_100_runs_2018_to_2019.csv', '2018-2019'),
    ('output_by_year/experiment_results_20_agents_100_runs_2019_to_2020.csv', '2019-2020')
]

# 调用函数
calculate_and_plot_convergence_steps(file_info)

#%%
import pandas as pd
import os
import matplotlib.pyplot as plt


def calculate_average_and_increment(folder_path):
    # 初始化一个空的DataFrame来存储结果
    all_years_data = pd.DataFrame()

    # 遍历文件夹中的每个CSV文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            try:
                # 提取年份
                year = file_name.split('_')[0]  # 从文件名的第一个部分提取年份

                # 读取CSV文件
                file_path = os.path.join(folder_path, file_name)
                df = pd.read_csv(file_path)

                # 将数据添加到汇总DataFrame中
                df['Year'] = int(year)  # 添加年份列
                all_years_data = pd.concat([all_years_data, df], ignore_index=True)
            except Exception as e:
                print(f"Error reading {file_name}: {e}")

    # 确保'Year'列是数值类型
    all_years_data['Year'] = all_years_data['Year'].astype(int)

    # 计算每年的平均香农多样性指数
    mean_shannon_per_year = all_years_data.groupby('Year')['Shannon Diversity Index'].mean()

    # 输出每年的平均香农多样性指数
    print("每年的平均香农多样性指数：")
    print(mean_shannon_per_year)

    # 计算每个年份间隔的增量
    increments = mean_shannon_per_year.diff().dropna()  # 使用diff()计算增量并移除NaN

    # 输出每个年份间隔的增量
    print("\n每个年份间隔的平均香农多样性增量：")
    print(increments)

    return mean_shannon_per_year, increments


def plot_shannon_diversity_increments(mean_shannon_per_year, increments):
    # 创建柱状图
    plt.figure(figsize=(10, 6))

    # 使用颜色表示增量的正负
    colors = ['red' if inc < 0 else 'green' for inc in increments]  # 增量为负时使用红色，正时使用绿色

    plt.bar(increments.index, increments, color=colors)  # 直接绘制所有年份间隔

    # 在每个柱子上显示增量的具体值
    for i, inc in enumerate(increments):
        plt.text(increments.index[i], inc, f'{inc:.3f}', ha='center', va='bottom' if inc > 0 else 'top', fontsize=12)

    # 设置图表标题和标签
    # plt.title('Average Shannon Diversity Index Increment (Year to Year)', fontsize=18)
    plt.xlabel('Year Range', fontsize=30)
    plt.ylabel('Average Shannon Diversity Increment', fontsize=20)

    # 自定义横坐标标签为 '16-17', '17-18', '18-19', '19-20'
    labels = []
    for year in increments.index:
        previous_year = year - 1
        label = f"{str(previous_year)[-2:]}-{str(year)[-2:]}"
        labels.append(label)

    plt.xticks(increments.index, labels, fontsize=20)  # 使用自定义标签
    plt.yticks(fontsize=20)
    plt.grid(True)

    # 展示图像
    plt.tight_layout()
    plt.show()


# 调用计算和绘制函数
folder_path = './by-year'
mean_shannon_per_year, increments = calculate_average_and_increment(folder_path)
plot_shannon_diversity_increments(mean_shannon_per_year, increments)

#%%
import pandas as pd

# 读取CSV文件
csv_filename = 'output_by_year_v4/experiment_results_v5_20_agents_50_runs_2017_to_2017.csv'
df = pd.read_csv(csv_filename)

# # 过滤掉 Convergence Ratio 不等于 1 的行
# df = df.loc[df['Convergence Ratio'] == 1]

# 获取所有实验的列名
num_agents = 20  # 代理数量
initial_mean_columns = [f'Agent {i} Initial average mean' for i in range(num_agents)]
final_mean_columns = [f'Agent {i} Final average mean' for i in range(num_agents)]

# 计算每次实验中，20个代理的初始mean的平均值
df['Initial Mean Avg (per experiment)'] = df[initial_mean_columns].mean(axis=1)

# 计算每次实验中，20个代理的最终mean的平均值
df['Final Mean Avg (per experiment)'] = df[final_mean_columns].mean(axis=1)

# 计算每次实验的初始mean和最终mean的变化值（最终mean - 初始mean）
df['Mean Change (per experiment)'] = df['Final Mean Avg (per experiment)'] - df['Initial Mean Avg (per experiment)']

# 计算所有100次实验的初始均值的平均值
overall_initial_mean_avg = df['Initial Mean Avg (per experiment)'].mean()

# 计算所有100次实验的最终均值的平均值
overall_final_mean_avg = df['Final Mean Avg (per experiment)'].mean()

# 计算100次实验的总体mean变化值
overall_mean_change = df['Mean Change (per experiment)'].mean()

# 输出结果
print(f"每次实验20个代理的初始mean平均值:\n{df['Initial Mean Avg (per experiment)']}\n")
print(f"每次实验20个代理的最终mean平均值:\n{df['Final Mean Avg (per experiment)']}\n")
print(f"每次实验的mean变化值:\n{df['Mean Change (per experiment)']}\n")
print(f"100次实验的初始mean总体平均值: {overall_initial_mean_avg}")
print(f"100次实验的最终mean总体平均值: {overall_final_mean_avg}")
print(f"100次实验的mean变化总体平均值: {overall_mean_change}")

# # 保存结果到CSV
# df.to_csv('output_by_year_v4/experiment_results_v4_20_agents_100_runs_with_avg_and_change_filtered.csv', index=False)

#%%
import pandas as pd
import matplotlib.pyplot as plt

# 定义文件路径和年份列表
file_info = [
    ('output_by_year_v4/experiment_results_v5_30_agents_20_runs_2016_to_2016.csv', '16-17'),
    ('output_by_year_v4/experiment_results_v5_20_agents_50_runs_2017_to_2017.csv', '17-18'),
    # ('output_by_year_v4/experiment_results_v5_20_agents_50_runs_2018_to_2018.csv', '2018'),
    ('output_by_year_v4/experiment_results_v4_20_agents_50_runs_2019_to_2019.csv', '18-19'),
    ('output_by_year_v4/experiment_results_v4_20_agents_50_runs_2020_to_2020.csv', '19-20')
]

# 初始化存储每年 mean_change 的列表
overall_mean_changes = []
years = []

# 遍历每个文件并计算 overall_mean_change
for file_path, year_range in file_info:
    # 读取CSV文件
    df = pd.read_csv(file_path)

    # # 过滤掉 Convergence Ratio 不等于 1 的行
    # df = df.loc[df['Convergence Ratio'] == 1]

    # 获取所有实验的列名
    num_agents = 20  # 代理数量
    initial_mean_columns = [f'Agent {i} Initial average mean' for i in range(num_agents)]
    final_mean_columns = [f'Agent {i} Final average mean' for i in range(num_agents)]

    # 计算每次实验中，20个代理的初始mean的平均值
    df['Initial Mean Avg (per experiment)'] = df[initial_mean_columns].mean(axis=1)

    # 计算每次实验中，20个代理的最终mean的平均值
    df['Final Mean Avg (per experiment)'] = df[final_mean_columns].mean(axis=1)

    # 计算每次实验的初始mean和最终mean的变化值（最终mean - 初始mean）
    df['Mean Change (per experiment)'] = df['Final Mean Avg (per experiment)'] - df['Initial Mean Avg (per experiment)']

    # 计算100次实验的总体mean变化值
    overall_mean_change = df['Mean Change (per experiment)'].mean()

    # 记录每年的 mean_change 和年份
    overall_mean_changes.append(overall_mean_change)
    years.append(year_range)

# 绘制直方图
plt.figure(figsize=(10, 6))
plt.bar(years, overall_mean_changes, color='green')
plt.title('model', fontsize=18)
plt.xlabel('Year Range', fontsize=14)
plt.ylabel('Overall average SDI  Change Change', fontsize=14)
plt.grid(True, axis='y')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('overall_mean_change_histogram.png', dpi=300)
plt.show()

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 定义文件路径和年份列表
file_info = [
    ('output_by_year_v4/experiment_results_v5_20_agents_200_runs_2016_to_2016.csv', '16-17'),
    ('output_by_year_v4/experiment_results_v5_20_agents_50_runs_2017_to_2017.csv', '17-18'),
    ('output_by_year_v4/experiment_results_v4_20_agents_50_runs_2019_to_2019.csv', '18-19'),
    ('output_by_year_v4/experiment_results_v4_20_agents_50_runs_2020_to_2020.csv', '19-20')
]

# 初始化存储每年 mean_change、标准误差（SE）和年份的列表
overall_mean_changes = []
standard_errors = []
years = []

# 遍历每个文件并计算 overall_mean_change 和标准误差
for file_path, year_range in file_info:
    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 获取所有实验的列名
    num_agents = 20  # 代理数量
    initial_mean_columns = [f'Agent {i} Initial average mean' for i in range(num_agents)]
    final_mean_columns = [f'Agent {i} Final average mean' for i in range(num_agents)]

    # 计算每次实验中，20个代理的初始mean的平均值
    df['Initial Mean Avg (per experiment)'] = df[initial_mean_columns].mean(axis=1)

    # 计算每次实验中，20个代理的最终mean的平均值
    df['Final Mean Avg (per experiment)'] = df[final_mean_columns].mean(axis=1)

    # 计算每次实验的初始mean和最终mean的变化值（最终mean - 初始mean）
    df['Mean Change (per experiment)'] = df['Final Mean Avg (per experiment)'] - df['Initial Mean Avg (per experiment)']

    # 计算 mean_change 的总体平均值和标准误差
    overall_mean_change = df['Mean Change (per experiment)'].mean()
    standard_error = df['Mean Change (per experiment)'].std() / np.sqrt(len(df))  # 计算标准误差

    # 记录每年的 mean_change、标准误差 和年份
    overall_mean_changes.append(overall_mean_change)
    standard_errors.append(standard_error)
    years.append(year_range)

# 绘制带置信区间的条形图
plt.figure(figsize=(10, 6))
plt.bar(years, overall_mean_changes, yerr=[1.645 * se for se in standard_errors], color='green', capsize=5)  # capsize 是误差棒的顶端宽度
plt.title('Model Mean Change with 90% Confidence Intervals', fontsize=18)
plt.xlabel('Year Range', fontsize=14)
plt.ylabel('Overall average SDI Change', fontsize=14)
plt.grid(True, axis='y')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
# plt.savefig('overall_mean_change_with_confidence_intervals.png', dpi=300)
plt.show()

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# File paths and year labels
file_info = [
    ('output_by_year_v4/experiment_results_v5_30_agents_20_runs_2016_to_2016.csv', '16-17'),
    ('output_by_year_v4/experiment_results_v5_20_agents_50_runs_2017_to_2017.csv', '17-18'),
    ('output_by_year_v4/experiment_results_v4_20_agents_50_runs_2019_to_2019.csv', '18-19'),
    ('output_by_year_v4/experiment_results_v4_20_agents_50_runs_2020_to_2020.csv', '19-20')
]

# Lists to store extracted data
mean_changes = []
years = []

# Step 1: Data extraction from each file
for file_path, year_range in file_info:
    df = pd.read_csv(file_path)
    num_agents = 30 if '30_agents' in file_path else 20  # Number of agents based on the file name

    # Calculate Mean Change for each experiment and agent
    initial_mean_columns = [f'Agent {i} Initial average mean' for i in range(num_agents)]
    final_mean_columns = [f'Agent {i} Final average mean' for i in range(num_agents)]
    df['Mean Change (per agent)'] = df[final_mean_columns].mean(axis=1) - df[initial_mean_columns].mean(axis=1)

    # Store data
    mean_changes.extend(df['Mean Change (per agent)'].values)
    years.extend([year_range] * len(df))  # Add year label for each data point

# Convert the data to arrays for easier processing
mean_changes = np.array(mean_changes)
years = np.array(years)

# Step 2: Sliding window fit
window_size = 30  # Adjust window size for smoothing
smoothed_mean_changes = uniform_filter1d(mean_changes, size=window_size)

# Step 3: Plotting the sliding window curve and yearly observation points
plt.figure(figsize=(10, 6))
plt.plot(smoothed_mean_changes, 'k-', label='Sliding Window Fit')  # Black solid line for the sliding window

# Add yearly observation points
for year in set(years):
    year_mask = (years == year)
    overall_mean_change = np.mean(mean_changes[year_mask])  # Yearly observation
    plt.plot(np.mean(np.where(year_mask)), overall_mean_change, 'ko', markersize=10, label=f'{year} Observation')

# Step 4: Add confidence intervals
# Add confidence intervals
for ci, ci_label in zip([1.645, 1.96], ['90% CI', '95% CI']):  # 90% and 95% confidence intervals
    ci_values = ci * np.std(mean_changes) / np.sqrt(len(mean_changes))
    plt.fill_between(range(len(smoothed_mean_changes)), 
                     smoothed_mean_changes - ci_values, 
                     smoothed_mean_changes + ci_values, 
                     alpha=0.3, linestyle='--', label=ci_label)


# Labels and legend
plt.title('Mean Change with 90% and 95% Confidence Intervals')
plt.xlabel('Year Range')
plt.ylabel('Overall Mean Change')
plt.xticks(ticks=[0, len(mean_changes)//3, 2*len(mean_changes)//3, len(mean_changes)], 
           labels=['16-17', '17-18', '18-19', '19-20'])

plt.legend()
plt.grid(True)
plt.tight_layout()

# # Save and show plot
# plt.savefig('mean_change_sliding_window_with_CI.png', dpi=300)
plt.show()

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# File paths and year labels
file_info = [
    ('output_by_year_v4/experiment_results_v5_40_agents_20_runs_2016_to_2016.csv', '16-17'),
    ('output_by_year_v4/experiment_results_v5_20_agents_50_runs_2017_to_2017.csv', '17-18'),
    ('output_by_year_v4/experiment_results_v4_20_agents_50_runs_2019_to_2019.csv', '18-19'),
    ('output_by_year_v4/experiment_results_v4_20_agents_50_runs_2020_to_2020.csv', '19-20')
]

# Define colors for different years
colors = ['red', 'blue', 'green', 'purple']

# Lists to store extracted data
mean_changes = []
years = []

# Step 1: Data extraction from each file
for file_path, year_range in file_info:
    df = pd.read_csv(file_path)
    num_agents = 30 if '30_agents' in file_path else 20  # Number of agents based on the file name

    # Calculate Mean Change for each experiment and agent
    initial_mean_columns = [f'Agent {i} Initial average mean' for i in range(num_agents)]
    final_mean_columns = [f'Agent {i} Final average mean' for i in range(num_agents)]
    df['Mean Change (per agent)'] = df[final_mean_columns].mean(axis=1) - df[initial_mean_columns].mean(axis=1)

    # Store data
    mean_changes.extend(df['Mean Change (per agent)'].values)
    years.extend([year_range] * len(df))  # Add year label for each data point

# Convert the data to arrays for easier processing
mean_changes = np.array(mean_changes)
years = np.array(years)

# Step 2: Sliding window fit
window_size = 50  # Adjust window size for smoothing
smoothed_mean_changes = uniform_filter1d(mean_changes, size=window_size)

# Make sure smoothed_mean_changes and CIs have the same length
valid_length = len(smoothed_mean_changes)

# Step 3: Dynamic CI for sliding window
ci_90 = []
ci_95 = []

# Iterate over the data in the sliding window
for i in range(len(mean_changes) - window_size + 1):
    window_data = mean_changes[i:i + window_size]

    # Calculate standard deviation for the current window
    window_std = np.std(window_data)

    # 90% and 95% confidence intervals based on the sliding window data
    ci_90.append(1.645 * window_std / np.sqrt(window_size))
    ci_95.append(1.96 * window_std / np.sqrt(window_size))

# Extend CI arrays to match the valid length of the smoothed data
ci_90 = np.pad(ci_90, (window_size // 2, window_size // 2), 'edge')[:valid_length]
ci_95 = np.pad(ci_95, (window_size // 2, window_size // 2), 'edge')[:valid_length]

# Step 4: Plotting the sliding window curve and yearly observation points
plt.figure(figsize=(10, 6))
plt.plot(smoothed_mean_changes, 'k-', label='Sliding Window Fit')  # Black solid line for the sliding window

# Add yearly observation points with different colors
for year, color in zip(set(years), colors):
    year_mask = (years == year)
    overall_mean_change = np.mean(mean_changes[year_mask])  # Yearly observation
    plt.plot(np.mean(np.where(year_mask)), overall_mean_change, 'o', markersize=10, color=color,
             label=f'{year} Observation')

# Step 5: Plotting confidence intervals for sliding window
plt.fill_between(range(valid_length),
                 smoothed_mean_changes - ci_90,
                 smoothed_mean_changes + ci_90,
                 alpha=0.3, linestyle='--', label='90% CI')

plt.fill_between(range(valid_length),
                 smoothed_mean_changes - ci_95,
                 smoothed_mean_changes + ci_95,
                 alpha=0.2, linestyle='--', label='95% CI')
# Step 4: Plotting the sliding window curve and yearly observation points
plt.style.use('default')  # Set white background
# Labels and legend
plt.title('Mean Change with 90% and 95% Confidence Intervals')
plt.xlabel('Year Range')
plt.ylabel('Overall Mean Change')
plt.xticks(ticks=[0, len(mean_changes) // 3, 2 * len(mean_changes) // 3, len(mean_changes)],
           labels=['16-17', '17-18', '18-19', '19-20'])

plt.legend()
plt.grid(True)
plt.tight_layout()

# Save and show plot
plt.savefig('mean_change_sliding_window_with_CI-50-colored-v2.png', dpi=300)
plt.show()

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# File paths and year labels
file_info = [
    ('output_by_year_v4/experiment_results_v5_40_agents_20_runs_2016_to_2016.csv', '2017-12-31'),
    ('output_by_year_v4/experiment_results_v5_20_agents_50_runs_2017_to_2017.csv', '2018-12-31'),
    ('output_by_year_v4/experiment_results_v4_20_agents_50_runs_2019_to_2019.csv', '2019-12-31'),
    ('output_by_year_v4/experiment_results_v4_20_agents_50_runs_2020_to_2020.csv', '2020-12-31')
]

# Define colors for different years
colors = ['red', 'blue', 'green', 'purple']

# Lists to store extracted data
mean_changes = []
years = []

# Step 1: Data extraction from each file
for file_path, year_label in file_info:
    df = pd.read_csv(file_path)
    num_agents = 30 if '30_agents' in file_path else 20  # Number of agents based on the file name

    # Calculate Mean Change for each experiment and agent
    initial_mean_columns = [f'Agent {i} Initial average mean' for i in range(num_agents)]
    final_mean_columns = [f'Agent {i} Final average mean' for i in range(num_agents)]
    df['Mean Change (per agent)'] = df[final_mean_columns].mean(axis=1) - df[initial_mean_columns].mean(axis=1)

    # Store data
    mean_changes.extend(df['Mean Change (per agent)'].values)
    years.extend([year_label] * len(df))  # Add year label for each data point

# Convert the data to arrays for easier processing
mean_changes = np.array(mean_changes)
years = np.array(years)

# Step 2: Sliding window fit
window_size = 50  # Adjust window size for smoothing
smoothed_mean_changes = uniform_filter1d(mean_changes, size=window_size)

# Ensure smoothed_mean_changes and ci_95 have the same length
valid_length = len(smoothed_mean_changes)

# Step 3: Dynamic CI for sliding window
ci_95 = []

# Iterate over the data in the sliding window
for i in range(len(mean_changes) - window_size + 1):
    window_data = mean_changes[i:i + window_size]

    # Calculate standard deviation for the current window
    window_std = np.std(window_data)

    # 95% confidence interval based on the sliding window data
    ci_95.append(1.96 * window_std / np.sqrt(window_size))

# Extend ci_95 array to match the valid length of the smoothed data
ci_95 = np.pad(ci_95, (window_size // 2, window_size // 2), 'edge')[:valid_length]

# Step 4: Plotting the sliding window curve and yearly observation points
plt.figure(figsize=(10, 6))
plt.plot(smoothed_mean_changes, 'k-', label='Sliding Window Fit')  # Black solid line for the sliding window

# Collect x_positions and labels for x-ticks
x_positions = []
labels = []
unique_years = sorted(set(years))
for year_label, color in zip(unique_years, colors):
    year_mask = (years == year_label)
    overall_mean_change = np.mean(mean_changes[year_mask])  # Yearly observation
    x_position = np.mean(np.where(year_mask))  # x position where the observation point is plotted
    x_positions.append(x_position)
    labels.append(year_label)
    plt.plot(x_position, overall_mean_change, 'o', markersize=10, color=color,
             label=f'{year_label} Observation')

# Set x-ticks at the positions of the observation points
plt.xticks(ticks=x_positions, labels=labels)

# Step 5: Plotting confidence intervals for sliding window
plt.fill_between(range(valid_length),
                 smoothed_mean_changes - ci_95,
                 smoothed_mean_changes + ci_95,
                 alpha=0.2, linestyle='--', label='95% CI')

# Set style for white background
plt.style.use('default')

# Labels and legend
plt.title('Mean Change with 95% Confidence Intervals')
plt.xlabel('Observation Date')
plt.ylabel('Overall Mean Change')

plt.legend()
plt.grid(True)
plt.tight_layout()

# Do not save the plot
plt.savefig('mean_change_sliding_window_with_95_CI_colored_years_corrected_v3.png', dpi=300)
plt.show()

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# 文件路径和年份标签
file_info = [
    ('output_by_year_v4/experiment_results_v5_30_agents_20_runs_2016_to_2016.csv', '2017-12-31'),
    ('output_by_year_v4/experiment_results_v5_20_agents_50_runs_2017_to_2017.csv', '2018-12-31'),
    ('output_by_year_v4/experiment_results_v4_20_agents_50_runs_2019_to_2019.csv', '2019-12-31'),
    ('output_by_year_v4/experiment_results_v4_20_agents_50_runs_2020_to_2020.csv', '2020-12-31')
]

# 为不同的年份定义颜色
colors = ['red', 'blue', 'green', 'purple']

# 存储提取的数据
mean_changes = []
years = []

# 步骤1：从每个文件中提取数据
for file_path, year_label in file_info:
    df = pd.read_csv(file_path)
    num_agents = 30 if '30_agents' in file_path else 20  # 根据文件名确定代理数量

    # 计算每个实验和代理的平均变化
    initial_mean_columns = [f'Agent {i} Initial average mean' for i in range(num_agents)]
    final_mean_columns = [f'Agent {i} Final average mean' for i in range(num_agents)]
    df['Mean Change (per agent)'] = df[final_mean_columns].mean(axis=1) - df[initial_mean_columns].mean(axis=1)

    # 存储数据
    mean_changes.extend(df['Mean Change (per agent)'].values)
    years.extend([year_label] * len(df))  # 为每个数据点添加年份标签

# 将数据转换为数组以便于处理
mean_changes = np.array(mean_changes)
years = np.array(years)

# 步骤2：滑动窗口拟合
window_size = 30  # 调整窗口大小以平滑曲线
smoothed_mean_changes = uniform_filter1d(mean_changes, size=window_size)

# 确保 smoothed_mean_changes 和 ci_95 具有相同的长度
valid_length = len(smoothed_mean_changes)

# 步骤3：为滑动窗口计算动态置信区间
ci_95 = []

# 迭代滑动窗口中的数据
for i in range(len(mean_changes) - window_size + 1):
    window_data = mean_changes[i:i + window_size]

    # 计算当前窗口的标准差
    window_std = np.std(window_data)

    # 基于滑动窗口数据的95%置信区间
    ci_95.append(1.96 * window_std / np.sqrt(window_size))

# 扩展 ci_95 数组以匹配平滑数据的有效长度
ci_95 = np.pad(ci_95, (window_size // 2, window_size // 2), 'edge')[:valid_length]

# 步骤4：绘制滑动窗口曲线和年度观测点
plt.figure(figsize=(10, 6))
plt.plot(smoothed_mean_changes, 'k-', label='Sliding Window Fit')  # 滑动窗口的黑色实线

# 收集 x 轴位置和标签
x_positions = [0]  # 在位置 0 添加 '2016-12-23' 标签
labels = ['2016-12-23']
unique_years = sorted(set(years))
for year_label, color in zip(unique_years, colors):
    year_mask = (years == year_label)
    overall_mean_change = np.mean(mean_changes[year_mask])  # 年度观测值
    x_position = np.mean(np.where(year_mask))  # 观测点的 x 轴位置
    x_positions.append(x_position)
    labels.append(year_label)
    plt.plot(x_position, overall_mean_change, 'o', markersize=10, color=color,
             label=f'{year_label} Observation')

# 在观测点的位置设置 x 轴刻度，包括起始标签
plt.xticks(ticks=x_positions, labels=labels,rotation=45)

# 步骤5：绘制滑动窗口的置信区间
plt.fill_between(range(valid_length),
                 smoothed_mean_changes - ci_95,
                 smoothed_mean_changes + ci_95,
                 alpha=0.2, linestyle='--', label='95% CI')

# 设置白色背景样式
plt.style.use('default')

# 添加标题和坐标轴标签
plt.title('Mean Change with 95% Confidence Intervals')
plt.xlabel('Observation Date')
plt.ylabel('Overall Mean Change')

plt.legend()
plt.grid(True)
plt.tight_layout()

# 不保存图像
plt.savefig('mean_change_sliding_window_with_95_CI_colored_years_corrected_v2.png', dpi=300)
plt.show()

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# File paths and year labels
file_info = [
    ('output_by_year_v4/experiment_results_v5_30_agents_50_runs_2016_to_2016.csv', '2017-12-31'),
    ('output_by_year_v4/experiment_results_v5_20_agents_50_runs_2017_to_2017.csv', '2018-12-31'),
    ('output_by_year_v4/experiment_results_v4_20_agents_50_runs_2019_to_2019.csv', '2019-12-31'),
    ('output_by_year_v4/experiment_results_v4_20_agents_50_runs_2020_to_2020.csv', '2020-12-31')
]

# Lists to store extracted data
mean_changes = []
years = []

# Step 1: Data extraction from each file
for file_path, year_label in file_info:
    df = pd.read_csv(file_path)
    num_agents = 30 if '30_agents' in file_path else 20  # Number of agents based on the file name

    # Calculate Mean Change for each experiment and agent
    initial_mean_columns = [f'Agent {i} Initial average mean' for i in range(num_agents)]
    final_mean_columns = [f'Agent {i} Final average mean' for i in range(num_agents)]
    df['Mean Change (per agent)'] = df[final_mean_columns].mean(axis=1) - df[initial_mean_columns].mean(axis=1)

    # Store data
    mean_changes.extend(df['Mean Change (per agent)'].values)
    years.extend([year_label] * len(df))  # Add year label for each data point

# Convert the data to arrays for easier processing
mean_changes = np.array(mean_changes)
years = np.array(years)

# Step 2: Sliding window fit
window_size = 30  # Adjust window size for smoothing
smoothed_mean_changes = uniform_filter1d(mean_changes, size=window_size)

# Ensure smoothed_mean_changes and ci_95 have the same length
valid_length = len(smoothed_mean_changes)

# Step 3: Dynamic CI for sliding window
ci_95 = []

# Iterate over the data in the sliding window
for i in range(len(mean_changes) - window_size + 1):
    window_data = mean_changes[i:i + window_size]

    # Calculate standard deviation for the current window
    window_std = np.std(window_data)

    # 95% confidence interval based on the sliding window data
    ci_95.append(1.96 * window_std / np.sqrt(window_size))

# Extend ci_95 array to match the valid length of the smoothed data
ci_95 = np.pad(ci_95, (window_size // 2, window_size // 2), 'edge')[:valid_length]

# Step 4: Plotting the sliding window curve and yearly observation points
plt.figure(figsize=(10, 6))
plt.plot(smoothed_mean_changes, 'k-', label='Sliding Window Fit')  # Black solid line for the sliding window

# Collect x_positions and labels for x-ticks
x_positions = []
labels = []
unique_years = sorted(set(years))
for year_label in unique_years:
    year_mask = (years == year_label)
    overall_mean_change = np.mean(mean_changes[year_mask])  # Yearly observation
    x_position = np.mean(np.where(year_mask))  # x position where the observation point is plotted
    x_positions.append(x_position)
    labels.append(year_label)
# Only one label for all observations (to avoid multiple legend entries for different years)
    plt.plot(x_position, overall_mean_change, 'ko', markersize=10, label='Yearly Observation' if year_label == unique_years[0] else "")

# Set x-ticks at the positions of the observation points
plt.xticks(ticks=x_positions, labels=labels,fontsize=16)
plt.yticks(fontsize=16)
# Step 5: Plotting confidence intervals for sliding window
plt.fill_between(range(valid_length),
                 smoothed_mean_changes - ci_95,
                 smoothed_mean_changes + ci_95,
                 alpha=0.2, linestyle='--', label='95% CI')

# Set style for white background
plt.style.use('default')

# Labels and legend
plt.title('Mean Change with 95% Confidence Intervals', fontsize=20)
plt.xlabel('Observation Date', fontsize=20)
plt.ylabel('Overall Mean Change', fontsize=20)

plt.legend()
plt.grid(True)
plt.tight_layout()

# # Save and show plot
plt.savefig('mean_change_sliding_window_with_95_CI_colored_years_corrected_v5.png', dpi=300)
plt.show()
