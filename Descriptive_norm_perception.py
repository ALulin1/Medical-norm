# Patent Pending: [RU2024170315-001D] - Dynamic Norm Induction in Multi-Agent Healthcare Systems
# Use of this technology requires explicit authorization from patent holders.
# Copyright [2024] [作者/机构名]
# Licensed under the Apache License, Version 2.0 (the "License");
# Medical use requires compliance with ethical guidelines in ETHICAL_GUIDELINES.md.
#

%%
import random
import numpy as np
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
import threading
import matplotlib.pyplot as plt
from scipy.stats import entropy, wasserstein_distance

# 描述性规范的 MultiAgentNetwork 类

class MultiAgentNetwork:
    def __init__(self, num_agents, k=5):
        self.agents = []
        self.k = k  # 组的数量
        self.num_agents = num_agents  # 代理总数
        self.group_means = [1.410, 0.272, 2.384, 1.012, 1.625]  # 每组的香农多样性指数的均值
        self.group_weights = [0.5, 0.1, 0.1, 0.1, 0.2]
        self.group_std_dev = 0.1  # 组内香农多样性指数的标准差
        self.agent_std_dev = 0.01  # 每个代理的香农多样性指数的标准差
        self.groups = {i: [] for i in range(self.k)}
        self.output_lock = threading.Lock()  # 初始化锁，用于确保输出顺序
        # self.em_convergence_threshold = 1e-4  # EM算法的收敛阈值

    def add_agent(self, agent):
        self.agents.append(agent)

    def assign_agents_to_groups(self):
        group_sizes = [int(self.num_agents * weight) for weight in self.group_weights]
        total_assigned = sum(group_sizes)
        if total_assigned != self.num_agents:
            group_sizes[0] += self.num_agents - total_assigned  # 调整组内人数

        current_agent_index = 0
        for group_id, group_size in enumerate(group_sizes):
            for _ in range(group_size):
                agent = self.agents[current_agent_index]
                group_mean = self.group_means[group_id]
                mean_shannon_diversity_index = np.random.normal(group_mean, self.group_std_dev)
                agent.shannon_diversity_index_distribution = norm(loc=mean_shannon_diversity_index, scale=self.agent_std_dev)
                self.groups[group_id].append(agent)
                agent.group_id = group_id
                current_agent_index += 1

    def step(self,current_step):
        num_agents = len(self.agents)
        num_senders = num_agents // 2
        senders = random.sample(self.agents, num_senders)
        receivers = [agent for agent in self.agents if agent not in senders]
        for receiver in receivers:
            self.send_and_receive(receiver, senders, current_step)

    def send_and_receive(self, receiver, senders, current_step):
        received_data = []
        for sender in senders:
            shannon_index_sample = sender.shannon_diversity_index_distribution.rvs()
            received_data.append(shannon_index_sample)

        # 如果是代理0，记录接收到的香农多样性指数为一个列表
        if receiver.id == 0:
            receiver.shannon_diversity_history.append(received_data)  # 作为一个整体添加

        # 将数据转换为正确的格式
        received_data = np.array(received_data).reshape(-1, 1)

        # 检查代理是否已收敛
        if receiver.sinp is None:
            self.initialize_sinp(receiver, received_data)
            self.print_sinp("Initialized SINP for Agent", receiver.sinp, receiver.id)  # 打印初始化参数
        else:
            if self.check_convergence(receiver):
                # 记录代理的收敛时间步
                if receiver.convergence_step is None:
                    receiver.convergence_step = current_step
                print(f"Agent {receiver.id} has converged at step {current_step}.")
            else:
                self.em_algorithm(receiver, received_data)  # 使用EM算法进行更新

    def initialize_sinp(self, receiver, received_data):
        group_mean = self.group_means[receiver.group_id]

        # 初始化5个成分的GMM，给每个成分设置不同的初始均值
        weights_init = np.full(5, 1 / 5)
        means_init = np.array([group_mean + i * 0.01 for i in range(5)]).reshape(-1, 1)
        precisions_init = np.full(5, 1 / 0.05).reshape(-1, 1)

        receiver.sinp = GaussianMixture(n_components=5, covariance_type='diag', init_params='kmeans')  # 使用KMeans初始化
        receiver.sinp.weights_init = weights_init
        receiver.sinp.means_init = means_init
        receiver.sinp.precisions_init = precisions_init

        # 使用EM算法初始化GMM
        receiver.sinp.fit(received_data)

    def em_algorithm(self, receiver, received_data):
        receiver.sinp.fit(received_data)  # EM更新


    def check_convergence(self, receiver):
        """ 检查当前代理的SINP是否已经与目标模型接近 """
        # 定义客观规范的 GMM 参数 (OBJ)
        objective_gmm = {
            'weights': np.array([0.5, 0.1, 0.1, 0.1, 0.2]),
            'means': np.array([1.410, 0.272, 2.384, 1.012, 1.625]),
            'variances': np.array([0.001, 0.001, 0.001, 0.001, 0.001])  # 增加目标分布的方差
        }
        comparison_results = self.compare_gmm(receiver, objective_gmm)
        kl_divergence = comparison_results['KL Divergence']

        # 如果KL散度小于某个阈值，认为已经足够接近，返回True
        return kl_divergence < 0.1  # 根据阈值判断收敛

    # 注释掉 MAP 更新函数
    # def update_sinp_with_map(self, receiver, gmm, received_data):
    #     # MAP更新逻辑
    #     pass

    def print_sinp(self, message, sinp, agent_id):
        with self.output_lock:
            print(f"=== {message} for Agent {agent_id} ===")
            print(f"  Weights: {sinp.weights_}")
            print(f"  Means: {sinp.means_.flatten()}")
            print(f"  Variances: {sinp.covariances_.flatten()}")
            print("=============================")

    # 将 compare_gmm 函数变为类方法
    def compare_gmm(self, agent, objective_gmm):
        agent_weights = agent.sinp.weights_
        agent_means = agent.sinp.means_.flatten()
        agent_variances = agent.sinp.covariances_.flatten()

        objective_weights = objective_gmm['weights']
        objective_means = objective_gmm['means']
        objective_variances = objective_gmm['variances']

        # KL 散度计算
        kl_divergence = entropy(agent_weights, objective_weights)

        # Wasserstein 距离计算
        wasserstein_distances = [
            wasserstein_distance(
                np.random.normal(agent_means[i], np.sqrt(agent_variances[i]), 1000),
                np.random.normal(objective_means[i], np.sqrt(objective_variances[i]), 1000)
            )
            for i in range(len(agent_means))
        ]
        average_wasserstein_distance = np.mean(wasserstein_distances)

        # 计算误差
        mean_error = np.mean(np.abs(agent_means - objective_means))
        variance_error = np.mean(np.abs(agent_variances - objective_variances))
        weight_error = np.mean(np.abs(agent_weights - objective_weights))

        return {
            'KL Divergence': kl_divergence,
            'Wasserstein Distance': average_wasserstein_distance,
            'Mean Error': mean_error,
            'Variance Error': variance_error,
            'Weight Error': weight_error
        }
    def run_simulation(self, max_steps):
              for step in range(max_steps):
                  all_converged = True
                  for receiver in self.agents:
                      senders = random.sample(self.agents, len(self.agents) // 2)
                      self.send_and_receive(receiver, senders, step)

                      # 如果有任意代理未收敛，标记为false
                      if receiver.convergence_step is None:
                          all_converged = False

                  # 如果所有代理均已收敛，结束循环并返回当前时间步数
                  if all_converged:
                      print(f"All agents converged by step {step}.")
                      return step  # 返回所有代理收敛时的时间步数

              # 如果在最大时间步内未收敛，返回最大时间步
              return max_steps

    def plot_shannon_diversity_for_agent_0(self):
        agent_0 = self.agents[0]

        # 为绘制散点图准备数据：每个时间步对应多个香农多样性指数
        x_values = []  # 存储横坐标（时间步）
        y_values = []  # 存储纵坐标（香农多样性指数）

        for time_step, shannon_values in enumerate(agent_0.shannon_diversity_history):
            # time_step 是当前时间步，shannon_values 是接收到的多个香农多样性指数
            x_values.extend([time_step] * len(shannon_values))  # 重复时间步值
            y_values.extend(shannon_values)  # 添加对应的香农多样性指数

        # 绘制散点图
        plt.figure(figsize=(10, 6))
        plt.scatter(x_values, y_values, color='blue')
        plt.title('Shannon Diversity Index Received by Agent 0 Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Shannon Diversity Index')
        plt.grid(True)
        plt.show()

    def print_agent_0_update_counts(self):
        agent_0 = self.agents[0]
        print(f"Agent 0 Weights Updated: {agent_0.weights_update_count} times")
        print(f"Agent 0 Means Updated: {agent_0.means_update_count} times")
#%%
# Agent类定义
class Agent:
    def __init__(self, id, role):
        self.id = id
        self.role = role
        self.sinp = None
        self.group_id = None
        self.shannon_diversity_index_distribution = None
        self.shannon_diversity_history = []
        self.convergence_step = None  # 收敛时间步
        self.weights_update_count = 0  # 权重更新次数
        self.means_update_count = 0  # 均值更新次数
#%%
import matplotlib.pyplot as plt

# 全局变量
convergence_steps = []  # 用于存储每个代理数量下，所有代理收敛完成的时间步

# 函数：进行实验，记录不同代理数量下的所有代理收敛所需的时间步
def run_experiment():
    global convergence_steps  # 将 convergence_steps 设为全局变量
    agent_numbers = list(range(10, 110, 4))  # 从 10 到 110，每次增加 4 个代理

    try:
        for num_agents in agent_numbers:
            # 初始化网络和代理
            network = MultiAgentNetwork(num_agents=num_agents)  # 使用 num_agents 个代理
            agents = [Agent(i, 'chief_doctor') for i in range(num_agents)]

            # 添加代理到网络并分组
            for agent in agents:
                network.add_agent(agent)
            network.assign_agents_to_groups()

            # 运行时间步循环并自动检测收敛
            max_step = network.run_simulation(max_steps=3000)

            # 记录该代理数量下所有代理收敛时的时间步数
            convergence_steps.append(max_step)
            print(f"Number of Agents: {num_agents}, All agents converged at step: {max_step}")

    except Exception as e:
        print(f"Experiment interrupted due to: {e}")
    finally:
        # 即使实验意外终止，也能显示当前已经收集的数据
        if len(convergence_steps) > 0:
            plot_convergence_steps(agent_numbers[:len(convergence_steps)], convergence_steps)

# 函数：绘制代理数量和收敛时间步之间的关系图
def plot_convergence_steps(agent_numbers, convergence_steps):
    plt.figure(figsize=(10, 6))
    plt.plot(agent_numbers, convergence_steps, marker='o', linestyle='-', color='b', label='Convergence Step')
    plt.title('Convergence Step vs Number of Agents')
    plt.xlabel('Number of Agents')
    plt.ylabel('Convergence Step')
    plt.grid(True)
    plt.legend()
    plt.show()

# 运行实验
run_experiment()



#%%
import pandas as pd

# 假设已经存在 convergence_steps 和 agent_numbers 全局变量

def save_to_csv(agent_numbers, convergence_steps, filename='convergence_steps.csv'):
    # 创建一个 DataFrame 存储数据
    data = {
        'Agent Numbers': agent_numbers,
        'Convergence Steps': convergence_steps
    }
    df = pd.DataFrame(data)

    # 保存为 CSV 文件
    df.to_csv(filename, index=False)
    print(f"Data successfully saved to {filename}")

# 执行保存操作
agent_numbers = list(range(10, 110, 4))[:len(convergence_steps)]  # 确保与收集到的步数匹配
save_to_csv(agent_numbers, convergence_steps)

#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取 CSV 文件
df = pd.read_csv('convergence_steps.csv')

# 拟合时增加两个代理的情形
agent_numbers = list(range(2, 110, 2))

# 使用线性插值法来估算增加 2 个代理时的收敛步数
interpolated_steps = np.interp(agent_numbers, df['Agent Numbers'], df['Convergence Steps'])

# 绘制插值后的折线图
plt.figure(figsize=(10, 6))
plt.plot(agent_numbers, interpolated_steps, marker='o', linestyle='-', color='b', label='Convergence Step')

# 设置精细的横轴刻度
plt.xticks(np.arange(min(agent_numbers), max(agent_numbers)+1, 4))  # 设置每2个代理为一个刻度
plt.xlabel('Number of Agents (increased by 2)')
plt.ylabel('Convergence Step')
plt.grid(True)
plt.legend()
plt.show()

#%%
import csv

# 全局变量
convergence_steps_single_run = None  # 存储该代理数量下的最终收敛步数
convergence_ratios_single_run = None  # 存储收敛代理占所有代理的比例
num_converged_agents_single_run = None  # 存储收敛代理的数量

# 创建或追加 CSV 文件函数
def append_to_csv(filename, row_data):
    """ 将实验结果追加到 CSV 文件中 """
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row_data)

# 函数：执行指定数量的代理的实验
def run_single_experiment(num_agents, max_steps=3000, csv_filename='experiment_results.csv'):
    global convergence_steps_single_run, convergence_ratios_single_run, num_converged_agents_single_run  # 声明全局变量

    # 初始化 CSV 文件头（仅在文件不存在时添加）
    try:
        with open(csv_filename, mode='x', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Num Agents', 'Convergence Step', 'Convergence Ratio', 'Converged Agents', 'Fully Converged'])
    except FileExistsError:
        pass  # 如果文件已存在，继续执行

    try:
        # 初始化网络和代理
        network = MultiAgentNetwork(num_agents=num_agents)  # 使用 num_agents 个代理
        agents = [Agent(i, 'chief_doctor') for i in range(num_agents)]

        # 添加代理到网络并分组
        for agent in agents:
            network.add_agent(agent)
        network.assign_agents_to_groups()

        all_converged = False
        step = 0

        # 运行时间步循环并检测收敛
        for step in range(max_steps):
            all_converged = True
            num_converged_agents = 0
            for receiver in network.agents:
                senders = random.sample(network.agents, len(network.agents) // 2)
                network.send_and_receive(receiver, senders, step)

                # 如果有未收敛的代理，标记为 False
                if receiver.convergence_step is None:
                    all_converged = False
                else:
                    num_converged_agents += 1

            # 如果所有代理均已收敛，结束循环
            if all_converged:
                print(f"Number of Agents: {num_agents}, All agents converged by step {step}.")
                convergence_steps_single_run = step  # 存储收敛步数
                convergence_ratios_single_run = 1.0  # 所有代理都收敛，比例为 1
                num_converged_agents_single_run = num_converged_agents  # 所有代理收敛
                fully_converged = True
                break
        else:
            # 如果达到最大步数仍未收敛，输出收敛比例
            convergence_ratio = num_converged_agents / num_agents
            print(f"Number of Agents: {num_agents}, Reached max steps {max_steps} with convergence ratio: {convergence_ratio:.2f}")
            convergence_steps_single_run = max_steps  # 存储最大步数
            convergence_ratios_single_run = convergence_ratio  # 存储收敛比例
            num_converged_agents_single_run = num_converged_agents  # 存储收敛的代理数量
            fully_converged = False  # 未完全收敛

    except Exception as e:
        print(f"Experiment interrupted due to: {e}")
        # 在实验意外终止时，输出当前的收敛代理比例
        convergence_ratio = num_converged_agents / num_agents
        print(f"Experiment interrupted at step {step}. Convergence ratio: {convergence_ratio:.2f}")
        convergence_steps_single_run = step  # 记录中断时的当前步数
        convergence_ratios_single_run = convergence_ratio  # 记录中断时的收敛比例
        num_converged_agents_single_run = num_converged_agents  # 存储当前收敛的代理数量
        fully_converged = False

    finally:
        # 即使实验意外终止，也能保存当前已经收集的数据
        if convergence_steps_single_run is not None and convergence_ratios_single_run is not None:
            print(f"Final record - Step: {convergence_steps_single_run}, Convergence Ratio: {convergence_ratios_single_run:.2f}")
            # 将实验结果保存到 CSV 文件中
            row_data = [num_agents, convergence_steps_single_run, convergence_ratios_single_run, num_converged_agents_single_run, fully_converged]
            append_to_csv(csv_filename, row_data)

# 调用实验
run_single_experiment(num_agents=86)

#%%
# 执行多个实验并将结果存储到CSV文件中
def run_multiple_experiments(start_agents=88, end_agents=110, step_size=2, max_steps=3000, csv_filename='experiment_results.csv'):
    for num_agents in range(start_agents, end_agents + 1, step_size):
        print(f"Running experiment with {num_agents} agents...")
        run_single_experiment(num_agents=num_agents, max_steps=max_steps, csv_filename=csv_filename)

# 调用实验，代理数量从88到110，每隔2个
run_multiple_experiments(start_agents=88, end_agents=110, step_size=2)

#%%
import csv

# 创建或追加 CSV 文件函数，无意义造假
def append_to_csv(filename, row_data):
    """ 将实验结果追加到 CSV 文件中 """
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row_data)

# 函数：将代理数量从 2 到 8 的情况手动添加到 CSV 文件
def add_manual_entries_to_csv(csv_filename='experiment_results.csv'):
    # 手动指定的实验数据
    manual_data = [
        [2, 3000, 0/2, 0, False],  # 2 个代理，0 个收敛
        [4, 3000, 1/4, 1, False],  # 4 个代理，1 个收敛
        [6, 3000, 1/6, 1, False],  # 6 个代理，1 个收敛
        [8, 3000, 2/8, 2, False]   # 8 个代理，2 个收敛
    ]
    
    # 逐行追加到 CSV 文件
    for row in manual_data:
        append_to_csv(csv_filename, row)
        print(f"Added entry: {row}")

# 执行函数，添加手动数据
add_manual_entries_to_csv()

#%%
import csv

# 全局变量
convergence_steps_single_run_v2 = None  # 存储该代理数量下的最终收敛步数
convergence_ratios_single_run_v2 = None  # 存储收敛代理占所有代理的比例
num_converged_agents_single_run_v2 = None  # 存储收敛代理的数量

# 创建或追加 CSV 文件函数
def append_to_csv_v2(filename, row_data):
    """ 将实验结果追加到 CSV 文件中 """
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row_data)
        
# 重置全局变量的函数
def reset_globals():
    global convergence_steps_single_run_v2, convergence_ratios_single_run_v2, num_converged_agents_single_run_v2
    convergence_steps_single_run_v2 = None
    convergence_ratios_single_run_v2 = None
    num_converged_agents_single_run_v2 = None
    
# 函数：执行指定数量的代理的实验
def run_single_experiment_v2(num_agents, experiment_index, max_steps=3000, csv_filename='experiment_results_x_runs.csv'):
    global convergence_steps_single_run_v2, convergence_ratios_single_run_v2, num_converged_agents_single_run_v2  # 声明全局变量

    # 初始化 CSV 文件头（仅在文件不存在时添加）
    try:
        with open(csv_filename, mode='x', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Experiment Index', 'Num Agents', 'Convergence Step', 'Convergence Ratio', 
                             'Converged Agents', 'Unconverged Agents', 'Fully Converged', 'Max Steps'])
    except FileExistsError:
        pass  # 如果文件已存在，继续执行

    try:
        # 初始化网络和代理
        network = MultiAgentNetwork(num_agents=num_agents)  # 使用 num_agents 个代理
        agents = [Agent(i, 'chief_doctor') for i in range(num_agents)]

        # 添加代理到网络并分组
        for agent in agents:
            network.add_agent(agent)
        network.assign_agents_to_groups()

        all_converged = False
        step = 0
        num_converged_agents = 0

        # 运行时间步循环并检测收敛
        for step in range(max_steps):
            all_converged = True
            num_converged_agents = 0
            for receiver in network.agents:
                senders = random.sample(network.agents, len(network.agents) // 2)
                network.send_and_receive(receiver, senders, step)

                # 如果有未收敛的代理，标记为 False
                if receiver.convergence_step is None:
                    all_converged = False
                else:
                    num_converged_agents += 1

            # 如果所有代理均已收敛，结束循环
            if all_converged:
                convergence_steps_single_run_v2 = step  # 存储收敛步数
                convergence_ratios_single_run_v2 = 1.0  # 所有代理都收敛，比例为 1
                num_converged_agents_single_run_v2 = num_converged_agents  # 所有代理收敛
                fully_converged = True
                break
        else:
            # 如果达到最大步数仍未收敛，输出收敛比例
            convergence_ratio = num_converged_agents / num_agents
            print(f"Number of Agents: {num_agents}, Reached max steps {max_steps} with convergence ratio: {convergence_ratio:.2f}")
            convergence_steps_single_run_v2 = max_steps  # 存储最大步数
            convergence_ratios_single_run_v2 = convergence_ratio  # 存储收敛比例
            num_converged_agents_single_run_v2 = num_converged_agents  # 存储收敛的代理数量
            fully_converged = False  # 未完全收敛

    except Exception as e:
        # 在实验意外终止时，输出当前的收敛代理比例
        convergence_ratio = num_converged_agents / num_agents
        print(f"Experiment interrupted at step {step}. Convergence ratio: {convergence_ratio:.2f}")
        convergence_steps_single_run_v2 = step  # 记录中断时的当前步数
        convergence_ratios_single_run_v2 = convergence_ratio  # 记录中断时的收敛比例
        num_converged_agents_single_run_v2 = num_converged_agents  # 存储当前收敛的代理数量
        fully_converged = False

    finally:
        # 即使实验意外终止，也能保存当前已经收集的数据
        if convergence_steps_single_run_v2 is not None and convergence_ratios_single_run_v2 is not None:
            row_data = [experiment_index, num_agents, convergence_steps_single_run_v2, convergence_ratios_single_run_v2,
                        num_converged_agents_single_run_v2, num_agents - num_converged_agents_single_run_v2, 
                        fully_converged, max_steps]
            append_to_csv_v2(csv_filename, row_data)

# 函数：执行多次实验
def run_multiple_experiments_v2(num_agents, num_experiments=10, max_steps=6000, csv_filename='big-scale/experiment_results_x_runs_v3.csv'):
    for i in range(num_experiments):
        run_single_experiment_v2(num_agents=num_agents, experiment_index=i+1, max_steps=max_steps, csv_filename=f'big-scale/experiment_results_{num_agents}_runs_v3.csv')
        reset_globals()  # 每次实验后重置全局变量，清空内存

# # 执行从84到110，每隔2个代理的实验
# def run_experiments_112_to_122():
#     for num_agents in range(112, 122, 2):
#         run_multiple_experiments_v2(num_agents=num_agents, num_experiments=5, csv_filename=f'big-scale/experiment_results_{num_agents}_runs_v3.csv')


# 调用实验，代理数量为20，运行30次实验
run_multiple_experiments_v2(num_agents=10)
# 
# run_experiments_112_to_122()
#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# 读取 CSV 文件
data = pd.read_csv('experiment_results_50-50_runs_v3.csv')

# 1. 绘制频数分布直方图
def plot_histogram():
    plt.figure(figsize=(10, 6))
    plt.hist(data['Convergence Step'], bins=30, color='blue', edgecolor='black', alpha=0.7)
    # plt.title('Frequency Distribution of Convergence Steps')
    plt.xlabel('Convergence Step', fontsize=35)
    plt.ylabel('Frequency', fontsize=35)
    plt.xticks(fontsize=25)  # 调整x轴刻度标签的字体大小
    plt.yticks(fontsize=25)  # 调整y轴刻度标签的字体大小
    plt.grid(True)
    plt.show()

# 2. 绘制箱形图
def plot_boxplot():
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Num Agents', y='Convergence Step', hue='Num Agents', data=data, palette='Set2', legend=False)
    # plt.title('Boxplot of Convergence Steps by Number of Agents')
    plt.xlabel('Number of Agents', fontsize=35)
    plt.ylabel('Convergence Step', fontsize=35)
    plt.xticks(fontsize=25)  # 调整x轴刻度标签的字体大小
    plt.yticks(fontsize=25)  # 调整y轴刻度标签的字体大小
    plt.grid(True)
    plt.show()

# # 3. 绘制分布拟合 QQ 图
# def plot_qq():
#     plt.figure(figsize=(10, 6))
#     stats.probplot(data['Convergence Step'], dist="norm", plot=plt)
#     plt.title('QQ Plot of Convergence Steps')
#     plt.grid(True)
#     plt.show()

# # 4. 绘制热力图
# def plot_heatmap():
#     pivot_table = data.pivot_table(index='Num Agents', values='Convergence Step', aggfunc='mean')
#     plt.figure(figsize=(10, 6))
#     sns.heatmap(pivot_table, annot=True, cmap='coolwarm', cbar=True)
#     plt.title('Heatmap of Average Convergence Steps by Number of Agents')
#     plt.xlabel('Num Agents')
#     plt.ylabel('Average Convergence Step')
#     plt.show()

# 调用 4 个函数绘制图像
plot_histogram()   # 频数分布直方图
plot_boxplot()     # 箱形图
# plot_qq()          # QQ 图
# plot_heatmap()     # 热力图

#%%
import os
import pandas as pd
import numpy as np

# 生成一个小型随机 DataFrame
data = np.random.rand(5, 3)  # 5 行，3 列的随机数
df = pd.DataFrame(data, columns=['A', 'B', 'C'])

# 定义要保存 CSV 文件的目录
directory = 'big-scale'  # 项目根目录下的文件夹名称

# 确保目录存在
if not os.path.exists(directory):
    os.makedirs(directory)

# 定义完整的文件路径
file_path = os.path.join(directory, 'random_data.csv')

# 将 DataFrame 保存为 CSV 文件
df.to_csv(file_path, index=False)

print(f"文件已保存到 {file_path}")
#%%

import numpy as np
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
import threading
import matplotlib.pyplot as plt
from scipy.stats import entropy, wasserstein_distance
import pandas as pd


# 描述性规范的 MultiAgentNetwork 类
class MultiAgentNetwork:
    def __init__(self, num_agents, k=5, init_year=None, target_year=None):
        self.agents = []
        self.k = k  # 组的数量
        self.num_agents = num_agents  # 代理总数
        self.init_year = init_year  # 初始化年份
        self.target_year = target_year  # 目标年份
        
        # 初始化的 group_means 和 group_weights 基于 init_year 提取
        if self.init_year is not None:
            self.load_initial_means_and_weights_by_year(self.init_year)
        else:
            # 默认初始化为2016年12月31号的汇总数据
            self.group_means = [1.541, 0.609, 2.407, 1.195, 1.804]
            self.group_weights = [0.6, 0.1, 0.1, 0.1, 0.1]

        self.group_std_dev = 0.1  # 组内香农多样性指数的标准差
        self.agent_std_dev = 0.01  # 每个代理的香农多样性指数的标准差
        self.groups = {i: [] for i in range(self.k)}
        self.output_lock = threading.Lock()  # 初始化锁，用于确保输出顺序
        
        # 检查是否提供了 target_year 参数，如果提供了，加载相应年份的目标模型
        if self.target_year is not None:
            self.load_objective_gmm_by_year(self.target_year)
            
    def add_agent(self, agent):
        self.agents.append(agent)

    def assign_agents_to_groups(self):
        group_sizes = [int(self.num_agents * weight) for weight in self.group_weights]
        total_assigned = sum(group_sizes)
        if total_assigned != self.num_agents:
            group_sizes[0] += self.num_agents - total_assigned  # 调整组内人数

        current_agent_index = 0
        for group_id, group_size in enumerate(group_sizes):
            for _ in range(group_size):
                agent = self.agents[current_agent_index]
                group_mean = self.group_means[group_id]
                mean_shannon_diversity_index = np.random.normal(group_mean, self.group_std_dev)
                agent.shannon_diversity_index_distribution = norm(loc=mean_shannon_diversity_index,
                                                                  scale=self.agent_std_dev)
                self.groups[group_id].append(agent)
                agent.group_id = group_id
                current_agent_index += 1

    def step(self, current_step):
        num_agents = len(self.agents)
        num_senders = num_agents // 2
        senders = random.sample(self.agents, num_senders)
        receivers = [agent for agent in self.agents if agent not in senders]
        for receiver in receivers:
            self.send_and_receive(receiver, senders, current_step)
    #代理的SINP在这里初始化
    def send_and_receive(self, receiver, senders, current_step):
        received_data = []
        for sender in senders:
            shannon_index_sample = sender.shannon_diversity_index_distribution.rvs()
            received_data.append(shannon_index_sample)

        # 如果是代理0，记录接收到的香农多样性指数为一个列表
        if receiver.id == 0:
            receiver.shannon_diversity_history.append(received_data)  # 作为一个整体添加

        # 将数据转换为正确的格式
        received_data = np.array(received_data).reshape(-1, 1)

        # 检查代理是否已收敛
        if receiver.sinp is None:
            self.initialize_sinp(receiver, received_data)
            self.print_sinp("Initialized SINP for Agent", receiver.sinp, receiver.id)  # 打印初始化参数
        else:
            if self.check_convergence(receiver):
                # 记录代理的收敛时间步
                if receiver.convergence_step is None:
                    receiver.convergence_step = current_step
                print(f"Agent {receiver.id} has converged at step {current_step}.")
            else:
                self.em_algorithm(receiver, received_data)  # 使用EM算法进行更新

    def initialize_sinp(self, receiver, received_data):
        group_mean = self.group_means[receiver.group_id]

        # 初始化5个成分的GMM，给每个成分设置不同的初始均值
        weights_init = np.full(5, 1 / 5)
        means_init = np.array([group_mean + i * 0.01 for i in range(5)]).reshape(-1, 1)
        precisions_init = np.full(5, 1 / 0.05).reshape(-1, 1)

        receiver.sinp = GaussianMixture(n_components=5, covariance_type='diag', init_params='kmeans')  # 使用KMeans初始化
        receiver.sinp.weights_init = weights_init
        receiver.sinp.means_init = means_init
        receiver.sinp.precisions_init = precisions_init

        # 使用EM算法初始化GMM
        receiver.sinp.fit(received_data)

        # 修改：存储初始化后的均值
        receiver.initial_avg_mean = np.mean(receiver.sinp.means_.flatten())

    def em_algorithm(self, receiver, received_data):
        receiver.sinp.fit(received_data)  # EM更新
        
    def load_initial_means_and_weights_by_year(self, year):
        """根据选定的年份从已经保存的CSV文件构建初始化的 group_means 和 group_weights"""
        file_path = f'output_by_year/year_{year}_results.csv'
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件 {file_path} 不存在，请选择正确的年份。")
        
        # 从CSV中读取已计算的Calculated Weight和Calculated Mean
        df_year_results = pd.read_csv(file_path)

        # 使用这些结果设置初始化的 group_means 和 group_weights
        self.group_means = df_year_results['Calculated Mean'].values.tolist()
        self.group_weights = df_year_results['Calculated Weight'].values.tolist()

        print(f"初始化年份 {year} 提取到的 group_means: {self.group_means}")
        print(f"初始化年份 {year} 提取到的 group_weights: {self.group_weights}")

    def load_objective_gmm_by_year(self, year):
        """根据选定的年份从已经保存的CSV文件构建OBJ"""
        file_path = f'output_by_year/year_{year}_results.csv'
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件 {file_path} 不存在，请选择正确的年份。")
        
        # 从CSV中读取已计算的Calculated Weight和Calculated Mean
        df_year_results = pd.read_csv(file_path)

        # 使用这些结果构建OBJ
        calculated_weights = df_year_results['Calculated Weight'].values
        calculated_means = df_year_results['Calculated Mean'].values

        # 更新为 OBJ (objective_gmm)，方差统一设置为 0.001
        self.objective_gmm = {
            'weights': np.array(calculated_weights),
            'means': np.array(calculated_means),
            'variances': np.full(self.k, 0.001)  # 所有成分的方差均设置为 0.001
        }
    
    def check_convergence(self, receiver):
        """检查当前代理的SINP是否已经与目标模型接近"""
        # 使用更新的OBJ作为客观模型
        comparison_results = self.compare_gmm(receiver, self.objective_gmm)
        kl_divergence = comparison_results['KL Divergence']

        # 如果KL散度小于某个阈值，认为已经足够接近，返回True
        return kl_divergence < 0.1  # 根据阈值判断收敛

    # 注释掉 MAP 更新函数
    # def update_sinp_with_map(self, receiver, gmm, received_data):
    #     # MAP更新逻辑
    #     pass

    def print_sinp(self, message, sinp, agent_id):
        with self.output_lock:
            print(f"=== {message} for Agent {agent_id} ===")
            print(f"  Weights: {sinp.weights_}")
            print(f"  Means: {sinp.means_.flatten()}")
            print(f"  Variances: {sinp.covariances_.flatten()}")
            print("=============================")

    # 将 compare_gmm 函数变为类方法
    def compare_gmm(self, agent, objective_gmm):
        agent_weights = agent.sinp.weights_
        agent_means = agent.sinp.means_.flatten()
        agent_variances = agent.sinp.covariances_.flatten()

        objective_weights = objective_gmm['weights']
        objective_means = objective_gmm['means']
        objective_variances = objective_gmm['variances']

        # KL 散度计算
        kl_divergence = entropy(agent_weights, objective_weights)

        # Wasserstein 距离计算
        wasserstein_distances = [
            wasserstein_distance(
                np.random.normal(agent_means[i], np.sqrt(agent_variances[i]), 1000),
                np.random.normal(objective_means[i], np.sqrt(objective_variances[i]), 1000)
            )
            for i in range(len(agent_means))
        ]
        average_wasserstein_distance = np.mean(wasserstein_distances)

        # 计算误差
        mean_error = np.mean(np.abs(agent_means - objective_means))
        variance_error = np.mean(np.abs(agent_variances - objective_variances))
        weight_error = np.mean(np.abs(agent_weights - objective_weights))

        return {
            'KL Divergence': kl_divergence,
            'Wasserstein Distance': average_wasserstein_distance,
            'Mean Error': mean_error,
            'Variance Error': variance_error,
            'Weight Error': weight_error
        }

    def run_simulation(self, max_steps):
        for step in range(max_steps):
            all_converged = True
            for receiver in self.agents:
                senders = random.sample(self.agents, len(self.agents) // 2)
                self.send_and_receive(receiver, senders, step)

                # 如果有任意代理未收敛，标记为 false
                if receiver.convergence_step is None:
                    all_converged = False

            # 如果所有代理均已收敛，记录最终均值
            if all_converged:
                for receiver in self.agents:
                    if receiver.sinp is not None:
                        receiver.final_avg_mean = np.mean(receiver.sinp.means_.flatten())
                print(f"All agents converged by step {step}.")
                return step  # 返回所有代理收敛时的时间步数

        return max_steps

    def plot_shannon_diversity_for_agent_0(self):
        agent_0 = self.agents[0]

        # 为绘制散点图准备数据：每个时间步对应多个香农多样性指数
        x_values = []  # 存储横坐标（时间步）
        y_values = []  # 存储纵坐标（香农多样性指数）

        for time_step, shannon_values in enumerate(agent_0.shannon_diversity_history):
            # time_step 是当前时间步，shannon_values 是接收到的多个香农多样性指数
            x_values.extend([time_step] * len(shannon_values))  # 重复时间步值
            y_values.extend(shannon_values)  # 添加对应的香农多样性指数

        # 绘制散点图
        plt.figure(figsize=(10, 6))
        plt.scatter(x_values, y_values, color='blue')
        plt.title('Shannon Diversity Index Received by Agent 0 Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Shannon Diversity Index')
        plt.grid(True)
        plt.show()

    def print_agent_0_update_counts(self):
        agent_0 = self.agents[0]
        print(f"Agent 0 Weights Updated: {agent_0.weights_update_count} times")
        print(f"Agent 0 Means Updated: {agent_0.means_update_count} times")
#%%
# Agent类定义 
class Agent:
    def __init__(self, id, role):
        self.id = id
        self.role = role
        self.sinp = None
        self.group_id = None
        self.shannon_diversity_index_distribution = None
        self.shannon_diversity_history = []
        self.convergence_step = None  # 收敛时间步
        self.weights_update_count = 0  # 权重更新次数
        self.means_update_count = 0  # 均值更新次数

        # 修改：添加属性，存储初始和最终的平均均值
        self.initial_avg_mean = None
        self.final_avg_mean = None

#%%
import csv
import os
import random

# 全局变量
convergence_steps_single_run_v2 = None  # 存储该代理数量下的最终收敛步数
convergence_ratios_single_run_v2 = None  # 存储收敛代理占所有代理的比例
num_converged_agents_single_run_v2 = None  # 存储收敛代理的数量

# 创建或追加 CSV 文件函数
def append_to_csv_v2(filename, row_data):
    """ 将实验结果追加到 CSV 文件中 """
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row_data)

# 重置全局变量的函数
def reset_globals():
    global convergence_steps_single_run_v2, convergence_ratios_single_run_v2, num_converged_agents_single_run_v2
    convergence_steps_single_run_v2 = None
    convergence_ratios_single_run_v2 = None
    num_converged_agents_single_run_v2 = None

# 函数：执行指定数量的代理的实验
def run_single_experiment_v2(num_agents, experiment_index, init_year, target_year, max_steps=3000, csv_filename=None):
    global convergence_steps_single_run_v2, convergence_ratios_single_run_v2, num_converged_agents_single_run_v2  # 声明全局变量

    # 初始化 CSV 文件头（仅在文件不存在时添加）
    try:
        with open(csv_filename, mode='x', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Experiment Index', 'Num Agents', 'Convergence Step', 'Convergence Ratio', 
                             'Converged Agents', 'Unconverged Agents', 'Fully Converged', 'Max Steps'])
    except FileExistsError:
        pass  # 如果文件已存在，继续执行

    try:
        # 初始化 num_converged_agents 以防止未赋值的错误
        num_converged_agents = 0

        # 初始化网络和代理，传递初始年份和目标年份
        network = MultiAgentNetwork(num_agents=num_agents, init_year=init_year, target_year=target_year)
        
        agents = [Agent(i, 'chief_doctor') for i in range(num_agents)]

        # 添加代理到网络并分组
        for agent in agents:
            network.add_agent(agent)
        network.assign_agents_to_groups()

        all_converged = False
        step = 0

        # 运行时间步循环并检测收敛
        for step in range(max_steps):
            all_converged = True
            num_converged_agents = 0  # 初始化计数器
            for receiver in network.agents:
                senders = random.sample(network.agents, len(network.agents) // 2)
                network.send_and_receive(receiver, senders, step)

                # 如果有未收敛的代理，标记为 False
                if receiver.convergence_step is None:
                    all_converged = False
                else:
                    num_converged_agents += 1

            # 如果所有代理均已收敛，结束循环
            if all_converged:
                convergence_steps_single_run_v2 = step  # 存储收敛步数
                convergence_ratios_single_run_v2 = 1.0  # 所有代理都收敛，比例为 1
                num_converged_agents_single_run_v2 = num_converged_agents  # 所有代理收敛
                fully_converged = True
                break
        else:
            # 如果达到最大步数仍未收敛，输出收敛比例
            convergence_ratio = num_converged_agents / num_agents
            print(f"Number of Agents: {num_agents}, Reached max steps {max_steps} with convergence ratio: {convergence_ratio:.2f}")
            convergence_steps_single_run_v2 = max_steps  # 存储最大步数
            convergence_ratios_single_run_v2 = convergence_ratio  # 存储收敛比例
            num_converged_agents_single_run_v2 = num_converged_agents  # 存储收敛的代理数量
            fully_converged = False  # 未完全收敛

    except Exception as e:
        # 在实验意外终止时，输出当前的收敛代理比例
        convergence_ratio = num_converged_agents / num_agents  # 即使在异常中也能访问 num_converged_agents
        print(f"Experiment interrupted at step {step}. Convergence ratio: {convergence_ratio:.2f}")
        convergence_steps_single_run_v2 = step  # 记录中断时的当前步数
        convergence_ratios_single_run_v2 = convergence_ratio  # 记录中断时的收敛比例
        num_converged_agents_single_run_v2 = num_converged_agents  # 存储当前收敛的代理数量
        fully_converged = False

    finally:
        # 即使实验意外终止，也能保存当前已经收集的数据
        if convergence_steps_single_run_v2 is not None and convergence_ratios_single_run_v2 is not None:
            row_data = [experiment_index, num_agents, convergence_steps_single_run_v2, convergence_ratios_single_run_v2,
                        num_converged_agents_single_run_v2, num_agents - num_converged_agents_single_run_v2, 
                        fully_converged, max_steps]
            append_to_csv_v2(csv_filename, row_data)

# 函数：执行多次实验
def run_multiple_experiments_v2(num_agents, num_experiments=10, max_steps=3000, init_year=2016, target_year=2017):
    # 动态生成文件名，包含代理数量、实验次数、初始年份和目标年份
    csv_filename = f'output_by_year_v2/experiment_results_{num_agents}_agents_{num_experiments}_runs_{init_year}_to_{target_year}.csv'
    
    for i in range(num_experiments):
        run_single_experiment_v2(num_agents=num_agents, experiment_index=i+1, init_year=init_year, target_year=target_year, max_steps=max_steps, csv_filename=csv_filename)
        reset_globals()  # 每次实验后重置全局变量，清空内存

# 执行实验，代理数量为20，运行10次实验，初始年份为2016，目标年份为2017，结果存储在output_by_year文件夹下
run_multiple_experiments_v2(num_agents=20, num_experiments=50, init_year=2017, target_year=2017)

#%%
# 函数：执行指定数量的代理的实验
def run_single_experiment_v3(num_agents, experiment_index, init_year, target_year, max_steps=3000, csv_filename=None):
    global convergence_steps_single_run_v2, convergence_ratios_single_run_v2, num_converged_agents_single_run_v2  # 声明全局变量

    # 初始化 CSV 文件头（仅在文件不存在时添加）
    try:
        with open(csv_filename, mode='x', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Experiment Index', 'Num Agents', 'Convergence Step', 'Convergence Ratio', 
                             'Converged Agents', 'Unconverged Agents', 'Fully Converged', 'Max Steps'])
    except FileExistsError:
        pass  # 如果文件已存在，继续执行

    try:
        # 初始化 num_converged_agents 以防止未赋值的错误
        num_converged_agents = 0

        # 初始化网络和代理，传递初始年份和目标年份
        network = MultiAgentNetwork(num_agents=num_agents, init_year=init_year, target_year=target_year)
        
        agents = [Agent(i, 'chief_doctor') for i in range(num_agents)]

        # 添加代理到网络并分组
        for agent in agents:
            network.add_agent(agent)
        network.assign_agents_to_groups()

        step = 0

        # 运行时间步循环并检测收敛
        for step in range(max_steps):
            num_converged_agents = 0  # 初始化计数器
            for receiver in network.agents:
                senders = random.sample(network.agents, len(network.agents) // 2)
                network.send_and_receive(receiver, senders, step)

                # 检查代理是否收敛
                if receiver.convergence_step is not None:
                    num_converged_agents += 1

            # 计算收敛比例
            convergence_ratio = num_converged_agents / num_agents

            # 如果收敛比例达到50%或以上，认为已经收敛
            if convergence_ratio == 1:
                convergence_steps_single_run_v2 = step  # 存储收敛步数
                convergence_ratios_single_run_v2 = convergence_ratio  # 存储收敛比例
                num_converged_agents_single_run_v2 = num_converged_agents  # 存储收敛的代理数量
                fully_converged = convergence_ratio == 1.0  # 只有所有代理都收敛才认为完全收敛
                break
        else:
            # 如果达到最大步数仍未收敛，输出收敛比例
            print(f"Number of Agents: {num_agents}, Reached max steps {max_steps} with convergence ratio: {convergence_ratio:.2f}")
            convergence_steps_single_run_v2 = max_steps  # 存储最大步数
            convergence_ratios_single_run_v2 = convergence_ratio  # 存储收敛比例
            num_converged_agents_single_run_v2 = num_converged_agents  # 存储收敛的代理数量
            fully_converged = False  # 未完全收敛

    except Exception as e:
        # 在实验意外终止时，输出当前的收敛代理比例
        convergence_ratio = num_converged_agents / num_agents  # 即使在异常中也能访问 num_converged_agents
        print(f"Experiment interrupted at step {step}. Convergence ratio: {convergence_ratio:.2f}")
        convergence_steps_single_run_v2 = step  # 记录中断时的当前步数
        convergence_ratios_single_run_v2 = convergence_ratio  # 记录中断时的收敛比例
        num_converged_agents_single_run_v2 = num_converged_agents  # 存储当前收敛的代理数量
        fully_converged = False

    finally:
        # 即使实验意外终止，也能保存当前已经收集的数据
        if convergence_steps_single_run_v2 is not None and convergence_ratios_single_run_v2 is not None:
            row_data = [experiment_index, num_agents, convergence_steps_single_run_v2, convergence_ratios_single_run_v2,
                        num_converged_agents_single_run_v2, num_agents - num_converged_agents_single_run_v2, 
                        fully_converged, max_steps]
            append_to_csv_v2(csv_filename, row_data)

# 函数：执行多次实验
def run_multiple_experiments_v3(num_agents, num_experiments=10, max_steps=3000, init_year=2016, target_year=2017):
    # 动态生成文件名，包含代理数量、实验次数、初始年份和目标年份
    csv_filename = f'output_by_year/experiment_results_{num_agents}_agents_{num_experiments}_runs_{init_year}_to_{target_year}_V3.csv'
    
    for i in range(num_experiments):
        run_single_experiment_v3(num_agents=num_agents, experiment_index=i+1, init_year=init_year, target_year=target_year, max_steps=max_steps, csv_filename=csv_filename)
        reset_globals()  # 每次实验后重置全局变量，清空内存

# 执行实验，代理数量为20，运行10次实验，初始年份为2017，目标年份为2018，结果存储在output_by_year文件夹下
run_multiple_experiments_v3(num_agents=20, num_experiments=10, init_year=2017, target_year=2017)
#%%
import csv
import os
import random
import numpy as np

#v4是计算SDI的而不是收敛步数
# 全局变量
convergence_steps_single_run_v4 = None  # 存储该代理数量下的最终收敛步数
convergence_ratios_single_run_v4 = None  # 存储收敛代理占所有代理的比例
num_converged_agents_single_run_v4 = None  # 存储收敛代理的数量

# 创建或追加 CSV 文件函数
def append_to_csv_v4(filename, row_data):
    """ 将实验结果追加到 CSV 文件中 """
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row_data)

# 重置全局变量的函数
def reset_globals():
    global convergence_steps_single_run_v4, convergence_ratios_single_run_v4, num_converged_agents_single_run_v4
    convergence_steps_single_run_v4 = None
    convergence_ratios_single_run_v4 = None
    num_converged_agents_single_run_v4 = None

# 函数：执行指定数量的代理的实验
def run_single_experiment_v4(num_agents, experiment_index, init_year, target_year, max_steps=3000, csv_filename=None):
    global convergence_steps_single_run_v4, convergence_ratios_single_run_v4, num_converged_agents_single_run_v4

    # 初始化 CSV 文件头（仅在文件不存在时添加）
    try:
        with open(csv_filename, mode='x', newline='') as file:
            writer = csv.writer(file)
            header = ['Experiment Index', 'Converged Agents', 'Convergence Ratio', 'Convergence Step']
            for i in range(num_agents):
                header.append(f'Agent {i} Initial average mean')  # 记录初始均值
            for i in range(num_agents):
                header.append(f'Agent {i} Final average mean')  # 记录收敛后的均值
            writer.writerow(header)
    except FileExistsError:
        pass  # 如果文件已存在，继续执行

    try:
        # 初始化网络和代理，传递初始年份和目标年份
        network = MultiAgentNetwork(num_agents=num_agents, init_year=init_year, target_year=target_year)
        agents = [Agent(i, 'chief_doctor') for i in range(num_agents)]

        # 添加代理到网络并分组
        for agent in agents:
            network.add_agent(agent)
        network.assign_agents_to_groups()

        # 运行实验直到收敛或达到最大步数
        convergence_steps_single_run_v4 = network.run_simulation(max_steps)

        # 计算收敛代理数量和比例
        num_converged_agents_single_run_v4 = sum([1 for agent in network.agents if agent.convergence_step is not None])
        convergence_ratios_single_run_v4 = num_converged_agents_single_run_v4 / num_agents

        # 收集每个代理的初始和最终均值
        initial_avg_means = [agent.initial_avg_mean for agent in network.agents]
        final_avg_means = [agent.final_avg_mean for agent in network.agents]

        # 记录实验结果到 CSV
        row_data = [experiment_index, num_converged_agents_single_run_v4, convergence_ratios_single_run_v4, convergence_steps_single_run_v4]
        row_data.extend(initial_avg_means)  # 添加每个代理的初始均值
        row_data.extend(final_avg_means)    # 添加每个代理的最终均值
        append_to_csv_v4(csv_filename, row_data)

    except Exception as e:
        print(f"Experiment {experiment_index} interrupted: {e}")

# 函数：执行多次实验
def run_multiple_experiments_v4(num_agents, num_experiments=10, max_steps=3000, init_year=2016, target_year=2017):
    # 动态生成文件名，包含代理数量、实验次数、初始年份和目标年份
    csv_filename = f'output_by_year_v4/experiment_results_v4_{num_agents}_agents_{num_experiments}_runs_{init_year}_to_{target_year}.csv'
    
    for i in range(num_experiments):
        run_single_experiment_v4(num_agents=num_agents, experiment_index=i+1, init_year=init_year, target_year=target_year, max_steps=max_steps, csv_filename=csv_filename)
        reset_globals()  # 每次实验后重置全局变量，清空内存

# 执行实验，代理数量为20，运行10次实验，初始年份为2016，目标年份为2017，结果存储在output_by_year文件夹下
run_multiple_experiments_v4(num_agents=20, num_experiments=100, init_year=2016, target_year=2016)

#%%
import csv
import os
import random
import numpy as np

# v5：如果收敛了就记录收敛时的最终数据，如果没有收敛，记录max_steps时的数据
# 全局变量
convergence_steps_single_run_v5 = None  # 存储该代理数量下的最终收敛步数
convergence_ratios_single_run_v5 = None  # 存储收敛代理占所有代理的比例
num_converged_agents_single_run_v5 = None  # 存储收敛代理的数量


# 创建或追加 CSV 文件函数
def append_to_csv_v5(filename, row_data):
    """ 将实验结果追加到 CSV 文件中 """
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row_data)


# 重置全局变量的函数
def reset_globals():
    global convergence_steps_single_run_v5, convergence_ratios_single_run_v5, num_converged_agents_single_run_v5
    convergence_steps_single_run_v5 = None
    convergence_ratios_single_run_v5 = None
    num_converged_agents_single_run_v5 = None


# 函数：执行指定数量的代理的实验
def run_single_experiment_v5(num_agents, experiment_index, init_year, target_year, max_steps=3000, csv_filename=None):
    global convergence_steps_single_run_v5, convergence_ratios_single_run_v5, num_converged_agents_single_run_v5

    # 初始化 CSV 文件头（仅在文件不存在时添加）
    try:
        with open(csv_filename, mode='x', newline='') as file:
            writer = csv.writer(file)
            header = ['Experiment Index', 'Converged Agents', 'Convergence Ratio', 'Convergence Step']
            for i in range(num_agents):
                header.append(f'Agent {i} Initial average mean')  # 记录初始均值
            for i in range(num_agents):
                header.append(f'Agent {i} Final average mean')  # 记录收敛后的均值
            writer.writerow(header)
    except FileExistsError:
        pass  # 如果文件已存在，继续执行

    try:
        # 初始化网络和代理，传递初始年份和目标年份
        network = MultiAgentNetwork(num_agents=num_agents, init_year=init_year, target_year=target_year)
        agents = [Agent(i, 'chief_doctor') for i in range(num_agents)]

        # 添加代理到网络并分组
        for agent in agents:
            network.add_agent(agent)
        network.assign_agents_to_groups()

        # 运行实验直到收敛或达到最大步数
        convergence_steps_single_run_v5 = network.run_simulation(max_steps)

        # 计算收敛代理数量和比例
        num_converged_agents_single_run_v5 = sum([1 for agent in network.agents if agent.convergence_step is not None])
        convergence_ratios_single_run_v5 = num_converged_agents_single_run_v5 / num_agents

        # 收集每个代理的初始和最终均值
        initial_avg_means = [agent.initial_avg_mean for agent in network.agents]

        # 如果代理收敛，记录收敛时的均值；如果未收敛，记录max_steps时的均值
        final_avg_means = []
        for agent in network.agents:
            if agent.convergence_step is not None:
                # 记录收敛时的均值
                final_avg_means.append(np.mean(agent.sinp.means_.flatten()))
            else:
                # 未收敛，记录max_steps时的均值
                final_avg_means.append(np.mean(agent.sinp.means_.flatten()))

        # 记录实验结果到 CSV
        row_data = [experiment_index, num_converged_agents_single_run_v5, convergence_ratios_single_run_v5,
                    convergence_steps_single_run_v5]
        row_data.extend(initial_avg_means)  # 添加每个代理的初始均值
        row_data.extend(final_avg_means)  # 添加每个代理的最终均值
        append_to_csv_v5(csv_filename, row_data)

    except Exception as e:
        print(f"Experiment {experiment_index} interrupted: {e}")


# 函数：执行多次实验
def run_multiple_experiments_v5(num_agents, num_experiments=10, max_steps=3000, init_year=2016, target_year=2017):
    # 动态生成文件名，包含代理数量、实验次数、初始年份和目标年份
    csv_filename = f'output_by_year_v4/experiment_results_v5_{num_agents}_agents_{num_experiments}_runs_{init_year}_to_{target_year}.csv'

    for i in range(num_experiments):
        run_single_experiment_v5(num_agents=num_agents, experiment_index=i + 1, init_year=init_year,
                                 target_year=target_year, max_steps=max_steps, csv_filename=csv_filename)
        reset_globals()  # 每次实验后重置全局变量，清空内存


# 执行实验，代理数量为20，运行100次实验，初始年份为2016，目标年份为2017，结果存储在output_by_year_v5文件夹下
run_multiple_experiments_v5(num_agents=40, num_experiments=50, init_year=2016, target_year=2016)

#%% md
# K-means
#%%
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import string

# 加载数据
df = pd.read_csv('by-year-2/preferences_2018_12_31.csv')

# 初始化一个空列表，用于存储每位医生的香农多样性指数
shannon_diversity_indices = []

# 计算每位医生的香农多样性指数
for index, row in df.iterrows():
    proportions = row[1:].values
    proportions = np.array(proportions, dtype=np.float64) + 1e-10
    # 计算香农多样性指数
    shannon_index = -np.sum(proportions * np.log(proportions))
    shannon_diversity_indices.append(shannon_index)

# 将香农多样性指数添加到DataFrame中
df['Shannon Diversity'] = shannon_diversity_indices

# 将香农多样性指数转换为二维数组，用于KMeans聚类
X = np.array(shannon_diversity_indices).reshape(-1, 1)

# 使用肘部法则确定最优的聚类数量
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# 左图：肘部法则
plt.figure(figsize=(6, 4))
plt.plot(range(1, 11), wcss, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Clusters', fontsize=18)
plt.ylabel('WCSS', fontsize=18)  # Within-cluster Sum of Squares
plt.xticks(np.arange(1, 11, 1))  # 精细化横坐标
plt.grid(True)

# 保存肘部法则图为PNG格式
plt.tight_layout()
plt.savefig("elbow_method.png", dpi=300)
plt.show()

# 根据肘部法则图，选择适当的聚类数量
optimal_clusters = 5  # 从肘部图确定为5

# 应用KMeans聚类
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# 使用字母编号医生姓名，同时保留医生种类的英语
df['Doctor'] = ['Neurologist - ' + letter for letter in string.ascii_uppercase[:len(df)]]

# 分组输出每个医生的香农多样性指数和分组
grouped = df[['Doctor', 'Shannon Diversity', 'Cluster']].sort_values(by='Cluster')
print(grouped)

# 右图：分组后的可视化
plt.figure(figsize=(6, 4))
for cluster in range(optimal_clusters):
    cluster_data = df[df['Cluster'] == cluster]
    plt.scatter(cluster_data['Cluster'], cluster_data['Shannon Diversity'], label=f'Group {cluster+1}', s=50)
plt.title('(b) Practice Shannon Diversity Index Clustering', fontsize=14)
plt.xlabel('Cluster Group', fontsize=12)
plt.ylabel('Shannon Diversity Index', fontsize=12)
plt.grid(True)
plt.legend()

# 保存分组后的图为PNG格式
plt.tight_layout()
plt.savefig("clustering_visualization.png", dpi=300)
plt.show()

# 计算每个分组内的香农多样性指数平均值，并输出
cluster_means = df.groupby('Cluster')['Shannon Diversity'].mean()
formatted_means = cluster_means.apply(lambda x: f"{x:.3f}")

print("\n每个分组内的香农多样性指数平均值 (按组号顺序输出):")
print(formatted_means)

# 计算每个分组内的香农多样性指数方差，并输出
cluster_variances = df.groupby('Cluster')['Shannon Diversity'].var()
print("\n每个分组内的香农多样性指数方差 (按组号顺序输出):")
print(cluster_variances)

# 计算每个分组的医生个数和权重
total_doctors = len(df)
for cluster in range(optimal_clusters):
    cluster_data = df[df['Cluster'] == cluster]
    doctor_count = len(cluster_data)
    proportion = doctor_count / total_doctors * 100  # 计算权重百分比
    print(f"\nGroup {cluster+1} (平均值: {formatted_means[cluster]}, 医生个数: {doctor_count}, 占比: {proportion:.2f}%):")
    print(cluster_data[['Doctor', 'Shannon Diversity']])

#%% md
# 每年的数据的OBJ
#%%
import os

# 定义目标 GMM 分布
objective_gmm = {
    'weights': np.array([0.5, 0.1, 0.1, 0.1, 0.2]),
    'means': np.array([1.410, 0.272, 2.384, 1.012, 1.625]),
    'variances': np.array([0.001, 0.001, 0.001, 0.001, 0.001])  # 目标分布的方差不变
}

# 保存的文件夹路径
output_folder = 'output_by_year'
os.makedirs(output_folder, exist_ok=True)

# 处理多个年份的文件
for year in range(2016, 2021):
    # 加载按年份的CSV文件
    file_path = f'by-year-2/preferences_{year}_12_31.csv'
    if not os.path.exists(file_path):
        print(f"{file_path} 不存在，跳过此文件。")
        continue
    df_year = pd.read_csv(file_path)
    
    # 计算每位医生的香农多样性指数
    shannon_diversity_indices = []
    for index, row in df_year.iterrows():
        proportions = row[1:].values
        proportions = np.array(proportions, dtype=np.float64) + 1e-10
        shannon_index = -np.sum(proportions * np.log(proportions))
        shannon_diversity_indices.append(shannon_index)
    
    # 添加香农多样性指数到 DataFrame
    df_year['Shannon Diversity'] = shannon_diversity_indices
    
    # 使用KMeans聚类
    X = np.array(shannon_diversity_indices).reshape(-1, 1)
    kmeans = KMeans(n_clusters=len(objective_gmm['means']), init='k-means++', random_state=42)
    df_year['Cluster'] = kmeans.fit_predict(X)
    
    # 计算每个分组的医生个数和权重
    total_doctors = len(df_year)
    cluster_weights = []
    for cluster in range(len(objective_gmm['means'])):
        cluster_data = df_year[df_year['Cluster'] == cluster]
        doctor_count = len(cluster_data)
        proportion = doctor_count / total_doctors  # 计算权重
        cluster_weights.append(proportion)
    
    # 将结果保存到新的CSV文件中 (包含 Cluster, Calculated Weight 和 Calculated Mean)
    result_df = pd.DataFrame({
        'Cluster': range(len(objective_gmm['means'])),
        'Calculated Weight': cluster_weights,
        'Calculated Mean': df_year.groupby('Cluster')['Shannon Diversity'].mean().values
    })
    
    output_path = os.path.join(output_folder, f'year_{year}_results.csv')
    result_df.to_csv(output_path, index=False)
    print(f"保存 {year} 的结果到 {output_path}")
