# Copyright [2024] [CHAO LI/ITMO National Center for Cognitive Research]
# Licensed under the Apache License, Version 2.0 (the "License");
# Medical use requires compliance with ethical guidelines in ETHICAL_GUIDELINES.md.


# Setting up the norm-augmented Markov game environment-设置规范增强的马尔科夫博弈环境
import random
import numpy as np
from itertools import combinations
from functools import partial
import concurrent.futures
from threading import Lock
from collections import Counter
#
class MultiAgentMarkovGame:

    def __init__(self, num_chief_doctors, num_assistant_doctors, diagnostic_methods, top_3_methods, gamma=0.9, temperature=5):
        self.diagnostic_methods = diagnostic_methods
        self.top_3_methods = top_3_methods
        self.gamma = gamma
        self.temperature = temperature

        # 初始化代理
        self.agents = []
        for i in range(num_chief_doctors):
            self.agents.append(ChiefDoctor(i, diagnostic_methods, top_3_methods))
        for i in range(num_assistant_doctors):
            self.agents.append(AssistantDoctor(i + num_chief_doctors, diagnostic_methods, top_3_methods))

        # 初始化时间步
        self.time_step = 0
        self.print_lock = Lock()  # 初始化锁
        self.index_counter = Counter()
        self.beliefs_history = []

    # def record_beliefs(self):
    #     # 记录当前所有代理的禁令信念
    #     current_beliefs = []
    #     for agent in self.agents:
    #         beliefs = {
    #             'must_use_top_3_diagnostics': agent.prior_beliefs[must_use_top_3_diagnostics],
    #             'role': agent.role  # 记录代理的角色
    #         }
    #         current_beliefs.append(beliefs)
    #     self.beliefs_history.append(current_beliefs)

    def record_beliefs(self):
        # 记录当前所有代理的规范信念
        current_beliefs = []
        for agent_id, agent in enumerate(self.agents):  # 使用 enumerate 获取 agent_id
            beliefs = {
                'agent_id': agent_id,  # 添加代理的唯一标识符
                'must_use_top_3_diagnostics': agent.prior_beliefs[must_use_top_3_diagnostics],
                'must_attend_teaching_activity': agent.prior_beliefs[must_attend_teaching_activity],
                'must_use_headshake_every_10_steps': agent.prior_beliefs[must_use_headshake_every_10_steps],
                'must_use_sakkad_if_even_step': agent.prior_beliefs[must_use_sakkad_if_even_step],
                'must_use_babinskogo_veylia_if_not_used_recently': agent.prior_beliefs[must_use_babinskogo_veylia_if_not_used_recently],
                'must_use_shagovyi_if_not_used_last_time': agent.prior_beliefs[must_use_shagovyi_if_not_used_last_time],
                'must_use_khalmagi_every_20_steps': agent.prior_beliefs[must_use_khalmagi_every_20_steps],
                'must_use_romberga_if_not_used_recently': agent.prior_beliefs[must_use_romberga_if_not_used_recently],
                'role': agent.role  # 记录代理的角色
            }
            current_beliefs.append(beliefs)

        # 将当前信念记录到信念历史中
        self.beliefs_history.append(current_beliefs)


    def get_initial_state(self, agent):
        initial_week = 1
        initial_role = agent.role
        return (initial_week, initial_role)

    def reset(self):
        self.time_step = 0
        for agent in self.agents:
            agent.reset()

        print_lock = Lock()

    def safe_print(self,*args, **kwargs):
        with self.print_lock:
            print(*args, **kwargs)

    #这里只有更新规则信念的时候，是多代理马尔科夫博弈的，代理本身诊断那就是单纯的马尔科夫MDP
    #在step函数中，使用concurrent.futures.ThreadPoolExecutor来并行执行多个代理的RTDP递归。
    #我直接删掉了1.的diagonose函数，并直接使用rtdp作为主导
    def step(self):
        # 每2个时间步进行一次RTDP
        #难道是并行执行除了问题？
        if self.time_step % 2 == 0:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(self.run_rtdp_for_agent, agent): agent for agent in self.agents}
                for future in concurrent.futures.as_completed(futures):
                    agent = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        print(f'Agent {agent.id} generated an exception: {exc}')

        # 更新历史记录
        for agent in self.agents:
            agent.update_history(self.time_step)

        self.record_beliefs()  # 记录信念
        self.time_step += 1

    def get_current_week(self):
        return (self.time_step % 4) + 1  # 保持周数循环在1到4之间

        #新增的函数
    def run_rtdp_for_agent(self, agent):
        if not agent.diagnostics_used:
            state = self.get_initial_state(agent)
        else:
            state = (self.get_current_week(), agent.role)  # 调整以使用辅助函数获取当前星期
        self.rtdp(agent, state, depth=3)

    def update_agent_beliefs(self, agent):
        #汤普森抽样15步在这
        if self.time_step % 15 == 0:
            norm = agent.thompson_sampling()
        else:
            norm = agent.sampled_norm  # 使用上一次抽样的规范
        self.update_beliefs(agent, norm, agent.get_history())

    #增加对隐性规范的判定，禁令、义务、隐性规范全部放在这里
    #这里的作用就是配合calculate_reward在rtdp函数里更新reward值，这里有汤普森抽样
    def apply_norms(self,agent):

        norm_violation_cost=0

        if  must_use_top_3_diagnostics(agent):
#             print('违反禁令')
            norm_violation_cost -=3
#         else:
#             print("没有违反")

        #暂时忽略教学成本,其实这里没有联系到代理的动作选择
        if must_attend_teaching_activity(self.time_step, agent):
            norm_violation_cost -=1

        #这个增加的所谓隐性规范内容，基于我们完整的实验方向，也许是不需要的
#         if chief_doctor_can_choose_additional_diagnostics(agent):
#             agent.update_diagnostics_used(agent._select_additional_diagnostics(self.time_step))

#         if assistant_doctor_can_add_diagnostic_post_teaching(self.time_step,agent):
#             agent.update_diagnostics_used(agent._select_additional_diagnostics(self.time_step))

        diagnostics_used_count = len(agent.diagnostics_used[-1].split(','))
        usage_cost = diagnostics_used_count * 0.01
        # 判断非必须5的诊断方案数量并计算额外奖励，这里有问题，计算方法
        #至少我们已经解决了只更新其中一个的问题.这是重大前进
#         non_mandatory_diagnostics_count = len([diag for diag in agent.diagnostics_used if diag not in agent.top_3_methods])
#         additional_reward = 0
#         for _ in range(non_mandatory_diagnostics_count):
#             if random.random() < 0.5:
#                 additional_reward += 0.1

        return norm_violation_cost + usage_cost



    #从Q表来看，学习到了规范1，但是这里的信念更新有错误
    #一定要调用玻尔兹曼分布获得的概率来计算
    #遵守规范给奖励，违反规范给惩罚，然后更新到Q表里，再用Q表的内容计算玻尔兹曼分布概率，从而更新规范信念
    #其实玻尔兹曼分布是核心。
#     def update_beliefs(self, agent, norm, history):
#         prior = agent.prior_beliefs[norm]
#         likelihood = 1
#         current_week = self.get_current_week()  # 使用函数来保持逻辑一致

#         #其实就是判断，在当前，其他所有代理的Q表里，使用玻尔兹曼分布选到与该代理同样动作（遵守该规范的动作？）的概率是多少,这句话非常对
#         #但是还是更新的不准确，遵守该规范的动作而不是同样的动作
#         #因为规范、似然值、Q表没有对应到一起
#         for other_agent in self.agents:
#             if other_agent != agent:
#                 state = (current_week, other_agent.role)

#                 #应该去找到遵守规范对应动作对应的Q表位置
#                 action = agent.diagnostics_used[-1]

#                 if state in other_agent.q_values:
# #                     other_agent_q_value = other_agent.q_values[state][action]
# #                     exp_val = np.exp(other_agent_q_value / self.temperature)

#                     #变成了一个如何控制信念正确增长的问题
# #                     print(f"exp_val:{exp_val}")
#                     total_exp = sum(np.exp(other_agent.q_values[state][action] / self.temperature)
#                                     for action in other_agent.q_values[state])
#                     action_probabilities = [(action, np.exp(other_agent.q_values[state][action] / self.temperature) / total_exp)
#                                         for action in other_agent.q_values[state] if total_exp > 0]
# # #                     print(f"total_exp:{total_exp}")
# #                     prob=exp_val/total_exp
# # #                     print(f"prob:{prob}")

#                     likelihood *= exp_val

# #         print(f"似然值: {likelihood}")

#         # 归一化likelihood
#         if likelihood > 0:
#             posterior = likelihood * prior
#         else:
#             posterior = prior

#         # Update the agent's belief for the norm
#         agent.prior_beliefs[norm] = posterior

#         # Normalize the beliefs to sum to 1
#         total = sum(agent.prior_beliefs.values())
#         for key in agent.prior_beliefs:
#             agent.prior_beliefs[key] /= total


    #既然观察，Q表，代理最后都收敛到正确的策略，那么这里采用最新的
    #根据代理最近采取的行动来增加似然值
    #是合理的，因为随着时间步推进，我们认为代理每次选择的动作，是趋向于固定的
    #这里也许可以使用一些工程办法，使得更新稍微准确一些，或者至少捕捉到更新准确的那一刻
#     def update_beliefs(self, agent, norm, history):
#         prior = agent.prior_beliefs[norm]
#         likelihood = 1
#         current_week = self.get_current_week()
#         regularization_lambda = 0.000001

#         for other_agent in self.agents:
#             if other_agent != agent:
#                 state = (current_week, other_agent.role)

#                 # 确保只考虑符合当前规范的行动
#                 if state in other_agent.q_values:
#                     # 计算所有可能行动的指数值
#                     exp_vals = np.array([np.exp(q_val / self.temperature) for q_val in other_agent.q_values[state].values()])
#                     # 归一化这些值得到概率分布
#                     if np.sum(exp_vals) > 0:
#                         prob_vals = exp_vals / np.sum(exp_vals)
#                     else:
#                         prob_vals = np.zeros_like(exp_vals)
#                     # 如果代理的最近行动在规范的行动列表中，计算这些行动的概率
#                     if agent.diagnostics_used[-1] in other_agent.q_values[state]:
#                         #可以成功定位，因为prob_vals也是按顺序被计算出来的，但是index是随机的
#                         action_index=list(other_agent.q_values[state].keys()).index(agent.diagnostics_used[-1])
#                         likelihood *= prob_vals[action_index]
# #                         print(f"似然值: {likelihood}")
# #                         self.index_counter[action_index] += 1
# #                         self.safe_print(list(other_agent.q_values[state].keys()).index(agent.diagnostics_used[-1]))

#         # 引入L2正则化项
#         l2_regularization_term = regularization_lambda * np.sum(np.square(list(agent.prior_beliefs.values())))
#         # 计算后验概率
#         if likelihood > 0:
#             posterior = likelihood * prior - l2_regularization_term
#         else:
#             posterior = prior
#         # 更新代理的信念
#         agent.prior_beliefs[norm] = posterior

#         # 归一化信念
#         total = sum(agent.prior_beliefs.values())
#         for key in agent.prior_beliefs:
#             agent.prior_beliefs[key] /= total

    #带锁版本
    def update_beliefs(self, agent, norm, history):
        prior = agent.prior_beliefs[norm]
        likelihood = 1
        current_week = self.get_current_week()

        # Adam-like parameters
        beta1 = 0.9  # 动量系数
        beta2 = 0.999  # RMSprop系数
        epsilon = 1e-8  # 防止除零的小量
        alpha = 0.001  # 初始学习率

        # 存储动量和二阶矩估计
        if not hasattr(agent, 'm'):
            agent.m = {k: 0 for k in agent.prior_beliefs}
            agent.v = {k: 0 for k in agent.prior_beliefs}
            agent.t = 0  # 时间步数，用于偏差校正

        # 初始化锁定机制
        if not hasattr(agent, 'locked_norms'):
            agent.locked_norms = {}

        # 计算似然值
        for other_agent in self.agents:
            if other_agent != agent:
                state = (current_week, other_agent.role)

                # 确保只考虑符合当前规范的行动
                if state in other_agent.q_values:
                    # 计算所有可能行动的指数值
                    exp_vals = np.array([np.exp(q_val / self.temperature) for q_val in other_agent.q_values[state].values()])
                    # 归一化这些值得到概率分布
                    if np.sum(exp_vals) > 0:
                        prob_vals = exp_vals / np.sum(exp_vals)
                    else:
                        prob_vals = np.zeros_like(exp_vals)
                    # 如果代理的最近行动在规范的行动列表中，计算这些行动的概率
                    if agent.diagnostics_used[-1] in other_agent.q_values[state]:
                        # 关键在这里，在于这里，获取其他代理最近的行动更新似然值
                        action_index = list(other_agent.q_values[state].keys()).index(agent.diagnostics_used[-1])
                        likelihood *= prob_vals[action_index]

        # 更新时间步
        agent.t += 1

        # 计算当前梯度
        gradient = likelihood * prior - prior

        # 动量更新
        agent.m[norm] = beta1 * agent.m[norm] + (1 - beta1) * gradient
        agent.v[norm] = beta2 * agent.v[norm] + (1 - beta2) * (gradient ** 2)

        # 偏差修正
        m_hat = agent.m[norm] / (1 - beta1 ** agent.t)
        v_hat = agent.v[norm] / (1 - beta2 ** agent.t)

        # 自适应学习率
        adaptive_learning_rate = alpha / (np.sqrt(v_hat) + epsilon)

        # 计算更新值，加入自适应学习率和动量
        posterior_update = adaptive_learning_rate * m_hat

        # 更新后验概率
        posterior = prior + posterior_update

        # 确保 posterior 在有效范围内
        posterior = min(max(posterior, 0), 1)

        # 如果该规范的信念已经接近 1，则锁定
        if posterior >= 0.95:
            agent.locked_norms[norm] = posterior

        # 只有在规范未被锁定时，才更新信念
        if norm not in agent.locked_norms:
            agent.prior_beliefs[norm] = posterior

        # 归一化信念：只归一化未锁定的规范信念
        unlocked_norms = [n for n in agent.prior_beliefs if n not in agent.locked_norms]
        total_unlocked_beliefs = sum(agent.prior_beliefs[n] for n in unlocked_norms)

        if total_unlocked_beliefs > 0:
            for n in unlocked_norms:
                agent.prior_beliefs[n] /= total_unlocked_beliefs

        # 锁定的规范保持不变
        for locked_norm, locked_value in agent.locked_norms.items():
            agent.prior_beliefs[locked_norm] = locked_value


    #无锁优秀版本
    # def update_beliefs(self, agent, norm, history):
    #     prior = agent.prior_beliefs[norm]
    #     likelihood = 1
    #     current_week = self.get_current_week()

    #     # Adam-like parameters
    #     beta1 = 0.9  # 动量系数
    #     beta2 = 0.999  # RMSprop系数
    #     epsilon = 1e-8  # 防止除零的小量
    #     alpha = 0.001  # 初始学习率

    #     # 存储动量和二阶矩估计
    #     if not hasattr(agent, 'm'):
    #         agent.m = {k: 0 for k in agent.prior_beliefs}
    #         agent.v = {k: 0 for k in agent.prior_beliefs}
    #         agent.t = 0  # 时间步数，用于偏差校正

    #     for other_agent in self.agents:
    #         if other_agent != agent:
    #             state = (current_week, other_agent.role)

    #             # 确保只考虑符合当前规范的行动
    #             if state in other_agent.q_values:
    #                 # 计算所有可能行动的指数值
    #                 exp_vals = np.array([np.exp(q_val / self.temperature) for q_val in other_agent.q_values[state].values()])
    #                 # 归一化这些值得到概率分布
    #                 if np.sum(exp_vals) > 0:
    #                     prob_vals = exp_vals / np.sum(exp_vals)
    #                 else:
    #                     prob_vals = np.zeros_like(exp_vals)
    #                 # 如果代理的最近行动在规范的行动列表中，计算这些行动的概率
    #                 if agent.diagnostics_used[-1] in other_agent.q_values[state]:
    #                     #关键在这里，在于这里，获取其他代理最近的行动更新似然值
    #                     action_index = list(other_agent.q_values[state].keys()).index(agent.diagnostics_used[-1])
    #                     likelihood *= prob_vals[action_index]

    #     # 更新时间步
    #     agent.t += 1

    #     # 计算当前梯度
    #     gradient = likelihood * prior - prior

    #     # 动量更新
    #     agent.m[norm] = beta1 * agent.m[norm] + (1 - beta1) * gradient
    #     agent.v[norm] = beta2 * agent.v[norm] + (1 - beta2) * (gradient ** 2)

    #     # 偏差修正
    #     m_hat = agent.m[norm] / (1 - beta1 ** agent.t)
    #     v_hat = agent.v[norm] / (1 - beta2 ** agent.t)

    #     # 自适应学习率
    #     adaptive_learning_rate = alpha / (np.sqrt(v_hat) + epsilon)

    #     # 计算更新值，加入自适应学习率和动量
    #     posterior_update = adaptive_learning_rate * m_hat

    #     # 更新后验概率
    #     posterior = prior + posterior_update

    #     # 确保posterior在有效范围内
    #     posterior = min(max(posterior, 0), 1)

    #     # 更新代理的信念
    #     agent.prior_beliefs[norm] = posterior

    #     # 归一化信念
    #     total = sum(agent.prior_beliefs.values())
    #     if total > 0:
    #         for key in agent.prior_beliefs:
    #             agent.prior_beliefs[key] /= total

    def print_index_usage(self):
        total_counts = sum(self.index_counter.values())
        if total_counts > 0:
            sorted_indexes = sorted(self.index_counter.items(), key=lambda x: x[1], reverse=True)
            print("Index Usage Proportions:")
            for index, count in sorted_indexes:
                print(f"Index {index}: {count / total_counts:.2%}")
        else:
            print("No indexes have been recorded yet.")

    def rtdp(self, agent, state, depth=3):
        if depth == 0:
            return

        #这里已经获取了action,要不就action没排序？？
        action = self.select_action(agent, state)
        #获取了action后，要直接更新到diagnostics_used里去
        agent.update_diagnostics_used([action])  # 更新诊断方法的使用记录

#         print(f"Agent: {agent.role}, diagnostics_used: {agent.diagnostics_used}")

        #把对规范的检验导致的reward变化，放在这里就行了
        reward = self.calculate_reward(agent)
        next_state = self.get_next_state(agent, state, action)
        q_value =reward + self.gamma * agent.v_values.get(next_state, 0)

        #更新代理自己的
        agent.q_values[state][action] = q_value
        agent.v_values[state] = max(agent.q_values[state].values())

        #更新对规范信念
        self.update_agent_beliefs(agent)

        # 检查是否每100个时间步
        if self.time_step % 100 == 0:
            agent.norm_practice()

        self.rtdp(agent, next_state, depth - 1)

    #调用q值计算概率，这里是正确的，
    def select_action(self, agent, state):
        actions = list(agent.q_values[state].keys())
        probabilities = [np.exp(agent.q_values[state][a] / self.temperature) for a in actions]
        probabilities /= np.sum(probabilities)
        # 确保概率和为1
        probabilities[-1] = 1 - np.sum(probabilities[:-1])
        action=random.choices(actions, probabilities)[0]
#          # 获取最大概率值及其索引
#         max_prob = np.max(probabilities)
#         max_index = np.argmax(probabilities)

#         # 输出相关信息
#         print(f"Agent: {agent.role}, Max Probability: {max_prob}, Index: {max_index}")
        return action


    #是reward的问题！！！！因为之前计算了全局的所谓R值
    def calculate_reward(self, agent):
        reward =  -1 * self.apply_norms(agent)
    # 调试输出: 打印奖励值
    #print(f"Calculated reward for agent {agent.id} with role {agent.role}: {agent.reward}")
        return reward


    #这里应该怎么添加状态转移函数？这里偷懒不添加状态转移概率函数
    #问题出这里，没有状态转移函数！！！！！！但是状态还是在变化
    def get_next_state(self, agent, state, action):
        week, role = state
        next_week = (week + 1 if week < 4 else 1)
        return (next_week, role)

    def render(self):
        print(f"Time Step: {self.time_step}")
        for agent in self.agents:
            print(f"Agent {agent.role} Beliefs:")
            sorted_beliefs = agent.sort_beliefs()
            for norm, belief in sorted_beliefs:
                print(f"Norm: {norm.__name__}, Belief: {belief}")

    def plot_beliefs(self):
        num_agents = len(self.agents)
        cols = 4
        rows = (num_agents + cols - 1) // cols

        fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(20, max(20, 5 * rows)))
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
        axs = axs.flatten()

        for index, agent in enumerate(self.agents):
            ax = axs[index]
            beliefs = {agent.norm_labels[norm]: belief for norm, belief in agent.prior_beliefs.items()}
            names = list(beliefs.keys())
            values = list(beliefs.values())

            ax.bar(names, values, color='skyblue')
            ax.set_title(f'Agent {agent.id} ({agent.role}) Beliefs')
            ax.set_ylim([0, 1])
            ax.tick_params(axis='x', rotation=45)

        for index in range(len(self.agents), len(axs)):
            axs[index].set_visible(False)

        plt.show()

class Agent:
    def __init__(self, id, role, diagnostic_methods, top_3_methods):
        self.id = id
        self.role = role  # 添加角色属性
        self.diagnostic_methods = diagnostic_methods
        self.top_3_methods = top_3_methods
        self.diagnostics_used = []
        self.diagnostics_used_last = []  # 上次诊断记录
        self.norms = [must_use_top_3_diagnostics, must_attend_teaching_activity, must_use_headshake_every_10_steps,
                      must_use_sakkad_if_even_step, must_use_babinskogo_veylia_if_not_used_recently,
                      must_use_shagovyi_if_not_used_last_time, must_use_khalmagi_every_20_steps,
                      must_use_romberga_if_not_used_recently]
                # Assign custom labels to norms
        self.norm_labels = {
            must_use_top_3_diagnostics: 'A',
            must_attend_teaching_activity: 'B',
            must_use_headshake_every_10_steps: '1',
            must_use_sakkad_if_even_step: '2',
            must_use_babinskogo_veylia_if_not_used_recently: '3',
            must_use_shagovyi_if_not_used_last_time: '4',
            must_use_khalmagi_every_20_steps: '5',
            must_use_romberga_if_not_used_recently: '6'
        }

        self.q_values = {}
        self.v_values = {}
        self.initialize_q_v_values()

        # Initialize prior beliefs with normalization
        self.prior_beliefs = {norm: 0.05 for norm in self.norms}
        total = sum(self.prior_beliefs.values())
        for key in self.prior_beliefs:
            self.prior_beliefs[key] /= total

        self.history = []
        self.sampled_norm = None  # 初始化 sampled_norm 属性
    # #无锁优秀班班
    # def norm_practice(self):

    #     # 获取信念值并排序，取最高的两个
    #     sorted_beliefs = sorted(self.prior_beliefs.items(), key=lambda item: item[1], reverse=True)
    #     top_two_norms = sorted_beliefs[:2]  # 取前两个最高信念

    #     for norm, value in top_two_norms:
    #         # 判断是否为禁令
    #         if norm not in [must_attend_teaching_activity]:
    #             # 如果不是禁令或义务，减少30%信念
    #             self.prior_beliefs[norm] *= 0.7
    #                 # 如果不是参加教学活动的规范，减少15%信念
    #         # if norm not in [must_attend_teaching_activity]:
    #         #     self.prior_beliefs[norm] *= 0.85
    #     # 重新归一化信念确保总和为1
    #     total = sum(self.prior_beliefs.values())
    #     if total > 0:
    #         for key in self.prior_beliefs:
    #             self.prior_beliefs[key] /= total

    #有锁版本
    def norm_practice(self):
        # 初始化锁定机制，如果还没有锁定任何规范
        if not hasattr(self, 'locked_norms'):
            self.locked_norms = {}  # 用于记录已锁定的核心规范
        if not hasattr(self, 'core_norms'):
            self.core_norms = [must_use_top_3_diagnostics, must_attend_teaching_activity]  # 核心规范列表

        # 获取信念值并排序，取信念最高的两个规范
        sorted_beliefs = sorted(self.prior_beliefs.items(), key=lambda item: item[1], reverse=True)
        top_two_norms = sorted_beliefs[:2]  # 取前两个信念最高的规范

        # 处理信念更新前，确保核心规范之一是否接近1，来进行锁定
        for norm in self.core_norms:
            # 如果某个核心规范信念值接近 1 且尚未锁定，则将其锁定
            if self.prior_beliefs[norm] >= 0.95 and norm not in self.locked_norms:
                self.locked_norms[norm] = self.prior_beliefs[norm]  # 锁定该规范

        # 对非核心规范以及未锁定的核心规范进行信念的正常更新
        for norm, value in top_two_norms:
            if norm not in self.locked_norms:
                if norm not in self.core_norms:  # 如果不是核心规范，减少信念值
                    self.prior_beliefs[norm] *= 0.7  # 减少30%信念

        # 重新归一化信念，确保总和为1，但不包括已锁定的规范
        unlocked_norms = [n for n in self.prior_beliefs if n not in self.locked_norms]
        total_unlocked_beliefs = sum(self.prior_beliefs[n] for n in unlocked_norms)

        # 归一化未锁定的规范信念
        if total_unlocked_beliefs > 0:
            for norm in unlocked_norms:
                self.prior_beliefs[norm] /= total_unlocked_beliefs

        # 锁定的规范信念保持不变
        for locked_norm, locked_value in self.locked_norms.items():
            self.prior_beliefs[locked_norm] = locked_value


    def initialize_q_v_values(self):
        all_combinations = self.generate_diagnostic_combinations()
        # Initialize Q and V tables only for the agent's own role
        for week in range(1, 5):
            state = (week, self.role)
            self.q_values[state] = {','.join(comb): 1 for comb in all_combinations}
            self.v_values[state] = 1

    #每次生成的都是相同的
    def generate_diagnostic_combinations(self, min_combination_size=2, max_combination_size=5):
        all_combinations = []
        for i in range(min_combination_size, max_combination_size + 1):
            for combination in combinations(self.diagnostic_methods, i):
                all_combinations.append(combination)
        return all_combinations


#     def update_reward(self, reward):
#         self.reward += reward

    def reset(self):
        self.reward = 1
        self.diagnostics_used = []
        self.diagnostics_used_last = []
        self.history = []

    def get_history(self):
        # 返回代理的状态-行动历史
        return self.history

    #汤普森抽样
    def thompson_sampling(self):
        self.sampled_norm = random.choice(self.norms)
        return self.sampled_norm

    def sort_beliefs(self):
        sorted_beliefs = sorted(self.prior_beliefs.items(), key=lambda item: item[1], reverse=True)
        return sorted_beliefs

    def update_diagnostics_used(self, diagnostics):

        self.diagnostics_used.extend(diagnostics)

        if len(self.diagnostics_used)> 1:
            self.diagnostics_used_last = self.diagnostics_used[-2]


    def update_history(self, time_step):
        self.history.append((time_step, list(self.diagnostics_used)))

    def __str__(self):
        return f"Agent {self.id} ({self.role}): Reward {self.reward}"

    def print_q_v_tables(self):
        print(f"Agent {self.id} ({self.role}) Q and V Tables:")
        print("Q-table:")
        for state, actions in self.q_values.items():
            print(f"State {state}:")
            for action, value in actions.items():
                print(f"  Action {action}: {value:.2f}")
        print("V-table:")
        for state, value in self.v_values.items():
            print(f"  State {state}: {value:.2f}")
        print("\n")

class ChiefDoctor(Agent):
    def __init__(self, id, diagnostic_methods, top_3_methods):
        super().__init__(id, 'chief_doctor', diagnostic_methods, top_3_methods.tolist())


    def _select_additional_diagnostics(self,time_step):
        num_mandatory = len(self.top_3_methods)
        num_additional = random.randint(0, 5 - num_mandatory)
        available_diagnostics = [m for m in self.diagnostic_methods if m not in self.top_3_methods]
        return random.sample(available_diagnostics, k=num_additional)

class AssistantDoctor(Agent):
    def __init__(self, id, diagnostic_methods, top_3_methods):
        super().__init__(id, 'assistant_doctor', diagnostic_methods, top_3_methods.tolist())

    def _select_additional_diagnostics(self, time_step):
        if time_step % 31 == 0 and time_step!=0:
            available_diagnostics = [m for m in self.diagnostic_methods if m not in self.top_3_methods]
            if available_diagnostics:
                return random.choice(available_diagnostics)
        return None

#     def complete_teaching_activity(self):
#         self.update_reward(1-0.5)  # 完成教学活动的奖励和成本


class Patient:
    def __init__(self):
        self.severity = random.random()  # 随机生成一个病情严重程度

import matplotlib.pyplot as plt
env = MultiAgentMarkovGame(2,8,diagnostic_methods, top_3_methods)
# 运行环境
env.reset()
for _ in range(5000):
    env.step()
#     env.render()
# env.plot_beliefs()
# env.agents[1].print_q_v_tables()
# for agent in env.agents:
#     print(f"Agent {agent.id} Beliefs:")
#     sorted_beliefs = agent.sort_beliefs()
#     for norm, belief in sorted_beliefs:
#         print(f"Norm: {norm.__name__}, Belief: {belief}")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_beliefs_history(env):
    # 提取信念历史数据
    beliefs_history = env.beliefs_history
    time_steps = range(len(beliefs_history))  # 时间步

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 6))  # 调整图形大小

    # 用于存储图例信息
    lines = []

    # 设置线条颜色的透明度
    alpha_value = 0.8

    # 绘制每个代理的信念曲线，根据角色设置颜色和线条样式
    for agent_id, agent_history in enumerate(zip(*beliefs_history)):
        beliefs = [agent['must_attend_teaching_activity'] for agent in agent_history]

        # 根据代理的角色设置颜色，主治医生为绿色，助理医师为黄色，调整颜色透明度和线条粗细
        if agent_history[0]['role'] == 'chief_doctor':
            line, = ax.plot(time_steps, beliefs, color='#66C2A5', lw=3, alpha=alpha_value)  # 增加线条粗细和透明度
            lines.append(line)
        elif agent_history[0]['role'] == 'assistant_doctor':
            line, = ax.plot(time_steps, beliefs, color='#FFD92F', lw=3, alpha=alpha_value)  # 增加线条粗细和透明度
            lines.append(line)

    # 设置x轴和y轴标签，字体大小为20
    ax.set_ylabel('Belief', fontsize=35)
    ax.set_xlabel('Time Step', fontsize=20)
    plt.xticks(fontsize=25)  # 调整x轴刻度标签的字体大小
    plt.yticks(fontsize=25)  # 调整y轴刻度标签的字体大小

    # 添加网格线
    ax.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)

    # 调整刻度字体大小为20
    ax.tick_params(axis='both', which='major', labelsize=18)

    # 显示图例，设置透明背景并调整字体
    legend = ax.legend([lines[0], lines[-1]], ['Chief Doctor (Green)', 'Assistant Doctor (Yellow)'], fontsize=25, loc='lower right')
    legend.get_frame().set_alpha(0.5)  # 使图例背景透明

    # 调整图形布局
    plt.tight_layout()

    # 显示图形
    plt.show()

# 调用绘图函数
plot_beliefs_history(env)


import pandas as pd

def save_beliefs_to_csv_pandas(env, file_name='beliefs_history.csv'):
    # 获取信念历史数据
    beliefs_history = env.beliefs_history

    # 创建一个列表来存储所有信念记录
    rows = []

    # 遍历每个时间步和代理的信念
    for time_step, beliefs in enumerate(beliefs_history):
        for agent_beliefs in beliefs:
            row = {
                'Time Step': time_step,
                'Agent ID': agent_beliefs['agent_id'],
                'must_use_top_3_diagnostics': agent_beliefs['must_use_top_3_diagnostics'],
                'must_attend_teaching_activity': agent_beliefs['must_attend_teaching_activity'],
                'must_use_headshake_every_10_steps': agent_beliefs['must_use_headshake_every_10_steps'],
                'must_use_sakkad_if_even_step': agent_beliefs['must_use_sakkad_if_even_step'],
                'must_use_babinskogo_veylia_if_not_used_recently': agent_beliefs['must_use_babinskogo_veylia_if_not_used_recently'],
                'must_use_shagovyi_if_not_used_last_time': agent_beliefs['must_use_shagovyi_if_not_used_last_time'],
                'must_use_khalmagi_every_20_steps': agent_beliefs['must_use_khalmagi_every_20_steps'],
                'must_use_romberga_if_not_used_recently': agent_beliefs['must_use_romberga_if_not_used_recently'],
                'Role': agent_beliefs['role']
            }
            rows.append(row)

    # 使用 pandas 将数据转换为 DataFrame 并保存为 CSV 文件
    df = pd.DataFrame(rows)
    df.to_csv(file_name, index=False)

# 调用函数保存信念历史到CSV文件
save_beliefs_to_csv_pandas(env)


def plot_agent_beliefs(agent_id, env):
    # 提取信念历史数据
    beliefs_history = env.beliefs_history
    time_steps = range(len(beliefs_history))  # 时间步

    # 获取指定代理的信念历史
    agent_beliefs = [time_step[agent_id] for time_step in beliefs_history]  # 提取每个时间步特定代理的信念

    # 设置线条颜色和样式的字典
    norms_colors = {
        'must_use_top_3_diagnostics': '#66C2A5',  # 绿色
        'must_attend_teaching_activity': '#FFD92F',  # 黄色
        'must_use_headshake_every_10_steps': '#8DA0CB',  # 蓝色
        'must_use_sakkad_if_even_step': '#E78AC3',  # 粉色
        'must_use_babinskogo_veylia_if_not_used_recently': '#FC8D62',  # 橙色
        'must_use_shagovyi_if_not_used_last_time': '#A6D854',  # 浅绿
        'must_use_khalmagi_every_20_steps': '#E5C494',  # 浅棕
        'must_use_romberga_if_not_used_recently': '#B3B3B3'  # 灰色
    }

    # 核心规范和其他规范
    core_norms = ['must_use_top_3_diagnostics', 'must_attend_teaching_activity']

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制每个规范的信念曲线
    lines = []  # 用于存储核心规范的线条
    for norm, color in norms_colors.items():
        # 使用字典键访问每个规范的信念值
        beliefs = [agent_belief[norm] for agent_belief in agent_beliefs]
        line, = ax.plot(time_steps, beliefs, color=color, lw=3, alpha=0.8)

        # 只存储核心规范的线条，用于图例
        if norm in core_norms:
            lines.append(line)

    # 设置x轴和y轴标签，字体大小为20
    ax.set_ylabel('Belief', fontsize=35)
    ax.set_xlabel('Time Step', fontsize=35)
    plt.xticks(fontsize=25)  # 调整x轴刻度标签的字体大小
    plt.yticks(fontsize=25)  # 调整y轴刻度标签的字体大小

    # 添加网格线
    ax.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)

    # 调整刻度字体大小
    ax.tick_params(axis='both', which='major', labelsize=18)

    # 显示图例，只标注核心规范，其他规范统一标记为 "Other Norms"
    legend = ax.legend([lines[0], lines[1]], ['Must Use Top 3 Diagnostics', 'Must Attend Teaching Activity'], fontsize=20, loc='center right')
    legend.get_frame().set_alpha(0.5)  # 使图例背景透明

    # 调整图形布局
    plt.tight_layout()

    # 显示图形
    plt.show()

# 调用绘图函数，假设你要展示第一个代理的信念变化
plot_agent_beliefs(agent_id=1, env=env)
plot_agent_beliefs(agent_id=2, env=env)

def calculate_and_sort_average_beliefs(env):
    # 初始化字典来存储每个规范的总信念值
    total_beliefs = {
        'must_use_top_3_diagnostics': 0,
        'must_attend_teaching_activity': 0,
        'must_use_headshake_every_10_steps': 0,
        'must_use_sakkad_if_even_step': 0,
        'must_use_babinskogo_veylia_if_not_used_recently': 0,
        'must_use_shagovyi_if_not_used_last_time': 0,
        'must_use_khalmagi_every_20_steps': 0,
        'must_use_romberga_if_not_used_recently': 0
    }

    # 统计代理数量
    num_agents = len(env.agents)

    # 遍历所有代理并累加他们对每个规范的信念值
    for agent in env.agents:
        total_beliefs['must_use_top_3_diagnostics'] += agent.prior_beliefs[must_use_top_3_diagnostics]
        total_beliefs['must_attend_teaching_activity'] += agent.prior_beliefs[must_attend_teaching_activity]
        total_beliefs['must_use_headshake_every_10_steps'] += agent.prior_beliefs[must_use_headshake_every_10_steps]
        total_beliefs['must_use_sakkad_if_even_step'] += agent.prior_beliefs[must_use_sakkad_if_even_step]
        total_beliefs['must_use_babinskogo_veylia_if_not_used_recently'] += agent.prior_beliefs[must_use_babinskogo_veylia_if_not_used_recently]
        total_beliefs['must_use_shagovyi_if_not_used_last_time'] += agent.prior_beliefs[must_use_shagovyi_if_not_used_last_time]
        total_beliefs['must_use_khalmagi_every_20_steps'] += agent.prior_beliefs[must_use_khalmagi_every_20_steps]
        total_beliefs['must_use_romberga_if_not_used_recently'] += agent.prior_beliefs[must_use_romberga_if_not_used_recently]

    # 计算每个规范的平均信念值
    avg_beliefs = {norm: total / num_agents for norm, total in total_beliefs.items()}

    # 按降序对平均信念值进行排序
    sorted_avg_beliefs = sorted(avg_beliefs.items(), key=lambda item: item[1], reverse=True)

    # 打印排序结果
    print("Average beliefs for each norm (sorted in descending order):")
    for norm, avg_belief in sorted_avg_beliefs:
        print(f"{norm}: {avg_belief:.4f}")

# 调用函数计算并输出平均信念值
calculate_and_sort_average_beliefs(env)


# import numpy as np
# import matplotlib.pyplot as plt

# def plot_beliefs_history(env):
#     # 提取信念历史数据
#     beliefs_history = env.beliefs_history
#     time_steps = range(len(beliefs_history))  # 时间步

#     # 初始化空列表用于存储每个时间步的所有代理信念
#     all_beliefs = []

#     for time_step in time_steps:
#         # 在每个时间步收集所有代理的信念
#         step_beliefs = [agent['must_use_top_3_diagnostics'] for agent in beliefs_history[time_step]]
#         all_beliefs.append(step_beliefs)

#     # 转换为numpy数组，方便计算均值和标准差
#     all_beliefs = np.array(all_beliefs)

#     # 计算每个时间步的均值和标准差
#     mean_beliefs = np.mean(all_beliefs, axis=1)
#     std_beliefs = np.std(all_beliefs, axis=1)

#     # 创建图形
#     fig, ax = plt.subplots(figsize=(10, 5))

#     # 绘制均值曲线，颜色设置为红色，无透明度
#     ax.plot(time_steps, mean_beliefs, label='Mean Belief', color='red')

#     # 绘制标准差阴影区域，颜色设置为浅红色，透明度增加
#     ax.fill_between(time_steps, mean_beliefs - std_beliefs, mean_beliefs + std_beliefs, color='red', alpha=0.15)

#     # 去掉图形标题，设置x和y轴标签，字体放大3倍
#     ax.set_ylabel('Belief', fontsize=20)
#     ax.set_xlabel('Time Step', fontsize=20)

#     # 添加注释：代理数量为30，字体放大3倍
#     ax.text(0.95, 0.55, 'Number of Agents: 30',
#             horizontalalignment='right',
#             verticalalignment='bottom',
#             transform=ax.transAxes,
#             fontsize=20)

#     # 增加透明度较高的淡灰色网格线
#     ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.3)

#     # 显示图例，字体放大3倍
#     ax.legend(fontsize=30)

#     # 调整图形布局
#     plt.tight_layout()

#     # 显示图形
#     plt.show()

# # 调用绘图函数
# plot_beliefs_history(env)
