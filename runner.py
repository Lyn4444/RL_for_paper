from tqdm import tqdm
from agent import Agent
from common.replay_buffer import Buffer
import torch
import os
import numpy as np
import matplotlib.pyplot as plt


# 学习环境，在环境中执行智能体的行为，并管理训练和评估过程
class Runner:
    def __init__(self, args, env):
        self.args = args
        # 噪声率，用于探索
        self.noise = args.noise_rate
        # 探索率
        self.epsilon = args.epsilon
        # 单个回合的最大时间步长
        self.episode_limit = args.max_episode_len
        # 环境对象
        self.env = env
        # 初始化所有智能体
        self.agents = self._init_agents()
        # 创建 Buffer 对象用于存储经验
        self.buffer = Buffer(args)
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    # 初始化所有智能体添加到列表
    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = Agent(i, self.args)
            agents.append(agent)
        return agents

    def run(self):
        # 创建列表 returns 用于存储评估结果
        returns = []
        # 对每个时间步长进行循环
        for time_step in tqdm(range(self.args.time_steps)):
            # 如果当前时间步长是回合长度的倍数，重置环境
            if time_step % self.episode_limit == 0:
                s = self.env.reset()
            # 用于存储动作
            u = []
            actions = []
            with torch.no_grad():
                for agent_id, agent in enumerate(self.agents):
                    # 通过传入状态、噪声和探索率选择动作
                    action = agent.select_action(s[agent_id], self.noise, self.epsilon)
                    u.append(action)
                    actions.append(action)
            # 每个非智能体进行随机动作生成，要修改
            # 非智能体可以生成随机动作，增加环境的动态性和复杂性，是环境的一部分
            for i in range(self.args.n_agents, self.args.n_players):
                actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
            # 执行动作 actions，获取环境返回的下一状态 s_next、奖励 r、是否结束标志 done 以及额外信息 info
            s_next, r, done, info = self.env.step(actions)
            # 将当前状态、动作、奖励和下一状态存储到缓冲区 buffer 中
            self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents], s_next[:self.args.n_agents])
            s = s_next
            # 如果缓冲区中存储的经验数量大于或等于batch_size采样
            if self.buffer.current_size >= self.args.batch_size:
                transitions = self.buffer.sample(self.args.batch_size)
                for agent in self.agents:
                    other_agents = self.agents.copy()
                    other_agents.remove(agent)
                    agent.learn(transitions, other_agents)
            # 如果当前时间步长大于0且是评估周期的倍数，进行评估并将返回值添加到 returns 列表中
            if time_step > 0 and time_step % self.args.evaluate_rate == 0:
                returns.append(self.evaluate())
                # 绘制评估结果图形，要修改，可视化
                plt.figure()
                plt.plot(range(len(returns)), returns)
                plt.xlabel('episode * ' + str(self.args.evaluate_rate / self.episode_limit))
                plt.ylabel('average returns')
                plt.savefig(self.save_path + '/plt.png', format='png')
            # 更新噪声和探索率，逐步减小但不小于0.05
            self.noise = max(0.05, self.noise - 0.0000005)
            self.epsilon = max(0.05, self.epsilon - 0.0000005)
            # 保存评估结果到文件
            np.save(self.save_path + '/returns.pkl', returns)

    # 负责智能体在环境中的评估过
    # def evaluate(self):
        # returns 用于存储每个评估回合的总奖励
        returns = []
        # 每个评估回合 episode 进行循环
        for episode in range(self.args.evaluate_episodes):
            # 重置环境
            s = self.env.reset()
            # 初始化累积奖励为0
            rewards = 0
            # 对每个时间步长进行循环
            for time_step in range(self.args.evaluate_episode_len):
                # 渲染环境的状态，可视化模拟过程，以帮助理解智能体的行为和环境的变化
                self.env.render()
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id], 0, 0)
                        actions.append(action)
                for i in range(self.args.n_agents, self.args.n_players):
                    actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                s_next, r, done, info = self.env.step(actions)
                rewards += r[0]
                s = s_next
            returns.append(rewards)
            print('Returns is', rewards)
        return sum(returns) / self.args.evaluate_episodes
