import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        world.dim_c = 2
        num_agents = 3
        num_landmarks = 3
        world.agents = [Agent() for i in range(num_agents)]
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        weights = [1.0, 1.5, 2.0]  # 示例权重
        for i, agent in enumerate(world.agents):
            agent.name = f'agent_{i}'
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
            agent.task_done = False  # 增加任务状态
            agent.weight = weights[i]  # 初始化权重
            agent.color = np.array([0.35, 0.35, 0.85])  # 设置颜色
        for i, landmark in enumerate(world.landmarks):
            landmark.name = f'landmark_{i}'
            landmark.collide = False
            landmark.movable = False
            landmark.color = np.array([0.25, 0.25, 0.25])  # 设置颜色
        self.reset_world(world)
        return world

    def reset_world(self, world):
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.task_done = False
            agent.R_s = np.random.uniform(0, 1)
            agent.R_b = np.random.uniform(0, 1)
            agent.T_s = np.random.uniform(20, 100)
            agent.T_b = np.random.uniform(20, 100)
            agent.heat_absorbed = 0.0
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        total_reward = 0
        for i, agent in enumerate(world.agents):
            # 根据每个智能体的任务量计算奖励
            agent_reward = self.calculate_agent_reward(agent, world)
            total_reward += agent.weight * agent_reward
        return total_reward

    def calculate_agent_reward(self, agent, world):
        # 示例奖励计算逻辑，可以根据实际需求修改
        reward = 0
        for landmark in world.landmarks:
            dist = np.sqrt(np.sum(np.square(agent.state.p_pos - landmark.state.p_pos)))
            reward -= dist
        return reward

    # 注意维度匹配
    def observation(self, agent, world):
        entity_pos = [entity.state.p_pos - agent.state.p_pos for entity in world.landmarks]
        other_pos = [other.state.p_pos - agent.state.p_pos for other in world.agents if other is not agent]
        task_status = [np.array([other.task_done], dtype=np.float32) for other in world.agents]
        R_s = np.array([agent.R_s], dtype=np.float32)
        R_b = np.array([agent.R_b], dtype=np.float32)
        T_s = np.array([agent.T_s], dtype=np.float32)
        T_b = np.array([agent.T_b], dtype=np.float32)
        heat_absorbed = np.array([agent.heat_absorbed], dtype=np.float32)
        return np.concatenate(
            [agent.state.p_pos] + entity_pos + other_pos + task_status + [R_s, R_b, T_s, T_b, heat_absorbed])

    def is_task_done(self, agent, world):
        if agent.agent_id == 0:
            if agent.R_s > 0.8 and agent.T_s > 80:
                agent.task_done = True
                self.collect_benchmark_data(agent, world)
            else:
                agent.task_done = False
        elif agent.agent_id == 1:
            if agent.R_b > 0.8 and agent.T_b > 80:
                agent.task_done = True
                self.collect_benchmark_data(agent, world)
            else:
                agent.task_done = False
        elif agent.agent_id == 2:
            if agent.heat_absorbed > 100:
                agent.task_done = True
                self.collect_benchmark_data(agent, world)
            else:
                agent.task_done = False
        else:
            agent.task_done = False

    def collect_benchmark_data(self, agent, world):
        # 收集基准数据(示例，应该改为实际需求)
        data = {}
        # 计算每个智能体与目标之间的最小距离
        data['min_dist_to_landmark'] = min([np.sqrt(np.sum(np.square(agent.state.p_pos - landmark.state.p_pos))) for landmark in world.landmarks])
        # 计算智能体之间的碰撞次数
        data['collisions'] = sum([1 for a in world.agents if self.is_collision(a, agent)])
        # 你可以根据需要添加更多的基准数据
        print(f"Benchmark data for agent {agent.name}: {data}")

    # 在训练和测试过程中收集agent任务完成时
    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False