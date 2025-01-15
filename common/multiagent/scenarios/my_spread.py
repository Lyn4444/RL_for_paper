import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    # def make_world(self):
    #     world = World()
    #     world.dim_c = 2
    #     num_agents = 3
    #     num_landmarks = 3
    #     world.agents = [Agent() for i in range(num_agents)]
    #     world.landmarks = [Landmark() for i in range(num_landmarks)]
    #     for i, agent in enumerate(world.agents):
    #         agent.name = f'agent_{i}'
    #         agent.collide = True
    #         agent.silent = True
    #         agent.size = 0.15
    #         agent.task_done = False  # 增加任务状态
    #     for i, landmark in enumerate(world.landmarks):
    #         landmark.name = f'landmark_{i}'
    #         landmark.collide = False
    #         landmark.movable = False
    #     self.reset_world(world)
    #     return world
    
    def make_world(self):
        world = World()
        world.dim_c = 2
        num_agents = 3
        num_landmarks = 3
        world.agents = [Agent() for i in range(num_agents)]
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        weights = [1.0, 1.0, 1.0]  # 示例权重
        for i, agent in enumerate(world.agents):
            agent.name = f'agent_{i}'
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
            agent.task_done = False  # 增加任务状态
            agent.weight = weights[i]  # 初始化权重
        for i, landmark in enumerate(world.landmarks):
            landmark.name = f'landmark_{i}'
            landmark.collide = False
            landmark.movable = False
        self.reset_world(world)
        return world

# 不全
    def reset_world(self, world):
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.task_done = False
            # 需要根据数据情况修改
            agent.R_s = np.random.uniform(0, 1)
            agent.R_b = np.random.uniform(0, 1)
            agent.T_s = np.random.uniform(20, 100)
            agent.T_b = np.random.uniform(20, 100)
            agent.heat_absorbed = 0.0
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)


    def reward(self, agent, world):
        rew = 0
        agents = world.agents
        # 假设每个 agent 有一个权重属性 weight
        if all(a.task_done for a in agents):
            for a in agents:
                rew += 10 * a.weight  # 根据每个 agent 的权重给予奖励
        return rew

# 不全
    def observation(self, agent, world):
        entity_pos = [entity.state.p_pos - agent.state.p_pos for entity in world.landmarks]
        other_pos = [other.state.p_pos - agent.state.p_pos for other in world.agents if other is not agent]
        task_status = [other.task_done for other in world.agents]
        # 需要添加的观测量
        R_s = [agent.R_s]
        R_b = [agent.R_b]
        T_s = [agent.T_s]
        T_b = [agent.T_b]
        heat_absorbed = [agent.heat_absorbed]
        return np.concatenate([agent.state.p_pos] + entity_pos + other_pos + task_status + R_s + R_b + T_s + T_b + heat_absorbed)

# 判断任务完成条件
    def is_task_done(self, agent):
        # 根据 agent 的 ID 或其他属性判断任务是否完成（需要修改判断条件）
        if agent.agent_id == 0:
            if agent.R_s > 0.8 and agent.T_s > 80:
                agent.task_done = True
            else:
                agent.task_done = False
        elif agent.agent_id == 1:
            if agent.R_b > 0.8 and agent.T_b > 80:
                agent.task_done = True
            else:
                agent.task_done = False
        elif agent.agent_id == 2:
            if agent.heat_absorbed > 100:
                agent.task_done = True
            else:
                agent.task_done = False
        else:
            agent.task_done = False