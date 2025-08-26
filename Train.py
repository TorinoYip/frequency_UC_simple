# -*- coding: utf-8 -*-
import time
from Controller import Controller
from Env.Grid import Grid
from Agent.Brain import Brain
from Utils.DataUtils import DataManager
from Teacher import Teacher

if __name__ == '__main__':
    config = Controller('GRL Train')  # 初始化控制器为智能体训练模式
    env = Grid(config)  # 初始化环境
    agent = Brain(config, env)  # 初始化智能体
    data = DataManager(config)  # 初始化数据管理器
    teacher = Teacher(config)

    # 训练
    print('Train Start')
    print('===== ENV:%s ===== Algorithm:%s ===== Device:%s =====\n' % (config.ENV_NAME, config.ALGO_NAME, config.DEVICE))
    for episode in range(config.EPISODE_NUM):  # 训练TRAIN_NUM个回合
        t = time.time()  # 初始化本回合开始时间
        ep_reward = 0  # 记录每个回合的奖励
        state = env.Reset()  # 重置环境,即开始新的回合
        for step in range(env.case.seq_len):
            teacher.GetLabel(env)
            dis_action, con_action = agent.SampleAction(state)  # 获取Agent的动作
            next_state, reward, done = env.Step(dis_action, con_action)  # 环境更新

            agent.memory.Push((state, dis_action, con_action, teacher.Teach(reward, env), next_state, done))  # 将轨迹置入经验缓存池
            agent.Learn()  # 智能体参数更新

            state = next_state  # 进入下一状态
            ep_reward += reward

            print('\rStep[%d/%d]: Reward=%.4f, Teach Loss=%.4f' % (step + 1, env.case.seq_len, reward, teacher.loss), end='')

            if done:  # 判断是否结束当前回合
                break

        mean_reward = ep_reward / (step + 1)  # 计算回合内平均奖励
        agent.Save(episode, mean_reward)  # 保存智能体中神经网络
        agent.Step(mean_reward)  # 更新动态学习率，更新贪心系数
        data.SaveGRLTrainRecord(episode, mean_reward, env.result.sum(axis=0))  # 保存当前回合训练结果

        print('\rEP[%d/%d]: Reward=%.4f, Epsilon=%.4f, Alpha=%.4f, Time=%.4f'
              % (episode + 1, config.EPISODE_NUM, mean_reward, agent.discrete_agent.epsilon, agent.continue_agent.alpha.item(), time.time() - t))
    print('Best Performance' + str(agent.best))
