import time

import numpy as np

from Controller import Controller
from Env.Grid import Grid
from Agent.Brain import Brain
from Utils.DataUtils import DataManager
from Utils.ChartUtils import DrawHistory, DrawStepResults, DrawReward

if __name__ == '__main__':
    config = Controller('GRL Test')  # 初始化控制器为智能体测试模式
    env = Grid(config)  # 初始化环境
    agent = Brain(config, env)  # 初始化智能体
    data = DataManager(config)  # 初始化数据管理器

    # 训练
    print('Test Start')
    print('===== ENV:%s ===== Algorithm:%s ===== Device:%s =====\n' % (config.ENV_NAME, config.ALGO_NAME, config.DEVICE))
    t = time.time()
    reward_list = np.zeros(shape=(env.reward_num, env.case.seq_len))  # 记录每个回合的奖励
    state = env.Reset()  # 重置环境,即开始新的回合
    for step in range(env.case.seq_len):
        dis_action, con_action = agent.PredictAction(state)
        next_state, reward, done = env.Step(dis_action, con_action)

        state = next_state  # 更新状态

        print('Step[%d/%d]: Reward=%.4f' % (step + 1, env.case.seq_len, reward))
        if done:
            break

    data.SaveTestResult(env)  # 保存历史记录
