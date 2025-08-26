import torch
import numpy as np
import os
import time

from Utils.FileUtils import CreateDir


class Controller:
    def __init__(self, mode='GRL Train'):
        # 设定当前模式
        # 'Frequency Data'：频率响应数据集
        # 'Frequency Net Train'：频率响应神经网络训练
        # 'Frequency Net Test'：频率响应神经网络测试
        # 'GRL Train'：GRL训练
        # 'GRL Test'：GRL测试
        self.MODE = mode

        # ========================= 基本信息 ========================= #
        self.ENV_NAME = 'Production Simulation'  # 环境名称
        self.ALGO_NAME = 'HGRL'  # 算法名称
        self.CASE_NAME = 'IEEE39' #  'IEEE39'  #算例名称
        self.DAY_NUM = 1  # 求解时间长度（单位：天）

        self.LOAD_AGENT_MODE = 'Latest'  # 'Best'  # 读取最新保存结果还是最好的保存结果
        self.AGENT_VERSION = 18  # 当前Agent训练版本
        self.USE_NEW_AGENT = True  # False  # 训练GRL时是否采用新智能体，否则继续在上次训练的基础上进行训练

        self.LOAD_FREQ_NET_MODE = 'Best'  # 'Latest'  # 读取最新保存结果还是最好的保存结果
        self.FREQ_NET_VERSION = 1  # 当前频率响应神经网络训练版本
        self.USE_NEW_FREQ_NET = True  # False  # 训练Expert时是否采用新智能体，否则继续在上次训练的基础上进行训练

        self.TEACH = True
        if self.TEACH:
            self.ALGO_NAME += '-Teach'

        # ========================= 存储路径 ========================= #
        abs_path = os.path.abspath(os.curdir)
        if self.MODE in ['Frequency Data', 'Frequency Net Train', 'Frequency Net Test']:
            abs_path = abs_path[:-len('FrequencyModel\\')]

        # 算例存储路径
        self.CASE_PATH = abs_path + '\\Case\\' + self.CASE_NAME + '\\'  # 算例存储路径
        self.ANDES_PATH = self.CASE_PATH + self.CASE_NAME + '-ANDES.xlsx'

        # 结果输出路径
        self.OUTPUT_PATH = CreateDir(abs_path + '\\Results\\' + self.CASE_NAME + '\\')  # 结果输出路径
        self.FREQ_DATASET_PATH = CreateDir(self.OUTPUT_PATH + 'FrequencyDataset\\', self.MODE == 'Frequency Data')
        self.SAMPLE_PATH = CreateDir(self.FREQ_DATASET_PATH + 'Samples\\', self.MODE == 'Frequency Data')
        self.FREQ_NET_PATH = CreateDir(self.OUTPUT_PATH + 'FrequencyNet-V' + str(self.FREQ_NET_VERSION) + '\\', self.MODE == 'Frequency Net Train')
        self.GRL_PATH = CreateDir(self.OUTPUT_PATH + self.ALGO_NAME + '-V' + str(self.AGENT_VERSION) + '\\', self.MODE == 'GRL Train')

        # 潮流计算结果输出路径
        # os.devnull：直接丢弃pypower潮流输出结果
        # None：pypower潮流输出结果直接输出至控制台（仅调试时推荐）
        # 'PowerFlow.txt'：pypower潮流输出结果直接输出PowerFlow.txt（仅批量调试时推荐）
        self.POWER_FLOW_OUTPUT_PATH = os.devnull  # None  # 'PowerFlow.txt'  #

        # ========================= 训练参数设置 ========================= #
        if self.MODE in ['Frequency Data']:
            self.SAMPLE_NUM = 300  # 样本数量
            self.SPLIT_RATE = 0.9  # 训练集测试集划分

            self.SIMULATION_TIME = 10  # 暂态仿真时间
            self.DISTURBANCE_TIME = 0.001  # 扰动发生时间

            self.MAX_DISTURBANCE_RATE, self.MIN_DISTURBANCE_RATE = 1.1, 0.9  # 数据集样本生成噪声（扰动）
            self.SAMPLE_NOISE_MAX, self.SAMPLE_NOISE_MIN = 1.1, 0.9  # 数据集样本生成噪声（新能源出力、负荷）
            self.SAVE_FORMAT = 'csv' # 新作改动：csv文件

        elif self.MODE in ['Frequency Net Train', 'Frequency Net Test']:
            self.EPOCH_NUM = 3000  # 频率响应神经网络训练轮次
            self.BATCH_SIZE = 512  # 批尺寸
            self.RANDOM_BATCH = True  # 是否从数据集中随机抽样获取batch

            self.SAVE_CYCLE = 50  # 保存周期

            # 选择损失函数
            # 'MAELoss':平均绝对值损失，用于回归任务
            # 'MSELoss':平均均方差损失，用于回归任务
            # 'SmoothL1Loss':MAE与MSE的结合，用于回归任务
            # 'MAPELoss':平均绝对百分比误差，用于回归任务，可用于计算百分比精度
            # 'BCELoss':二元交叉熵损失，用于二分类任务
            # 'CrossEntropyLoss'# 交叉熵损失，用于多分类任务
            self.LOSS = 'SmoothL1Loss'

            self.OPTIMIZER = 'Adam'  # 'SGD'  #
            self.LR = 1e-2  # 初始学习率
            self.DYNAMIC_LR = True  # 是否使用动态学习率
            self.MIN_LR = 1e-4  # 最小学习率
            self.PATIENCE = 100  # 动态学习率调整门槛（超过PATIENCE轮训练奖励没有提高/损失没有下降则减小学习率）
            self.DAMP_FACTOR = 0.1  # 动态学习率衰减比例
            self.COLD_DOWN = 100  # 动态学习率最小调整间隔

            self.GRAD_CLIP = True
            self.MAX_NORM = 5  # 梯度裁剪最大范数
            self.NORM_TYPE = 2  # 梯度裁剪范数类型

            self.DROPOUT = 0  # 随机丢失率（防止过拟合）
            self.L1_NORM = 0  # 1e-4
            self.L2_NORM = 0

            # 频率响应神经网络超参数
            self.DISTURBANCE_HIDDEN = 16  # 扰动嵌入维度
            self.FREQ_NET_GE_LAYER = 2  # Expert中图嵌入器层数
            self.FREQ_NET_MLP_LAYER = 4  # Expert中MLP层数
            self.FREQ_NET_HIDDEN = 32  # Expert中隐藏层维度（简便起见EXPERT中所有隐藏层使用同一维度）

            self.GNN_NORM = 'GraphNorm'  # 图神经网络后标准化方法
            self.LINEAR_NORM = 'LayerNorm'  # MLP后标准化方法

            self.ACT = 'RReLU'  # 选择激活函数
            self.RES = False  # 是否使用残差结构

        elif self.MODE in ['GRL Train', 'GRL Test']:
            self.EPISODE_NUM = 30000  # 训练回合数

            self.MEMORY_CAP = 2048  # 经验缓冲池容量
            self.BATCH_SIZE = 128  # 批尺寸

            self.SAVE_CYCLE = 50  # 保存周期

            self.DELTA_F_MAX = 0.2  # 频率最大偏差控制标准

            self.NOISE_MAX, self.NOISE_MIN = 1.1, 0.9  # SAC生成均方差的log的上下限

            self.GAMMA = 0.9  # 折扣因子
            self.TAU = 1e-2  # 缓步更新系数

            self.ALPHA = 0.2  # SAC初始温度系数
            self.ADAPTIVE_ALPHA = True  # 采用动态温度系数
            self.ALPHA_LR = 1e-3  # SAC动态温度系数学习率
            self.ALPHA_MAX, self.ALPHA_MIN = np.inf, -np.inf  # 动态温度系数限值

            self.EPSILON_START, self.EPSILON_END = 0.5, 0.05  # DDQN贪心系数最大最小值
            self.EPSILON_DECAY = 1000  # DDQN贪心系数衰减因子

            # 选择损失函数
            # 'MAELoss':平均绝对值损失，用于回归任务
            # 'MSELoss':平均均方差损失，用于回归任务
            # 'SmoothL1Loss':MAE与MSE的结合，用于回归任务
            # 'MAPELoss':平均绝对百分比误差，用于回归任务，可用于计算百分比精度
            # 'BCELoss':二元交叉熵损失，用于二分类任务
            # 'CrossEntropyLoss'# 交叉熵损失，用于多分类任务
            self.LOSS = 'SmoothL1Loss'

            # GRL训练参数
            self.OPTIMIZER = 'Adam'  # 'SGD'
            self.LR = 1e-2  # 初始学习率
            self.DYNAMIC_LR = True  # 是否使用动态学习率
            self.MIN_LR = 1e-5  # 最小学习率
            self.PATIENCE = 500  # 动态学习率调整门槛（超过PATIENCE轮训练奖励没有提高/损失没有下降则减小学习率）
            self.DAMP_FACTOR = 0.1  # 动态学习率衰减比例
            self.COLD_DOWN = 500  # 动态学习率最小调整间隔

            self.GRAD_CLIP = True
            self.MAX_NORM = 5  # 梯度裁剪最大范数
            self.NORM_TYPE = 2  # 梯度裁剪范数类型

            self.DROPOUT = 0.01 #  随机丢失率（防止过拟合）
            self.L1_NORM = 0
            self.L2_NORM = 0

            self.LOG_STD_MAX, self.LOG_STD_MIN = 2, -20  # SAC生成均方差的log的上下限

            # 示教学习系数
            self.W_START, self.W_END = 0.9, 0.1
            self.W_DECAY = 3000  # 衰减因子越大，衰减越慢

            self.DATE_EMBED = False  # 是否进行日期嵌入
            self.TIME_HIDDEN = 64  # （日期）时间嵌入维度
            self.DISTURBANCE_HIDDEN = 64  # 扰动嵌入维度
            self.HEAD = 4  # Q网络中图嵌入器注意力头数

            # DDQN神经网络超参数
            self.QNET_GE_LAYER = 3  # Q网络中图嵌入器层数
            self.QNET_MLP_LAYER = 6  # Q网络中MLP层数
            self.QNET_HIDDEN = 256  # Q网络中隐藏层维度（简便起见Actor中所有隐藏层使用同一维度）

            # SAC神经网络超参数
            self.ACTOR_GE_LAYER = 3  # Actor中图嵌入器层数
            self.ACTOR_MLP_LAYER = 6  # Actor中MLP层数
            self.ACTOR_HIDDEN = 256 # Actor中隐藏层维度（简便起见Actor中所有隐藏层使用同一维度）

            self.CRITIC_GE_LAYER = 3  # Critic中图嵌入器层数
            self.CRITIC_MLP_LAYER = 6  # Critic中MLP层数
            self.CRITIC_HIDDEN = 256  # Critic中隐藏层维度（简便起见Actor中所有隐藏层使用同一维度）

            self.GNN_NORM = 'GraphNorm'  # 图神经网络后标准化方法
            self.LINEAR_NORM = 'LayerNorm'  # MLP后标准化方法

            self.ACT = 'ReLU'  # 激活函数
            self.RES = False  # 是否使用残差结构

            # 奖励系数
            self.PUNISH_MAX = -30000
            self.PUNISH = {
                'Power Flow Diverge': -self.PUNISH_MAX,  # 潮流不收敛（负）奖励
                'Power Imbalance': 15000,  # 系统功率缺额（负）奖励
                'Load Cut': 0,  # 切负荷（负）奖励
                'RES Cut': 700,  # 切新能源（负）奖励
                'Cost': 200,  # 系统成本（负）奖励
                'Thermal On': 100,  # 火电开机（负）奖励
                'Branch Overlimit': 10000,  # 支路潮流越限（负）奖励
                'PFR Shortage': 3000,  # 一次调频容量不足（负）奖励
                'Frequency Overlimit': 1000  # 支路潮流越限（负）奖励
            }

        # ========================= 训练设备 ========================= #
        use_gpu = True  # False  # 是否使用GPU加速训练
        if use_gpu and torch.cuda.is_available():  # 检测是否能使用GPU
            self.DEVICE = torch.device('cuda')
            self.USE_GPU = True
        else:
            self.DEVICE = torch.device('cpu')
            self.USE_GPU = False

        # ========================= 随机数种子 ========================= #
        self.RANDOM_SEED = int(time.time())  # 设置随机数种子为当前时间戳
        np.random.seed(self.RANDOM_SEED)
        torch.manual_seed(self.RANDOM_SEED)
        if self.USE_GPU:
            torch.cuda.manual_seed(self.RANDOM_SEED)

        self.PrintSetting()

    def PrintSetting(self):
        if self.MODE == 'Frequency Data':  # 打印专家经验数据集设置
            with open(self.FREQ_DATASET_PATH + 'Setting.txt', 'w') as txt:
                txt.write('=' * 30 + 'Frequency Dataset Setting' + '=' * 30 + '\n')
                txt.write('Sample Num: ' + str(self.SAMPLE_NUM) + '\n')
                txt.write('Split Rate: ' + str(self.SPLIT_RATE) + '\n')
                txt.write('-' * 30 + '\n')

                txt.write('Simulation Time: ' + str(self.SIMULATION_TIME) + '\n')
                txt.write('Disturbance Time' + str(self.DISTURBANCE_TIME) + '\n')
                txt.write('Min Disturbance Rate' + str(self.MIN_DISTURBANCE_RATE) + '\n')
                txt.write('Max Disturbance Rate' + str(self.MAX_DISTURBANCE_RATE) + '\n')
                txt.write('Sample Noise Max: ' + str(self.SAMPLE_NOISE_MAX) + '\n')
                txt.write('Sample Noise Min: ' + str(self.SAMPLE_NOISE_MIN) + '\n')
                txt.write('-' * 30 + '\n')

        elif self.MODE == 'Frequency Net Train' and self.USE_NEW_FREQ_NET:
            with open(self.FREQ_NET_PATH + 'Setting.txt', 'w') as txt:
                txt.write('=' * 30 + 'Frequency Network Training Setting' + '=' * 30 + '\n')

                txt.write('Epoch Num : ' + str(self.EPOCH_NUM) + '\n')
                txt.write('Batch Size : ' + str(self.BATCH_SIZE) + '\n')
                txt.write('Random Batch : ' + str(self.RANDOM_BATCH) + '\n')
                txt.write('-' * 30 + '\n')

                txt.write('Loss Function: ' + str(self.LOSS) + '\n')
                txt.write('Optimizer: ' + str(self.OPTIMIZER) + '\n')
                txt.write('Learning Rate: ' + str(self.LR) + '\n')
                txt.write('Dynamic Learning Rate: ' + str(self.DYNAMIC_LR) + '\n')
                if self.DYNAMIC_LR:
                    txt.write('Min Learning Rate: ' + str(self.MIN_LR) + '\n')
                    txt.write('Patience: ' + str(self.PATIENCE) + '\n')
                    txt.write('Damp Factor: ' + str(self.DAMP_FACTOR) + '\n')
                    txt.write('Cold Down: ' + str(self.COLD_DOWN) + '\n')
                txt.write('-' * 30 + '\n')

                txt.write('Grad Clip: ' + str(self.GRAD_CLIP) + '\n')
                if self.GRAD_CLIP:
                    txt.write('Max Norm: ' + str(self.MAX_NORM) + '\n')
                    txt.write('Norm Type: ' + str(self.NORM_TYPE) + '\n')
                txt.write('-' * 30 + '\n')

                txt.write('Dropout: ' + str(self.DROPOUT) + '\n')
                txt.write('L1 Norm: ' + str(self.L1_NORM) + '\n')
                txt.write('L2 Norm' + str(self.L2_NORM) + '\n')
                txt.write('-' * 30 + '\n')

                txt.write('Linear Layer Norm: ' + str(self.LINEAR_NORM) + '\n')
                txt.write('GNN Layer Norm: ' + str(self.GNN_NORM) + '\n')
                txt.write('Use Residual Structure: ' + str(self.RES) + '\n')
                txt.write('-' * 30 + '\n')

                txt.write('Random Seed: ' + str(self.RANDOM_SEED) + '\n')
                txt.write('Device: ' + str(self.DEVICE) + '\n')
                txt.write('-' * 30 + '\n')

        elif self.MODE == 'GRL Train' and self.USE_NEW_AGENT:
            with open(self.GRL_PATH + 'Setting.txt', 'w') as txt:
                txt.write('=' * 30 + 'Agent Training Setting' + '=' * 30 + '\n')

                txt.write('Episode Num: ' + str(self.EPISODE_NUM) + '\n')
                txt.write('Memory Cap: ' + str(self.MEMORY_CAP) + '\n')
                txt.write('Batch Size: ' + str(self.BATCH_SIZE) + '\n')
                txt.write('-' * 30 + '\n')

                txt.write('Noise Max: ' + str(self.NOISE_MAX) + '\n')
                txt.write('Noise Min: ' + str(self.NOISE_MIN) + '\n')
                txt.write('-' * 30 + '\n')

                txt.write('Loss Function: ' + str(self.LOSS) + '\n')
                txt.write('Optimizer: ' + str(self.OPTIMIZER) + '\n')
                txt.write('Learning Rate: ' + str(self.LR) + '\n')
                txt.write('Dynamic Learning Rate: ' + str(self.DYNAMIC_LR) + '\n')
                if self.DYNAMIC_LR:
                    txt.write('Min Learning Rate: ' + str(self.MIN_LR) + '\n')
                    txt.write('Patience: ' + str(self.PATIENCE) + '\n')
                    txt.write('Damp Factor: ' + str(self.DAMP_FACTOR) + '\n')
                    txt.write('Cold Down: ' + str(self.COLD_DOWN) + '\n')
                txt.write('-' * 30 + '\n')

                txt.write('Discount Factor: ' + str(self.GAMMA) + '\n')
                txt.write('TAU: ' + str(self.TAU) + '\n')
                txt.write('-' * 30 + '\n')

                txt.write('Initial Greedy Coefficient: ' + str(self.EPSILON_START) + '\n')
                txt.write('Min Greedy Coefficient: ' + str(self.EPSILON_END) + '\n')
                txt.write('Greedy Coefficient Decay Factor: ' + str(self.EPSILON_END) + '\n')
                txt.write('-' * 30 + '\n')

                txt.write('Entropy Coefficient: ' + str(self.ALPHA) + '\n')
                txt.write('Entropy Coefficient Max: ' + str(self.ALPHA_MAX) + '\n')
                txt.write('Entropy Coefficient Min: ' + str(self.ALPHA_MIN) + '\n')
                if self.ADAPTIVE_ALPHA:
                    txt.write('Entropy Coefficient Learning Rate: ' + str(self.ALPHA_LR) + '\n')
                txt.write('-' * 30 + '\n')

                txt.write('Action Log Std Max: ' + str(self.LOG_STD_MAX) + '\n')
                txt.write('Action Log Std Min: ' + str(self.LOG_STD_MIN) + '\n')
                txt.write('-' * 30 + '\n')

                txt.write('Linear Layer Norm: ' + str(self.LINEAR_NORM) + '\n')
                txt.write('GNN Layer Norm: ' + str(self.GNN_NORM) + '\n')
                txt.write('Use Residual Structure: ' + str(self.RES) + '\n')
                txt.write('-' * 30 + '\n')

                txt.write('Power Flow Diverge Reward: ' + str(self.PUNISH['Power Flow Diverge']) + '\n')
                txt.write('Power Imbalance Reward: ' + str(self.PUNISH['Power Imbalance']) + '\n')
                txt.write('Load Cut Reward: ' + str(self.PUNISH['Load Cut']) + '\n')
                txt.write('RES Cut Reward: ' + str(self.PUNISH['RES Cut']) + '\n')
                txt.write('Cost Reward: ' + str(self.PUNISH['Cost']) + '\n')
                txt.write('Thermal On Reward: ' + str(self.PUNISH['Thermal On']) + '\n')
                txt.write('Branch Power Flow Overlimit Reward: ' + str(self.PUNISH['Branch Overlimit']) + '\n')
                txt.write('Frequency Min Overlimit Reward: ' + str(self.PUNISH['Frequency Overlimit']) + '\n')
                txt.write('-' * 30 + '\n')

                txt.write('Random Seed: ' + str(self.RANDOM_SEED) + '\n')
                txt.write('Device: ' + str(self.DEVICE) + '\n')
                txt.write('-' * 30 + '\n')
