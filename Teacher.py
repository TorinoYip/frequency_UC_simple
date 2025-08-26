import numpy as np
import gurobipy as gp
import logging

import os
import sys
from contextlib import contextmanager

from Utils.MathUtils import ExpNorm


class Teacher:
    def __init__(self, config):
        self.teach = config.TEACH

        self.w_start, self.w_end = config.W_START, config.W_END
        self.w_decay = config.W_DECAY
        self.w = self.w_start

        self.punish = config.PUNISH_MAX

        self.count = 0

        self.loss = None
        self.success, self.label = None, None

    def GetLabel(self, env):
        model = gp.Model()  # 创建模型
        model.Params.LogToConsole = 0  # 关键：关掉控制台日志
        model.Params.OutputFlag = 0  # 保险起见，也关 OutputFlag（作用相同）

        # ======= 创建优化变量 ======= #
        # 决策变量
        Pt = model.addVars(env.thermal.unit_num, vtype=gp.GRB.CONTINUOUS, lb=0, ub=np.inf, name='Thermal Power')  # 火电机组出力
        St = model.addVars(env.thermal.unit_num, vtype=gp.GRB.BINARY, lb=0, ub=np.inf, name='Thermal State')  # 火电机组启停
        Ps = model.addVar(vtype=gp.GRB.CONTINUOUS, name='Slack Power')  # 平衡电机组出力
        Ph = model.addVars(env.hydro.num, vtype=gp.GRB.CONTINUOUS, lb=0, ub=np.inf, name='Hydro Power')  # 水电机组出力

        P_bus = model.addVars(env.case.bus_num, vtype=gp.GRB.CONTINUOUS, lb=-np.inf, ub=np.inf, name='Bus Inject Power')  # 节点注入功率
        P_branch = model.addVars(env.case.branch_num, vtype=gp.GRB.CONTINUOUS, lb=-np.inf, ub=np.inf, name='Branch Power Flow')  # 节点注入功率
        Slack_1 = model.addVars(env.case.branch_num, vtype=gp.GRB.CONTINUOUS, lb=0, ub=np.inf, name='Branch Power Flow')
        Slack_2 = model.addVars(env.case.branch_num, vtype=gp.GRB.CONTINUOUS, lb=0, ub=np.inf, name='Branch Power Flow')
        # ======= 创建目标函数 ======= #
        energy_cost = sum(Pt[i] * unit.energy_cost for i, unit in enumerate(env.thermal.unit_list)) + Ps * env.thermal.slack.energy_cost
        on_cost = sum(St[i] * unit.on_cost for i, unit in enumerate(env.thermal.unit_list))
        slack = sum(Slack_1[l] + Slack_2[l]  for l in range(env.case.branch_num)) * 1000000
        model.setObjective(energy_cost + on_cost + slack, gp.GRB.MINIMIZE)  # 总发电成本最小

        # ======= 创建约束条件 ======= #
        model.addConstr((sum(Pt[i] for i in range(env.thermal.unit_num)) + sum(Ph[i] for i in range(env.hydro.num)) + Ps == env.load.P_predict.sum() - env.res.P_predict.sum()), name='Power Balance')  # 功率平衡约束

        model.addConstrs((Pt[i] <= St[i] * unit.P_max for i, unit in enumerate(env.thermal.unit_list)), name='Thermal Power Max')  # 机组最大功率约束
        model.addConstrs((Pt[i] >= St[i] * unit.P_min for i, unit in enumerate(env.thermal.unit_list)), name='Thermal Power Min')  # 机组最小功率约束

        model.addConstr((Ps <= env.thermal.slack.P_max), name='Slack Power Max')  # 机组最小功率约束
        model.addConstr((Ps >= env.thermal.slack.P_min), name='Slack Power Min')  # 机组最小功率约束

        model.addConstrs((Ph[i] <= unit.P_max for i, unit in enumerate(env.hydro.unit_list)), name='Hydro Power Max')  # 机组最大功率约束
        model.addConstrs((Ph[i] >= unit.P_min for i, unit in enumerate(env.hydro.unit_list)), name='Hydro Power Min')  # 机组最小功率约束

        for i in range(env.case.bus_num):
            model.addConstr((P_bus[i] == sum(Pt[j] for j in env.thermal.GetBusUnit(i)) + sum(Ph[j] for j in env.hydro.GetBusUnit(i)) + (Ps if i == env.thermal.slack.bus else 0) + sum(env.res.P_predict[j] for j in env.res.GetBusRES(i)) + sum(env.load.P_predict[j] for j in env.load.GetBusLoad(i))), name='Bus Inject Power')

        for l in range(env.case.branch_num):
            model.addConstr(P_branch[l] == sum(P_bus[i] * env.case.ptfd[l, i] for i in range(env.case.bus_num)), name='Branch Max')  # 支路潮流最大功率约束
            model.addConstr((P_branch[l] <= env.case.P_branch_max[l] + Slack_1[l]), name='Branch Max')  # 支路潮流最大功率约束
            model.addConstr((P_branch[l] >= -env.case.P_branch_max[l] - Slack_2[l]), name='Branch Min')  # 支路潮流最大功率约束

        # ======= 求解 ======= #
        model.optimize()

        # ======= 导出求解变量 ======= #
        self.success = (model.status == gp.GRB.OPTIMAL)  # 求解成功
        if self.success:
            Pt_ = np.array([Pt[i].x for i in range(env.thermal.unit_num)])
            Ph_ = np.array([Ph[i].x for i in range(env.hydro.num)])
            Ps_ = Ps.x
            self.label = np.hstack([Pt_, Ph_, [Ps_]])
        else:  # 模型无解
            self.label = np.zeros(env.thermal.unit_num + env.hydro.num + 1)

    def Teach(self, env_reward, env):
        self.count += 1
        if self.teach and self.w > 0 and self.success:
            result = np.hstack([env.thermal.P_unit, env.hydro.P, [env.thermal.P_slack]])
            self.loss = np.clip(abs(result - self.label) / (self.label + 1e-30), 0, 1).mean()
            teach_reward = ExpNorm(self.loss) * self.punish
            self.w = self.w_end + (self.w_start - self.w_end) * np.exp(- self.count / self.w_decay).round(2)
            return self.w * teach_reward + (1 - self.w) * env_reward
        return env_reward

@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout