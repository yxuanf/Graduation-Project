"""
@Description: OOBN节点的状态变化以及查询
@Author  : yxuanf
@Time    : 2024/4/15
@Site    : yxuanf@nudt.edu.cn
@File    : oobnQuery.py 
"""
import os
import threading
import time

import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from oobn.build.encapsulation import bnClass
from oobn.train.buildOOBN import ooBN
from oobn.train.instantiation import instantiation
from pgmpy.inference import VariableElimination, BeliefPropagation
from oobn.utils.queryUtils import QueryUtils


class ooBNQuery:
    """
     @ description: 推理框架
    """

    def __init__(self, model: ooBN) -> None:
        self.model = model
        self.simplifiedModel = model.simplifiedOOBN
        self.instanceNode = self.model.objectNodes
        self.originalModel = model.originalBN
        self.globalInference = VariableElimination(model=self.simplifiedModel)
        self.vEInference = VariableElimination(model=self.originalModel)
        self.cTPInference = BeliefPropagation(model=self.originalModel)

    def globalPropagation(self, observed: dict, counts=500):
        """
        @ description: 全局传播算法
        Args:
            queryNodes:
            observed:
            counts:
        """
        observedNodes = set(observed.keys())
        queryNodes = set(self.simplifiedModel.nodes) - observedNodes
        posterior = dict()
        ans = dict()
        resultTime = list()
        for count in tqdm(range(1, counts + 1)):
            t1 = time.perf_counter() * 1000
            for node in queryNodes:
                q = self.globalInference.query(variables=[node],
                                               evidence=observed,
                                               joint=False)
                if count == 1:
                    ans.update(q)
            t2 = time.perf_counter() * 1000
            resultTime.append((t2 - t1) / len(queryNodes) * 2)
        return ans, resultTime

    @staticmethod
    def localInference(queryNodes: list, instanceNode: instantiation, posterior: dict, counts=100):
        """
        @ description: 局部推理
        Args:
            counts: 运行次数
            queryNodes: 所查询的节点
            instanceNode: 所查询的对象
            posterior:
        """
        for node in queryNodes:
            if node not in instanceNode.classBN.BN.nodes():
                raise ValueError(f"所查询的节点不在对象{instanceNode.classBN.name}中")
        virtualEvidence = dict()
        for node in list(posterior.keys()):
            virtualEvidence.update({f"virtual_node_of_{node}": "state0"})
        newModel = QueryUtils.addVirtualNode(instanceNode, posterior)
        ans = list()
        resultTime = list()
        for count in tqdm(range(1, counts + 1)):
            t1 = time.perf_counter() * 1000
            inferenceMethod = VariableElimination(newModel)
            q = inferenceMethod.query(
                variables=queryNodes,
                evidence=virtualEvidence,
                joint=False
            )
            t2 = time.perf_counter() * 1000
            resultTime.append(t2 - t1)
            if not ans:
                ans = q
        return ans, resultTime

    def queryVE(self, queryNodes: list, observed: dict, counts=500):
        """
        @ description: VE推理算法
        Args:
            queryNodes: 待查询的节点
            observed: 观测节点
            counts: 次数

        Returns: 查询结果 与 消耗时间

        """
        for node in queryNodes:
            if node not in self.originalModel.nodes():
                raise ValueError("所查询的节点不在原始网络中")
        ans = list()
        resultTime = list()
        for count in tqdm(range(1, counts + 1)):
            t1 = time.perf_counter() * 1000
            for node in queryNodes:
                q = self.vEInference.query(variables=[node],
                                           evidence=observed,
                                           joint=False)
                if count == 1:
                    ans.append(q)
            t2 = time.perf_counter() * 1000
            resultTime.append(t2 - t1)
        return ans, resultTime

    def queryCTP(self, queryNodes: list, observed: dict, counts=500):
        """
        @ description: 团树传播算法
        Args:
            queryNodes: 待查询的节点
            observed: 观测节点
            counts: 次数

        Returns: 查询结果 与 消耗时间

        """
        for node in queryNodes:
            if node not in self.originalModel.nodes():
                raise ValueError("所查询的节点不在原始网络中")
        ans = list()
        resultTime = list()
        for count in tqdm(range(1, counts + 1)):
            t1 = time.perf_counter() * 1000
            for node in queryNodes:
                q = self.cTPInference.query(variables=[node],
                                            evidence=observed,
                                            joint=False)
                if count == 1:
                    ans.append(q)
            t2 = time.perf_counter() * 1000
            resultTime.append((t2 - t1) / len(queryNodes))
        return ans, resultTime


if __name__ == '__main__':
    class1 = bnClass("./XML/originalOOBN/停车开关组件.xdsl")
    class2 = bnClass("./XML/originalOOBN/启动供油组件.xdsl")
    class3 = bnClass("./XML/originalOOBN/故障燃油排放阀.xdsl")
    class4 = bnClass("./XML/originalOOBN/温度调节器.xdsl")
    class5 = bnClass("./XML/originalOOBN/燃油系统class.xdsl")
    class6 = bnClass("./XML/originalOOBN/燃油调节器.xdsl")
    class7 = bnClass("./XML/originalOOBN/自动燃油分配器.xdsl")
    class8 = bnClass("./XML/originalOOBN/节流组件.xdsl")
    class9 = bnClass("./XML/originalOOBN/齿轮泵.xdsl")
    class10 = bnClass("./XML/originalOOBN/齿轮泵1.1.xdsl")
    class11 = bnClass("./XML/originalOOBN/齿轮泵1.2.xdsl")
    instantiation11 = instantiation(class11)
    instanceNodeList = [instantiation(class1), instantiation(class2), instantiation(class3), instantiation(class4),
                        instantiation(class5), instantiation(class6), instantiation(class7), instantiation(class8),
                        instantiation(class9), instantiation(class10), instantiation11]

    ooBNTest = ooBN(instanceNodeList)
    queryTest = ooBNQuery(model=ooBNTest)
    nodes = ["Low_fuel_injector_pressure", "Parking_switch",
             "Injector_pressure_fluctuation", "Start_fueling",
             "No_working_signal", "Thermostat_check_abnormality", "Faulty_fuel_drain_valve",
             "Temperature_regulator",
             "Fuel_System",
             "Pressure_rise_limiter", "Fuel_regulator",
             "Automatic_fuel_dispenser",
             "Throttle_component",
             "Gear_pump"
             ]

    # 观测数据
    observedData = {
        "Gear_pump_wear_1": "劣化",
        "Fine_Oil_Filter_1": "正常",
        "Gear_pump_wear_2": "劣化",
        "Fine_Oil_Filter_2": "脏污",
        "Damaged_sealing_ring": "未发生",
        "Insufficient_traffic": "未发生",
        "Begrime": "未发生",
        "Start_fuel_valve": "正常",
        "Check_valve": "正常",
        "Thermostat_circuit_disconnected": "未发生",
        "On_signal_off": "未发生",
        "Distribution_valve": "卡滞在关闭位置",
        "Autostarter": "正常",
        "Short_circuit_or_open_circuit": "未发生",
        "Throttle": "积垢"
    }
    result, t = queryTest.queryVE(nodes, observedData)
    # for r in result.values():
    #     print(r)
    print("VE算法耗费时间")
    # for r in result:
    #     for v in r.values():
    #         print(v)
    print(t)
    print("----------------------------------------")

    # result, t = queryTest.queryBP(nodes, observedData)
    # # for r in result.values():
    # #     print(r)
    # print("BF算法耗费时间")
    # for r in result:
    #     for v in r.values():
    #         print(v)
    # print(t)
    # print("----------------------------------------")

    result, t = queryTest.globalPropagation(observedData)
    for r in result:
        for v in r.values():
            print(v)
    print(t)
    print("----------------------------------------")

    # result, t = ooBNQuery.localInference(["Iron_filings_1", "Shaft_seal_damaged_1"], ooBNTest.objectNodes[-2],
    #                                      {"Gear_pump_wear_1": np.array([0, 1, 0]),
    #                                       "Fine_Oil_Filter_1": np.array([0, 1])})
    # for r in result.values():
    #     print(r)
    # print(t)
    # print("----------------------------------------")
    #
    # result, t = ooBNQuery.localInference(["Iron_filings_2", "Pressure_fluctuations_2"], ooBNTest.objectNodes[-1],
    #                                      {"Fine_Oil_Filter_2": np.array([1, 0]),
    #                                       "Gear_pump_wear_2": np.array([0, 1, 0])})
    # for r in result.values():
    #     print(r)
    # print(t)
    # print("----------------------------------------")
    #
    # result, t = ooBNQuery.localInference(["Pressure_rise_limiter"], ooBNTest.objectNodes[5],
    #                                      {"Throttle_component": np.array([0.9, 0.1])})
    # for r in result.values():
    #     print(r)
    # print(t)
    # print("----------------------------------------")
