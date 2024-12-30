import numpy as np
import pandas as pd

import oobn.build.inheritance
from oobn.build.encapsulation import bnClass
from oobn.build.inheritance import inheritBNClass
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.factors.discrete import TabularCPD, DiscreteFactor
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork


class instantiation(object):
    """
    @ description: 类的实例化（生成对象）
    """

    def __init__(self, classBN: bnClass):
        self.values = None
        self.classBN = classBN
        self._statement = None
        self._subStatement = None
        self.simplifiedModel = None

        # 对象输入节点的条件概率表
        self.inputNodesCPDs = None if not self.checkModel() else [self.classBN.BN.get_cpds(inputNode) for inputNode in
                                                                  self.classBN.inputNodes]
        self.inputNodes = self.classBN.inputNodes
        self.outputNode = self.classBN.outputNode
        self.encapsulationNodes = set(list(self.classBN.BN.nodes)) - set(self.classBN.inputNodes) - set(self.classBN.outputNode)

    def checkModel(self):
        """
        @ description: Check the model for various errors.
        Returns: True or False
        """
        return self.classBN.BN.check_model()

    @property
    def instanceNode(self):
        """
        @ description:
        Returns:
        """
        if self.checkModel():
            return self.classBN.BN
        raise ValueError("不符合贝叶斯网络的定义")

    def trainMLE(self, csvPath: str):
        """
        @ description: 类的实例化
        Args:
            csvPath: 数据路径

        Returns:

        """
        # 先去除cpd
        self.classBN.BN.remove_cpds(*self.classBN.BN.get_cpds())
        data = pd.read_csv(csvPath, delimiter=' ')
        self.classBN.BN.fit(data, estimator=MaximumLikelihoodEstimator)
        for cpd in self.classBN.BN.get_cpds():
            print(cpd)
        if self.classBN.BN.check_model():
            self.inputNodesCPDs = [self.classBN.BN.get_cpds(inputNode) for inputNode in self.classBN.inputNodes]
            print("训练完成")
        else:
            ValueError("训练失败，请稍后重试")


    def _getCpdValues(self, nodes: list, index: int) -> None:
        """
        @Description:  计算简化后对象的cpd（回溯算法）
        Args:
            nodes: 对象的输入节点
            index: 传入输入节点的序号
        Returns: 简化后对象的cpd
        """
        if index == len(nodes):
            states = self._subStatement.copy()
            evidence = dict()
            for i in range(len(states)):

                evidence[self.classBN.inputNodes[i]] = states[i]
            self._statement.append(evidence)
            inferMethod = VariableElimination(model=self.classBN.BN)
            q = inferMethod.query(variables=[self.classBN.outputNode], evidence=evidence)
            self.values[len(self._statement) - 1] = q.values
            return

        for state in self.classBN.BN.get_cpds(nodes[index]).state_names[nodes[index]]:
            self._subStatement.append(state)
            self._getCpdValues(nodes, index + 1)
            self._subStatement.pop()

    def getSimplifiedModel(self):
        """
        Returns: 返回一个 tuple (简化后的对象, 简化后对象输出节点的cpd）
        """
        self._statement = list()
        self._subStatement = list()
        self.values = np.zeros((self.classBN.inputNodesCardProduct, self.classBN.outputNodeCard))
        self._getCpdValues(self.inputNodes, 0)

        # 矩阵转置
        self.values = np.transpose(self.values).tolist()

        # 简化后输出节点的条件概率表
        simplifiedCPD = TabularCPD(
            variable=self.classBN.outputNode,
            variable_card=self.classBN.outputNodeCard,
            values=self.values,
            evidence=self.classBN.inputNodes,
            evidence_card=self.classBN.inputNodesCard,
            state_names=self.classBN.interfaceState,
        )

        self.simplifiedModel = self._getSimplifiedModel()

        if not self.inputNodesCPDs:
            # 更新对象输入节点的先验概率
            self.setInputNodesCPDS()

        # 设置输入节点的概率表
        for cpd in self.inputNodesCPDs:
            self.simplifiedModel.add_cpds(cpd)

        # 设置输出节点的概率表
        self.simplifiedModel.add_cpds(simplifiedCPD)
        self.simplifiedModel.name = "simplifiedObject_" + self.classBN.name
        return self.simplifiedModel, simplifiedCPD

    def _getSimplifiedModel(self) -> BayesianNetwork:
        """
        Returns: 返回简化后的对象结构（输入节点 -> 输出节点）
        """
        edges = list()
        for inputNode in self.inputNodes:
            edges.append((inputNode, self.outputNode))
        return BayesianNetwork(edges)

    def setInputNodesCPDS(self, cpds: list[DiscreteFactor] = None):
        """
        @description: 设置或更新输入节点的先验概率
        Args:
        """
        model = self.classBN.BN
        if not model.check_model():
            raise ValueError("请先完成对象的实例化")
        if cpds is not None:
            for cpd in cpds:
                variable = cpd.variables[0]
                variable_card = cpd.cardinality[0]
                values = cpd.values.reshape(-1, 1)
                state_names = cpd.state_names
                if variable not in self.classBN.inputNodes:
                    continue
                self.classBN.BN.remove_cpds(variable)
                new_cpd = TabularCPD(
                    variable=variable,
                    variable_card=variable_card,
                    values=values,
                    state_names=state_names,
                )
                self.classBN.BN.add_cpds(new_cpd)

        self.inputNodesCPDs = [self.classBN.BN.get_cpds(inputNode) for inputNode in self.classBN.inputNodes]


if __name__ == '__main__':
    path1 = "./XML/originalOOBN/温度调节器.xdsl"
    dataPath = "./data/test/燃烧状态.csv"
    basicClass = bnClass(path1)
    instanceNode = instantiation(basicClass)
    instanceNode.trainMLE(dataPath)
    infer = VariableElimination(model=instanceNode.classBN.BN)
    cpdList = list()
    for node in instanceNode.classBN.inputNodes:
        cpdList.append(infer.query(variables=[instanceNode.classBN.inputNodes[0]]))
    instanceNode.setInputNodesCPDS(cpdList)
    for node in instanceNode.classBN.inputNodes:
        cpd = instanceNode.classBN.BN.get_cpds(node)
        print(cpd)
    var = instanceNode.classBN.BN.check_model()
    print(instanceNode)
