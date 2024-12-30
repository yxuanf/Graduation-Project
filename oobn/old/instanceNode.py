"""
@Description: OOBN model
@Author  : yxuanf
@Time    : 2024/3/10
@Site    : yxuanf@nudt.edu.cn
@File    : instanceNode.py
@note    : 将一个普通的贝叶斯网络转化为OOBN(单个对象)
"""

import pickle
import networkx as nx
import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from genie2pgm.simplemodel import SimpleDiscreteModel
from genie2pgm.exception import GenieException


class OrdinaryObject:
    """
    @Description: 对象的定义
    """

    def __init__(self, xmlPath: str) -> None:
        self._infer = None
        self.xmlPath = xmlPath
        # 输入节点
        self._inputNodes = None
        # 输出节点
        self._outPutNode = None
        # bayesianNetwork
        self._BN = self.getModel(xmlPath)

        # 对象的名字
        self._name = str(self._outPutNode)

        # 输出节点的cpd
        self._childCpds = self._BN.get_cpds(node=self._outPutNode)

        # 所有输入节点的cpd
        self._parentsCpds = [self._BN.get_cpds(node) for node in (self._inputNodes)]

        # 输入节点 + 输出节点
        self._interfaceNodes = self._inputNodes + [self._outPutNode]

        # 输出节点的先验概率
        self.priorOfOutPutNode = {self._name: self.childPriorProb()}

        # 输入节点的先验概率
        self.priorProbOfInputNodes = self.parentPriorProb()

        # all prior prob of interface
        self.priorProbOfInterface = self.priorOfOutPutNode.copy()
        self.priorProbOfInterface.update(self.parentPriorProb())

        # 接口节点的所有状态
        self.interfaceState = {
            interfaceNode: self._BN.get_cpds(interfaceNode).state_names[interfaceNode]
            for interfaceNode in self._interfaceNodes
        }

        # num of output
        self.childCard = self._childCpds.variable_card

        # num of input
        self.parentsCard = np.array(
            [self._BN.get_cpds(node=inputNode).variable_card for inputNode in self._inputNodes]
        )

        #  num of state combinations of input node
        self.parentsCardProduct = np.prod(self.parentsCard)
        self.__statement = list()
        self.__subStatement = list()
        self._simplifiedModel = None

    # 获取genie对应的贝叶斯网络模型
    def getModel(self, xmlPath: str) -> BayesianNetwork:
        try:
            module = SimpleDiscreteModel(xmlPath)
            module.add_cpd(module.model, module.getcpd())
            self._inputNodes, self._outPutNode = module.getInterface()
            return module.model
        except:
            raise GenieException("pgmpy2genie 模块解析出现错误！")

    def childPriorProb(self) -> dict:
        """
        @Description: 获取输出节点的先验概率
        """
        prob = dict()
        infer = VariableElimination(model=self._BN)
        ans = infer.query(variables=[self._outPutNode], joint=False)
        for q in ans.values():
            name = q.variables[0]
            for i in range(len(q.values)):
                prob[q.state_names[name][i]] = q.values[i]
        return prob

    def parentPriorProb(self) -> dict:
        """
        @Description:
        Returns:
        """
        prob = dict()
        length = len(self._parentsCpds)
        for i in range(length):
            nodeName = self._parentsCpds[i].variable
            # {'Fuel_L': ['good', 'bad']}
            stateName = self._parentsCpds[i].state_names[nodeName]
            prior = self._parentsCpds[i].values
            prob[nodeName] = {stateName[j]: prior[j] for j in range(len(stateName))}
        return prob

    @property
    def name(self) -> str:
        return self._name

    @property
    def parents(self) -> list:
        return self._inputNodes

    @property
    def child(self) -> str:
        return self._outPutNode

    @property
    def initialModel(self) -> BayesianNetwork:
        """_summary_

        Returns:
            BayesianNetwork: 返回初始的bayes网络
        """
        return self._BN

    @property
    def parentState(self) -> dict:
        parentStateDict = dict()
        for parentCpd in self._parentsCpds:
            parentStateDict.update(parentCpd.name_to_no)
        return parentStateDict

    @property
    def simplifiedModel(self) -> tuple[BayesianNetwork, TabularCPD]:
        """
        Returns: 返回一个二元组 （简化的对象， 简化后输出节点的条件概率表）
        """
        self._infer = VariableElimination(model=self._BN)
        self.values = np.zeros((self.parentsCardProduct, self.childCard))

        self.__getCpdValues(self._inputNodes, 0)

        # 矩阵转置
        self.values = np.transpose(self.values).tolist()

        # 简化后输出节点的条件概率表
        self._simplifiedCPD = TabularCPD(
            variable=self._outPutNode,
            variable_card=self.childCard,
            values=self.values,
            evidence=self._inputNodes,
            evidence_card=self.parentsCard,
            state_names=self.interfaceState,
        )
        self._simplifiedModel = self.__getSimplifiedModel()
        for cpd in self._parentsCpds:
            self._simplifiedModel.add_cpds(cpd)

        self._simplifiedModel.add_cpds(self._simplifiedCPD)
        self._simplifiedModel.name = self._name
        return self._simplifiedModel, self._simplifiedCPD

    @property
    def childCpds(self):
        return self._childCpds

    @property
    def parentsCpds(self):
        return self._parentsCpds

    @property
    def encapsulatedNode(self) -> dict[str, list[str]]:
        """
        Returns:   返回该oobn的封装节点
        """
        return {self._name: list(set(list(self._BN.nodes())) - set(self._interfaceNodes))}

    def __getCpdValues(self, nodes: list, index: int) -> None:
        """
        @Description:  计算简化后对象的cpd（回溯算法）
        Args:
            nodes: 对象的输入节点
            index: 传入输入节点的序号
        Returns: 简化后对象的cpd
        """
        if index == len(nodes):
            states = self.__subStatement.copy()
            evidence = dict()
            for i in range(len(states)):
                evidence[self._inputNodes[i]] = states[i]
            self.__statement.append(evidence)
            q = self._infer.query(variables=[self._outPutNode], evidence=evidence)
            self.values[len(self.__statement) - 1] = q.values
            return

        for state in self._BN.get_cpds(nodes[index]).state_names[nodes[index]]:
            self.__subStatement.append(state)
            self.__getCpdValues(nodes, index + 1)
            self.__subStatement.pop()

    #
    def __getSimplifiedModel(self) -> BayesianNetwork:
        """
        Returns: 返回简化后的对象结构（输入节点-> 输出节点）
        """
        edges = list()
        for inputNode in self._inputNodes:
            edges.append((inputNode, self._outPutNode))
        return BayesianNetwork(edges)


if __name__ == "__main__":
    xmlpath = "./XML/OOBN/燃烧状态.xdsl"
    oobn1 = OrdinaryObject(xmlpath)
    model, a = oobn1.simplifiedModel
    for node in list(model.nodes()):
        print(model.get_cpds(node))
    print(BayesianNetwork.check_model(model))
    b = oobn1.priorOfOutPutNode
    print(model.name)
