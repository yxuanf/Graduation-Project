"""
@Description: Noisymax model
@Author  : yxuanf
@Time    : 2024/3/27
@Site    : yxuanf@nudt.edu.cn
@File    : buildOOBN.py
@note    : 将多个object组装为OOBN
"""

import numpy as np
from oobn.old.instanceNode import OrdinaryObject
from pgmpy.models import BayesianNetwork


class BuildOOBN:
    """
    @Description: 根据提供的对象构建面向对象贝叶斯网络
    """

    def __init__(self, oobnList: list[OrdinaryObject]) -> None:
        # 对象的列表
        self._oobnNodes = oobnList
        self._BN = self.buildStructure()
        self.addCPD()
        # 返回oobn中的各个对象的输出节点的先验概率
        self.priorprobOfoobnNode = self.getPriorProb()

    def getPriorProb(self) -> dict[str, dict[str, np.float64]]:
        """
        Returns: 返回oobn中的各个对象的输出节点的先验概率
        """
        ans = dict()
        for instanceNode in self.instanceNodes:
            ans[instanceNode.name] = instanceNode.priorOfOutPutNode
        return ans

    @property
    def model(self) -> BayesianNetwork:
        """_summary_

        Returns:
            BayesianNetwork: 返回简化的bayes
        """
        return self._BN

    @property
    def EncapsulatedNode(self) -> list[dict[str, list[str]]]:
        """_summary_
        返回封装节点
        Returns:
            list[dict[str, list[str]]] :[dict[str, list[encapsulatedNode]]]
        """
        encapsulatedNodeList = list()
        for instanceNode in self._oobnNodes:
            encapsulatedNodeList.append(instanceNode.encapsulatedNode)
        return encapsulatedNodeList

    def oobnNodes(self) -> dict[str, OrdinaryObject]:
        """
        Returns:
        """
        oobnDict = dict()
        for node in self._oobnNodes:
            oobnDict[node.name] = node
        return oobnDict

    def buildStructure(self) -> BayesianNetwork:
        """_summary_
        Returns:
            BayesianNetwork: 返回oobn的模型结构
        """
        model = BayesianNetwork()
        for node in self._oobnNodes:
            model.add_node(node.name)
            for parent in node.parents:
                model.add_edge(parent, node.name)
        return model

    def addCPD(self) -> None:
        """
        Returns:
        """
        # oobn节点名称（以输出节点为名称)
        nameList = [node.name for node in self._oobnNodes]
        # 简化模型中根节点的bayes节点
        childOfRoot = {self._BN.get_children(root)[0] for root in self._BN.get_roots()}
        # childOfRoot所对应的oobn节点
        oobnWithRoot = [oobn for oobn in self._oobnNodes if oobn.name in childOfRoot]
        # oobn中root节点所对应的cpd
        parentCpdDict = {
            parent: [cpd for cpd in oobn._parentsCpds if cpd.variable == parent][0]
            for oobn in oobnWithRoot
            for parent in oobn.parents
        }
        for node in self._BN.nodes:
            # 若 node 为 oobn 某对象的输出节点
            if node in nameList:
                self._BN.add_cpds(
                    self._oobnNodes[nameList.index(node)].simplifiedModel[-1]
                )
            # 根节点所对应的cpd
            else:
                self._BN.add_cpds(parentCpdDict[node])


if __name__ == "__main__":
    oobnNodes = [
        OrdinaryObject("./XML/OOBN/齿轮箱震动.xdsl"),
        OrdinaryObject("./XML/OOBN/动力装置.xdsl"),
        OrdinaryObject("./XML/OOBN/燃油管道.xdsl"),
        OrdinaryObject("./XML/OOBN/燃烧状态.xdsl"),
        OrdinaryObject("./XML/OOBN/喷油器.xdsl"),
    ]
    oobn = BuildOOBN(oobns=oobnNodes)
    model = oobn.priorprobOfoobnNode
    print(model["Fuel_L"].values())
    print(model)
    # infer = VariableElimination(model)
    # answers = infer.query(
    #     variables=["Fuel_injector"],
    #     evidence={"FuelL_Block": "good", "fuel_supply_advance_angle": "bad"},
    #     joint=False,
    # )
    # for ans in answers.values():
    #     ans
    # ans = list(answers.values())[0].values
    # print("-----------------------------------------------")
