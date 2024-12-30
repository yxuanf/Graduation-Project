from oobn.build.encapsulation import bnClass
from oobn.build.inheritance import inheritBNClass
from oobn.train.instantiation import instantiation
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork


class ooBN(object):
    """
    @ description: 构建oobn
    """

    def __init__(self, objectNodes: list[instantiation]):
        self.objectNodes = objectNodes
        self.checkObjectNodes()
        self.simplifiedOOBN = None
        self.originalBN = None
        self.connectedNodes = set()
        self.simplifiedObjectNodes = list()
        self.merge()
        self.addCPDs2SimplifiedOOBN()

    def checkObjectNodes(self):
        """
        @ description: 判断类是否完成了对应的实例化
        Returns:
        """
        for node in self.objectNodes:
            if not node.classBN.BN.check_model():
                print(f"类{node.classBN.name} 还未实例化，请先完成实例化")
                return False
        return True

    def getSimplifiedOOBNStructure(self):
        """
        @ description: 返回oobn所对应的简化贝叶斯网络的模型
        Returns:
        """
        model = BayesianNetwork()
        for node in self.objectNodes:
            model.add_node(node.classBN.name)
            for inputNode in node.classBN.inputNodes:
                model.add_edge(inputNode, node.classBN.name)
        self.simplifiedOOBN = model

    def addCPDs2SimplifiedOOBN(self):
        """
        @ description:为oobn对应的简化贝叶斯网络添加对应的条件概率表
        """
        if self.simplifiedOOBN is None:
            self.getSimplifiedOOBNStructure()

        # STEP1：oobn节点名称（以输出节点为名称）
        nameList = [instanceNode.outputNode for instanceNode in self.objectNodes]

        # STEP2: 获取简化BN中输入节点的子节点（即判断简化BN的根节点属于哪个对象）
        childrenOfRootNodes = {self.simplifiedOOBN.get_children(root)[0] for root in self.simplifiedOOBN.get_roots()}

        # STEP3：childOfRoot所对应的oobn节点
        rootOOBNList = [objectNode for objectNode in self.objectNodes if objectNode.outputNode in childrenOfRootNodes]

        # STEP4: oobn中root节点所对应的cpd
        cpdOfRootNode = {
            inputNode: [cpd for cpd in instanceNode.inputNodesCPDs if cpd.variable == inputNode][0]
            for instanceNode in rootOOBNList
            for inputNode in instanceNode.inputNodes
        }

        for node in self.simplifiedOOBN.nodes:
            # 若 node 为 oobn 某对象的输出节点
            if node in nameList:
                self.simplifiedOOBN.add_cpds(
                    self.objectNodes[nameList.index(node)].getSimplifiedModel()[-1]
                )
            # 根节点所对应的cpd
            else:
                self.simplifiedOOBN.add_cpds(cpdOfRootNode[node])

    def merge(self):
        """
        @ description: 将各个对象拼接组装为普通的贝叶斯网络,
        """
        ordinaryBN = BayesianNetwork()
        inputNodeSet = set()
        outputNodeSet = set()
        for objectNode in self.objectNodes:
            ordinaryBN.add_edges_from(objectNode.classBN.BN.edges())
            inputNodeSet.update(objectNode.inputNodes)
            outputNodeSet.add(objectNode.outputNode)

        # 获取相互连接的接口节点
        self.connectedNodes = inputNodeSet.intersection(outputNodeSet)

        # 更新相互连接接口（输入）节点的cpd
        for objectNode in self.objectNodes:
            for node in objectNode.classBN.BN.nodes():
                if node in self.connectedNodes and node != objectNode.outputNode:
                    continue
                else:
                    # 更新cpd
                    ordinaryBN.add_cpds(objectNode.classBN.BN.get_cpds(node))

        # 设置原始的贝叶斯网络
        self.originalBN = ordinaryBN
        # 基于完整的贝叶斯网络调整各个对象对应输入接口的先验概率
        infer = VariableElimination(model=self.originalBN)
        for objectNode in self.objectNodes:
            for inputNode in objectNode.inputNodes:
                q = infer.query(variables=[inputNode])
                objectNode.setInputNodesCPDS([q])

    # def updateObjectCPDs(self):
    #     """
    #     @ description: 基于完整的贝叶斯网络调整各个对象对应输入接口的先验概率
    #     """
    #     if not self.originalBN:
    #         self.merge()
    #     infer = VariableElimination(model=self.originalBN)
    #     for objectNode in self.objectNodes:
    #         for inputNode in objectNode.inputNodes:
    #             q = infer.query(variables=[inputNode])
    #             objectNode.setInputNodesCPDS([q])

    def adjustOOBN(self, initialObject: instantiation, targetObject: instantiation):
        """
        @ description:
        Args:
            initialObject: 原始对象
            targetObject:  替换后的对象
        """
        if initialObject not in self.objectNodes:
            raise ValueError(f"{initialObject}并不存在此 oobn中")

        # 检查子类的合法性
        if not targetObject.classBN.is_descendant(initialObject.classBN):
            raise ValueError("扩展类必须是原类的子类")

        # 改变实例节点
        self.objectNodes.remove(initialObject)
        self.objectNodes.append(targetObject)

        # 重构简化模型
        self.getSimplifiedOOBNStructure()

        # 更改原始模型并更新各个对象输入节点的概率
        self.merge()

        # 为简化贝叶斯网络添加对应的条件概率表
        self.addCPDs2SimplifiedOOBN()


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
    # class12 = bnClass("./XML/originalOOBN/齿轮泵1.3.xdsl")
    instantiation11 = instantiation(class11)
    class12 = inheritBNClass("./XML/originalOOBN/齿轮泵1.3.xdsl")
    class12.setParent(class11)
    instantiation12 = instantiation(class12)
    instanceNodeList = [instantiation(class1), instantiation(class2), instantiation(class3), instantiation(class4),
                        instantiation(class5), instantiation(class6), instantiation(class7), instantiation(class8),
                        instantiation(class9), instantiation(class10), instantiation11]

    oobnTest = ooBN(instanceNodeList)
    a = oobnTest.simplifiedOOBN.get_cpds("Gear_pump_2")
    print(oobnTest.objectNodes[-3].inputNodesCPDs[-1])
    oobnTest.adjustOOBN(instantiation11, instantiation12)
    b = oobnTest.simplifiedOOBN.get_cpds("Gear_pump_2")
    print(oobnTest.objectNodes[-3].inputNodesCPDs[-1])
