import numpy as np

from genie2pgm.simplemodel import SimpleDiscreteModel
from genie2pgm.exception import GenieException
from pgmpy.models import BayesianNetwork


class bnClass(object):
    """
    @ description: 定义基础类
    """

    def __init__(self, xmlPath: str) -> None:
        # path
        self.xmlPath = xmlPath
        # inputNodes
        self.inputNodes = list()
        # outputNode
        self.outputNode = None
        # bayesianNetwork
        self.BN = self.getModel(xmlPath)
        # name of class
        self.name = str(self.outputNode)
        # inputNodes + outputNode
        self.interfaceNodes = self.inputNodes + [self.outputNode]

        # encapsulationNode
        self.encapsulationNode = list(set(self.BN.nodes()) - set(self.interfaceNodes))

        # all statements of interfaceNodes
        self.interfaceState = {
            interfaceNode: self.BN.get_cpds(interfaceNode).state_names[interfaceNode]
            for interfaceNode in self.interfaceNodes
        }

        # The Num of outputNode’s states
        self.outputNodeCard = self.BN.get_cpds(self.outputNode).variable_card

        # The Num of inputNode’s states
        self.inputNodesCard = np.array(
            [self.BN.get_cpds(node=inputNode).variable_card for inputNode in self.inputNodes]
        )

        #  num of state combinations of input node
        self.inputNodesCardProduct = np.prod(self.inputNodesCard)

        # inheritance Chain
        # parent
        self.parent = None
        # children
        self.children = list()

    def getModel(self, xmlPath: str) -> BayesianNetwork:
        """
        @ description: 获取genie的网络结构
        """
        try:
            module = SimpleDiscreteModel(xmlPath)
            module.add_cpd(module.model, module.getcpd())
            self.inputNodes, self.outputNode = module.getInterface()
            return module.model
        except Exception as e:
            raise GenieException(f"pgmpy2genie 模块解析出现错误!:{str(e)}")

    def checkClass(self):
        """
        @description: 判断是否符合类的规范
        Returns: true or false
        """
        model = self.BN
        if len(model.get_children(self.outputNode)) != 0:
            return False
        for node in self.inputNodes:
            if len(model.get_parents(node)) != 0:
                return False
        return True

    def is_ancestor(self, target_class):
        """
        检查当前类是否是target_class父类。

        Args:
            target_class: 目标类（bnClass 实例）。

        Returns:
            bool: 如果 target_class 是 self子类，则返回 True；否则返回 False。
        """
        # 检查输入类型
        if not isinstance(self, bnClass) or not isinstance(target_class, bnClass):
            raise ValueError("输入的类必须是 bnClass 类型的实例！")

        # 检查直接子类关系
        if target_class in self.children:
            return True

        # 检查继承链中的子类关系（递归）
        for child in self.children:
            if child.is_ancestor(target_class):
                return True

        # 如果未找到子类关系，返回 False
        return False

    def is_descendant(self, target_class):
        """
        @description: 判断当前类是否是目标类的子类（直接或间接）
        Args:
            target_class: 目标类实例
        Returns:
            True 如果当前类是 target_class 的子类（直接或间接），否则 False
        """
        if not isinstance(target_class, bnClass):
            raise TypeError("target_class 必须是 bnClass 类型的实例")

        # 如果当前类的父类是目标类，则为直接子类
        if self.parent == target_class:
            return True

        # 如果存在父类，递归向上检查父类是否是目标类的子类
        if self.parent:
            return self.parent.is_descendant(target_class)

        # 否则，不是子类
        return False


if __name__ == '__main__':
    Path1 = "./XML/OOBN/燃烧状态.xml"
    instanceNode1 = bnClass(Path1)
    check_class = instanceNode1.checkClass()
    print(check_class)
    print(instanceNode1.BN.edges)
