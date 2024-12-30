import numpy as np
from oobn.build.encapsulation import bnClass
from pgmpy.factors.discrete import TabularCPD


class inheritBNClass(bnClass):
    def __init__(self, xmlPath: str):
        super(inheritBNClass, self).__init__(xmlPath)

    def validateInheritance(self, parentClass: bnClass):
        """
        @description: 验证子类是否符合继承父类的要求
        """
        # 父类的输入节点集合
        parent_input_nodes = set(parentClass.inputNodes)

        # 子类的输入节点集合
        child_input_nodes = set(self.inputNodes)

        # 检查子类的输入节点是否是父类的超集
        if not parent_input_nodes.issubset(child_input_nodes):
            raise ValueError(
                f"非法继承：子类的输入节点必须包含父类的所有输入节点。"
                f"父类: {parentClass.inputNodes}，子类: {self.inputNodes}"
            )

        # 检查相同部分的节点状态是否一致
        for node in parent_input_nodes:
            child_states = self.interfaceState[node]
            parent_states = parentClass.interfaceState[node]
            if child_states != parent_states:
                raise ValueError(
                    f"非法继承：节点 '{node}' 的状态不一致。子类: {child_states}，父类: {parent_states}"
                )

    def setParent(self, parentClass: bnClass) -> None:
        """
        @ description: 设置类的父类
        Args:
            parentClass: 所传入的父类
        Returns:
        """
        # 验证是否符合所设计的继承规则
        self.validateInheritance(parentClass)
        self.parent = parentClass
        if self not in parentClass.children:
            parentClass.children.append(self)
        else:
            ValueError(f"子类 '{self}' 已经存在于当前父类的子类集合中")

    def addInternalStructure(self, newNodes: list, newEdges: list):
        """
        @ description: 添加新的内部节点和边
        Args:
            newNodes: 新增节点列表
            newEdges: 新增边列表，格式为 [(parent, child), ...]
        """
        # 添加新的节点
        for node in newNodes:
            if node in self.BN.nodes:
                raise ValueError(f"节点 {node} 已存在于网络中")
            self.BN.add_node(node)

        # 添加新的边
        for edge in newEdges:
            parent, child = edge
            if parent not in self.BN.nodes or child not in self.BN.nodes:
                raise ValueError(f"边 {edge} 中的节点未定义")
            self.BN.add_edge(parent, child)

    def deleteInternalStructure(self, node_to_remove: str):
        """
        @ description: 删除类的内部节点及相关边
        Args:
            class_A: 需要修改的类（bnClass 实例）。
            node_to_remove: 要删除的内部节点名称。
        Returns:
            修改后的类（class_A）。
        """
        # 检查节点是否是接口节点
        if node_to_remove in self.interfaceNodes:
            raise ValueError(f"节点 {node_to_remove} 是接口节点，不能删除！")

        # 检查节点是否存在于网络中
        if node_to_remove not in self.BN.nodes:
            raise ValueError(f"节点 {node_to_remove} 不存在于网络中！")

        # 删除节点及其相关边
        self.BN.remove_node(node_to_remove)

    def modifyInputNodeState(self, original_inputNode: str, new_node: str):
        """
        修改类的非接口结构以实现接口节点状态的修改。

        Args:
            new_node:
            original_inputNode: 原始的接口节点（需要修改的节点名称）。
            new_node: 新的接口节点（新的状态）。
        Returns:
            修改后的类（class_A）。
        """
        # Step 1: 将新的节点添加为封装节点
        if new_node in self.BN.nodes:
            raise ValueError(f"节点 {new_node} 已存在于类 A 的结构中!")
        self.BN.add_node(new_node)

        # Step 2: 遍历原始节点的所有子节点
        for child in self.BN.get_children(original_inputNode):
            # Step 3: 删除原始节点与子节点之间的边
            self.BN.remove_edge(original_inputNode, child)
            # Step 4: 添加新的节点与子节点之间的边
            self.BN.add_edge(new_node, child)

        # Step 5: 在原始节点与新节点之间添加有向边
        self.BN.add_edge(original_inputNode, new_node)

    def modifyOutputNodeState(self, original_outputNode, new_node, state: list, cpd=None):
        """
        修改类的非接口结构以实现输出节点状态的修改。

        Args:
            state:
            original_outputNode:
            class_A: 需要修改的类（bnClass 实例）。
            original_node: 原始的输出节点（需要修改的节点名称）。
            new_node: 新的封装节点（用于等效改变输出节点状态）。

        """
        # Step 1: 将新的节点添加为封装节点
        if new_node in self.BN.nodes:
            raise ValueError(f"节点 {new_node} 已存在于类 A 的结构中！")
        self.BN.add_node(new_node)

        # Step 2: 遍历原始输出节点的所有父节点
        for parent in self.BN.get_parents(original_outputNode):
            # Step 3: 删除原输出节点与父节点之间的边
            self.BN.remove_edge(parent, original_outputNode)
            # Step 4: 添加新的封装节点与父节点之间的边
            self.BN.add_edge(parent, new_node)

        # Step 5: 在新节点与原输出节点之间添加方向边
        self.BN.add_edge(new_node, original_outputNode)

    def addInputNode(self, node_name: str, states: list, cpd=None):
        """
        @ description: 添加输入节点，并更新相关属性
        Args:
            cpd: 输入节点的cpd
            node_name: 新输入节点的名称
            states: 输入节点的所有可能状态（列表）{'Cylinder': ['good', 'bad'],...,}
        """
        # 检查节点是否已经存在
        if node_name in self.inputNodes:
            print(f"节点 {node_name} 已经存在于输入节点中，无法重复添加。")
            return
        # 添加到输入节点列表
        self.inputNodes.append(node_name)
        # 更新接口节点列表
        self.interfaceNodes = self.inputNodes + [self.outputNode]
        # 设置节点的状态
        self.interfaceState[node_name] = states

        # 如果该节点在模型中不存在，则需要创建对应的节点
        if node_name not in self.BN.nodes():
            self.BN.add_node(node_name)

        if cpd is None:
            # 添加一个默认的 CPD（条件概率分布），以确保模型一致性
            # 默认均匀分布
            cpd = np.full((len(states), 1), 1 / len(states))

        self.BN.add_cpds(
            TabularCPD(variable=node_name, variable_card=len(states), values=cpd, state_names={node_name: states})
        )

        # 更新输入节点的状态数量
        self.inputNodesCard = np.array(
            [self.BN.get_cpds(node=inputNode).variable_card for inputNode in self.inputNodes]
        )
        print(f"成功添加输入节点：{node_name}，状态：{states}")


if __name__ == "__main__":
    Path1 = "./XML/OOBN/燃烧状态.xdsl"
    instanceNode1 = bnClass(Path1)
    inheritanceNode1 = inheritBNClass(instanceNode1)
    inheritanceNode2 = inheritBNClass(inheritanceNode1)
    # inheritanceNode2.addInputNode("A", ["good", "bad"])
    # print(inheritanceNode1.parent)
    # print(instanceNode1.children)
    # print(instanceNode1.is_ancestor(inheritanceNode2))
    # print(inheritanceNode2.is_descendant(instanceNode1))
    # print(inheritanceNode1.is_ancestor(inheritanceNode2))
    # print(inheritanceNode2.BN.get_cpds("A"))
    inheritanceNode2.modifyInputNodeState('Fuel_L', 'new_Fuel_L')
