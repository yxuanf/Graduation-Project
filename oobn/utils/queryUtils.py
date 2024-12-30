import numpy as np

from oobn.train.instantiation import instantiation
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from oobn.old.instanceNode import OrdinaryObject
from pgmpy.models import BayesianNetwork


class QueryUtils:
    """
    Query utils class
    """

    @staticmethod
    def addVirtualNode(instanceNode: instantiation, post: dict[str, np.ndarray], alpha=0.8) -> BayesianNetwork:
        """_summary_

        虚拟节点为接口的子节点的原因：
            1、不用改变原始net的cpd
            2、所构造的cpd更小
        Args:
            post: 接口传入的概率
            alpha: 修正因子    alpha -> (0-1]
            instanceNode (BayesianNetwork): 传入的某个对象
            prior (dict[str, dict]): {nodeName: {nodeStates: prior}}
                                     待添加节点的先验概率（必须计算清楚）
            post = {
                "Fuel_injector": np.array([0.74783433, 0.25216567, 0]),
                "starting_injection_pressure": np.array([0.99543117, 0.0045688304]),
                "Poor_atom": np.array([0.99519341, 0.0048065881]),
                }
        Returns:
            BayesianNetwork: 返回添加了虚拟节点（虚拟节点为接口的子节点）的新对象
        """

        # name of interface
        nodeNames = list(post.keys())
        # create new model
        copiedModel = instanceNode.classBN.BN.copy()
        for nodeName in nodeNames:
            # prior of interface
            # priorProb = list(prior[nodeName].values())
            priorProb = list()
            for cpd in instanceNode.inputNodesCPDs:
                if cpd.variable == nodeName:
                    priorProb = list(cpd.values)
            postProb = post[nodeName]
            # Start from the first state where the posterior probability is not 0
            init_index = np.nonzero(postProb)[0][0]
            # init prior prob
            init_prior = priorProb[init_index]
            # init post prob
            init_post = postProb[init_index]
            # num of interface states
            length = postProb.size
            virtual_cpd = np.zeros((2, length))
            virtual_cpd[0][init_index] = 1
            for i in range(length):
                if i == init_index:
                    continue
                value = (postProb[i] * init_prior) / (priorProb[i] * init_post)
                virtual_cpd[0][i] = value
            max_value = np.max(virtual_cpd[0])
            # Zhang's originalOOBN
            virtual_cpd[0] = (alpha / max_value) * virtual_cpd[0]
            virtual_cpd[1] = 1 - virtual_cpd[0]

            # 添加虚拟节点
            copiedModel.add_edge(nodeName, f"virtual_node_of_{nodeName}")
            # 添加条件状态表
            cpd_virtual_node = TabularCPD(
                variable=f"virtual_node_of_{nodeName}",
                variable_card=2,
                values=virtual_cpd,
                evidence=[nodeName],
                evidence_card=[length],
                state_names={
                    f"virtual_node_of_{nodeName}": ["state0", "state1"],
                    nodeName: copiedModel.states[nodeName],
                },
            )
            copiedModel.add_cpds(cpd_virtual_node)
        return copiedModel

    @staticmethod
    def PIA_GBS_Time():
        """

        """
        pass


if __name__ == "__main__":
    # xmlpath = "./XML/originalOOBN/启动供油组件.xdsl"
    # oobn = OrdinaryObject(xmlpath)
    # model = oobn.initialModel
    # # prior = {oobn.name: oobn.priorProbOfParent}
    # prior = oobn.priorProbOfInterface
    # # prior.update(
    # #     {"Shaft": {"fault": 0.02, "deterioration": 0.03, "alert": 0.05, "good": 0.9}}
    # # )
    # postProbvalues = {
    #     "Fuel_injector": np.array([0.74783433, 0.25216567, 0]),
    #     "starting_injection_pressure": np.array([0.99543117, 0.0045688304]),
    #     "Poor_atom": np.array([0.99519341, 0.0048065881]),
    # }
    # evidence = {}
    # set_parent = set()
    # for parent in list(postProbvalues.keys()):
    #     evidence.update({f"virtual_node_of_{parent}": "state0"})
    #     set_parent.add(f"virtual_node_of_{parent}")
    # copied_model = OOBNUtils.addVirtualNode(instanceNode=model, prior=prior, post=postProbvalues)
    # infer = VariableElimination(model=copied_model)
    # q = infer.query(
    #     variables=list(set(list(copied_model.nodes)) - set_parent),
    #     evidence=evidence,
    #     joint=False,
    # )
    # for ans in q.values():
    #     print(ans)
    pass
