import numpy as np
import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame

from genie2pgm.simplemodel import SimpleDiscreteModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork


class Trainutils:
    """
    @ description: 训练工具类
    """

    def __init__(self):
        pass

    @staticmethod
    def GED(bn_s: BayesianNetwork, bn_t: BayesianNetwork):
        """
        @ description: 最小编辑距离
        Args:
            bn_t: 源网络
            bn_s: 目标网络
        return: 返回源网络与目标网络的最小编辑距离
        """
        node_s = set(bn_s.nodes())
        node_t = set(bn_t.nodes())
        all_nodes = list(node_s.union(node_t))
        n = len(all_nodes)
        adj_matrix_s = np.zeros((n, n))
        adj_matrix_t = np.zeros((n, n))

        node_to_index = {node: idx for idx, node in enumerate(all_nodes)}

        # 计算源网络的邻接矩阵
        for parent, child in bn_s.edges():
            parent_index = node_to_index[parent]
            child_index = node_to_index[child]
            adj_matrix_s[parent_index, child_index] = 1
            adj_matrix_s[child_index, parent_index] = -1

        # 计算目标网络的邻接矩阵
        for parent, child in bn_t.edges():
            parent_index = node_to_index[parent]
            child_index = node_to_index[child]
            adj_matrix_t[parent_index, child_index] = 1
            adj_matrix_t[child_index, parent_index] = -1

        a = np.sum(np.abs(adj_matrix_s - adj_matrix_t))
        b = np.sum(np.abs(adj_matrix_t + adj_matrix_s))
        c = (np.sum(np.abs(adj_matrix_s)) + np.sum(np.abs(adj_matrix_t)))
        s = (a + c - b) / 4
        return 1 / (1 + s)

    @staticmethod
    def kLDivergence(bn_s: BayesianNetwork, bn_t: BayesianNetwork):
        """
        @ description: 两个相同网络的KL散度
        Args:
            bn_s:
            bn_t:
        """
        total_kl = 0.0
        nodes = bn_t.nodes()
        for node in nodes:
            cpt_s = bn_s.get_cpds(node)
            cpt_t = bn_t.get_cpds(node)
            total_kl += Trainutils.kl_divergence_node(cpt_t, cpt_s)
        return total_kl

    @staticmethod
    def kl_divergence_node(cpt_s, cpt_t):
        """
        计算两个节点的KL散度。
        参数:
            cpd_t: 目标网络中某个节点的CPD
            cpd_s1: 参考网络中对应节点的CPD
        返回:
            float: 节点的KL散度
        """
        # 转换为数组形式
        theta_t = np.array(cpt_t.values)
        theta_s1 = np.array(cpt_s.values)

        # 确保维度一致
        if theta_t.shape != theta_s1.shape:
            raise ValueError("两个CPD的形状不一致")

        # 计算KL散度，避免log(0)或除以0
        kl = np.sum(theta_t * np.log(theta_t / theta_s1, where=(theta_t > 0)))
        return kl

    @staticmethod
    def compareLocalStructure(bn_s: BayesianNetwork, bn_t: BayesianNetwork, node: str):
        """

        Args:
            bn_s: 源网络
            bn_t: 目标网络
            node: 节点

        Returns:是否具有相同的局部结构

        """
        parents_BNs = set(bn_s.get_parents(node))
        parents_BNt = set(bn_t.get_parents(node))
        if parents_BNs != parents_BNt:
            return False

        for parent in parents_BNt:
            if (parent, node) not in bn_s.edges():
                return False

        return True

    @staticmethod
    def transferParams(bn_t: BayesianNetwork, bn_s: list[BayesianNetwork]):
        """
        @ description:  贝叶斯网络参数亲历学习
        Args:
            bn_t: 利用目标域训练的目标贝叶斯网络（与真实网络存在区别）
            bn_s: 源域贝叶斯网络
        """
        beta = 0.5
        # 目标域训练数据样本
        structSim = list()
        for bn in bn_s:
            ssc = Trainutils.GED(bn, bn_t)
            structSim.append(ssc)
        result = dict()
        for i, node in enumerate(bn_t.nodes()):
            valid_bnDict = list()
            cpt_List = list()
            for l, bn in enumerate(bn_s):
                if node in bn.nodes() and Trainutils.compareLocalStructure(bn, bn_t, node):
                    valid_bnDict.append(bn)
                    cpt_List.append(bn.get_cpds(node).values)
                else:
                    cpt_List.append(np.zeros_like(bn_t.get_cpds(node).values))
            w_s = list()
            w_p = list()
            for l, bn in enumerate(bn_s):
                if bn_s[l] not in valid_bnDict:
                    w_s.append(0)
                    w_p.append(0)
                else:
                    cpt_t = bn_t.get_cpds(node)
                    cpt_s = bn.get_cpds(node)
                    paramSim = Trainutils.kl_divergence_node(cpt_s, cpt_t)
                    w_s.append(structSim[l])
                    w_p.append(np.exp(-paramSim))
            w_s = [beta * w / sum(w_s) for w in w_s]
            w_p = [(1 - beta) * w / sum(w_p) for w in w_p]
            w = [a + b for a, b in zip(w_p, w_s)]
            ans = np.zeros_like(bn_t.get_cpds(node).values)
            for l in range(len(bn_s)):
                ans += w[l] * cpt_List[l]
            result.update({node: ans})
        return result

    @staticmethod
    def transferParams_beta(bn_t: BayesianNetwork, bn_s: list[BayesianNetwork], beta=0.5):
        """
        @ description:  贝叶斯网络参数亲历学习
        Args:
            beta:
            bn_t: 利用目标域训练的目标贝叶斯网络（与真实网络存在区别）
            bn_s: 源域贝叶斯网络
        """
        structSim = list()
        for bn in bn_s:
            ssc = Trainutils.GED(bn, bn_t)
            structSim.append(ssc)
        result = dict()
        for i, node in enumerate(bn_t.nodes()):
            valid_bnDict = list()
            cpt_List = list()
            for l, bn in enumerate(bn_s):
                if node in bn.nodes() and Trainutils.compareLocalStructure(bn, bn_t, node):
                    valid_bnDict.append(bn)
                    cpt_List.append(bn.get_cpds(node).values)
                else:
                    cpt_List.append(np.zeros_like(bn_t.get_cpds(node).values))
            w_s = list()
            w_p = list()
            for l, bn in enumerate(bn_s):
                if bn_s[l] not in valid_bnDict:
                    w_s.append(0)
                    w_p.append(0)
                else:
                    cpt_t = bn_t.get_cpds(node)
                    cpt_s = bn.get_cpds(node)
                    paramSim = Trainutils.kl_divergence_node(cpt_s, cpt_t)
                    w_s.append(structSim[l])
                    w_p.append(np.exp(-paramSim))
            w_s = [beta * w / sum(w_s) for w in w_s]
            w_p = [(1 - beta) * w / sum(w_p) for w in w_p]
            w = [a + b for a, b in zip(w_p, w_s)]
            ans = np.zeros_like(bn_t.get_cpds(node).values)
            for l in range(len(bn_s)):
                ans += w[l] * cpt_List[l]
            result.update({node: ans})
        return result

    @staticmethod
    def getThreshold(bn_t: BayesianNetwork, epsilon=0.1, delta=0.05, lam=1):
        """

        Args:
            bn_t:
            epsilon:
            delta:
            lam:
        """
        K = Trainutils.max_state_count(bn_t)
        d = Trainutils.max_parent_count(bn_t)
        n = len(set(bn_t.nodes()))
        part1 = 1 / (2 * np.power(lam, 2 * (d + 1)))
        part2 = (1 + epsilon) ** 2 / epsilon ** 2
        part3 = np.log(n * np.power(K, d + 1) / delta)
        N_Tr = int(part1 * part2 * part3)

        return N_Tr

    @staticmethod
    def max_state_count(bn: BayesianNetwork):
        """
        @ description: 计算贝叶斯网络中变量的最大状态数量。
        Args:
            bn: BayesianNetwork 对象
        返回:
            int: 最大状态数量
        """
        max_states = 0
        for node in bn.nodes():
            cpd = bn.get_cpds(node)
            state_count = cpd.cardinality[0]  # cardinality[0] 表示当前节点的状态数量
            max_states = max(max_states, state_count)
        return max_states

    @staticmethod
    def max_parent_count(bn: BayesianNetwork):
        """
        @ description:计算贝叶斯网络中所有变量的最大父节点数量。
        Args:
            bn:BayesianNetwork

        Returns:最大父节点数量

        """
        max_parents = 0
        for node in bn.nodes():
            parents = bn.get_parents(node)
            parent_count = len(parents)
            max_parents = max(max_parents, parent_count)
        return max_parents


if __name__ == '__main__':
    # path_t = "./model/train/testKL/AsiaDiagnosis.xdsl"
    # path_s = "./model/train/testKL/AsiaDiagnosis-1.xdsl"
    #
    # model_s = SimpleDiscreteModel(path_s)
    # model_s.add_cpd(model_s.model, model_s.getcpd())
    # bnS = model_s.model
    #
    # model_t = SimpleDiscreteModel(path_t)
    # model_t.add_cpd(model_t.model, model_t.getcpd())
    # bnT = model_t.model
    # print(trainUtils.getThreshold(bnS))

    path_T = "./model/train/testTransfer/AsiaDiagnosis-T.xdsl"
    path_T1 = "./model/train/testTransfer/AsiaDiagnosis-T1.xdsl"
    path_S1 = "./model/train/testTransfer/AsiaDiagnosis-S1.xdsl"
    path_S2 = "./model/train/testTransfer/AsiaDiagnosis-S2.xdsl"
    path_S3 = "./model/train/testTransfer/AsiaDiagnosis-S3.xdsl"
    path_S4 = "./model/train/testTransfer/AsiaDiagnosis-S4.xdsl"
    data_path = "./model/train/testTransfer/AsiaDiagnosis-T.csv"

    model_T = SimpleDiscreteModel(path_T)
    model_T.add_cpd(model_T.model, model_T.getcpd())
    bnT = model_T.model

    model_T1 = SimpleDiscreteModel(path_T1)
    model_T1.add_cpd(model_T1.model, model_T1.getcpd())
    bnT1 = model_T1.model

    print(Trainutils.kLDivergence(bnT, bnT1))

    model_S1 = SimpleDiscreteModel(path_S1)
    model_S1.add_cpd(model_S1.model, model_S1.getcpd())
    bnS1 = model_S1.model

    model_S2 = SimpleDiscreteModel(path_S2)
    model_S2.add_cpd(model_S2.model, model_S2.getcpd())
    bnS2 = model_S2.model

    model_S3 = SimpleDiscreteModel(path_S3)
    model_S3.add_cpd(model_S3.model, model_S3.getcpd())
    bnS3 = model_S3.model

    model_S4 = SimpleDiscreteModel(path_S4)
    model_S4.add_cpd(model_S4.model, model_S4.getcpd())
    bnS4 = model_S4.model

    BN_S = [bnS1, bnS2, bnS3, bnS4]
    ans = Trainutils.transferParams(bnT1, BN_S)
    result = dict()
    print(Trainutils.getThreshold(bnT))
    alpha = np.exp(-20 / Trainutils.getThreshold(bnT))
    for node in ans:
        cpt = bnT1.get_cpds(node)
        pT = ans[node] * alpha + cpt.values * (1 - alpha)
        cpt.values = pT
        new_cpd = cpt
        bnT1.remove_cpds(bnT1.get_cpds(node))
        bnT1.add_cpds(new_cpd)
    print(bnT1.check_model())
    print(Trainutils.kLDivergence(bnT, bnT1))
