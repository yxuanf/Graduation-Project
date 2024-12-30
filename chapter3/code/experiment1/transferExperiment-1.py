import numpy as np
import pandas as pd

from genie2pgm.simplemodel import SimpleDiscreteModel
from oobn.utils.trainutils import Trainutils
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork


def compareDiffWeight(bn_t: BayesianNetwork, bn_T: BayesianNetwork, bn_s: list[BayesianNetwork]):
    """

    Args:
        bn_t: 待学习的目标网络
        bn_T: 已知参数的目标网络
        bn_s: 源网络集合
    """
    for _, j in enumerate(range(0, 11, 5)):
        beta = j / 10
        ans = Trainutils.transferParams_beta(bn_t, bn_s, beta)
        model = bn_t.copy()
        for node, value in ans.items():
            cpt = model.get_cpds(node)
            pT = value * alpha1 + cpt.values * (1 - alpha1)
            cpt.values = pT
            new_cpt = cpt
            model.remove_cpds(model.get_cpds(node))
            model.add_cpds(new_cpt)
        print(beta)
        print(Trainutils.kLDivergence(bn_T, model))


def compareTransfer(bn_t: BayesianNetwork, bn_T: BayesianNetwork, bn_s: list[BayesianNetwork]):
    """
    @ description:  比较是否进行参数学习
     Args:
        bn_t: 待学习的目标网络
        bn_T: 已知参数的目标网络
        bn_s: 源网络集合
    """
    ans = Trainutils.transferParams(bn_t, bn_s)
    model = bn_t.copy()
    for node, value in ans.items():
        cpt = model.get_cpds(node)
        pT = value * alpha1 + cpt.values * (1 - alpha1)
        cpt.values = pT
        new_cpd = cpt
        model.remove_cpds(bnT1.get_cpds(node))
        model.add_cpds(new_cpd)
    print(Trainutils.kLDivergence(bn_T, model))


if __name__ == '__main__':
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
    print(Trainutils.kLDivergence(bnT, bnT1))
    alpha1 = np.exp(-20 / Trainutils.getThreshold(bnT))
    alpha2 = 0.5
    # for node in ans:
    #     print(ans)
    # for node, value in ans.items():
    #     cpt = bnT1.get_cpds(node)
    #     pT = value * alpha1 + cpt.values * (1 - alpha1)
    #     cpt.values = pT
    #     new_cpd = cpt
    #     bnT1.remove_cpds(bnT1.get_cpds(node))
    #     bnT1.add_cpds(new_cpd)

    compareDiffWeight(bnT1, bnT, BN_S)

    # for _, j in enumerate(range(0, 11, 5)):
    #     beta = j / 10
    #     ans = Trainutils.transferParams_beta(bnT1, BN_S, beta)
    #     model = bnT1.copy()
    #     for node, value in ans.items():
    #         cpt = model.get_cpds(node)
    #         pT = value * alpha1 + cpt.values * (1 - alpha1)
    #         cpt.values = pT
    #         new_cpd = cpt
    #         model.remove_cpds(model.get_cpds(node))
    #         model.add_cpds(new_cpd)
    #     print(beta)
    #     print(Trainutils.kLDivergence(bnT, model))
    #     print(model.check_model())

    #     pT = ans[node] * alpha1 + cpt.values * (1 - alpha1)
    #     cpt.values = pT
    #     new_cpd = cpt
    #     bnT1.remove_cpds(bnT1.get_cpds(node))
    #     bnT1.add_cpds(new_cpd)
    # print(Trainutils.kLDivergence(bnT, bnT1))
