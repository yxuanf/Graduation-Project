import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from pgmpy.factors.discrete import TabularCPD
from genie2pgm.simplemodel import SimpleDiscreteModel
from oobn.utils.trainutils import Trainutils
from pgmpy.estimators import MaximumLikelihoodEstimator

if __name__ == '__main__':
    bnPath = './chapter3/model/originBN/燃油系统.xdsl'
    bnTrainPath = './chapter3/model/originBN/燃油系统.xdsl'
    trainDataPath = "./chapter3/model/trainBN/燃油系统.xdsl"
    data = pd.read_csv("./chapter3/data/trainData/燃油系统-5000.csv", delimiter=' ')
    originBN = SimpleDiscreteModel(bnPath)
    originBN.add_cpd(originBN.model, originBN.getcpd())
    bn = originBN.model
    print(bn.check_model())
    klTest = Trainutils.kLDivergence(bn, bn)
    # TrainBN = SimpleDiscreteModel(trainDataPath)
    # TrainBN.add_cpd(TrainBN.model, TrainBN.getcpd())
    # bnTrain = TrainBN.model

    bnTrain = bn.copy()
    bnTrain.remove_cpds(*bnTrain.get_cpds())
    timeList = list()
    for _ in tqdm(range(50)):
        start_time = time.perf_counter() * 1000
        bnTrain.remove_cpds(*bnTrain.get_cpds())
        bnTrain.fit(data, estimator=MaximumLikelihoodEstimator)
        end_time = time.perf_counter() * 1000
        timeList.append(end_time - start_time)
    print(sum(timeList) / len(timeList))
