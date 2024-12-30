import numpy as np
from matplotlib import pyplot as plt

from oobn.build.encapsulation import bnClass
from oobn.query.query import ooBNQuery
from oobn.train.buildOOBN import ooBN
from oobn.train.instantiation import instantiation
from oobn.utils.visualUtils import timeComparision, detailTime
from scipy.ndimage import gaussian_filter

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
    instantiation1 = instantiation(class1)
    instantiation2 = instantiation(class2)
    instantiation3 = instantiation(class3)
    instantiation4 = instantiation(class4)
    instantiation5 = instantiation(class5)
    instantiation6 = instantiation(class6)
    instantiation7 = instantiation(class7)
    instantiation8 = instantiation(class8)
    instantiation9 = instantiation(class9)
    instantiation10 = instantiation(class10)
    instantiation11 = instantiation(class11)
    instanceNodeList = [instantiation1, instantiation2, instantiation3, instantiation4,
                        instantiation5, instantiation6, instantiation7, instantiation8,
                        instantiation9, instantiation10, instantiation11]

    ooBNTest = ooBN(instanceNodeList)
    queryTest = ooBNQuery(model=ooBNTest)
    nodes = [
        "Pressure_rise_limiter",
        "Low_protection_value",
        "Iron_filings_1",
        "Low_fuel_injector_pressure",
        "No_working_signal",
        "Shaft_seal_damaged_2",
        "Fuel_System",
        "Oil_leak_2",
        "Injector_pressure_fluctuation",
        "No_power_on_signal",
        # "Unable_to_adjust_fuel",
        # "No_working_signal",
    ]
    # 底层叶子节点作为观测数据
    observedData = {
        "Gear_pump_wear_1": "劣化",
        "Fine_Oil_Filter_1": "正常",
        "Gear_pump_wear_2": "劣化",
        "Fine_Oil_Filter_2": "脏污",
        "Damaged_sealing_ring": "未发生",
        "Insufficient_traffic": "未发生",
        "Begrime": "未发生",
        "Start_fuel_valve": "正常",
        "Check_valve": "正常",
        "Thermostat_circuit_disconnected": "未发生",
        "On_signal_off": "未发生",
        "Distribution_valve": "卡滞在关闭位置",
        "Autostarter": "正常",
        "Short_circuit_or_open_circuit": "未发生",
        "Throttle": "积垢"
    }
    resultVE, tVE = queryTest.queryVE(nodes, observedData, counts=50)
    resultBP, tBP = queryTest.queryCTP(nodes, observedData, counts=50)
    resultGlobal, tGL = queryTest.globalPropagation(observedData, counts=50)
    p1 = resultGlobal["Throttle_component"].values
    result1, t1 = ooBNQuery.localInference(["Pressure_rise_limiter"], ooBNTest.objectNodes[5],
                                           {"Throttle_component": p1},
                                           counts=50)
    result2, t2 = ooBNQuery.localInference(["Low_protection_value"], ooBNTest.objectNodes[3],
                                           {"Thermostat_circuit_disconnected": np.array([0, 1]),
                                            "On_signal_off": np.array([0, 1])},
                                           counts=50)
    result3, t3 = ooBNQuery.localInference(["Iron_filings_1"], ooBNTest.objectNodes[-2],
                                           {"Gear_pump_wear_1": np.array([0, 1, 0]),
                                            "Fine_Oil_Filter_1": np.array([0, 1])},
                                           counts=50)
    result4, t4 = ooBNQuery.localInference(["Low_fuel_injector_pressure"], ooBNTest.objectNodes[0],
                                           {"Distribution_valve": np.array([1, 0]),
                                            "Autostarter": np.array([0, 1])},
                                           counts=50)
    result5, t5 = ooBNQuery.localInference(["No_working_signal"], ooBNTest.objectNodes[2],
                                           {"Short_circuit_or_open_circuit": np.array([0, 1]),
                                            "Throttle": np.array([1, 0])},
                                           counts=50)
    result6, t6 = ooBNQuery.localInference(["Oil_leak_2"], ooBNTest.objectNodes[-1],
                                           {"Gear_pump_wear_2": np.array([0, 1, 0]),
                                            "Fine_Oil_Filter_2": np.array([1, 0])},
                                           counts=50)
    result7, t7 = ooBNQuery.localInference(["Injector_pressure_fluctuation"], ooBNTest.objectNodes[1],
                                           {"Start_fuel_valve": np.array([0, 1]),
                                            "Check_valve": np.array([0, 1])},
                                           counts=50)

    print(f"BP算法的时间为{tBP}")
    print(f"VE算法的时间为{tVE}")
    print(f"全局算法的时间为{tGL}")
    print(f"局部推理算法时间为{t1}")
    t = [tGL[i] + max(values) for i, values in enumerate(zip(t1, t2, t3, t4, t5, t6, t7))]
    print(f"新算法的时间为{t}")
    timeComparision(tVE, tBP, t)
    detailTime(tGL, {"NLow_fuel_injector_pressure": t5, "Thermostat_check_abnormality": t6, "Fuel_System": t7,
                     "Shaft_seal_damaged": t1, "Valve_components": t2, "Thermostat_working": t3,
                     "Low_protection_value": t4, })
    print(f"CTP算法的平均时间为{np.mean(tBP)}")
    print(f"VE算法的平均时间为{np.mean(tVE)}")
    print(f"新算法的平均时间为{np.mean(t)}")
    plt.show()
