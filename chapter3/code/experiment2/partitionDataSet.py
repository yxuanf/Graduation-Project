import pandas as pd


def getGearPump1(path: str):
    """

    Args:
        path:
    """
    columns_to_select = ['Gear_pump_wear_1', 'Iron_filings_1', 'Breakdown_shutdown_1', 'Fine_Oil_Filter_1',
                         'Pressure_difference_1', 'Shaft_seal_damaged_1', 'Oil_leak_1', 'Gear_pump_1']
    output_file = './chapter3/data/trainData/GearPump1/GearPump1.csv'
    df = pd.read_csv(path, usecols=columns_to_select, delimiter=' ')
    df.to_csv(output_file, index=False)
    print(df.head())


def getGearPump2(path: str):
    """

    Args:
        path:
    """
    columns_to_select = ['Gear_pump_wear_2', 'Iron_filings_2', 'Breakdown_shutdown_2', 'Fine_Oil_Filter_2',
                         'Pressure_difference_2', 'Shaft_seal_damaged_2', 'Oil_leak_2', 'Gear_pump_2']
    output_file = './chapter3/data/trainData/GearPump2/GearPump2.csv'
    df = pd.read_csv(path, usecols=columns_to_select, delimiter=' ')
    df.to_csv(output_file, index=False)
    print(df.head())


def getGearPump(path: str):
    """

    Args:
        path:
    """
    columns_to_select = ['Gear_pump_wear_1', 'Gear_pump_wear_2', 'Gear_pump']
    output_file = './chapter3/data/trainData/GearPump/GearPump.csv'
    df = pd.read_csv(path, usecols=columns_to_select, delimiter=' ')
    df.to_csv(output_file, index=False)
    print(df.head())


def get_Faulty_fuel_drain_valve(path: str):
    """
    @ description: 故障燃油排放阀
    Args:
        path:
    """
    columns_to_select = ['Short_circuit_or_open_circuit', 'Throttle', 'No_working_signal',
                         'Thermostat_check_abnormality', 'Faulty_fuel_drain_valve']
    output_file = './chapter3/data/trainData/Faulty_fuel_drain_valve/Faulty_fuel_drain_valve.csv'
    df = pd.read_csv(path, usecols=columns_to_select, delimiter=' ')
    df.to_csv(output_file, index=False)
    print(df.head())


def get_Throttle_component(path: str):
    """
    @ description: 节流组件
    Args:
        path:
    """
    columns_to_select = ['Damaged_sealing_ring', 'Insufficient_traffic', 'Begrime',
                         'Thermostat_working', 'Abnormal_speed', 'Throttle_component']
    output_file = './chapter3/data/trainData/Throttle_component/Throttle_component.csv'
    df = pd.read_csv(path, usecols=columns_to_select, delimiter=' ')
    df.to_csv(output_file, index=False)
    print(df.head())


def get_Start_fueling(path: str):
    """
    @ description: 启动供油组件
    Args:
        path:
    """
    columns_to_select = ['Start_fuel_valve', 'Check_valve', 'Injector_pressure_fluctuation', 'Start_fueling']
    output_file = './chapter3/data/trainData/Start_fueling/Start_fueling.csv'
    df = pd.read_csv(path, usecols=columns_to_select, delimiter=' ')
    df.to_csv(output_file, index=False)
    print(df.head())


def get_Fuel_regulator(path: str):
    """
    @ description: 燃油调节器
    Args:
        path:
    """
    columns_to_select = ['Throttle_component', 'Pressure_rise_limiter', 'Valve_components', 'Unable_to_adjust_fuel',
                         'Fuel_regulator']
    output_file = './chapter3/data/trainData/Fuel_regulator/Fuel_regulator.csv'
    df = pd.read_csv(path, usecols=columns_to_select, delimiter=' ')
    df.to_csv(output_file, index=False)
    print(df.head())


def get_Fuel_System(path: str):
    """
    @ description: 燃油系统
    Args:
        path:
    """
    columns_to_select = ['Fuel_System', 'Gear_pump', 'Fuel_regulator', 'Start_fueling', 'Automatic_fuel_dispenser',
                         'Faulty_fuel_drain_valve']
    output_file = './chapter3/data/trainData/Fuel_System/Fuel_System.csv'
    df = pd.read_csv(path, usecols=columns_to_select, delimiter=' ')
    df.to_csv(output_file, index=False)
    print(df.head())


def get_Parking_switch(path: str):
    """
    @ description: 停车开关组件
    Args:
        path:
    """
    columns_to_select = ['Distribution_valve', 'Autostarter', 'Low_fuel_injector_pressure', 'Parking_switch', ]
    output_file = './chapter3/data/trainData/Parking_switch/Parking_switch.csv'
    df = pd.read_csv(path, usecols=columns_to_select, delimiter=' ')
    df.to_csv(output_file, index=False)
    print(df.head())


def get_Temperature_regulator(path: str):
    """
    @ description: 温度调节器
    Args:
        path:
    """
    columns_to_select = ['Thermostat_circuit_disconnected', 'On_signal_off', 'Thermostat_failure',
                         'Low_protection_value', 'No_power_on_signal', 'Temperature_regulator']
    output_file = './chapter3/data/trainData/Temperature_regulator/Temperature_regulator.csv'
    df = pd.read_csv(path, usecols=columns_to_select, delimiter=' ')
    df.to_csv(output_file, index=False)
    print(df.head())


def get_Automatic_fuel_dispenser(path: str):
    """
    @ description: 自动燃油分配器
    Args:
        path:
    """
    columns_to_select = ['Temperature_regulator', 'Parking_switch', 'Automatic_fuel_dispenser']
    output_file = './chapter3/data/trainData/Automatic_fuel_dispenser/Automatic_fuel_dispenser.csv'
    df = pd.read_csv(path, usecols=columns_to_select, delimiter=' ')
    df.to_csv(output_file, index=False)
    print(df.head())


if __name__ == '__main__':
    filePath = "./chapter3/data/trainData/燃油系统.csv"
    # getGearPump1(filePath)
    # getGearPump2(filePath)
    # getGearPump(filePath)
    # get_Faulty_fuel_drain_valve(filePath)
    # get_Throttle_component(filePath)
    # get_Start_fueling(filePath)
    # get_Fuel_regulator(filePath)
    # get_Fuel_System(filePath)
    # get_Parking_switch(filePath)
    # get_Temperature_regulator(filePath)
    get_Automatic_fuel_dispenser(filePath)