import numpy as np

class ThermalState:
    def __init__(self, date_time, Ti=None, Ta=None, Th=None, Te=None, Ts=None, hour=None):
        self.date_time = date_time
        self.T_indoor = Ti
        self.T_ambient = Ta #outdoor
        self.T_heater = Th
        self.T_envelope = Te
        self.P_heater = None
        self.T_set = Ts
        self.hour = hour
        self.reward = None
        self.comfort = None
        self.CO2 = None
        self.cost = None

    def get_RL_state(self):
        return np.array([self.T_ambient, self.T_envelope, self.T_heater, self.T_indoor, self.T_set, self.hour])
