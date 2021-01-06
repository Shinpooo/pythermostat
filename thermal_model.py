class ThermalModel:
    def __init__(self, params):
        for key in params:
            setattr(self, key, params[key])

    def step(self, Ta, Te, Th, Ti, Ph, time_step):
        dTi = ((Te - Ti) / (self.Rie * self.Ci) + (Th - Ti) / (self.Rih * self.Ci) + (Ta - Ti) / (self.Ria * self.Ci)) * time_step 
        dTe = ((Ti - Te) / (self.Rie * self.Ce) + (Ta - Te) / (self.Rea * self.Ce)) * time_step 
        dTh = ((Ti - Th) / (self.Rih * self.Ch) + (Ph) / (self.Ch)) * time_step
        return dTi, dTe, dTh