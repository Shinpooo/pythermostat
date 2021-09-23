import gym
from gym import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_management.database import Database
from simulation.thermal_state import ThermalState
from models.thermal_model import ThermalModel
import json
import os
from plot.plotter import Plotter


class ThermostatEnv(gym.Env):
  """Custom Environment that follows gym interface"""

  def __init__(self, start_date, end_date):
    with open('params.json') as json_file:
        params = json.load(json_file)
    self.thermal_model = ThermalModel(params)
    self.database = Database("data/demo_data.csv")
    self.time_step = 1
    self.start_date = start_date
    self.end_date = end_date
    self.date_range = pd.date_range(start=start_date, end=end_date,freq=str(self.time_step)+"H") 
    #Action: P_h
    self.action_space = spaces.Discrete(150)
    # TODO Normalize action space [-1,1] for continuous action
    # self.action_space = spaces.Box(low=-1, high=1, dtype=np.float64)
    # Example for using image as input:
    #State : [T_a, T_e, T_h, T_i, Ts, H]
    self.observation_space = spaces.Box(low=np.array([0, 0, 0, 10, 16, 0]), high=np.array([40, 30, 100, 25, 25, 23]), dtype=np.float64)
    self.thermal_states = []
    self.get_init_condtions()
    self.P_capacity = 2000
    


  def step(self, action):
    Ta = self.thermal_states[-1].T_ambient
    Te = self.thermal_states[-1].T_envelope
    Th = self.thermal_states[-1].T_heater
    Ti = self.thermal_states[-1].T_indoor
    Ts = self.thermal_states[-1].T_set
    Ph = action
    #Ph = (action + 1)*75
    dTi, dTe, dTh = self.thermal_model.step(Ta, Te, Th, Ti, Ph, self.time_step)
    Ti = Ti + dTi 
    #print(Ts)
    Te = Te + dTe 
    Th = Th + dTh
    reward = -(Ti - Ts)**2
     
    self.env_step += 1
    Ta = self.database.get_columns("Ta", self.date_range[self.env_step])
    #Ts = np.random.uniform(16,22)
    Ts = self.database.get_columns("Ts", self.date_range[self.env_step])
    hour_of_day = self.database.get_columns("Hour", self.date_range[self.env_step])
    
    self.thermal_states[-1].P_heater = Ph
    self.thermal_states[-1].reward = reward
    # print(self.thermal_states[-1].T_indoor)
    # print(self.thermal_states[-1].P_heater)
    done = False
    if self.env_step == len(self.date_range) - 1:
        done = True
    p_dt = self.date_range[self.env_step]
    self.thermal_states.append(ThermalState(date_time=self.date_range[self.env_step], Ta=Ta, Te=Te, Th=Th, Ti=Ti, Ts=Ts, hour=hour_of_day))
    next_state = self.thermal_states[-1].get_RL_state()
    return next_state, reward, done, {}

  def reset(self):
    self.env_step = 0
    hour_of_day = self.database.get_columns("Hour", self.start_date)
    self.thermal_states = [ThermalState(date_time=self.date_range[self.env_step], Ta=self.Ta0, Te=self.Te0, Th=self.Th0, Ti=self.Ti0, Ts=self.Ts0, hour=hour_of_day)]
    state = self.thermal_states[-1].get_RL_state()
    return state 

  def render(self):
    pass
    
  def get_init_condtions(self):
    self.Ta0 = self.database.get_columns("Ta", self.start_date)
    self.Th0 = self.database.get_columns("Th", self.start_date)
    self.Ti0 = self.database.get_columns("Ti", self.start_date)
    self.Ts0 = self.database.get_columns("Ts", self.start_date)
    self.Te0 = self.thermal_model.Te0


  def store_and_plot(self, result_folder):
    results = dict(
      dates = ["%s" % tstate.date_time for tstate in self.thermal_states],
      T_indoor = [tstate.T_indoor for tstate in self.thermal_states],
      T_ambient = [tstate.T_ambient for tstate in self.thermal_states],
      T_envelope = [tstate.T_envelope for tstate in self.thermal_states],
      T_heater = [tstate.T_heater for tstate in self.thermal_states],
      T_set = [tstate.T_set for tstate in self.thermal_states],
      P_heater = [float(tstate.P_heater) for tstate in self.thermal_states[:-1]]
    )
    if not os.path.isdir(result_folder):
      os.makedirs(result_folder)
    with open(result_folder + "/results.json", 'w') as jsonFile:
      json.dump(results, jsonFile)
    plotter = Plotter(results, result_folder)
    plotter.plot_results()
