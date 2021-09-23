from simulation.thermostat_env import ThermostatEnv
from dateutil.parser import isoparse
from controllers.opt_controller import OptController
from controllers.dqn_controller import DqnController
from controllers.valve_controller import ValveController


if  __name__ == "__main__":
    #Dates
    train_start_date = isoparse("2019-12-23T00:00:00")
    train_end_date = isoparse("2019-12-28T00:00:00")
    test_start_date = isoparse("2019-12-23T00:00:00")
    test_end_date = isoparse("2019-12-30T00:00:00")
    train_env = ThermostatEnv(train_start_date, train_end_date)
    test_env = ThermostatEnv(test_start_date, test_end_date)

    #Reinforcement learning control
    c1 = DqnController(train_env)
    c1.train()
    # c1.save()
    # c1.load()
    c1.set_env(test_env)
    c1.simulate()

    #Optimization control
    # c2 = OptController(test_env)
    # c2.simulate()

    # Valve  control
    c3 = ValveController(test_env)
    c3.simulate()

