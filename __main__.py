from simulation.thermostat_env import ThermostatEnv
from dateutil.parser import isoparse
from controllers.opt_controller import OptController
from controllers.dqn_controller import DqnController
from controllers.valve_controller import ValveController


if  __name__ == "__main__":
    train_start_date = isoparse("2019-12-23T00:00:00")
    train_end_date = isoparse("2019-12-28T00:00:00")
    test_start_date = isoparse("2019-12-23T00:00:00")
    test_end_date = isoparse("2019-12-28T00:00:00")
    train_env = ThermostatEnv(train_start_date, train_end_date)
    test_env = ThermostatEnv(test_start_date, test_end_date)
    #model = TRPO(MlpPolicy, train_env, verbose=1)
    c = OptController(test_env)
    # c.train()
    # c.save()
    # c.load()
    # c.set_env(test_env)
    c.simulate()
