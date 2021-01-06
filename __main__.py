from thermostat_env import ThermostatEnv
import pandas as pd
import numpy as np
from stable_baselines.deepq.policies import MlpPolicy
#from stable_baselines.common.policies import MlpPolicy
from stable_baselines import DQN, TRPO
from dateutil.parser import isoparse


if  __name__ == "__main__":
    # data = pd.read_csv('./data/demo_data.csv', index_col=0, parse_dates=True)
    # len_data = data.shape[0]
    # n_day_train = 25#0-33
    # dfs = np.split(data, [24*n_day_train], axis=0)
    # train_df = dfs[0]
    # test_df = dfs[1]
    # params = {
    #     "Ti0": train_df.iloc[0]['Ti'],
    #     "Th0": train_df.iloc[0]['Th'],
    #     "Te0":18.8064, #[10,25]
    #     "Ci":156.988,
    #     "Ch":2.55,
    #     "Ce":389.1556,
    #     "Rie":0.1106,
    #     "Rea":405778124793,
    #     "Ria":0.637186,
    #     "Rih":0.65
   
    train_start_date = isoparse("2019-12-23T00:00:00")
    train_end_date = isoparse("2020-01-10T00:00:00")
    test_start_date = isoparse("2020-01-10T00:00:00")
    test_end_date = isoparse("2020-01-11T00:00:00")
    train_env = ThermostatEnv(train_start_date, train_end_date)
    test_env = train_env#ThermostatEnv(test_start_date, test_end_date)
    #model = TRPO(MlpPolicy, train_env, verbose=1)
    model = DQN(MlpPolicy, train_env, verbose=1, tensorboard_log="./dqn_thermostat_tensorboard/")
    model.learn(total_timesteps=50000)
    model.save("dqn2.pk")
    #model = DQN.load("dqn.pk")
    state = test_env.reset()
    done = False
    while done is False:
        action, _state = model.predict(state)
        state, reward, done, info = test_env.step(action)
    test_env.render()
    print(test_env.thermal_states[-1].__dict__)