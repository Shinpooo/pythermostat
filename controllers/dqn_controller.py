

import logging
from stable_baselines.deepq.policies import MlpPolicy
#from stable_baselines.common.policies import MlpPolicy
from stable_baselines import DQN, TRPO

#  TODO Add constraint: when there are several storages, prevent the simulatenous charge and discharge of different
#  batteries The only differences between optimization lookahead and simulation are the presence of tolerances on net
#  import and capacity update

logger = logging.getLogger(__name__)  #: Logger


class DqnController:
    """
    Implements an optimizaton controller
    """

    def __init__(self, env):
        """
        :param: env: a microgrid environment
        :param: control_horizon: the number of lookahead steps in the optimization model.
        :param: simulation_horizon: the number of optimized actions that will be simulated.
        :param: options_filename: the agent options filename.
        :param: save_data: flag to save or not the data.
        :param: path_to_score_experience: the path where experiences are saved.
        :param: forecast_type: the type of forecast used for the lookahead.
        :param: n_test_episodes: The number of simulated episodes (useful only with noisy forecast).
        """
        self.env = env
        self.model = DQN(MlpPolicy, env, verbose=1, tensorboard_log="./dqn_thermostat_tensorboard/")
    
    def train(self):
        self.model.learn(total_timesteps=500000)
    
    def save(self):
        self.model.save("dqn.pk")
    
    def load(self):
        self.model = None
        self.model = DQN.load("dqn.pk")


    def simulate(self):
        state = self.env.reset()
        cumulative_reward = 0.0
        done = False
        while not done:
            action, _state = self.model.predict(state)
            state, reward, done, info = self.env.step(action)
            cumulative_reward += reward
        print(cumulative_reward)
        result_folder = "results/" + self.env.start_date.strftime("%m-%d-%Y") + "_to_" + self.env.end_date.strftime("%m-%d-%Y")
        self.env.store_and_plot(result_folder)

    def set_env(self, env):
        self.env = env