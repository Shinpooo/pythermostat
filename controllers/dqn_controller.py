

import logging
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN



logger = logging.getLogger(__name__)  #: Logger


class DqnController:
    """
    Implements an RL (DQN) controller
    """

    def __init__(self, env):
        """
        :param: env: a microgrid environment
        """
        self.env = env
        self.model = DQN(MlpPolicy, env, verbose=1, tensorboard_log="./dqn_thermostat_tensorboard/")
    
    @staticmethod
    def name():
        return "Dqn"
    
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
        P_consumed = 0.0
        done = False
        while not done:
            action, _state = self.model.predict(state)
            state, reward, done, info = self.env.step(action)
            cumulative_reward += reward
            P_consumed += action
        print("MSE Setpoint- realized: %.3f - Energy consumed: %.2f"%
              (cumulative_reward, P_consumed))
        result_folder = "results/" + self.name() + "/" + self.env.start_date.strftime("%m-%d-%Y") + "_to_" + self.env.end_date.strftime("%m-%d-%Y")
        self.env.store_and_plot(result_folder)

    def set_env(self, env):
        self.env = env
