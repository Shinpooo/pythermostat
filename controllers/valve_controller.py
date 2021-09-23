

class ValveController:
    """
    Implements a valve controller
    """

    def __init__(self, env):
        """
        :param: env: a thermostat environment
        """
        self.env = env

    @staticmethod
    def name():
        return "Valve"

    def train(self):
        pass

    def simulate(self):
        state = self.env.reset()
        cumulative_reward = 0.0
        done = False
        P_consumed = []
        while not done:
            if self.env.thermal_states[-1].T_set + 2 > self.env.thermal_states[-1].T_indoor:
                action = self.env.P_capacity*(0.3/5)
            else:
                action = 0 
            state, reward, done, info = self.env.step(action)
            P_consumed.append(action)
            cumulative_reward += reward
        print("MSE Setpoint- realized: %.3f - Energy consumed: %.2f"% (cumulative_reward, sum(P_consumed)))
        result_folder = "results/" + self.name() + "/" + self.env.start_date.strftime("%m-%d-%Y") + \
            "_to_" + self.env.end_date.strftime("%m-%d-%Y")
        self.env.store_and_plot(result_folder)

    def set_env(self, env):
        self.env = env
