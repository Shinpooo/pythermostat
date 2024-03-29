import logging
from pyomo.environ import *
from pyomo.opt import SolverFactory
import time


logger = logging.getLogger(__name__)  #: Logger


class OptController:
    """
    Implements an optimizaton controller
    """

    def __init__(self, env):
        """
        :param: env: a thermostat environment
        """
        self.env = env
        self.model = None
        self.control_horizon = len(self.env.date_range)

    @staticmethod
    def name():
        return "Opt"

    def simulate(self):
        state = self.env.reset()
        cumulative_reward = 0.0
        P_consumed = []
        done = False
        #while not done:
        self._create_model()
        logger.info("SOLVING: " + str(self.env.thermal_states[-1].date_time))
        # Take the optimized low level actions #
        p_heat_opt = self.get_optimal_action()
        # self.model.T_i.display()
        # self.model.T_s.display()
        for p_opt in p_heat_opt:
            logger.info("simulating for " + str(self.env.thermal_states[-1].date_time))
            next_state, reward, done, info = self.env.step(action=p_opt)  # Run the simulator in continuous mode (low_level_action)
            cumulative_reward += reward
            P_consumed.append(p_opt)
        print("MSE Setpoint- realized: %.3f - Energy consumed: %.2f"%
                  (cumulative_reward, sum(P_consumed)))
        result_folder = "results/" + self.name() + "/" + self.env.start_date.strftime("%m-%d-%Y") + \
            "_to_" + self.env.end_date.strftime("%m-%d-%Y")
        self.env.store_and_plot(result_folder)

    def _create_model(self):
        """
        Define the main elements of the optimization problem
            * the sets
            * the variables
            * the parameters
            * the constraints
            * the objective
        Create and update the model. Return nothing.
        """
        t_build = time.time()
        self.model = ConcreteModel()
        self._create_sets()
        self._create_parameters()
        self._create_variables()
        self._create_constraints()
        self._create_objective()

        t_build = time.time() - t_build

        logger.debug(
            "Time spent building the mathematical program: %gs" % t_build)

    def _create_sets(self):
        """
        Create the sets of the optimization model.
        Update the model.
        """
        self.model.Periods = RangeSet(self.control_horizon)  # The number of periods given to the optimization problem
        #self.model.SimulationPeriods = RangeSet(self.simulation_horizon)  # The number of  optimized actions steps that will be simulated

    def _create_parameters(self):
        """
        Create the parameters of the optimization model.
        Update the model.
        """
        T_ambiant = {p: Ta for (p, Ta) in zip(
            self.model.Periods, self.env.database.get_column("Ta", dt_from=self.env.date_range[0], dt_to=self.env.date_range[-1]))}
        T_set = {p: Ts for (p, Ts) in zip(
            self.model.Periods, self.env.database.get_column("Ts", dt_from=self.env.date_range[0], dt_to=self.env.date_range[-1]))}
        self.model.T_a = Param(self.model.Periods, initialize=T_ambiant)
        self.model.T_s = Param(self.model.Periods, initialize=T_set)
        self.model.Rei = Param(initialize=self.env.thermal_model.Rie)
        self.model.Ci = Param(initialize=self.env.thermal_model.Ci)
        self.model.Rih = Param(initialize=self.env.thermal_model.Rih)
        self.model.Ria = Param(initialize=self.env.thermal_model.Ria)
        self.model.Ce = Param(initialize=self.env.thermal_model.Ce)
        self.model.Rea = Param(initialize=self.env.thermal_model.Rea)
        self.model.Ch = Param(initialize=self.env.thermal_model.Ch)
        self.model.Te0 = Param(initialize=self.env.thermal_states[-1].T_envelope)
        self.model.Ti0 = Param(initialize=self.env.thermal_states[-1].T_indoor)
        self.model.Th0 = Param(initialize=self.env.thermal_states[-1].T_heater)
        


    def _create_variables(self):
        """
        Create the variables of the optimization model.
        Update the model.
        """
        self.model.T_i = Var(self.model.Periods)
        self.model.T_e = Var(self.model.Periods)
        self.model.T_h = Var(self.model.Periods)
        self.model.P_h = Var(self.model.Periods, within=NonNegativeReals)
 

    def _create_constraints(self):
        """
        Create the constraints of the optimization model.
        Update the model.
        """

        def T_indoor(m, p):
            # Indoor temperature model
            DT = self.env.time_step
            if p == 1:
                return m.T_i[p] == self.env.Ti0
            else:
                    return m.T_i[p] == ((m.T_e[p-1] - m.T_i[p-1])/(m.Rei*m.Ci) + (m.T_h[p-1] - m.T_i[p-1])/(m.Rih*m.Ci) + (m.T_a[p-1] - m.T_i[p-1])/(m.Ria*m.Ci))*DT + m.T_i[p-1]

        def T_envelope(m, p):
            # Envelope temperature model
            DT = self.env.time_step
            if p == 1:
                return m.T_e[p] == self.env.Te0
            else:
                return m.T_e[p] == ((m.T_i[p-1] - m.T_e[p-1])/(m.Rei*m.Ce) + (m.T_a[p-1] - m.T_e[p-1])/(m.Rea*m.Ce))*DT + m.T_e[p-1]

        def T_heater(m, p):
            DT = self.env.time_step
            # Off-grid assumption: The grid import is lost load and the import export is curtailment
            if p == 1:
                return m.T_h[p] == self.env.Th0
            else:
                return m.T_h[p] == ((m.T_i[p-1] - m.T_h[p-1])/(m.Rih*m.Ch) + (m.P_h[p-1])/m.Ch)*DT + m.T_h[p-1]
        
        def P_capacity(m, p):
            # heating power capacity constraint
            return m.P_h[p] <= self.env.P_capacity



        self.model.T_indoor_cstr = Constraint(
            self.model.Periods, rule=T_indoor)
        self.model.T_envelope_cstr = Constraint(
            self.model.Periods, rule=T_envelope)
        self.model.T_heater_cstr = Constraint(
            self.model.Periods, rule=T_heater)
        self.model.P_capacity_cstr = Constraint(self.model.Periods, rule=P_capacity)


    def _create_objective(self):
        """
        Create the objective function of the optimization model.
        Update the model.
        """

        def total_error(m):
            # Obj: MSE between set_temperature and realized one
            error = 0
            for p in range(1, self.control_horizon):
                error += 0.00001*m.P_h[p]
                if m.T_s[p] == 23:
                    error += (m.T_i[p+1] - m.T_s[p])**2
                # else:
                #     error += m.P_h[p]
            #error = sum((m.T_i[p+1] -  m.T_s[p])**2  for p in range(1, self.control_horizon))
            return error

        self.model.objFct = Objective(rule=total_error, sense=minimize)

    def get_optimal_action(self):
        """
        Solve the optimization problem.
        Return a list of optimized heating power.
        """
        #variables = dict(p_heater = [], T_indoor=[], T_heater = [], T_envelope = [])
        solver = SolverFactory("gurobi")
        results = solver.solve(self.model)
        p_heat_opt = []
        for p in range(1, self.control_horizon):
            p_heat_opt.append(value(self.model.P_h[p]))
            # variables["p_heater"].append(value(self.model.P_h[p]))
            # variables["T_indoor"].append(value(self.model.T_i[p]))
            # variables["T_heater"].append(value(self.model.T_h[p]))
            # variables["T_envelope"].append(value(self.model.T_e[p]))
        return p_heat_opt
