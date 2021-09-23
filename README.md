# Pythermostat

A tool to evaluate thermal controllers on a given thermal model
![image](https://user-images.githubusercontent.com/31999833/134508066-33b662ae-299a-410f-b951-b4a3c86c1df2.png)

## Controllers

1) Valve controller: Heat only below threshold temperature.
2) DQN controller: A reinforcement based controller, learning when to heat through experience.
3) Opt controller: An optimized control using MILP. Maximizing comfort and minimizing energy consumption.

## Model

The model used is the dark-grey box model. More on this [here](https://medium.com/analytics-vidhya/data-driven-thermal-models-for-buildings-15385f744fc5) and [here](https://github.com/czagoni/darkgreybox).

## Install & Getting started

Using python 3.7.9, install dependencies with `pip install -r requirements.txt`. Then you should be able to run the main file and get results in the results folder.

If wou want to run Opt controller, you'll need to install a solver: gurobi (or cbc, cplex, ...)

Dqn controller uses [Stable-baselines](https://stable-baselines.readthedocs.io/en/master/guide/install.html) librabry, you may want to install external dependencies such as mpi.
