import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_df = pd.read_csv('./data/demo_data.csv', index_col=0, parse_dates=True)   
input_X = train_df[['Ph', 'Ta', 'Th']]
input_y = train_df['Ti']
Ph = input_X["Ph"]
Ta = input_X["Ta"]
Th_mes = input_X["Th"].values
Ti_mes = input_y.values
print(Ti_mes)
num_rec = len(Ta)
rec_duration = 1
Ti0 = 18.1375
Te0 = 18.806405711348518 #[10,25]
Th0 = 18.1375
Ci = 156.988
Ch = 2.55
Ce = 389.1556
Rie = 0.1106
Rea = 405778124793
Ria = 0.637186
Rih = 0.65

Ti = np.zeros(num_rec)
Te = np.zeros(num_rec)
Th = np.zeros(num_rec)
Ti[0] = Ti0
Te[0] = Te0
Th[0] = Th0 

for i in range(1, num_rec):
    # the model equations
    dTi = ((Te[i-1] - Ti[i-1]) / (Rie * Ci) + (Th[i-1] - Ti[i-1]) / (Rih * Ci) +
            (Ta[i-1] - Ti[i-1]) / (Ria * Ci)) * rec_duration 
    dTe = ((Ti[i-1] - Te[i-1]) / (Rie * Ce) + (Ta[i-1] - Te[i-1]) / (Rea * Ce)) * rec_duration 
    dTh = ((Ti[i-1] - Th[i-1]) / (Rih * Ch) + (Ph[i-1]) / (Ch)) * rec_duration 

    Ti[i] = Ti[i-1] + dTi 
    Te[i] = Te[i-1] + dTe 
    Th[i] = Th[i-1] + dTh

# plt.plot(Ti, label="Timodeled")
# plt.plot(Ti_mes, label="Timeasured")
plt.plot(Th, label="Thmodeled")
plt.plot(Th_mes, label="Thmeasured")
plt.show()