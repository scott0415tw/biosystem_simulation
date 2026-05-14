from scipy.optimize import fmin as simplex
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize

df = pd.read_excel(r"C:\Users\USER\Desktop\biosystem_simulation\HW4\document\HW4-PROBLEM-2.xlsx")
x_time = df['Time (Day)'].values
y_16h = df['16-Hr Lighting'].values
y_24h = df['24-Hr Lighting'].values


def logistic_model(time, K, r, a):
    return K / (1 + a * np.exp(-r * time))

guess = [450, 0.3, 50]

# 16 hr
parameter_16, cov = optimize.curve_fit(logistic_model, x_time, y_16h, method='lm', p0 = guess )
print(parameter_16)
K_16h, r_16h, a_16h = parameter_16

fit_16h = logistic_model(x_time, K_16h, r_16h, a_16h)

# 24 hr

parameter_24, cov = optimize.curve_fit(logistic_model, x_time, y_24h, method='lm', p0 = guess )
print(parameter_24)
K_24h, r_24h, a_24h = parameter_24

fit_24h = logistic_model(x_time, K_24h, r_24h, a_24h)



# plot
plt.plot(x_time, y_16h, label = "16h data", marker = "o", linestyle = "none")
plt.plot(x_time, y_24h, label = "24h data", marker = "o", linestyle = "none")
plt.plot(x_time, fit_16h, label = "16h-fit")
plt.plot(x_time, fit_24h, label = "24h-fit")
plt.legend()
plt.ylabel("growth")
plt.xlabel("time")
plt.show()
