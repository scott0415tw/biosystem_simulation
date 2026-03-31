#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#%%
R = np.array([0, 36, 80, 155, 210, 240]).reshape(-1, 1)
I = np.array([8.0, 3.0, 5.0, 6.5, 4.0, 2.5])
E = np.array([0.0, 0.05, 0.10, 0.50, 1.75, 1.75])

# =========================
# Q1_(a) 線性回歸
# =========================
model_I = LinearRegression().fit(R, I)
model_E = LinearRegression().fit(R, E)

a_I = model_I.intercept_
b_I = model_I.coef_[0]

a_E = model_E.intercept_
b_E = model_E.coef_[0]

print("Immigration: I(R) = {:.5f} + ({:.5f})R".format(a_I, b_I))
print("Extinction : E(R) = {:.5f} + ({:.5f})R".format(a_E, b_E))



# =========================
# plot, I(R),  E(R)
# =========================
R_plot = np.linspace(0, 300, 100)

I_fit = a_I + b_I * R_plot
E_fit = a_E + b_E * R_plot

plt.figure()
plt.scatter(R, I, label="observed I")
plt.scatter(R, E, label="observed E")
plt.plot(R_plot, I_fit, label="Fitted I(R)")
plt.plot(R_plot, E_fit, label="Fitted E(R)")
plt.xlabel("Species number (R)")
plt.ylabel("Rate")
plt.legend()
plt.title("Immigration and Extinction (linear)")
plt.show()
#%%
# =========================
# Q1_(c) equilibrium
# I(R) = E(R)
# =========================

R_star = (a_I - a_E) / (b_E - b_I)
print("Equilibrium R* =", R_star)

#%%
# =========================
# Q1_(e) 0, 500 case
# =========================
dt = 0.1
T = 250

time = np.arange(0, T, dt)

def dR(R):
    return (a_I + b_I * R) - (a_E + b_E * R)
# I(t)-E(t)

def simulate(R0):
    R = np.zeros(len(time))
    R[0] = R0
    
    for t in range(len(time) - 1):
        R[t+1] = R[t] + dt * dR(R[t])
    
    return R

# 兩種初始條件
R0_case = simulate(0)
R500_case = simulate(500)
print("ok")


#%%

plt.figure()
plt.plot(time, R0_case, label="Start from 0")
plt.plot(time, R500_case, label="Start from 500")
plt.axhline(250, linestyle='--', label="~250 species", color = "red")
plt.axhline(R_star, linestyle='--', label="Equilibrium (model)")
plt.xlabel("Time")
plt.ylabel("Species number")
plt.legend()
plt.title("Species Dynamics")
plt.show()


# %%
# =========================
# Q2_(a): I(R),  E(R)
# =========================
# new model
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


R = np.array([0, 36, 80, 155, 210, 240])

# Immigration fit
def I_model_new(R, I0, a):
    return (I0 * np.exp(-a * R))

params_I, _ = curve_fit(I_model_new, R, I, p0=[8, 0.001])
I0= params_I[0]
a = params_I[1]

# Extinction 
def E_model_new(R, b):
    return b * R**2

params_E, _ = curve_fit(E_model_new, R, E)
b = params_E[0]

print(f"I = {round(I0, 5)}*exp(-{round(a, 5)} R)")
print(f"E = {round(b, 10)}*R^2") 

# %%
R_plot = np.linspace(0, 250, 200)

I_fit = (I0 * np.exp(-0.008 * R_plot))
E_fit = b * R_plot**2

plt.figure()
plt.scatter(R, I, label="Observed I")
plt.scatter(R, E, label="Observed E")
plt.plot(R_plot, I_fit, label="Fitted I(R)")
plt.plot(R_plot, E_fit, label="Fitted E(R)")
plt.xlabel("R")
plt.ylabel("Rate")
plt.text(x=8 , y = 8, s = f"I = {round(I0,2)}*exp(-{round(a, 6)}*R)")
plt.title("Immigration and Extinction (new)")
plt.legend()
plt.show()


#%%
# =========================
# Q2_(a) simulation 
# =========================
def dR_new(R):
    return (I0 * np.exp(-a * R)) - (b * R**2)

dt = 0.1
T = 250

time = np.arange(0, T, dt)

def simulate_new(R0):
    R = np.zeros(len(time))
    R[0] = R0
    
    for t in range(len(time) - 1):
        R[t+1] = R[t] + dt * dR_new(R[t])
    
    return R

R0_new= simulate_new(0)


#%%
plt.figure()

plt.plot(time, R0_case, label="original_model")
plt.plot(time, R0_new, label="new_model")

# plt.plot(time, R500_case, label="Start from 500")
# plt.axhline(250, linestyle='--', label="~250 species", color = "red")
plt.axhline(R_star, linestyle='--', label="Equilibrium (original model)")
plt.axhline(R_star, linestyle='--', label="Equilibrium (original model)")
plt.xlabel("Time")
plt.ylabel("Species number")
plt.legend()
plt.title("Species Dynamics")
plt.show()



# %%

from sklearn.metrics import mean_squared_error

# linear model
I1_pred = model_I.predict(R.reshape(-1,1))
E1_pred = model_E.predict(R.reshape(-1,1))

rmse_I1 = np.sqrt(mean_squared_error(I, I1_pred))
rmse_E1 = np.sqrt(mean_squared_error(E, E1_pred))

# new model
I2_pred = I_model_new(R, *params_I)
E2_pred = E_model_new(R, *params_E)

rmse_I2 = np.sqrt(mean_squared_error(I, I2_pred))
rmse_E2 = np.sqrt(mean_squared_error(E, E2_pred))

print(f"Model 1 (Linear) - RMSE I: {rmse_I1:.4f}, RMSE E: {rmse_E1:.4f}")
print(f"Model 2 (New)    - RMSE I: {rmse_I2:.4f}, RMSE E: {rmse_E2:.4f}")
# %%
