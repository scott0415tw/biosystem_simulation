#%%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

## parameters 
# sugar breakdown rate
a = 0.5

# fraction of sugar breakdown that yields alcohol
b = 1/2.0665

# fraction of sugar breakdown that yields C02
f = 0.9565/2.0665

# rate of yeast cell formation per unit breakdown of sugar
c = 0.11/2.0665      

# death rate of yeast cells per unit of alcohol  
d = 0.05

Temp_opt = 20
sigma = 8
#%%
def temperature(time):
    return 20 + 8*np.sin(2*np.pi*(time - 0.5))

def temp_factor(Temp):
    return 10 * np.exp(-0.1 * Temp)

def model(t, y):
    S, Y, A = y
    T = temperature(t)
    ft = temp_factor(T)
    
    dSdt = -a*b*S*Y - a*f*S*Y #sugar
    dYdt = a*c*ft*S*Y - d*Y #yeast
    dAdt = a*b*S*Y #alcohol
    
    return [dSdt, dYdt, dAdt]

# initial conditions
S0 = 100  # sugar
Y0 = 5   # yeast
A0 = 0

t_span = (0, 7)  # 1 week
t_eval = np.linspace(0, 7, 500)



sol = solve_ivp(model, t_span, [S0, Y0, A0], t_eval=t_eval)
#%%
# plot
plt.plot(sol.t, sol.y[0], label="Sugar")
plt.plot(sol.t, sol.y[1], label="Yeast")
plt.plot(sol.t, sol.y[2], label="Alcohol")
plt.legend()
plt.xlabel("Days")
plt.ylabel("Concentration")
plt.show()


#%%
t = np.linspace(0, 7, 500)
T = temperature(t)

plt.figure()
plt.plot(t, T)
plt.xlabel("Days")
plt.ylabel("Temperature (°C)")
plt.title("Temperature Oscillation Over 7 Days")
plt.show()
