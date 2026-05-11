import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from sklearn.linear_model import LinearRegression

density = np.array([4, 10, 30, 90, 173, 256])
eaten = np.array([2.5, 9.5, 12.5, 19.5, 21.5, 19.0] )




def mm_kinetic(v_max, d_mid, d_now):
    return v_max*(d_now/(d_mid + d_now))


# Lineweaver-Burke transform
## v = v_max*d/(d_mid + d)
## -> 1/v = (d_mid + d) /d + 1/v_max

x_lb = 1/density
y_lb = 1/eaten
model_lb = model_lb = LinearRegression().fit(x_lb.reshape(-1, 1), y_lb)

v_max_lb = 1 / model_lb.intercept_
d_mid_lb = model_lb.coef_[0] * vmax_lb
print(f"LB: Vmax = {v_max_lb:.2f}, Km = {d_mid_lb:.2f}")

fit_lb = mm_kinetic(v_max_lb, d_mid_lb, density)
print(fit_lb)

# Eadie-Hofstee transform
x_eh = eaten
y_eh = eaten/density

model_eh = LinearRegression().fit(x_eh.reshape(-1, 1), y_eh)




# plot
plt.plot(density, fit_lb, marker = "o", label = "Lineweaver-Burke")
plt.plot(density, eaten, label = "data", marker='x', linestyle='none', color = "red")
plt.legend()
plt.xlabel("density")
plt.ylabel("prey eaten")
plt.show()




