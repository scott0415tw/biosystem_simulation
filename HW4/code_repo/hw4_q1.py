import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import fmin as simplex
from sklearn.linear_model import LinearRegression

density = np.array([4, 10, 30, 90, 173, 256])
eaten = np.array([2.5, 9.5, 12.5, 19.5, 21.5, 19.0] )

# mm kinetic function
def mm_kinetic(d_now, v_max, d_mid):
    return v_max*(d_now/(d_mid + d_now))

# Lineweaver-Burke transform
## v = v_max*d/(d_mid + d)
## -> 1/v = (d_mid + v_max)*1/d + 1/v_max

x_lb = 1/density
y_lb = 1/eaten
model_lb = LinearRegression().fit(x_lb.reshape(-1, 1), y_lb)

v_max_lb = 1 / model_lb.intercept_
d_mid_lb = model_lb.coef_[0] * v_max_lb
print(f"LB: Vmax = {v_max_lb:.2f}, Km = {d_mid_lb:.2f}")

fit_lb = mm_kinetic(density, v_max_lb, d_mid_lb)
print(fit_lb)

# Eadie-Hofstee transform
# v/d = -1/d_mid*v + v_max/d_mid

x_eh = eaten
y_eh = eaten/density

model_eh = LinearRegression().fit(x_eh.reshape(-1, 1), y_eh)

d_mid_eh = -1/model_eh.coef_[0]
v_max_eh = model_eh.intercept_*d_mid_eh

fit_eh = mm_kinetic(density, v_max_eh, d_mid_eh)


# Levenberg-Marquardt

guess = [v_max_eh, d_mid_eh] # use eh method as initial guess
parameter_hat, cov = optimize.curve_fit(mm_kinetic, density, eaten, method='lm', p0 = guess )

v_max_lm, d_mid_lm = parameter_hat
fit_lm = mm_kinetic(density, v_max_lm, d_mid_lm)



# Nelder-Mead simplex

# 設定標準差 sigma，這裡假設所有點的權重相同（均為 1）
sigma = np.ones(len(density))

def loss_function(params, X, Y, Err):
    # params 是一個陣列，儲存演算法正在嘗試的 [v_max, d_mid]
    v_max_guess = params[0]
    d_mid_guess = params[1]
    
    chi2 = 0.0  # 初始化誤差值 (Chi-square)
    
    # 遍歷每一個數據點，計算預測值與實際值的差距
    for n in range(len(X)):
        # 呼叫你定義的模型，帶入目前的猜測參數
        y_model = mm_kinetic(X[n], v_max_guess, d_mid_guess)
        
        # 計算殘差平方和 (Residual Sum of Squares)
        # 公式：(實際值 - 模型值)^2 / 誤差^2
        chi2 += ((Y[n] - y_model) / Err[n])**2
        
    return chi2  # 回傳總誤差，演算法會試圖讓這個數字越小越好

x0 = [20.0, 10.0]

# 執行 Simplex 演算法
# args 帶入除了 params 以外的參數 (X 數據, Y 數據, 誤差陣列)
v_max_sim, d_mid_sim = simplex(loss_function, x0, args=(density, eaten, sigma))
fit_sim = mm_kinetic(density, v_max_sim, d_mid_sim)


# plot
print(f'EH:\nV_max = {v_max_eh}, D_mid = {d_mid_eh}\n')
print(f'LB:\nV_max = {v_max_lb}, D_mid = {d_mid_lb}\n')
print(f'LM:\nV_max = {v_max_lm}, D_mid = {d_mid_lm}\n')
print(f'nelder-mead:\nV_max = {v_max_sim}, D_mid = {d_mid_sim}\n')

plt.plot(density, fit_lb, marker = "o", label = "Lineweaver-Burke")
plt.plot(density, fit_eh, marker = "o", label = "Eadie-Hofstee")
plt.plot(density, fit_lm, marker = "o", label = "Levenberg-Marquardt")
plt.plot(density, fit_sim, marker = "o", label = "Nelder-mead simplex")
plt.plot(density, eaten, label = "data", marker='x', 
         linestyle='none', color = "black")
plt.legend()
plt.xlabel("density")
plt.ylabel("prey eaten")
plt.show()