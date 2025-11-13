import tensorflow as tf
from scipy.optimize import minimize
import numpy as np
from math import gamma
import numpy.linalg as LA
from statsmodels.tsa.stattools import acf
from scipy.stats import ks_2samp
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.ar_model import AutoReg
from tqdm import tqdm
from PINN_GAM_alg import LSTMDensePINN
from PINN_GAM_alg import FTV_AR_GAM

# Simulation
def frac_diff_gent(d, m):
    w = np.zeros(m + 1)
    w[0] = 1.0
    for k in range(1, m + 1):
        w[k] = w[k - 1] * (d - k + 1) / k
    return w

def simulate_ftvar_stock(T=1000, d=0.8, m=100, burnin=500, sigma=0.05, seed=42):
    total = T + burnin
    w = frac_diff_gent(d, m)
    # a1 = 0.5 + np.cos(2 * np.pi * np.arange(total) / T)*0.5
    a1 = 1 - 1*np.arange(total) / T
    x = np.zeros(total)
    eps = np.random.normal(0, sigma, total)
    
    for t in range(total):
        L = sum(w[j] * eps[t-j] for j in range(1, min(m, t)+1))
        AR = a1[t] * x[t-1] if t >= 1 else 0
        x[t] = AR + eps[t] + L
    return np.cumsum(x[burnin:])

PINNs_pred, GAM_pred, AR_pred, orig_pred = [], [], [], []
for _ in tqdm(range(50)):
    T = 1000
    price = simulate_ftvar_stock(T=T)
    temp_price = price[1:]- price[:-1]
    orig_pred.append(temp_price)

    # PINNs
    t_list = np.arange(0, len(price)-1, 1).reshape(-1,1)
    m = 50
    lags_r = 1
    model = LSTMDensePINN(lags_r=lags_r)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    epochs = 800
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            coef_all = model(t_list[m:])
            loss = TV_AR_loss(temp_price.reshape(-1,1), m, lags_r, coef_all)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    coef_all = model(t_list[m:])
    rect = TV_AR_loss(temp_price.reshape(-1,1), m, lags_r, coef_all, recont=True)
    PINNs_pred.append(rect)

    # GAM
    GAM_based = FTV_AR_GAM(lags_r, temp_price, np.arange(0, len(temp_price), 1)/len(temp_price))
    GAM_based.FTV_AR_GAM()
    rect = GAM_based.GAM_rection()
    GAM_pred.append(rect.reshape(-1,))

    # AR
    p = lags_r
    model = AutoReg(temp_price, lags=p)
    res = model.fit()

    X_reconstructed = np.zeros_like(temp_price)
    X_reconstructed[:p] = temp_price[:p]
    for t in range(p, len(temp_price)):
        X_reconstructed[t] = np.dot(res.params[1:], temp_price[t-p:t][::-1]) + res.params[0]  # intercept + lagged
    AR_pred.append(X_reconstructed)

PINNs_pred = np.array(PINNs_pred)
GAM_pred = np.array(GAM_pred)
AR_pred = np.array(AR_pred)
orig_pred = np.array(orig_pred)
PINNs_pred = np.array(PINNs_pred)
GAM_pred = np.array(GAM_pred)
AR_pred = np.array(AR_pred)
orig_pred = np.array(orig_pred)

print(mean_absolute_error(PINNs_pred, orig_pred[:, 50:]))
print(mean_absolute_error(GAM_pred, orig_pred[:, 50:]))
print(mean_absolute_error(AR_pred, orig_pred))

K = 50
PINNs_acf, GAM_acf, AR_acf = [], [], []
PINNs_ks, GAM_ks, AR_ks = [], [], []
for i in range(len(PINNs_pred)):
    acf_true = acf(orig_pred[i], nlags=K, fft=True)
    acf_PINN = acf(PINNs_pred[i], nlags=K, fft=True)
    acf_GAM = acf(GAM_pred[i], nlags=K, fft=True)
    acf_AR = acf(AR_pred[i], nlags=K, fft=True)

    PINNs_acf.append(mean_absolute_error(acf_PINN, acf_true))
    GAM_acf.append(mean_absolute_error(acf_GAM, acf_true))
    AR_acf.append(mean_absolute_error(acf_AR, acf_true))

    ks_PINN = orig_pred[i][50:] - PINNs_pred[i]
    ks_GAM = orig_pred[i][50:] - GAM_pred[i]
    ks_AR = orig_pred[i] - AR_pred[i]

    PINNs_ks.append(ks_2samp(np.random.normal(0, 0.1, len(orig_pred[i])), ks_PINN))
    GAM_ks.append(ks_2samp(np.random.normal(0, 0.1, len(orig_pred[i])), ks_GAM))
    AR_ks.append(ks_2samp(np.random.normal(0, 0.1, len(orig_pred[i])), ks_AR))

print(np.mean(PINNs_acf))
print(np.mean(GAM_acf))
print(np.mean(AR_acf))


# Real-world Application
import pandas as pd
data1 = pd.read_csv('VIX_History.csv')
price1 = data1['OPEN'].to_numpy()
data2 = pd.read_csv('sp500_2001.csv')
price2 = data2.iloc[3642:5907]
price2 = price2['Open'].to_numpy()

temp_price = np.log(price2[1:]) - np.log(price2[:-1])
temp_price = temp_price**2
temp_price = (temp_price - np.min(temp_price)) / (np.max(temp_price) - np.min(temp_price))

temp_vix = (price1 - np.min(price1)) / (np.max(price1) - np.min(price1))

t_list = np.arange(0, len(temp_price), 1).reshape(-1,1)
m = 50
lags_r = 1
model = LSTMDensePINN(lags_r=lags_r)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
epochs = 800
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        coef_all = model(t_list[m:])
        loss = TV_AR_loss(temp_price.reshape(-1,1), m, lags_r, coef_all)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
coef_all = model(t_list[m:])
rect_PINNs = TV_AR_loss(temp_price.reshape(-1,1), m, lags_r, coef_all, recont=True)
rect_PINNs = tf.nn.relu(rect_PINNs)

GAM_based = FTV_AR_GAM(lags_r, temp_price, np.arange(0, len(temp_price), 1)/len(temp_price), m=100)
GAM_based.FTV_AR_GAM()
rect_GAM = GAM_based.GAM_rection()
rect_GAM = rect_GAM.reshape(-1,)

print(mean_absolute_error(temp_price[50:], rect_PINNs))
print(mean_absolute_error(temp_price[100:], rect_GAM))
K = 50
acf_true = acf(temp_price[50:], nlags=K, fft=True)
acf_PINNs = acf(rect_PINNs, nlags=K, fft=True)
acf_GAM = acf(rect_GAM, nlags=K, fft=True)

print(mean_absolute_error(acf_true, acf_PINNs))
