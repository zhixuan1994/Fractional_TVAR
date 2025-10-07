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

class LSTMDensePINN(tf.keras.Model):
    def __init__(self, lags_r):
        super(LSTMDensePINN, self).__init__()
        self.dens1 = tf.keras.layers.Dense(16, activation='tanh')
        self.lstm1 = tf.keras.layers.LSTM(32, activation='sigmoid')
        self.out = tf.keras.layers.Dense(lags_r + 2, activation='tanh')
        
    def call(self, X):
        h = self.dens1(X)
        h = tf.expand_dims(h, axis=1)
        h = self.lstm1(h)
        u = self.out(h)
        return u    
    
def TV_AR_lagged(y_t, lags_r):
    y_out = []
    for i in range(len(y_t) - lags_r):
        y_temp = np.concatenate([[1], y_t[i: lags_r+i, :].reshape(-1,)])
        y_out.append(y_temp)
    return np.array(y_out)

def fraction_lag(lags_r, y_t, m):
    y_out = []
    for i in range(len(y_t) - m):
        y_out.append(y_t[i:m+i-lags_r-1, :].reshape(-1,))
    return np.array(y_out)

def frac_diff_weights(lags_r, d, m):
    weights = []
    for k in range(lags_r+1, m):
        k = m-k
        pi_k = gamma(k - d) / (gamma(-d) * gamma(k + 1))
        weights.append(pi_k)
    return weights

def const_d(d):
    return tf.math.reduce_variance(d)


def TV_AR_loss(y_t, m, lags_r, coef_all, alpha = 0.8, recont=False):
    right = TV_AR_lagged(y_t[m-lags_r:], lags_r)
    left = tf.cast(y_t[m:].reshape(-1,), dtype='float64')
    TV_AR_coef = coef_all[:, :-1]
    right = tf.reduce_sum(TV_AR_coef * right, axis=1)
    right = tf.cast(right, dtype='float64')
    d = tf.sigmoid(coef_all[:, -1])
    const_d_cont = const_d(d)
    const_d_cont = tf.cast(const_d_cont, dtype='float64')
    d = tf.reduce_mean(d)
    fraction_coef = frac_diff_weights(lags_r, d, m)
    fraction_coef = tf.reshape(fraction_coef, [1,-1])
    fraction_right = fraction_lag(lags_r, y_t, m)
    fraction_right = tf.reduce_sum(fraction_coef*fraction_right, axis=1)
    fraction_right = tf.cast(fraction_right, dtype='float64')
    if recont == True:
        return right*alpha + fraction_right*(1-alpha)
    else:
        return tf.reduce_mean(tf.square(left - right*alpha - fraction_right*(1-alpha))) + const_d_cont

class FTV_AR_GAM:
    def __init__(self, lag_r, series, t_list, m=50):
        self.lag_r = lag_r
        self.series_data_all = np.array(series)
        self.m = m
        self.t_list = t_list[self.m-self.lag_r:]
        self.series_data = self.series_data_all[self.m-self.lag_r:]
        
    def frac_diff_weights(self, lags_r, d, m):
        weights = []
        for k in range(lags_r+1, m):
            k = m-k
            pi_k = gamma(k - d) / (gamma(-d) * gamma(k + 1))
            weights.append(pi_k)
        return np.array(weights)

    def all_basis_func(self, t, k_order=3):
        t = 2*(t-0.5)
        t = t.reshape(-1,1)
        # Linear with order
        linear_list = t
        for linear_order in range(2, k_order):
            temp = (t)**linear_order
            linear_list = np.concatenate([linear_list, temp], axis=1)
        tanh = (np.exp(5*t)-np.exp(-5*t)) / (np.exp(5*t)+np.exp(-5*t))
        gaus = np.exp(-5*t**2)
        trad_exp = np.exp(t/5)
        trad_cos = np.cos(t/5)
        return np.concatenate([linear_list, gaus, trad_exp, tanh, trad_cos], axis=1)

    def TV_VAR_lagged(self, series_data, linear_max_order=3):
        t_train = []
        for i in range(self.lag_r):
            t_train.append(self.t_list[i:-self.lag_r+i])
        t_train = np.transpose(np.array(t_train))
        t_train_processed = self.all_basis_func(t_train[:, -1] + self.t_list[1]-self.t_list[0])
        for i in range(t_train.shape[1]):
            t_processed_temp = self.all_basis_func(t_train[:,i], k_order=linear_max_order)
            t_processed_kroned = np.array(np.kron(series_data[i], t_processed_temp[0])).reshape(1,-1)
            for j in range(1, t_train.shape[0]):
                t_kroned_temp = np.kron(series_data[i+j], t_processed_temp[j]).reshape(1,-1)
                t_processed_kroned = np.concatenate([t_processed_kroned, t_kroned_temp], axis=0)
            t_train_processed = np.concatenate([t_train_processed, t_processed_kroned], axis=1)
        return t_train_processed
    
    def fraction_lag(self, lags_r, y_t, m):
        y_out = []
        for i in range(len(y_t) - m):
            y_out.append(y_t[i:m+i-lags_r-1].reshape(-1,))
        return np.array(y_out)
    
    def linear_minimize_obj(self, x, A, y):
        x = x.reshape(-1,1)
        AR_x = x[:-1]
        TVAR_res = A.dot(AR_x)
        d = x[-1]
        fraction_coef = self.frac_diff_weights(self.lag_r, d, self.m)
        fraction_coef = fraction_coef.reshape(1,-1)
        fraction_right = self.fraction_lag(self.lag_r, self.series_data_all, self.m)
        fraction_right = np.sum(fraction_coef*fraction_right, axis=1)
        return LA.norm(TVAR_res + fraction_right.reshape(-1,1) - y)
    
    def GAM_rection(self):
        TV_VAR_data = self.TV_VAR_lagged(self.series_data)
        x = self.TVAR_x.reshape(-1,1)
        AR_x = x[:-1]
        TVAR_res = TV_VAR_data.dot(AR_x)
        d = x[-1]
        fraction_coef = self.frac_diff_weights(self.lag_r, d, self.m)
        fraction_coef = fraction_coef.reshape(1,-1)
        fraction_right = self.fraction_lag(self.lag_r, self.series_data_all, self.m)
        fraction_right = np.sum(fraction_coef*fraction_right, axis=1)
        return TVAR_res + fraction_right.reshape(-1,1)
    
    def pred_lag(self, series, time_t, t_h):
        TV_AR_data = self.all_basis_func(time_t+t_h).reshape(-1,)
        series = np.concatenate([[1], series])
        return np.kron(series, TV_AR_data).reshape(-1,)

    def GAM_predict(self, predict_step):
        AR_x = self.TVAR_x.reshape(-1,1)[:-1]
        d = self.TVAR_x.reshape(-1,1)[-1]
        fraction_coef = self.frac_diff_weights(self.lag_r, d, self.m)
        frac_data = self.series_data[-self.m-1:].reshape(-1,)
        TV_AR_data = self.series_data[-self.lag_r:]
        t_h = self.t_list[1]- self.t_list[0]
        pred_out = []
        current_time = self.t_list[-1]
        for _ in range(predict_step):
            fraction_right = self.fraction_lag(self.lag_r, frac_data[-self.m-1:], self.m)
            fraction_right = np.sum(fraction_coef*fraction_right)
            TV_AR_data = TV_AR_data[-self.lag_r:].reshape(-1)
            TV_AR_lagged_data = self.pred_lag(TV_AR_data, current_time, t_h)
            TVAR_res = TV_AR_lagged_data.dot(AR_x)
            current_time = current_time + t_h
            out = TVAR_res.reshape(-1,) + fraction_right.reshape(-1,)
            frac_data = np.concatenate([frac_data, out])
            TV_AR_data = np.concatenate([TV_AR_data, out])
            pred_out.append(out)
        return pred_out

    def FTV_AR_GAM(self):
        TV_VAR_data = self.TV_VAR_lagged(self.series_data)
        TV_VAR_target = self.series_data[self.lag_r:]
        bounds = [(None, None)] * TV_VAR_data.shape[1] + [(0.0, 1.0)]
        TV_VAR_x = minimize(self.linear_minimize_obj, np.ones(shape=TV_VAR_data.shape[1] + 1)/2, 
                            method='trust-constr', args = (TV_VAR_data, TV_VAR_target.reshape(-1,1)),
                            bounds=bounds).x
        self.TVAR_x = TV_VAR_x
    



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
print(mean_absolute_error(acf_true, acf_GAM))