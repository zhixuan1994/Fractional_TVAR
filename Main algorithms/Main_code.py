import tensorflow as tf
from scipy.optimize import minimize
import numpy as np
from math import gamma
import numpy.linalg as LA

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

    def TV_VAR_GAM_one_index(self):
        TV_VAR_data = self.TV_VAR_lagged(self.series_data)
        TV_VAR_target = self.series_data[self.lag_r:]
        bounds = [(None, None)] * TV_VAR_data.shape[1] + [(0.0, 1.0)]
        TV_VAR_x = minimize(self.linear_minimize_obj, np.ones(shape=TV_VAR_data.shape[1] + 1)/2, 
                            method='trust-constr', args = (TV_VAR_data, TV_VAR_target.reshape(-1,1)),
                            bounds=bounds).x
        self.TVAR_x = TV_VAR_x

    
