import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

import cvxpy as cvx # pip install cvxpy
from sklearn.covariance import LedoitWolf
import os

from Strategy_test import strategy_metrics

# reload Strategy_test module to ensure latest changes are applied
# import importlib
# import Strategy_test
# importlib.reload(Strategy_test)
# from Strategy_test import strategy_metrics 
# strategy = strategy_metrics(Daily_bar)
# sharpe_ratio = strategy.sharpe_ratio
# ret = strategy.calculate_returns()
# port_ret = strategy.portfolio

Daily_bar = pd.read_pickle("price_1d_20210101_20250817.pk")
print(Daily_bar.shape)

# outline the project
# signal generation
# 1. data preprocessing
# 2. feature engineering
# 3. signal generation
# 4. backtesting
# 5. portfolio optimization
# machine learning
# 1. data preprocessing
# 2. feature engineering
# 3. model training
# time series analysis
# 1. data preprocessing
# 2. feature engineering
# 3. model training
# 4. model evaluation
# 5. model deployment
# find multiple simple strategies
# 1. momentum
# 2. mean reversion
# 3. trend following
# 4. pairs trading
# 5. volatility arbitrage
# combine them with convex optimization

""" too slow """
# def convex_opt_t(i, constraints, tc_penalty = 1/100, comm_bps = 1e-4, ret = ret):
#     def get_tracking_error(w,ideal,sigma):    
#         tracking_error = cvx.quad_form(w,sigma) - 2 * ideal @ sigma @ w  
#         return tracking_error
#     def get_tcost(w,w_prev,comm_bps=1e-4,tc_penalty=1/100.):
#         tcost = tc_penalty*cvx.sum(cvx.abs(w - w_prev)*comm_bps) # only commission of 1bp
#         return tcost


#     ideal = port_weights.iloc[i].fillna(0)
#     w_prev = port_weights.iloc[i-1].fillna(0)
#     sigma_lw = LedoitWolf().fit(ret.fillna(0).values).covariance_
#     w = cvx.Variable(ret.shape[1])

#     tracking_error = get_tracking_error(w,ideal,sigma_lw)
#     tcost = get_tcost(w,w_prev, comm_bps, tc_penalty)

#     objective_tc = cvx.Minimize(tracking_error + tcost)
#     prob = cvx.Problem(objective_tc,constraints)
#     prob.solve()
#     w_cons = pd.Series(w.value,index=ideal.index)
#     return w_cons

# w_cons_list = []
# constraints = []
# fully_invested = cvx.sum(w) == 1
# constraints.append(fully_invested)
# long_only = w >= 0
# constraints.append(long_only)

# for i in range(1, port_weights.shape[0]):
#     w_cons = convex_opt_t(i, constraints)
#     w_cons_list.append(w_cons)
# w_cons_df = pd.DataFrame(w_cons_list, index=port_weights.index[2:])

# NOT WORKING FOR OSQP solver
# def get_tcost(w,w_prev, price_i, comm_bps=1e-4,tc_penalty=1/100.):
#     tcost = tc_penalty * cvx.sum(cvx.matmul(cvx.abs(w - w_prev), comm_bps / price_i))
#     return tcost
""" too slow """

""" convex opt accidentally create only sum(w) == 1 and very high sharpe """
# # only fully_invested constraints 
# # increase tc_penalty you trade less
# comm_bps = 35 * 1e-4
# tc_penalty = 0.2/100.
# # static covariance matrix
# sigma_lw = LedoitWolf().fit(ret.fillna(0).values).covariance_

# w_prev = port_weights.iloc[0]
# w_cons_list = [w_prev]
# for i in range(1, port_weights.shape[0]):
#     ideal = port_weights.iloc[i]
#     # sigma_lw = LedoitWolf().fit(ret.iloc[:i].fillna(0).values).covariance_
#     w = cvx.Variable(ret.shape[1])

#     tracking_error = get_tracking_error(w,ideal,sigma_lw)
    
#     # estimates tcost
#     price_i = Daily_bar['Adj Close'].iloc[i]
#     est_comm_bps = (comm_bps / price_i).mean()
#     tcost = get_tcost(w, w_prev, est_comm_bps, tc_penalty)

#     # fully_invested constraints
#     constraints = []
#     fully_invested = cvx.sum(w) == 1
#     constraints.append(fully_invested)
#     # long_only = w >= 0
#     # constraints.append(long_only)

#     objective_tc = cvx.Minimize(tracking_error + tcost)
#     prob = cvx.Problem(objective_tc,constraints)
#     prob.solve(warm_start=True)
#     w_cons = pd.Series(w.value,index=ideal.index)
#     w_prev = w_cons
#     print(f"Optimized weights for time {i}:")
#     w_cons_list.append(w_cons)

# port_weights_optimized = pd.DataFrame(w_cons_list, index=port_weights.index)
# strategy.summary_stats(port_weights_optimized)
""" convex opt accidentally create only sum(w) == 1 and very high sharpe """

""" convex opt with long short fully invested constraints """ 
# # tcost and tracking error
# def get_tracking_error(w,ideal,sigma):
#     param = cvx.Parameter(shape=sigma.shape, value=sigma, PSD=True)
#     tracking_error = cvx.quad_form(w,param) - 2 * ideal @ sigma @ w
#     return tracking_error

# def get_tcost(w,w_prev,comm_bps=1e-4,tc_penalty=1/100.):
#     tcost = tc_penalty*cvx.sum(cvx.abs(w - w_prev)*comm_bps) # only commission of 1bp
#     return tcost
# # increase tc_penalty you trade less
# comm_bps = 35 * 1e-4
# tc_penalty = 0.2/100.
# # static covariance matrix, rolling?
# sigma_lw = LedoitWolf().fit(ret.fillna(0).values).covariance_

# w_prev = port_weights.iloc[0]
# w_cons_list = [w_prev]
# for i in range(1, port_weights.shape[0]):
#     ideal = port_weights.iloc[i]
#     # sigma_lw = LedoitWolf().fit(ret.iloc[:i].fillna(0).values).covariance_
#     w = cvx.Variable(ret.shape[1])

#     tracking_error = get_tracking_error(w,ideal,sigma_lw)
    
#     # estimates tcost with average tcost
#     price_i = Daily_bar['Adj Close'].iloc[i]
#     est_comm_bps = (comm_bps / price_i).mean()
#     tcost = get_tcost(w, w_prev, est_comm_bps, tc_penalty)

#     objective_tc = cvx.Minimize(tracking_error + tcost)
#     prob = cvx.Problem(objective_tc)
#     prob.solve(warm_start=True)

#     # impose constraints for long short fully invested
#     sign = np.sign(w.value)
#     w = cvx.Variable(ret.shape[1])
#     constraints = []
#     constraints.append(cvx.multiply(sign,w) >= 0)
#     constraints.append(sign @ w == 1)

#     tracking_error = get_tracking_error(w,ideal,sigma_lw)
#     tcost = get_tcost(w, w_prev, est_comm_bps, tc_penalty)

#     objective_tc = cvx.Minimize(tracking_error + tcost)
#     prob = cvx.Problem(objective_tc,constraints)
#     prob.solve(warm_start=True)
#     w_cons = pd.Series(w.value,index=ideal.index)
    
#     w_prev = w_cons
#     print(f"Optimized weights for time {i}:")
#     w_cons_list.append(w_cons)

# port_weights_optimized = pd.DataFrame(w_cons_list, index=port_weights.index)
# strategy.summary_stats(port_weights_optimized)

# port_opt_return = (port_weights_optimized.shift() * ret).sum(axis=1)
# port_opt_return.cumsum().plot()
""" convex opt with long short fully invested constraints """

# turnover and net returns after simple tcost
# net_ret = strategy.net_returns(port_weights, commissions_bps=1, slippage_bps=0)
# sharpe_ratio(net_ret)

def returns(data):
    ret = data['Adj Close'] / data['Adj Close'].shift() - 1
    ret = ret.iloc[1:]
    return ret

# 1 day holding period returns
def portfolio(signal, ret):
    # calculate the weights based on the signal
    weights = signal.div(signal.abs().sum(axis=1), axis=0)
    # calculate the portfolio returns
    port_ret = (weights.shift() * ret).sum(axis=1)
    return port_ret

# sharpe ratio
def sharpe_ratio(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def signal_sr(signal, ret):
    weights = signal.div(signal.abs().sum(axis=1), axis=0)
    port_ret = (weights.shift() * ret).sum(axis=1)
    sr = sharpe_ratio(port_ret)
    return sr

ret = returns(Daily_bar)

# rankrize ema 20 returns and construct a simple long only strategy 0.82sr
ema_20_signal = ret.ewm(span = 20, adjust=False).mean()
signal = ema_20_signal.rank(axis=1, pct=True)
weights = signal.div(signal.abs().sum(axis=1), axis=0)
port_ret = (weights.shift() * ret).sum(axis=1)
sr = sharpe_ratio(port_ret)


# another simple signal based on volume rank 0.95 sr
volume_signal = Daily_bar['Volume'].rank(axis=1)
v_weights = volume_signal.div(volume_signal.abs().sum(axis=1), axis=0)
v_port_ret = (v_weights.shift() * ret).sum(axis=1)
sharpe_ratio(v_port_ret)

# info = {}
# for i in range(len(ret.columns)):
#     temp = yf.Ticker(ret.columns[i]).info
#     info[ret.columns[i]] = temp
import pickle
# with open("sp500_generalinfo_all.pk", "wb") as file:
#     pickle.dump(info, file)
with open('sp500_generalinfo_all.pk', 'rb') as file:
    info = pickle.load(file)

market_cap = pd.Series({k: v['marketCap'] for k, v in info.items()})

# volume signal with price i.e dollars traded
volume_signal_v2 = (Daily_bar['Volume'] * Daily_bar['Adj Close']/market_cap).rank(axis=1)
v_weights_v2 = volume_signal_v2.div(volume_signal_v2.abs().sum(axis=1), axis=0)
v_port_ret_v2 = (v_weights_v2.shift() * ret).sum(axis=1)
sharpe_ratio(v_port_ret_v2)

test = (Daily_bar['Volume'] * Daily_bar['Adj Close']/market_cap)

# trade when above average volume
test_signal = (test > test.rolling(window=20).mean()).astype(int)
test_weights = test_signal.div(test_signal.abs().sum(axis=1), axis=0)
test_ret = (test_weights.shift() * ret).sum(axis=1)
sharpe_ratio(test_ret)

# grid search for trade above certain percent of average volume
test_grid = {}
for i in np.arange(0.01, 0.3, 0.01):
    test_signal = (test > test.rolling(window=20).mean() * (1+i)).astype(int)
    test_weights = test_signal.div(test_signal.abs().sum(axis=1), axis=0)
    test_ret = (test_weights.shift() * ret).sum(axis=1)
    sr = sharpe_ratio(test_ret)
    test_grid[i] = sr
test_grid = pd.Series(test_grid)
test_grid

test_grid.plot(figsize=(10,6))

# grid search for both above certain percent of average volume and different rolling window average
test_grid_combined = {}
for i in np.arange(0.01, 1, 0.01):
    for j in np.arange(10, 51, 10):
        test_signal = (test > test.rolling(window=j).mean() * (1+i)).astype(int)
        test_weights = test_signal.div(test_signal.abs().sum(axis=1), axis=0)
        test_ret = (test_weights.shift() * ret).sum(axis=1)
        sr = sharpe_ratio(test_ret)
        test_grid_combined[(i,j)] = sr
test_grid_combined = pd.Series(test_grid_combined)
test_grid_combined.unstack()['20']
test_grid_combined.unstack().plot(figsize=(10,6))

# signal of 20% high volume
res = {}
for i in np.arange(0.01, 1, 0.01):
    high_volume_signal = (test > test.rolling(window=20).mean() * (1+i)).astype(int)
    high_volume_weights = high_volume_signal.div(high_volume_signal.abs().sum(axis=1), axis=0)
    high_volume_ret = (high_volume_weights.shift() * ret).sum(axis=1)
    res[(i, sharpe_ratio(high_volume_ret))] = high_volume_signal.sum(1).mean()
res = pd.Series(res)
# choose value greater than 50
res = res[res > 45]

# choose 0.2 and 0.45 create two signals
high_volume_signal_02 = (test > test.rolling(window=20).mean() * (1+0.2)).astype(int)
high_volume_weights_02 = high_volume_signal_02.div(high_volume_signal_02.abs().sum(axis=1), axis=0)
high_volume_ret_02 = (high_volume_weights_02.shift() * ret).sum(axis=1)
high_volume_signal_045 = (test > test.rolling(window=20).mean() * (1+0.45)).astype(int)
high_volume_weights_045 = high_volume_signal_045.div(high_volume_signal_045.abs().sum(axis=1), axis=0)
high_volume_ret_045 = (high_volume_weights_045.shift() * ret).sum(axis=1)

# combine the two signals and calculate the sharpe ratio
# combined_signal = signal.add(volume_signal, fill_value=0)/2
# signal_sr(combined_signal, ret)

# create a mean reversion signal based on zscore of 5 day returns
mean_reversion_signal = (ret.rolling(window=5).mean() - ret.rolling(window=5).mean().rolling(window=20).mean()) / ret.rolling(window=5).std().rolling(window=20).std()
mean_reversion_signal = mean_reversion_signal.rank(axis=1, pct=True)
signal_sr(mean_reversion_signal, ret)

# grid search of mean_reversion signal rolling window
mean_reversion_grid = {}
for i in np.arange(6, 25, 1):
    mean_reversion_signal = (ret.rolling(window=5).mean() - ret.rolling(window=5).mean().rolling(window=i).mean()) / ret.rolling(window=i).std()
    mean_reversion_signal = mean_reversion_signal.rank(axis=1, pct=True)
    sr = signal_sr(mean_reversion_signal, ret)
    mean_reversion_grid[i] = sr
mean_reversion_grid = pd.Series(mean_reversion_grid)

""" EMA grid search """
# create a signal entry when diff_ema_20_norm is above 0.02 and short when above 0.05
# sr 0.970
diff_ema_20_norm = Daily_bar['Adj Close'].ewm(span = 20, adjust=False).mean()
diff_ema_20_norm = (Daily_bar['Adj Close'] - diff_ema_20_norm) /diff_ema_20_norm

# grid searched for both above and below ema
# entry when 2 percent above 20 day ema and sell short when 5 percent above
ema_port_signal = (diff_ema_20_norm > 0.02).astype(int) - 2*(diff_ema_20_norm > 0.05).astype(int)
ema_port_ret = port_ret(ema_port_signal, ret)
sharpe_ratio(ema_port_ret)

# short when below -0.03 and long when below -0.04
# sr 0.975
trend_following_signal_short = -(diff_ema_20_norm < -0.03).astype(int) + 2*(diff_ema_20_norm < -0.04).astype(int)
ema_port_ret_short = port_ret(trend_following_signal_short, ret)
sharpe_ratio(ema_port_ret_short)

ema_combined_signal = ema_port_signal + trend_following_signal_short
ema_combined_port_ret = port_ret(ema_combined_signal, ret)
sharpe_ratio(ema_combined_port_ret)

# grid serach the best threshold for the trend following strategy
sr_grid = {}
for i in np.arange(0.01, 0.1, 0.01):
    for j in np.arange(0.01, 0.1, 0.01):
        if i >= j:
            continue
        trend_following_signal = (diff_ema_20_norm > i).astype(int) - 2*(diff_ema_20_norm > j).astype(int)
        ema_weights = trend_following_signal.div(trend_following_signal.abs().sum(axis=1), axis=0)
        ema_port_ret = (ema_weights.shift() * ret).sum(axis=1)
        sr = sharpe_ratio(ema_port_ret)
        sr_grid[(i,j)] = sr
sr_grid = pd.Series(sr_grid)

sr_grid = {}
for i in np.arange(0.01, 0.1, 0.01):
    for j in np.arange(0.01, 0.1, 0.01):
        if i >= j:
            continue
        trend_following_signal = -(diff_ema_20_norm < -i).astype(int) + 2*(diff_ema_20_norm < -j).astype(int)
        ema_weights = trend_following_signal.div(trend_following_signal.abs().sum(axis=1), axis=0)
        ema_port_ret = (ema_weights.shift() * ret).sum(axis=1)
        sr = sharpe_ratio(ema_port_ret)
        sr_grid[(-i,-j)] = sr
sr_grid = pd.Series(sr_grid)

""" ema grid search """


Portfolio_comb = [ema_port_ret, ema_port_ret_short]
Portfolio_comb = pd.concat(Portfolio_comb, axis=1)
Portfolio_comb.columns = ['ema_port_ret', 'ema_port_ret_short']

def simple_optimal_weights(rets):
    """
    Calculate optimal weights for a portfolio based on the covariance matrix and mean returns.
    Uses the formula w = inv(sigma) * mu / sum(abs(inv(sigma) * mu))
    """
    def optimal_weights(sigma, mu):
        wgt = np.linalg.inv(sigma) @ mu
        wgt = wgt/np.abs(wgt).sum()
        return wgt
    sigma = rets.cov()
    mu = rets.mean()
    wgt = optimal_weights(sigma, mu)
    comb = (rets*wgt).sum(axis = 1)
    return comb
comb_ret = simple_optimal_weights(Portfolio_comb)
sharpe_ratio(comb_ret)



cum_sum = comb_ret.cumsum()
drawdown = (cum_sum / cum_sum.expanding(min_periods=1).max() - 1)
info_summary = pd.read_pickle("info_summary_20250817.pk")


test = strategy_metrics(Daily_bar)
summary = test.summary_stats(Portfolio_comb)

# rolling sharpe for the stock
rolling_sharpe_20 = ret.rolling(20).mean()/ret.rolling(20).std()* np.sqrt(252)
#rolling_sharpe_175 = ret.rolling(175).mean()/ret.rolling(175).std()* np.sqrt(252)

# rank the rolling sharpe ratio 
weights = (rolling_sharpe_20 > 1) * 1
weights = weights.div(weights.sum(axis = 1), axis = 0)

# calculate the portfolio returns based on the weights
rolling_sharpe_20_port_ret = (weights.shift() * ret).sum(axis=1)
# calculate the sharpe ratio of the portfolio
sharpe_ratio(rolling_sharpe_20_port_ret)

grid_sr = {}
for i in range(20, 252, 5):
    rolling_sharpe = ret.rolling(i).mean()/ret.rolling(i).std()* np.sqrt(252)
    weights = (rolling_sharpe > 1) * 1
    weights.div(weights.sum(axis = 1), axis = 0)
    port_ret = (weights.shift() * ret).sum(axis=1)
    sr = sharpe_ratio(port_ret)
    grid_sr[i] = sr
grid_sr = pd.Series(grid_sr)

# rolling zscore strategy
zscore_10_252 = np.sqrt(10)*(ret.rolling(10,min_periods=1).mean() - ret.rolling(365,min_periods=10).mean())
zscore_10_252 = zscore_10_252 / ret.rolling(365,min_periods=10).std()
# tanh the zscore to avoid extreme values
zscore_10_252 = np.tanh(zscore_10_252)
# rank the zscore
zscore_signal = zscore_10_252.rank(axis=1, pct=True)
zscore_weights = zscore_signal.div(zscore_signal.abs().sum(axis=1), axis=0)
zscore_port_ret = (zscore_weights.shift() * ret).sum(axis=1)
sharpe_ratio(zscore_port_ret)


Portfolio_comb = [zscore_port_ret, ema_port_ret, ema_port_ret_short, v_port_ret]
Portfolio_comb = pd.concat(Portfolio_comb, axis=1)
Portfolio_comb.columns = ['zscore_port_ret', 'ema_port_ret', 'ema_port_ret_short', 'v_port_ret']
Portfolio_comb.corr()
comb_ret = simple_optimal_weights(Portfolio_comb)
sharpe_ratio(comb_ret)

for i in range(Portfolio_comb.shape[1]):
    # exclude ith column in the portfolio
    temp_comb = Portfolio_comb.drop(Portfolio_comb.columns[i], axis=1)
    temp_comb_ret = simple_optimal_weights(temp_comb)
    sr = sharpe_ratio(temp_comb_ret)
    print(f"Sharpe ratio excluding {Portfolio_comb.columns[i]}: {sr}")

Daily_bar_adjusted_open = Daily_bar['Open'] * (Daily_bar['Adj Close']/Daily_bar['Close'])

ret_daytime = Daily_bar['Adj Close'] / Daily_bar_adjusted_open - 1
ret_overnight = (1 + ret)/ (1 + ret_daytime) - 1

# signal based on both daytime and overnight returns
strategy = strategy_metrics(Daily_bar)

from scipy.stats import pearsonr
overnight_lead_daytime = pearsonr(ret_daytime.iloc[1:], ret_overnight.iloc[1:])[0]
#ret_daytime.iloc[1:].corrwith(ret_overnight.iloc[1:])
daytime_lead_overnight = pearsonr(ret_daytime.iloc[:-1], ret_overnight.iloc[1:])[0]


test = ret_daytime.iloc[1:].corrwith(ret_overnight.iloc[1:])
test2 = ret_daytime.iloc[:-1].corrwith(ret_overnight.iloc[1:])



Daily_bar_adjusted_open = Daily_bar['Open'] * (Daily_bar['Adj Close']/Daily_bar['Close'])
ret_daytime = Daily_bar['Adj Close'] / Daily_bar_adjusted_open - 1
ret_overnight = (1 + ret)/ (1 + ret_daytime) - 1
# signal overnight leads daytime
signal_dt_on = ((ret_overnight > 0) & (ret_daytime < 0)).astype(int) - ((ret_overnight < 0) & (ret_daytime > 0)).astype(int)
dt_on_port_weights = strategy.portfolio_weights(signal_dt_on)
strategy.summary_stats(dt_on_port_weights)



# buy and hold spy
spy = yf.download("SPY", start="2021-01-01", auto_adjust=False)
spy_ret = spy/spy.shift() - 1
sharpe_ratio(spy_ret)
spy_weights = pd.DataFrame(index = spy_ret.index, data = 1, columns = ['SPY'])

strategy.summary_stats(spy_weights,ret = spy_ret)


# 5 days reversal strategy
# 0.83 sr
signal_5d_reversal_signal = (-1 * ret.rolling(5).sum()).rank(1)
signal_5d_reversal_weights = strategy.portfolio_weights(signal_5d_reversal_signal)
signal_5d_reversal_ret = (signal_5d_reversal_weights.shift() * ret).sum(axis=1)
strategy.summary_stats(signal_5d_reversal_weights)

# grid search for different day reversal
grid_sr = {}
for i in range(2, 31):
    signal = (-1 * ret.rolling(i).sum()).rank(1)
    weights = strategy.portfolio_weights(signal)
    sr = sharpe_ratio((weights.shift() * ret).sum(axis=1))
    grid_sr[i] = sr
grid_sr = pd.Series(grid_sr)

# convex opt test
ideal = port_weights.iloc[-1].fillna(0)
w_prev = port_weights.iloc[-2].fillna(0)

sigma_lw = LedoitWolf().fit(ret.fillna(0).values).covariance_
w = cvx.Variable(ret.shape[1])


def get_tracking_error(w,ideal,sigma):    
    param = cvx.Parameter(shape=sigma.shape, value=sigma, PSD=True)
    tracking_error = cvx.quad_form(w,param) - 2 * ideal @ sigma @ w  - ideal @ sigma @ ideal
    return tracking_error
tracking_error = get_tracking_error(w,ideal,sigma_lw)

def get_tcost(w,w_prev,comm_bps=1e-4,tc_penalty=1/100.):
    tcost = tc_penalty*cvx.sum(cvx.abs(w - w_prev)*comm_bps) # only commission of 1bp
    return tcost
tcost = get_tcost(w,w_prev)


objective_tc = cvx.Minimize(tracking_error + tcost)
prob = cvx.Problem(objective_tc)
prob.solve()
w_tc = pd.Series(w.value,index=ideal.index)

# fully invested and long only constraints
constraints = []
fully_invested = cvx.sum(w) == 1
constraints.append(fully_invested)
long_only = w >= 0
constraints.append(long_only)

# re-solve the problem with constraints
prob = cvx.Problem(objective_tc,constraints)
prob.solve()
w_cons = pd.Series(w.value,index=ideal.index)


# information ratio
import statsmodels.api as sm    

X = spy_ret['Adj Close'].dropna()[:1159]
Y = comb_ret_less_strat.dropna()[1:]
X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()

beta_contr = model.params['SPY']*X['SPY']
prediction = model.params['SPY']*X['SPY'] + model.params['const']
alpha_contr = model.params['const'] + model.resid
beta_contr.mean()/beta_contr.std()*np.sqrt(252)
alpha_contr.mean()/alpha_contr.std()*np.sqrt(252)
model.tvalues['const']


# ica on returns
from sklearn.decomposition import FastICA
X = ret.fillna(0)[1:].values
ica = FastICA(n_components=5, random_state=0)
independent_components = ica.fit_transform(X)

# mixing_ is the matrix that maps the independent components to the observed signals
# For stock return as observed signals, mixing is factor_loadings that maps back your independent returns driver to your rets
# we can use the best performing component to construct the weights of stocks at time T=t using t prior returns perform ICA

# grid search to get a sense of separation sr
import collections
from sklearn.decomposition import FastICA, PCA
X = ret.fillna(0)[1:].values
grid_sr_ica = collections.defaultdict(list)
for j in range(1, 20): # j as number of components
    ica = FastICA(n_components=j, random_state=0)
    independent_components = ica.fit_transform(X)
    factor_loadings = ica.mixing_
    for i in range(factor_loadings.shape[1]): # i as component index
        weights = ica.mixing_[:, i]
        weights = weights / np.abs(weights).sum()
        portfolio_returns = (ret * weights).sum(axis=1)
        grid_sr_ica[j].append((i, sharpe_ratio(portfolio_returns)))

pca = PCA(n_components=133)
pca.fit(X[:252])
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
cumulative_variance
variance_threshold = 0.95
n_components_pca = np.argmax(cumulative_variance >= variance_threshold) + 1
n_components_pca



plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance * 100, 'b-')
plt.axhline(y=variance_threshold * 100, color='r', linestyle='--', label=f'{variance_threshold*100:.0f}% Threshold')
plt.axvline(x=n_components_pca, color='g', linestyle='--', label=f'n_components = {n_components_pca}')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance (%)')
plt.title('Scree Plot: PCA on Stock Returns')
plt.legend()
plt.grid(True)
plt.show()

# visualize grid_sr_ica absolute value
import matplotlib.pyplot as plt
for j in grid_sr_ica:
    components = [x[0] for x in grid_sr_ica[j]]
    srs = [x[1] for x in grid_sr_ica[j]]
    plt.plot(components, np.abs(srs), label=f'Components: {j}')
plt.xlabel('Independent Component')
plt.ylabel('Sharpe Ratio')
plt.title('Grid Search - ICA')
plt.legend()
plt.show()

# map each independent components back to original space
weights = independent_components[:, i].reshape(-1, 1) * ica.mixing_[:, i].reshape(1, -1)

spy = strategy.get_benchmark()
spy = spy.fillna(0)
spy = spy['SPY']

# for Method 1, ica.mixing_[:, i]. I would like to have different weights for the 1160 days. I would like to use the n-1 samples to perform ICA to long the best sharpe producing loadings and short the worst sharpe producing loadings combines a weights for Day n's portfolio weight.
def pca_ica_strategy(ret, min_period = 252, pca_threshold = 0.95, ica_n_components = 10, top_n=5):
    rets = ret.fillna(0)
    Strategy_weights = pd.DataFrame(index=ret.index, columns=ret.columns)
    for i in range(min_period, ret.shape[0]+1):
        # clean training data by removing market_betas
        train_data = rets[:i]
        market_factor = spy[:i]
        market_betas = np.array([np.cov(train_data[col], market_factor)[0, 1] /
                               np.var(market_factor) for col in train_data.columns])
        
        market_neutral = train_data.values - np.outer(market_factor, market_betas)

        # reduce dimensionality performs PCA with variance captured threshold
        pca = PCA()
        pca.fit(market_neutral)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components_pca = np.argmax(cumulative_variance >= pca_threshold) + 1

        X_reduced = pca.transform(market_neutral)[:, :n_components_pca]

        if n_components_pca < ica_n_components:
            n_components_pca = ica_n_components
            print(f"Warning: n_components_pca {n_components_pca} is less than ica_n_components {ica_n_components}. Setting n_components_pca to {ica_n_components}.")
        # perform ICA on the reduced data map back ica_mixing to original space
        ica = FastICA(n_components=ica_n_components, random_state=0, max_iter=2000)
        ica.fit_transform(X_reduced)
        ica_mixing = pca.components_[:n_components_pca, :].T @ ica.mixing_

        # Component analysis and weighting logic
        component_stats = {}
        
        for j in range(ica_n_components):
            w = ica_mixing[:, j]
            w = w / (np.abs(w).sum())

            # Calculate component returns
            port_ret = (market_neutral * w).sum(axis=1)

            # Alpha score based on market corr, kurtosis (tanh to bound -1,1), and autocorr
            market_corr = np.corrcoef(port_ret, market_factor)[0, 1]
            kurtosis = np.tanh(scipy.stats.kurtosis(port_ret))
            autocorr = np.corrcoef(port_ret[:-1], port_ret[1:])[0, 1]
            
            component_stats[j] = {
                'market_corr': market_corr,
                'kurtosis': kurtosis,
                'autocorr': autocorr,
                'weights': w
            }

        # Create lists of values
        market_corrs = [abs(comp['market_corr']) for comp in component_stats.values()] # Less is better
        kurtoses = [abs(comp['kurtosis']) for comp in component_stats.values()] # More is better
        autocorrs = [max(0, comp['autocorr']) for comp in component_stats.values()] # More is better
        rank_kurtosis = scipy.stats.rankdata(kurtoses) # High kurtosis gets high rank
        rank_autocorr = scipy.stats.rankdata(autocorrs) # High autocorr gets high rank
        rank_market_corr = scipy.stats.rankdata([-x for x in market_corrs]) # Low corr gets high rank

        # Now assign the combined rank score
        for idx, j in enumerate(component_stats.keys()):
            combined_rank_score = (rank_market_corr[idx] * 0.4 +
                                rank_kurtosis[idx] * 0.3 +
                                rank_autocorr[idx] * 0.3)
            
            component_stats[j]['alpha_score'] = combined_rank_score
        
        # Select components with best alpha characteristics
        sorted_components = sorted(component_stats.items(), 
                                 key=lambda x: x[1]['alpha_score'], 
                                 reverse=True)
        
        # Take top 2-3 alpha components
        top_components = sorted_components[:top_n]
        print(f"Top components at time {i}: {[comp[0] for comp in top_components]} with scores {[comp[1]['alpha_score'] for comp in top_components]}")

        if top_components:
            # Extract alpha scores and weights
            # alpha_scores = [comp[1]['alpha_score'] for comp in top_components]
            # component_weights = [comp[1]['weights'] for comp in top_components]
            
            # # Convert alpha scores to positive weights (softmax or normalized)
            # alpha_weights = np.array(alpha_scores)
            
            # # Normalize alpha weights to sum to 1
            # alpha_weights = alpha_weights / (alpha_weights.sum())
            
            # # Weight components by their alpha scores
            # final_weights = np.zeros_like(component_weights[0])
            # for alpha_w, comp_w in zip(alpha_weights, component_weights):
            #     final_weights += alpha_w * comp_w
            
            # Final normalization
            final_weights = np.mean([comp[1]['weights'] for comp in top_components], axis=0)
            final_weights = final_weights / (np.abs(final_weights).sum())
            Strategy_weights.iloc[i-1] = final_weights
    return Strategy_weights
ica_weights = pca_ica_strategy(ret, min_period=504, pca_threshold=0.95, ica_n_components=15, top_n = 3)
ica_weights.fillna(0, inplace=True)
ica_ret = (ica_weights.shift() * ret).sum(axis=1)
strategy.summary_stats(ica_weights)

for i in range(5, 30):
    for j in range(1, i//2):
        test = pca_ica_strategy(ret, min_period=504, pca_threshold=0.95, ica_n_components=i, top_n = 3)
        strategy.summary_stats(test)

# ICA pure alpha
import scipy.stats
spy = strategy.benchmark
spy = spy.fillna(0)
spy = spy['SPY']
def pure_alpha_ica_strategy(ret, rolling_window=504, ica_n_components=15, benchmark=spy):
    """
    Focus purely on extracting non-market, non-Gaussian components
    """
    rets = ret.fillna(0)
    Strategy_weights = pd.DataFrame(0, index=ret.index, columns=ret.columns, dtype=float)
    
    for i in range(rolling_window, ret.shape[0]):
        train_data = rets[i-rolling_window:i]
        
        # Remove market factor explicitly
        market_factor = benchmark[i-rolling_window:i]
        market_betas = np.array([np.cov(train_data[col], market_factor)[0, 1] /
                               np.var(market_factor) for col in train_data.columns])
        
        market_neutral = train_data.values - np.outer(market_factor, market_betas)
        
        # Direct ICA on market-neutral returns (no PCA)
        ica = FastICA(n_components=min(ica_n_components, train_data.shape[1]), 
                     max_iter=2000, tol=1e-6, random_state=0)
        ica.fit(market_neutral)
        
        # Analyze components for alpha potential
        alpha_components = []
        
        for j in range(ica.n_components):
            weights = ica.components_[j]
            
            # Test for alpha characteristics
            component_returns = market_neutral @ weights
            
            # We want components that are:
            # 1. Not just noise (significant non-Gaussianity)
            # 2. Not correlated with known factors
            # 3. Have some structure/persistence
            
            if (abs(scipy.stats.kurtosis(component_returns)) > 1.0 and
                abs(np.corrcoef(component_returns, market_factor)[0, 1]) < 0.2):
                alpha_components.append(weights)
        
        if alpha_components:
            # Combine alpha signals
            final_weights = np.mean(alpha_components, axis=0)
            final_weights = final_weights / np.abs(final_weights).sum()
            Strategy_weights.iloc[i] = final_weights
    
    return Strategy_weights
test = pure_alpha_ica_strategy(ret, rolling_window=504, ica_n_components=15, benchmark=spy)
strategy.summary_stats(test)

grid_search_pure_ica = {}
for i in range(1, 31):
    test = pure_alpha_ica_strategy(ret, rolling_window=504, ica_n_components=i, benchmark=spy)
    grid_search_pure_ica[i] = strategy.summary_stats(test)

# plot the ir and sr results

ir_values = [v['information_ratio'] for v in grid_search_pure_ica.values()]
sr_values = [v['sharpe'] for v in grid_search_pure_ica.values()]

plt.figure(figsize=(12, 6))
plt.plot(ir_values, label='Information Ratio', marker='o')
plt.plot(sr_values, label='Sharpe Ratio', marker='o')
plt.title('Grid Search Results for Pure ICA')
plt.xlabel('Number of Components')
plt.ylabel('Ratio')
plt.xticks(range(1, 31))
plt.legend()
plt.grid()
plt.show()

# grid search for optimal number of components, top_n and bot_n, rolling_window
grid_search_results = {}
for n_components in range(10, 31, 5):
    ica_weights = pca_ica_strategy(ret, rolling_window=504, ica_n_components=n_components, top_n=n_components//4, bot_n=n_components//4)
    ica_weights.fillna(0, inplace=True)
    summ = strategy.summary_stats(ica_weights)
    grid_search_results[n_components] = summ
    print(f"Completed n_components={n_components} with sr = {summ['sharpe']}, ir={summ['information_ratio']}")

# visualize grid search results


# Find the best hyperparameters
best_params = max(grid_search_results, key=lambda x: x[1])
print(f"Best Parameters: {best_params}")

# X_from_IC0 = independent_components[:, 0].reshape(-1, 1) * ica.mixing_[:, 0].reshape(1, -1)
X_from_IC0 = independent_components[:, 0].reshape(-1, 1) * ica.mixing_[:, 0].reshape(1, -1)



# below will Reconstruct the signals if n_components == 503
# ica j_mix
test = independent_components @ ica.mixing_.T + ica.mean_
test = pd.DataFrame(test, index=ret[1:].index, columns=ret[1:].columns)
strategy.summary_stats(strategy.portfolio_weights(test))



# back test strategy with new data
ret.columns
Daily_bar_new = yf.download(tickers=ret.columns.tolist(), start="2025-08-15", auto_adjust=False)
# append new data to existing data based on curr data's date index
test = pd.concat([Daily_bar, Daily_bar_new], axis=0)
test = test[~test.index.duplicated(keep='last')]

ret_backtest = Daily_bar_new['Adj Close'] / Daily_bar_new['Adj Close'].shift() - 1

def rebalance(curr_weights, ideal_weights):
    # Rebalance the portfolio
    return ideal_weights
