import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from Strategy_test import strategy_metrics
from joblib import Parallel, delayed


Daily_bar = pd.read_pickle("price_1d_20210101_20250817.pk")
strategy = strategy_metrics(Daily_bar)
ret = strategy.calculate_returns()
Daily_bar_adjusted_open = Daily_bar['Open'] * (Daily_bar['Adj Close']/Daily_bar['Close'])
ret_daytime = Daily_bar['Adj Close'] / Daily_bar_adjusted_open - 1
ret_overnight = (1 + ret)/ (1 + ret_daytime) - 1

log_ret_overnight = np.log1p(ret_overnight)
log_ret_daytime = np.log1p(ret_daytime)

def compute_corr(i, j, lead_data, lag_data, window):
    x = lead_data[i, :-1]
    y = lag_data[j, 1:]
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() > window * 0.8:
        return np.corrcoef(x[mask], y[mask])[0, 1]
    else:
        return 0.0

def construct_lead_lag_matrix_paper(ret_lead, ret_lag, window=60, n_jobs=20):
    # Use the most recent window of data
    lead_data = ret_lead.iloc[-window:].values.T
    lag_data = ret_lag.iloc[-window:].values.T
    n_stocks = lead_data.shape[0]
    # Remove stocks with too many NaN values
    valid_mask = (~np.isnan(lead_data)).sum(axis=1) > window * 0.8
    lead_data = lead_data[valid_mask]
    lag_data = lag_data[valid_mask]
    valid_stocks = ret_lead.columns[valid_mask]
    n_valid = len(valid_stocks)
    # Parallel computation
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(compute_corr)(i, j, lead_data, lag_data, window)
        for i in range(n_valid) for j in range(n_valid)
    )
    M = np.array(results).reshape((n_valid, n_valid))
    return M, valid_stocks

def directed_spectral_clustering_paper(M, n_clusters=2):
    # Hermitian matrix as in the paper
    eta = 0.1
    log_term = np.log((1 - eta) / (eta + 1e-10))
    H = (1j * log_term * (M - M.T) +
         np.log(1 / (4 * eta * (1 - eta) + 1e-10)) * (M + M.T))
    eigvals, eigvecs = eigh(H)
    v1 = eigvecs[:, -1]
    embedding = np.column_stack([np.real(v1), np.imag(v1)])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(embedding)
    labels = kmeans.labels_
    return labels


# Optimized function to output weights for each stock at each timestamp
def paper_lead_lag_strategy_weights(ret_overnight, ret_daytime, window=60, n_jobs=20):
    common_stocks = ret_overnight.columns.intersection(ret_daytime.columns)
    common_dates = ret_overnight.index.intersection(ret_daytime.index)
    ret_overnight = ret_overnight[common_stocks].loc[common_dates]
    ret_daytime = ret_daytime[common_stocks].loc[common_dates]
    n_dates = len(common_dates)
    weights_dict = {}
    for t in range(window, n_dates-1):
        M, valid_stocks = construct_lead_lag_matrix_paper(
            ret_overnight.iloc[:t], ret_daytime.iloc[:t], window, n_jobs=n_jobs)
        if len(valid_stocks) < 10:
            weights_dict[common_dates[t+1]] = pd.Series(0, index=common_stocks)
            continue
        labels = directed_spectral_clustering_paper(M)
        cluster_0 = np.where(labels == 0)[0]
        cluster_1 = np.where(labels == 1)[0]
        net_flow = M[cluster_0][:, cluster_1].sum() - M[cluster_1][:, cluster_0].sum()
        if net_flow > 0:
            C_lead, C_lag = cluster_0, cluster_1
        else:
            C_lead, C_lag = cluster_1, cluster_0
        w = pd.Series(0, index=valid_stocks, dtype=float)
        if len(C_lead) > 0 and len(C_lag) > 0:
            w.iloc[C_lag] = 1 / len(C_lag)
            w.iloc[C_lead] = -1 / len(C_lead)
            w = w - w.mean()
            w = w / w.abs().sum()
        # Reindex to all stocks, fill missing with 0
        w_full = pd.Series(0, index=common_stocks, dtype=float)
        w_full.loc[w.index] = w
        weights_dict[common_dates[t+1]] = w_full
        if (t-window) % 10 == 0:
            print(f"Processed {t-window+1}/{n_dates-window-1} days")
    weights_df = pd.DataFrame.from_dict(weights_dict, orient='index').sort_index()
    return weights_df


# Get weights for each stock at each timestamp
weights_df = paper_lead_lag_strategy_weights(log_ret_overnight, log_ret_daytime, window=60, n_jobs=20)

simple_ret_df = paper_lead_lag_strategy_weights(ret_overnight, ret_daytime, window=20, n_jobs=20)

# Compute portfolio returns from weights
portfolio_returns = (weights_df.shift() * log_ret_overnight.loc[weights_df.index]).sum(axis=1)
portfolio_returns.cumsum().plot(title="Paper-Style Lead-Lag Strategy Cumulative Return")
plt.show()
# weights_df contains the weights for each stock at each timestamp