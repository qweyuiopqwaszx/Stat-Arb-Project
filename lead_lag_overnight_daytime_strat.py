import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from scipy.linalg import eigh
import warnings
warnings.filterwarnings('ignore')
from Strategy_test import strategy_metrics


Daily_bar = pd.read_pickle("price_1d_20210101_20250817.pk")
strategy = strategy_metrics(Daily_bar)
ret = strategy.calculate_returns()
Daily_bar_adjusted_open = Daily_bar['Open'] * (Daily_bar['Adj Close']/Daily_bar['Close'])
ret_daytime = Daily_bar['Adj Close'] / Daily_bar_adjusted_open - 1
ret_overnight = (1 + ret)/ (1 + ret_daytime) - 1


def construct_lead_lag_matrix_fast(ret_lead, ret_lag, window=60):
    """
    FAST version: Construct lead-lag correlation matrix using vectorized operations
    """
    # Use the most recent window of data
    lead_data = ret_lead.iloc[-window:].values.T  # shape: (n_stocks, window)
    lag_data = ret_lag.iloc[-window:].values.T    # shape: (n_stocks, window)
    
    n_stocks = lead_data.shape[0]
    
    # Remove stocks with too many NaN values
    valid_mask = (~np.isnan(lead_data)).sum(axis=1) > window * 0.8
    lead_data = lead_data[valid_mask]
    lag_data = lag_data[valid_mask]
    valid_stocks = ret_lead.columns[valid_mask]
    n_valid = len(valid_stocks)
    
    # Precompute means and standard deviations
    lead_mean = np.nanmean(lead_data, axis=1, keepdims=True)
    lag_mean = np.nanmean(lag_data, axis=1, keepdims=True)
    
    lead_std = np.nanstd(lead_data, axis=1, keepdims=True)
    lag_std = np.nanstd(lag_data, axis=1, keepdims=True)
    
    # Replace NaN with mean for correlation calculation
    lead_data_norm = np.where(np.isnan(lead_data), lead_mean, lead_data)
    lag_data_norm = np.where(np.isnan(lag_data), lag_mean, lag_data)
    
    # Standardize
    lead_data_std = (lead_data_norm - lead_mean) / (lead_std + 1e-10)
    lag_data_std = (lag_data_norm - lag_mean) / (lag_std + 1e-10)
    
    # Vectorized correlation calculation
    M = np.dot(lead_data_std, lag_data_std.T) / (window - 1)
    
    # Create full matrix with zeros for invalid stocks
    M_full = np.zeros((n_stocks, n_stocks))
    valid_indices = np.where(valid_mask)[0]
    for i, idx_i in enumerate(valid_indices):
        for j, idx_j in enumerate(valid_indices):
            M_full[idx_i, idx_j] = M[i, j]
    
    return M_full

def d_le_sc_clustering_fast(A, n_iterations=5):
    """
    Optimized d-LE-SC algorithm
    """
    n = A.shape[0]
    eta = 0.1
    
    # Remove isolated nodes (stocks with very low connectivity)
    row_sums = A.sum(axis=1)
    col_sums = A.sum(axis=0)
    active_mask = (row_sums > np.percentile(row_sums, 25)) | (col_sums > np.percentile(col_sums, 25))
    A_active = A[active_mask][:, active_mask]
    n_active = A_active.shape[0]
    
    if n_active < 10:  # Not enough stocks for meaningful clustering
        return np.array([]), np.array([])
    
    for iteration in range(n_iterations):
        try:
            # Compute Hermitian matrix
            log_term = np.log((1 - eta) / (eta + 1e-10))
            H = (1j * log_term * (A_active - A_active.T) + 
                 np.log(1 / (4 * eta * (1 - eta) + 1e-10)) * (A_active + A_active.T))
            
            # Use eigh for Hermitian matrices (faster and more stable)
            eigvals, eigvecs = eigh(H)
            v1 = eigvecs[:, -1]  # Largest eigenvalue
            
            # Create embedding and cluster
            embedding = np.column_stack([np.real(v1), np.imag(v1)])
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(embedding)
            labels = kmeans.labels_
            
            # Identify lead and lag clusters
            cluster_0 = np.where(labels == 0)[0]
            cluster_1 = np.where(labels == 1)[0]
            
            net_flow = (A_active[cluster_0][:, cluster_1].sum() - 
                       A_active[cluster_1][:, cluster_0].sum())
            
            if net_flow > 0:
                C_lead_active, C_lag_active = cluster_0, cluster_1
            else:
                C_lead_active, C_lag_active = cluster_1, cluster_0
            
            # Update eta
            total_flow = (A_active[C_lead_active][:, C_lag_active].sum() + 
                         A_active[C_lag_active][:, C_lead_active].sum())
            if total_flow > 0:
                flow_lead_to_lag = A_active[C_lead_active][:, C_lag_active].sum()
                flow_lag_to_lead = A_active[C_lag_active][:, C_lead_active].sum()
                eta = min(flow_lead_to_lag / total_flow, flow_lag_to_lead / total_flow)
            
        except Exception as e:
            # Fallback: simple threshold-based clustering
            net_flows = A_active.sum(axis=1) - A_active.sum(axis=0)
            median_flow = np.median(net_flows)
            C_lead_active = np.where(net_flows > median_flow)[0]
            C_lag_active = np.where(net_flows <= median_flow)[0]
            break
    
    # Map back to original indices
    active_indices = np.where(active_mask)[0]
    C_lead = active_indices[C_lead_active] if len(C_lead_active) > 0 else np.array([])
    C_lag = active_indices[C_lag_active] if len(C_lag_active) > 0 else np.array([])
    
    return C_lead, C_lag

def overnight_lead_daytime_strategy_fast(ret_overnight, ret_daytime, window=60):
    """
    FAST main strategy implementation
    """
    # Ensure alignment
    common_stocks = ret_overnight.columns.intersection(ret_daytime.columns)
    common_dates = ret_overnight.index.intersection(ret_daytime.index)
    
    ret_overnight = ret_overnight[common_stocks].loc[common_dates]
    ret_daytime = ret_daytime[common_stocks].loc[common_dates]
    
    n_dates = len(common_dates)
    portfolio_returns = []
    dates = []
    
    print(f"Processing {n_dates - window} trading days...")
    
    for t in range(window, n_dates):
        if t % 50 == 0:
            print(f"Processing day {t-window}/{n_dates-window}")
        
        # Construct lead-lag matrix (FAST version)
        M = construct_lead_lag_matrix_fast(
            ret_overnight.iloc[:t], 
            ret_daytime.iloc[:t],
            window
        )
        
        A = np.abs(M)
        
        # Cluster stocks
        C_lead, C_lag = d_le_sc_clustering_fast(A)
        
        if len(C_lead) == 0 or len(C_lag) == 0:
            portfolio_returns.append(0)
            dates.append(common_dates[t])
            continue
        
        # Generate signal from top leaders
        lead_stocks = [common_stocks[i] for i in C_lead]
        signal = ret_overnight[lead_stocks].iloc[t-1].mean()
        
        # Calculate lag scores
        lag_scores = M[C_lead][:, C_lag].sum(axis=0)
        sorted_indices = np.argsort(lag_scores)[::-1]
        
        # Select top and bottom 20%
        n_select = max(1, int(0.2 * len(C_lag)))
        top_indices = C_lag[sorted_indices[:n_select]]
        bottom_indices = C_lag[sorted_indices[-n_select:]]
        
        # Get next day returns
        next_returns = ret_daytime.iloc[t]
        
        # Calculate portfolio return
        if signal > 0:
            port_return = (next_returns.iloc[top_indices].mean() - 
                          next_returns.iloc[bottom_indices].mean())
        else:
            port_return = (next_returns.iloc[bottom_indices].mean() - 
                          next_returns.iloc[top_indices].mean())
        
        portfolio_returns.append(port_return)
        dates.append(common_dates[t])
    
    return pd.Series(portfolio_returns, index=dates)

portfolio_returns = overnight_lead_daytime_strategy_fast(ret_overnight, ret_daytime, window = 30)