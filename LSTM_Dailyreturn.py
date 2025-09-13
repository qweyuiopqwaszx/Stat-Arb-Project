import pandas as pd
from Strategy_test import strategy_metrics
import numpy as np
from LSTM import StockLSTMTrainer
from sklearn.decomposition import PCA
import LSTM

# import importlib
# import LSTM
# importlib.reload(LSTM)
# from LSTM import StockLSTMTrainer

# Apply PCA before LSTM and map it back after model?
# 

Daily_bar = pd.read_pickle('price_1d_20210101_20250817.pk')

# Daily_bar = pd.read_pickle('price_1d_20180101_20250817.pk')

strategy = strategy_metrics(Daily_bar)
ret = strategy.calculate_returns()
ret = ret[1:]
ret = ret.loc[:, (ret.isnull().sum() < 1)]

# ret = ret[1:].fillna(0)

pca = PCA()
pca.fit(ret)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
n_components_pca = np.argmax(cumulative_variance >= 0.95) + 1

X_reduced = pca.transform(ret)[:, :n_components_pca]

X_reduced = pd.DataFrame(X_reduced, index=ret.index)

ret_appl = ret[['AAPL']]

# random seed
seed = 42

trainer = StockLSTMTrainer(ret,train_ratio = 0.8, sequence_length= 100, 
                           dropout_rate=0.5, batch_size=64, 
                           hidden_size=1, epochs=500, 
                           lr=0.0005, weight_decay=10e-4,
                           seed=seed, loss_fn= LSTM.SignAccuracyLoss())

trainer.train(additional_epochs=2000, early_stopping_patience=50)
# trainer.plot_train_loss()
trainer.plot_val_loss()

test_preds = trainer.predict()
test_actual = ret.iloc[-len(test_preds):]

# Manual R^2 calculation per column (stock), then average
sse_per_col = np.sum((test_preds - test_actual.values) ** 2, axis=0)
sst_per_col = np.sum((test_actual.values - np.mean(test_actual.values, axis=0)) ** 2, axis=0)
r2_per_col = 1 - (sse_per_col / sst_per_col)
avg_r2 = np.mean(r2_per_col)
print(f'Average Test R^2: {avg_r2:.6f}')

# Baseline and model MSE
mean_pred = np.mean(test_actual.values, axis=0)
mean_preds = np.tile(mean_pred, (len(test_actual), 1))
baseline_mse = np.mean((test_actual.values - mean_preds) ** 2)
model_mse = np.mean((test_actual.values - test_preds) ** 2)
print(f'Baseline MSE (predict mean): {baseline_mse:.6f}')
print(f'Model MSE: {model_mse:.6f}')

# Strategy evaluation
test_index = trainer.df.index[-len(test_preds):]
test_columns = trainer.stocks
test_preds_df = pd.DataFrame(test_preds, index=test_index, columns=test_columns)

thresh = 0.000
signals = np.where(test_preds > thresh, 1, 0) + np.where(test_preds < -thresh, -1, 0)
signals_df = pd.DataFrame(signals, index=test_index, columns=test_columns)
signals_weights = signals_df.div(signals_df.abs().sum(axis=1), axis=0)
portfolio_returns = (signals_weights.shift(1) * test_actual).sum(axis=1)
portfolio_returns.cumsum().plot()
strategy.summary_stats(signals_weights, ret = test_actual)

# Strategy2 with mean reversion on predicted mean
stock_pred_mean = test_preds_df.mean(0)
signals2 = np.where(test_preds_df > stock_pred_mean + thresh, -1, 0) + np.where(test_preds_df < stock_pred_mean - thresh, 1, 0)
signals2_df = pd.DataFrame(signals2, index=test_index, columns=test_columns)
signals2_weights = signals2_df.div(signals2_df.abs().sum(axis=1), axis=0)
portfolio_returns2 = (signals2_weights.shift(1) * test_actual).sum(axis=1)
portfolio_returns2.cumsum().plot()

# plot apple actual returns vs predicted returns
import matplotlib.pyplot as plt
    
plt.figure(figsize=(14, 7))
plt.plot(test_actual.index['AAPL'], test_actual['AAPL'], label='Actual Returns')
plt.plot(test_actual.index['AAPL'], test_preds_df['AAPL'], label='Predicted Returns', alpha=0.7)
plt.title('Actual vs Predicted Returns')
plt.legend()
plt.show()

# Save model and scaler
trainer.model.save('lstm_model.pth')


""" # LSTM with BCE """

seed = 42

trainer = StockLSTMTrainer(ret,train_ratio = 0.8, sequence_length= 100, 
                           dropout_rate=0.5, batch_size=64, 
                           hidden_size=16, epochs=500, 
                           lr=0.0005, weight_decay=10e-4,
                           seed=seed, loss_fn= LSTM.MarginSignLoss())

# seed_9_initial_state = trainer.initial_state_dict
# seed_42_initial_state = trainer.initial_state_dict

trainer.train(additional_epochs=2000, early_stopping_patience=5)
# trainer.plot_train_loss()
trainer.plot_val_loss()

test_preds = trainer.predict()
test_actual = ret.iloc[-len(test_preds):]

evaluate_signals(test_preds, test_actual, 0.001)

def evaluate_signals(test_preds, test_actual, thresh=0.001):
    # Sign accuracy and classification metrics
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

    pred_sign = (test_preds > 0).astype(int)
    true_sign = (test_actual > 0).values.astype(int)
    sign_accuracy = (pred_sign == true_sign).mean()
    print(f"Sign accuracy: {sign_accuracy:.4f}")

    # Flatten for metrics (multi-stock, multi-date)
    y_true_flat = true_sign.flatten()
    y_pred_flat = pred_sign.flatten()

    precision = precision_score(y_true_flat, y_pred_flat, zero_division=0)
    recall = recall_score(y_true_flat, y_pred_flat, zero_division=0)
    f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0)

    # For AUC-ROC, use the raw predictions if available, else use binary
    try:
        auc_roc = roc_auc_score(y_true_flat, test_preds.flatten())
    except Exception:
        auc_roc = roc_auc_score(y_true_flat, y_pred_flat)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")

    # Strategy evaluation
    test_index = trainer.df.index[-len(test_preds):]
    test_columns = trainer.stocks

    signals = np.where(test_preds > thresh, 1, 0) + np.where(test_preds < -thresh, -1, 0)
    signals_df = pd.DataFrame(signals, index=test_index, columns=test_columns)
    signals_weights = signals_df.div(signals_df.abs().sum(axis=1), axis=0)
    portfolio_returns = (signals_weights.shift(1) * test_actual).sum(axis=1)
    portfolio_returns.cumsum().plot()
    print(strategy.summary_stats(signals_weights, ret = test_actual))

