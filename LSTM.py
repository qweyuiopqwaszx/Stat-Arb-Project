import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import copy

# synthetic data For demonstration purpose
# dates = pd.date_range('2020-01-01', periods=1000)
# stocks = ['AAPL', 'MSFT', 'GOOG']
# data = np.random.randn(1000, len(stocks)) * 0.01
# df = pd.DataFrame(data, index=dates, columns=stocks)

# naive Sharpe ratio loss
class SharpeRatioLoss(torch.nn.Module):
    def __init__(self, eps=1e-10):
        super().__init__()
        self.eps = eps  # to avoid division by zero

    def forward(self, preds, targets):
        returns = preds - targets
        mean_return = torch.mean(returns)
        std_return = torch.std(returns) + self.eps
        sharpe = mean_return / std_return
        # We want to maximize Sharpe, so loss is negative Sharpe
        loss = -sharpe
        return loss

# binary cross-entropy loss
class SignAccuracyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()  # combines sigmoid + BCE

    def forward(self, preds, targets):
        # preds, targets: (batch_size, num_stocks)
        # Convert targets to 0 (negative) and 1 (positive)
        sign_targets = (targets > 0).float()
        # Optionally, you can scale preds to logits for BCEWithLogitsLoss
        return self.bce(preds, sign_targets)

# Margin-based sign loss
class MarginSignLoss(torch.nn.Module):
    def __init__(self, margin=0.01):
        super().__init__()
        self.margin = margin

    def forward(self, preds, targets):
        # Encourage preds * targets > margin (same sign, large enough)
        # Use softplus for smooth, differentiable penalty
        loss = torch.nn.functional.softplus(self.margin - preds * targets)
        return loss.mean()


# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take last output
        out = self.dropout(out)
        out = self.fc(out)
        return out

class StockLSTMTrainer:
    def __init__(self, df, sequence_length=20, train_ratio=0.7, 
                 batch_size=128, hidden_size=32, epochs=20, 
                 lr=0.001, dropout_rate=0.2, weight_decay=1e-4, seed=None,
                 loss_fn=nn.MSELoss()):
        self.df = df.copy()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sequence_length = sequence_length
        self.train_ratio = train_ratio
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.dropout_rate = dropout_rate
        self.stocks = df.columns.tolist()
        self.scaler = StandardScaler()
        self.model = None
        self.optimizer = None
        self.loss_fn = loss_fn
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.train_loss = []
        self.val_loss = []
        self.weight_decay = weight_decay
        # Set random seed for reproducibility if provided
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            import random as pyrandom
            pyrandom.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self._prepare_data()
        self._build_model()

    def _prepare_data(self):
        # Split raw data first
        split_idx = int(self.train_ratio * len(self.df))
        train_data = self.df.iloc[:split_idx]
        test_data = self.df.iloc[split_idx:]

        # Fit scaler only on training data
        self.scaler.fit(train_data.values)
        scaled_train = self.scaler.transform(train_data.values)
        scaled_test = self.scaler.transform(test_data.values)

        # Build sequences for train
        X_train, y_train = [], []
        for i in range(self.sequence_length, len(scaled_train)):
            X_train.append(scaled_train[i-self.sequence_length:i])
            y_train.append(scaled_train[i])
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Build sequences for test
        X_test, y_test = [], []
        for i in range(self.sequence_length, len(scaled_test)):
            X_test.append(scaled_test[i-self.sequence_length:i])
            y_test.append(scaled_test[i])
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        # Validation split from training set
        split = int(0.9 * len(X_train))  # 90% train, 10% val
        X_tr, X_val = X_train[:split], X_train[split:]
        y_tr, y_val = y_train[:split], y_train[split:]

        self.X_train_t = torch.tensor(X_tr, dtype=torch.float32).to(self.device)
        self.y_train_t = torch.tensor(y_tr, dtype=torch.float32).to(self.device)
        self.X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        self.y_val_t = torch.tensor(y_val, dtype=torch.float32).to(self.device)
        self.X_test_t = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        self.y_test_t = torch.tensor(y_test, dtype=torch.float32).to(self.device)

        train_ds = TensorDataset(self.X_train_t, self.y_train_t)
        val_ds = TensorDataset(self.X_val_t, self.y_val_t)
        test_ds = TensorDataset(self.X_test_t, self.y_test_t)
        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=False)
        self.val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

    def _build_model(self):
        input_size = len(self.stocks)
        output_size = len(self.stocks)
        self.model = LSTMModel(input_size, self.hidden_size, output_size, dropout_rate=self.dropout_rate).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.initial_state_dict = copy.deepcopy(self.model.state_dict())

    def train(self, additional_epochs=0, early_stopping_patience=10):
        total_epochs = self.epochs + additional_epochs
        best_val_loss = float('inf')
        patience_counter = 0
        for epoch in range(total_epochs):
            self.model.train()
            for xb, yb in self.train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(xb)
                loss = self.loss_fn(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            # Validation loss
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in self.val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    pred = self.model(xb)
                    val_loss = self.loss_fn(pred, yb)
                    val_losses.append(val_loss.item())
            avg_val_loss = np.mean(val_losses)
            self.train_loss.append(loss.item())
            self.val_loss.append(avg_val_loss)
            print(f"Epoch {epoch+1}, Train Loss: {loss.item():.6f}, Val Loss: {avg_val_loss:.6f}")
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1} (no improvement in val loss for {early_stopping_patience} epochs)")
                    break

    def plot_train_loss(self):
        plt.plot(self.train_loss)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
    
    def plot_val_loss(self):
        plt.plot(self.val_loss)
        plt.title('Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

    def predict(self):
        self.model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in self.test_loader:
                xb = xb.to(self.device)
                pred = self.model(xb)
                preds.append(pred.cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        preds = self.scaler.inverse_transform(preds)
        return preds


# Usage example:
# trainer = StockLSTMTrainer(df)
# trainer.train()
# trainer.plot_train_loss()



# trainer.train(additional_epochs=1000)
# trainer.predict()
# test = trainer.predict()


