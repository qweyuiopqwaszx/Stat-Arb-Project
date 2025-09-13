# trainer that has positive return
# seed = 42
# trainer = StockLSTMTrainer(ret,train_ratio = 0.8, sequence_length= 100, 
#                            dropout_rate=0.5, batch_size=64, 
#                            hidden_size=1, epochs=500, 
#                            lr=0.0005, weight_decay=10e-4,
#                            seed=seed)
# trainer = StockLSTMTrainer(ret,train_ratio = 0.8, sequence_length= 20, 
#                            dropout_rate=0.5, batch_size=64, 
#                            hidden_size=1, epochs=500, 
#                            lr=0.0005, weight_decay=10e-4,
#                            seed=seed)
# trainer = StockLSTMTrainer(ret,train_ratio = 0.8, sequence_length= 50, 
#                            dropout_rate=0.5, batch_size=64, 
#                            hidden_size=1, epochs=500, 
#                            lr=0.0005, weight_decay=20e-4,
#                            seed=seed)
# trainer = StockLSTMTrainer(ret,train_ratio = 0.8, sequence_length= 100, 
#                            dropout_rate=0.5, batch_size=64, 
#                            hidden_size=1, epochs=500, 
#                            lr=0.0005, weight_decay=10e-4,
#                            seed=seed, loss_fn= LSTM.SignAccuracyLoss())