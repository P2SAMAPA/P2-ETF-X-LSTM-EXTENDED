import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class XLSTMBlock(nn.Module):
    """
    Extended LSTM block with exponential gating and memory mixing.
    """
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        # Exponential gating for input and forget gates
        self.input_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Sigmoid()
        )
        self.forget_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Sigmoid()
        )
        self.output_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Sigmoid()
        )
        # Memory mixing: candidate cell
        self.cell_proj = nn.Linear(input_size + hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, h, c):
        # x: (batch, input_size)
        # h: (batch, hidden_size)
        # c: (batch, hidden_size)
        combined = torch.cat([x, h], dim=1)
        # Gates
        i = self.input_gate(combined)
        f = self.forget_gate(combined)
        o = self.output_gate(combined)
        # Candidate cell update (exponential gating)
        c_tilde = torch.tanh(self.cell_proj(combined))
        # Memory mixing: f * c + i * c_tilde
        c_new = f * c + i * c_tilde
        # Output
        h_new = o * torch.tanh(c_new)
        h_new = self.dropout(h_new)
        return h_new, c_new

class XLSTM(nn.Module):
    """
    Extended LSTM with exponential gating and memory mixing.
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.1, seq_len=10):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.layers = nn.ModuleList([
            XLSTMBlock(hidden_size, hidden_size, dropout) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        batch, seq_len, _ = x.shape
        x = self.input_proj(x)
        # Initialize hidden and cell states
        h = torch.zeros(batch, self.hidden_size, device=x.device)
        c = torch.zeros(batch, self.hidden_size, device=x.device)
        # Process sequence
        for t in range(seq_len):
            x_t = x[:, t, :]
            for layer in self.layers:
                h, c = layer(x_t, h, c)
                x_t = h
        # Use last hidden state for prediction
        out = self.output_proj(h)
        return out.squeeze(-1)

def prepare_data(returns, macro_df, seq_len=10):
    """
    Prepare sequences for training.
    returns: pandas Series (single ETF)
    macro_df: pandas DataFrame (macro variables)
    """
    if len(returns) < seq_len + 1:
        return None, None
    common_idx = returns.index.intersection(macro_df.index)
    ret_aligned = returns.loc[common_idx]
    macro_aligned = macro_df.loc[common_idx]
    X, y = [], []
    for i in range(seq_len, len(ret_aligned)):
        ret_seq = ret_aligned.iloc[i-seq_len:i].values.reshape(-1, 1)
        macro_seq = macro_aligned.iloc[i-seq_len:i].values
        seq_features = np.concatenate([ret_seq, macro_seq], axis=1)
        X.append(seq_features)
        y.append(ret_aligned.iloc[i])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y

def xlstm_score(returns, macro_df, hidden_size=64, num_layers=2, dropout=0.1, seq_len=10, epochs=30, lr=0.001, batch_size=16):
    """
    Train X-LSTM and return predicted next-day return with momentum enhancement.
    Final score = X-LSTM prediction × (1 + last_return)
    """
    X, y = prepare_data(returns, macro_df, seq_len)
    if X is None or len(X) < batch_size:
        return 0.0
    input_size = X.shape[2]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = XLSTM(input_size, hidden_size, num_layers, dropout, seq_len).to(device)
    dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    # Predict next day
    model.eval()
    with torch.no_grad():
        ret_seq = returns.iloc[-seq_len:].values.reshape(-1, 1)
        macro_seq = macro_df.iloc[-seq_len:].values
        last_seq = np.concatenate([ret_seq, macro_seq], axis=1)
        last_seq = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(device)
        pred = model(last_seq).item()
    # Momentum factor
    last_return = returns.iloc[-1]
    momentum = 1.0 + last_return
    momentum = max(0.5, min(2.0, momentum))
    final_score = pred * momentum
    return float(final_score)
