# X-LSTM Extended for ETFs

Implements the Extended LSTM (X-LSTM) with exponential gating and memory mixing – a revived recurrent architecture that outperforms Transformers and Mamba. The model predicts next‑day ETF returns from sequences of returns and macro variables, enhanced with momentum.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Exponential gating and memory mixing
- Configurable hidden size, layers, sequence length
- Score = X-LSTM prediction × (1 + last_return)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-x-lstm-extended-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (slower due to neural net training)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High positive score → ETF expected to rise tomorrow with momentum confirmation.
- Negative score → expected to fall.

## Requirements

See `requirements.txt`.
