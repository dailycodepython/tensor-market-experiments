# Tensor Market Experiments

An experimental repository for exploring the multidimensional structure of financial markets using tensor algebra. Rather than flattening market data into 2D matrices or isolated time series, this project models market dynamics as higher-order tensors ($\text{Assets} \times \text{Features} \times \text{Time}$ and $\text{Assets} \times \text{Assets} \times \text{Time}$), preserving cross-sectional and temporal dependencies simultaneously.

## Implemented Methods

* **Data Tensorization (`data/loader.py`)** — Aggregates market metrics (log returns, normalized volume, intraday volatility) into spatiotemporal data cubes and constructs rolling correlation tensors.
* **Tucker Decomposition / HOSVD (`methods/decompositions.py`)** — Compresses the market tensor, filters noise, and extracts orthogonal factor spaces along each mode (asset clusters, temporal regimes, and feature profiles).
* **Canonical Polyadic Decomposition / PARAFAC (`methods/decompositions.py`)** — Decomposes tensors into rank-1 components to uncover latent drivers jointly affecting groups of equities and metrics.
* **Tensor Train Decomposition (`methods/decompositions.py`)** — Represents high-dimensional arrays as a chain of 3D cores to mitigate the curse of dimensionality.
* **Bilinear Matrix Autoregression / TAR (`methods/autoregression.py`)** — Estimates the matrix-valued autoregressive model $X_t = A X_{t-1} B^T + E_t$ via Alternating Least Squares (ALS) to forecast market slices while preserving matrix geometry.
* **Regime-Shift & Anomaly Detection (`methods/anomalies.py`)** — Tracks market stress via the singular value entropy (effective rank) of rolling correlation slices and detects local anomalies as residuals from low-rank Tucker approximations.

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the baseline experiment pipeline:
```bash
python main.py
```

## Disclaimer
This repository is strictly for mathematical research and experimental purposes. Nothing here constitutes financial advice or a live trading system.


## Дисклеймер
Проект носит исключительно исследовательский и математический характер и не является торговой системой или инвестиционной рекомендацией.
