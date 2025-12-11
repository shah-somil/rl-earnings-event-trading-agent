# Earnings Event Trading Agent (EETA) v2.0

## Multi-Agent Reinforcement Learning for Intelligent Earnings-Based Trading

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Overview

EETA is a sophisticated multi-agent reinforcement learning system that learns optimal trading strategies around corporate earnings announcements. The system combines:

- **Deep Q-Networks (DQN)** for position selection
- **Thompson Sampling** for position sizing under uncertainty
- **Cost-Aware Orchestration** of specialized analysis agents
- **Walk-Forward Validation** to prevent look-ahead bias

### Key Features

- ✅ 5 trading actions (equity + simulated volatility plays)
- ✅ 36 carefully engineered features
- ✅ Walk-forward validation methodology
- ✅ SPY benchmark comparison
- ✅ Ablation studies proving component value
- ✅ Built-in risk controls with kill switch

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    COST-AWARE ORCHESTRATOR                          │
│  Conditionally runs agents based on confidence levels               │
└────────────────────────────┬────────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  HISTORICAL   │   │   SENTIMENT   │   │    MARKET     │
│    AGENT      │   │    AGENT      │   │    AGENT      │
│               │   │               │   │               │
│ • Beat rate   │   │ • News tone   │   │ • VIX level   │
│ • Avg move    │   │ • Attention   │   │ • SPY trend   │
│ • Consistency │   │ • Revisions   │   │ • Regime      │
└───────┬───────┘   └───────┬───────┘   └───────┬───────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
                ┌───────────────────────┐
                │    36-DIM STATE       │
                │    VECTOR             │
                └───────────┬───────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                                       ▼
┌───────────────────┐               ┌───────────────────┐
│  POSITION SELECTOR│               │   SIZE OPTIMIZER  │
│      (DQN)        │               │ (Thompson Sampling)│
│                   │               │                   │
│  36 → 128 → 64 → 5│               │  Beta distributions│
└─────────┬─────────┘               └─────────┬─────────┘
          │                                   │
          └───────────────┬───────────────────┘
                          ▼
                ┌───────────────────────┐
                │   RISK CONTROLLER     │
                │   (Hard Limits)       │
                │                       │
                │  • Max 5% per trade   │
                │  • Daily loss 3%      │
                │  • Drawdown 10%       │
                └───────────────────────┘
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/earnings-trading-agent.git
cd earnings-trading-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Build Dataset

```python
from src.data import build_dataset

# Build earnings dataset (uses free yfinance data)
dataset = build_dataset(
    start_date="2019-01-01",
    end_date="2024-12-01",
    cache_path="data/processed/earnings_dataset.parquet"
)

print(f"Dataset size: {len(dataset)} earnings events")
```

### Train the Agent

```python
from src.training import EETATrainer
from src.utils import get_config

# Load configuration
config = get_config()

# Initialize trainer
trainer = EETATrainer(
    data=dataset,
    config=config._raw,
    experiment_name="experiment_001"
)

# Train
results = trainer.train(n_episodes=100)

print(f"Best Sharpe: {results['best_sharpe']:.2f}")
```

### Run Walk-Forward Validation

```python
from src.training import WalkForwardValidator, create_agent_factory, train_fold, test_fold

# Create validator
validator = WalkForwardValidator(
    data=dataset,
    min_train_years=3,
    test_years=1
)

# Run validation
results = validator.run_validation(
    agent_factory=create_agent_factory(config._raw),
    train_fn=train_fold,
    test_fn=test_fold,
    config=config._raw
)

print(f"Mean Test Sharpe: {results['aggregate']['mean_sharpe']:.2f}")
```

---

## 📊 Action Space

| ID | Action | Description | When Optimal |
|----|--------|-------------|--------------|
| 0 | NO_TRADE | Skip this earnings | Low confidence |
| 1 | LONG_STOCK | Buy shares | High confidence bullish |
| 2 | SHORT_STOCK | Short shares | High confidence bearish |
| 3 | LONG_VOL | Long volatility (straddle-like) | Big move expected |
| 4 | SHORT_VOL | Short volatility (condor-like) | Small move expected |

---

## 📈 State Space (36 Features)

### Historical Features (0-11)
- Beat rate, average moves, consistency, guidance impact, trends

### Sentiment Features (12-19)  
- News sentiment, volume, analyst revisions, attention

### Market Features (20-27)
- VIX level/percentile, SPY momentum, market regime

### Technical Features (28-33)
- RSI, trend strength, volume ratio, momentum

### Meta Features (34-35)
- Signal agreement, overall confidence

---

## 📁 Project Structure

```
earnings-trading-agent/
├── src/
│   ├── agents/           # Analysis agents
│   │   ├── historical_agent.py
│   │   ├── sentiment_agent.py
│   │   ├── market_agent.py
│   │   └── orchestrator.py
│   ├── rl/               # Reinforcement learning
│   │   ├── dqn.py
│   │   ├── thompson.py
│   │   └── replay_buffer.py
│   ├── environment/      # Trading environment
│   │   ├── trading_env.py
│   │   └── action_simulator.py
│   ├── data/             # Data pipeline
│   │   ├── sources.py
│   │   ├── preprocessor.py
│   │   └── dataset_builder.py
│   ├── training/         # Training infrastructure
│   │   ├── train.py
│   │   ├── curriculum.py
│   │   └── walk_forward.py
│   ├── evaluation/       # Evaluation
│   │   ├── metrics.py
│   │   ├── benchmarks.py
│   │   └── ablation.py
│   ├── risk/             # Risk management
│   │   └── controller.py
│   └── utils/            # Utilities
│       ├── config.py
│       └── logging.py
├── configs/
│   └── default.yaml
├── data/
├── experiments/
├── demo/
└── notebooks/
```

---

## 🔬 Evaluation

### Benchmarks

The system is compared against:
- Buy & Hold SPY
- Random Agent
- Always Long
- Momentum Strategy
- Beat Rate Strategy

### Ablation Studies

Proves each component's contribution:
- Full System (baseline)
- No Historical Agent
- No Sentiment Agent
- No Thompson Sampling
- Random Actions

---

## ⚠️ Risk Controls

Hard-coded safety limits that **cannot** be overridden:

| Control | Limit | Purpose |
|---------|-------|---------|
| Max Position | 5% | Prevent concentration |
| Daily Loss | 3% | Daily circuit breaker |
| Max Drawdown | 10% | Capital preservation |
| Consecutive Losses | 5 | Cool-down trigger |

---

## 📚 References

- [DQN Paper](https://arxiv.org/abs/1312.5602) - Mnih et al. 2013
- [Thompson Sampling](https://arxiv.org/abs/1707.02038) - Tutorial
- [Walk-Forward Validation](https://quantstart.com/articles/Walk-Forward-Optimisation/) - QuantStart

---

## 📄 License

MIT License - see LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines first.

---

## 📧 Contact

For questions and feedback, please open an issue on GitHub.
