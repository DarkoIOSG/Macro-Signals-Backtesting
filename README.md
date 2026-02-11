# Macro-Signals-Backtesting

bitcoin-return-prediction/
│
├── README.md                          # Project overview, setup instructions
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package installation configuration
├── .gitignore                        # Git ignore file
├── .env.example                      # Example environment variables
│
├── config/                           # Configuration files
│   ├── __init__.py
│   ├── data_config.yaml             # Data paths, date ranges
│   ├── model_config.yaml            # Model hyperparameters
│   └── feature_config.yaml          # Feature engineering settings
│
├── data/                            # Data directory (add to .gitignore)
│   ├── raw/                         # Original data files
│   │   ├── prices/
│   │   ├── vix/
│   │   ├── hy_spreads/
│   │   ├── mvrv/
│   │   └── volume/
│   ├── processed/                   # Cleaned/processed data
│   └── features/                    # Engineered features
│
├── notebooks/                       # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_results_analysis.ipynb
│   └── README.md                    # Notebook descriptions
│
├── src/                             # Source code
│   ├── __init__.py
│   │
│   ├── data/                        # Data handling
│   │   ├── __init__.py
│   │   ├── loader.py               # Load raw data from files
│   │   ├── preprocessor.py         # Clean and preprocess data
│   │   └── validator.py            # Data quality checks
│   │
│   ├── features/                    # Feature engineering
│   │   ├── __init__.py
│   │   ├── price_features.py       # BTC price-based features
│   │   ├── macro_features.py       # VIX, HY spread features
│   │   ├── onchain_features.py     # MVRV, volume features
│   │   ├── target_builder.py       # Build forward return targets
│   │   └── feature_selector.py     # Feature selection logic
│   │
│   ├── models/                      # Model implementations
│   │   ├── __init__.py
│   │   ├── base_model.py           # Abstract base class
│   │   ├── ridge_model.py          # Ridge regression
│   │   ├── random_forest_model.py  # Random Forest
│   │   ├── gradient_boosting_model.py  # Gradient Boosting
│   │   └── ensemble_model.py       # Model ensemble
│   │
│   ├── training/                    # Training logic
│   │   ├── __init__.py
│   │   ├── trainer.py              # Training orchestration
│   │   ├── hyperparameter_tuner.py # Grid search, CV
│   │   └── cross_validator.py      # Time series CV
│   │
│   ├── evaluation/                  # Model evaluation
│   │   ├── __init__.py
│   │   ├── metrics.py              # R², RMSE, MAE, etc.
│   │   ├── backtester.py           # Backtest strategies
│   │   └── visualizer.py           # Plotting functions
│   │
│   └── utils/                       # Utility functions
│       ├── __init__.py
│       ├── config_loader.py        # Load YAML configs
│       ├── logger.py               # Logging setup
│       └── helpers.py              # Common utilities
│
├── scripts/                         # Executable scripts
│   ├── download_data.py            # Download external data
│   ├── prepare_dataset.py          # Full data pipeline
│   ├── train_models.py             # Train all models
│   ├── evaluate_models.py          # Compare model performance
│   └── run_backtest.py             # Run backtesting
│
├── tests/                           # Unit tests
│   ├── __init__.py
│   ├── test_data/                  # Test data handling
│   │   ├── test_loader.py
│   │   └── test_preprocessor.py
│   ├── test_features/              # Test feature engineering
│   │   ├── test_price_features.py
│   │   └── test_macro_features.py
│   ├── test_models/                # Test models
│   │   ├── test_ridge_model.py
│   │   └── test_trainer.py
│   └── conftest.py                 # Pytest fixtures
│
├── results/                         # Model outputs (add to .gitignore)
│   ├── models/                     # Saved model files
│   │   ├── ridge_model.pkl
│   │   ├── rf_model.pkl
│   │   └── gb_model.pkl
│   ├── predictions/                # Prediction outputs
│   ├── metrics/                    # Performance metrics (CSV/JSON)
│   └── figures/                    # Plots and visualizations
│
├── reports/                         # Analysis reports
│   ├── model_comparison.md
│   ├── feature_importance_analysis.md
│   └── backtesting_results.md
│
└── docs/                            # Documentation
    ├── architecture.md              # System architecture
    ├── data_dictionary.md           # Data field descriptions
    ├── model_details.md             # Model explanations
    └── api_reference.md             # Code API documentation