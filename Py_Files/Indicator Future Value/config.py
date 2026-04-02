# ==========================================
# CONFIGURATION VARIABLES
# ==========================================

# To run: py -m streamlit run app.py
# Data needs to be comma delimited and have the following columns: Date, Open, High, Low, Close, Volume, Spread
# Time,Open,High,Low,Close,Volume,Spread
# 2024/02/29 22:45,1.08029,1.0805,1.08021,1.08047,2297,20
# 2024/02/29 23:00,1.08053,1.08054,1.08026,1.08033,369,5

# Phase 1: Image Generation Settings
CANDLES_BEFORE_TRIGGER = 20  # X: Number of candles to show before the trigger
CANDLES_AFTER_TRIGGER = 30   # Y: Number of candles to show after the trigger

# Phase 3: Backtesting & Money Management Settings
STARTING_BALANCE = 10000.0   # Standardized simulation starting balance
RISK_PER_TRADE_PERCENT = 1.0 # Z%: Risk per trade (e.g., 1.0 means 1% of current balance)

# Asset Properties
POINT_VALUE = 0.00001            # Value of one point (e.g., in USD)
LOT_SIZE = 100000            # Lot size (standard lot in Forex is typically 100,000)

# Default Strategy Name (used for folder naming)
STRATEGY_NAME = "TemplateStrategy"
