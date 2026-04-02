# ==========================================
# CONFIGURATION VARIABLES
# ==========================================

# Phase 1: Image Generation Settings
CANDLES_BEFORE_TRIGGER = 20  # X: Number of candles to show before the trigger
CANDLES_AFTER_TRIGGER = 10   # Y: Number of candles to show after the trigger

# Phase 3: Backtesting & Money Management Settings
STARTING_BALANCE = 10000.0   # Standardized simulation starting balance
RISK_PER_TRADE_PERCENT = 1.0 # Z%: Risk per trade (e.g., 1.0 means 1% of current balance)

# Asset Properties
POINT_VALUE = 1.0            # Value of one point (e.g., in USD)
LOT_SIZE = 100000            # Lot size (standard lot in Forex is typically 100,000)

# Default Strategy Name (used for folder naming)
STRATEGY_NAME = "TemplateStrategy"
