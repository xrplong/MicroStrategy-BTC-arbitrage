# Basic MicroStrategy-BTC arbitrage script

# **Explanation of Files and Output**

**1. <u> main.py** </u>\
Main and only script.

Input data:
- SYMBOL_MSTR = "MSTR"
- SYMBOL_BTC = "BTC-USD"
- START_DATE = "2020-01-01" 
- END_DATE = "2025-11-08"

- LOOKBACK = 40  # Days for rolling mean/std of mNAV
- Z_ENTRY = 1.2  # Enter when |z| > Z_ENTRY - default 1.5
- Z_EXIT = 1  # Exit when |z| < Z_EXIT - default 0.5
- STOP_LOSS_Z = 3.0  # Hard stop if z > STOP_LOSS_Z (optional)
- BETA_LOOKBACK = 40  # For beta hedging

- INITIAL_CAPITAL = 100_000

Output data (Performance matrics):
- Total return
- Annualised return
- Sharpe ratio
- Sortino ratio
- Max drawdown
- Final capital
  
Output data (Plots over time axis):
- mNAV Ratio & Z-Score
- Signals (+1, 0, -1 )
- Equity curve

