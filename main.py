import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import warnings

warnings.filterwarnings("ignore")

# ======================
# 1. CONFIGURATION
# ======================
SYMBOL_MSTR = "MSTR"
SYMBOL_BTC = "BTC-USD"
START_DATE = "2020-01-01"  # year-month-day
END_DATE = "2025-11-08"

# Strategy Parameters
LOOKBACK = 40  # Days for rolling mean/std of mNAV
Z_ENTRY = 1.2  # Enter when |z| > Z_ENTRY - default 1.5
Z_EXIT = 1  # Exit when |z| < Z_EXIT - default 0.5
STOP_LOSS_Z = 3.0  # Hard stop if z > STOP_LOSS_Z (optional)
BETA_LOOKBACK = 40  # For beta hedging

# Initial capital
INITIAL_CAPITAL = 100_000


# ======================
# 2. FETCH MSTR BTC HOLDINGS (HISTORICAL)
# ======================
def get_mstr_btc_holdings():
    """
    Returns DataFrame with date and BTC holdings.
    Sources: Public filings, press releases, and known accumulation schedule.
    """
    # Known major accumulation points (manually curated from SEC filings & announcements)
    holdings = [
        # Date,       BTC Held,     Notes
        ("2020-08-11", 250, "Initial purchase"),
        ("2020-09-14", 38250, "Second purchase"),
        ("2020-12-04", 40470, ""),
        ("2020-12-20", 70470, ""),
        ("2021-01-23", 71425, ""),
        ("2021-02-24", 90531, ""),
        ("2021-05-18", 92079, ""),
        ("2021-06-21", 105085, ""),
        ("2021-12-30", 124391, ""),
        ("2022-11-09", 130000, "Approx"),
        ("2023-09-24", 152800, "Approx"),
        ("2023-12-27", 189150, ""),
        ("2024-03-25", 214400, ""),
        ("2024-06-20", 226500, ""),
        ("2024-09-20", 252220, ""),
        ("2024-10-31", 252220, "Last before Nov raise"),
        ("2024-11-18", 331200, "Post $1.1B raise"),
        ("2024-12-23", 446400, "Post $2.1B raise"),
        ("2025-03-10", 478300, "Estimate"),
        ("2025-06-15", 568205, "Estimate"),
        ("2025-09-01", 610000, "Estimate"),
        ("2025-11-08", 641205, "Current reported"),
    ]

    df = pd.DataFrame(holdings, columns=["date", "btc_held", "notes"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return df


# ======================
# 3. DOWNLOAD PRICE DATA
# ======================
print("Downloading price data...")
mstr = yf.download(SYMBOL_MSTR, start=START_DATE, end=END_DATE)
btc = yf.download(SYMBOL_BTC, start=START_DATE, end=END_DATE)

# Get close prices
mstr_price = mstr[('Close', 'MSTR')].rename('mstr_price')
btc_price = btc[('Close', 'BTC-USD')].rename('btc_price')

# Merge price data and start date when MSTR began holding BTC "2020-08-11"
data = pd.concat([mstr_price, btc_price], axis=1).dropna()
data = data[data.index >= "2022-08-11"]


# ======================
# 4. INTERPOLATE BTC HOLDINGS DAILY
# ======================
holdings_df = get_mstr_btc_holdings()
daily_holdings = holdings_df.resample('D').ffill()
daily_holdings = daily_holdings.reindex(data.index, method='ffill').fillna(method='bfill')

# Forward fill to cover all trading days
data["btc_held"] = daily_holdings["btc_held"]


# ======================
# 5. COMPUTE MARKET CAP & mNAV RATIO
# ======================
# Get shares outstanding (approximate, changes with issuances)
# We use historical approximation based on market cap / price
data["shares_out"] = data["mstr_price"] * 1e6 / data["mstr_price"].iloc[0] * 170e6 / data["mstr_price"].iloc[0]
# Better: use actual reported shares when possible
# For simplicity, use known values and interpolate

known_shares = {
    "2020-08-11": 9.7e6,
    "2021-01-01": 9.7e6,
    "2022-01-01": 11.0e6,
    "2023-01-01": 12.2e6,
    "2024-01-01": 16.8e6,
    "2024-11-01": 250e6,  # Post split & raises
    "2025-01-01": 270e6,
    "2025-11-08": 278e6,  # Latest estimate
}

shares_df = pd.DataFrame([
    (pd.to_datetime(date), shares) for date, shares in known_shares.items()
], columns=["date", "shares"]).set_index("date")

daily_shares = shares_df.resample('D').ffill()
daily_shares = daily_shares.reindex(data.index, method='ffill')

data['shares_out'] = daily_shares['shares']

# Market Cap
data['market_cap'] = data['mstr_price'] * data['shares_out']

# BTC NAV = BTC Held * BTC Price
data['btc_nav'] = data['btc_held'] * data['btc_price']

# mNAV Ratio = Market Cap / BTC NAV
data['mnav_ratio'] = data['market_cap'] / data['btc_nav']
data['premium_pct'] = (data['mnav_ratio'] - 1) * 100


# ======================
# 6. COMPUTE Z-SCORE OF mNAV
# ======================
data['mnav_mean'] = data['mnav_ratio'].rolling(LOOKBACK).mean()
data['mnav_std'] = data['mnav_ratio'].rolling(LOOKBACK).std()
data['z_score'] = (data['mnav_ratio'] - data['mnav_mean']) / data['mnav_std']

# ======================
# 7. BETA HEDGING (MSTR vs BTC)
# ======================
# Compute rolling beta: MSTR return ~ beta * BTC return
# where beta(mstr_return) = covar(mstr_return, btc_return)/var(btc_return)
returns = np.log(data[['mstr_price', 'btc_price']]).diff()
returns.columns = ['ret_mstr', 'ret_btc']

data = data.join(returns)

data['beta'] = (
    data['ret_mstr'].rolling(BETA_LOOKBACK)
    .cov(data['ret_btc'])
    / data['ret_btc'].rolling(BETA_LOOKBACK).var()
)

# Fill early beta
data['beta'] = data['beta'].fillna(1.0)


# ======================
# 8. GENERATE SIGNALS
# ======================
data['signal'] = 0

# Long MSTR / Short BTC when z < -Z_ENTRY (undervalued)
# Short MSTR / Long BTC when z > +Z_ENTRY (overvalued)
data.loc[data['z_score'] < -Z_ENTRY, 'signal'] = 1
data.loc[data['z_score'] > Z_ENTRY, 'signal'] = -1

# Exit when |z| < Z_EXIT
data['prev_signal'] = data['signal'].shift(1)
exit_cond = (data['prev_signal'] != 0) & (data['z_score'].abs() < Z_EXIT)
data.loc[exit_cond, 'signal'] = 0

# # Optional: Stop loss
# stop_cond = (data['prev_signal'] == 1) & (data['z_score'] < -STOP_LOSS_Z)
# data.loc[stop_cond, 'signal'] = 0
# stop_cond = (data['prev_signal'] == -1) & (data['z_score'] > STOP_LOSS_Z)
# data.loc[stop_cond, 'signal'] = 0

# Forward fill signal (hold until exit)
data['signal'] = data['signal'].replace(to_replace=0, method='ffill').fillna(0)

# ======================
# 9. BACKTEST: PORTFOLIO SIMULATION
# ======================
capital = INITIAL_CAPITAL
portfolio = pd.DataFrame(index=data.index)
portfolio['capital'] = capital
portfolio['position_mstr'] = 0.0
portfolio['position_btc'] = 0.0
portfolio['pnl'] = 0.0

position_mstr = 0
position_btc = 0

for i in range(1, len(data)):

    # Get current and previous date
    date = data.index[i]
    prev_date = data.index[i - 1]

    # Get current and previous signals
    signal = data['signal'].iloc[i]
    prev_signal = data['signal'].iloc[i - 1]

    # Get current returns and beta
    mstr_ret = data['ret_mstr'].iloc[i]
    btc_ret = data['ret_btc'].iloc[i]
    beta = data['beta'].iloc[i - 1]

    # Calculate PnL
    pnl = position_mstr * mstr_ret + position_btc * btc_ret

    # Close previous position if signal changes
    if signal != prev_signal:
        # Realize PnL
        portfolio.loc[date, 'pnl'] = pnl
        # Update portfolio capital
        portfolio.loc[date, 'capital'] = portfolio['capital'].iloc[i - 1] + pnl
        # Reset positions
        position_mstr = 0
        position_btc = 0

    # Keep previous position if signal same
    else:
        # Update PnL and portfolio capital
        portfolio.loc[date, 'pnl'] = pnl
        portfolio.loc[date, 'capital'] = portfolio['capital'].iloc[i - 1] + pnl

    # Open new position (signal=0 -> signal !=0)
    if signal != 0 and prev_signal == 0:
        notional = portfolio['capital'].iloc[i - 1] * 0.95  # Use 95% of cash in portfolio
        dollar_mstr = notional if signal == 1 else -notional
        dollar_btc = -dollar_mstr * beta if signal == 1 else dollar_mstr * beta

        position_mstr = dollar_mstr / data['mstr_price'].iloc[i]
        position_btc = dollar_btc / data['btc_price'].iloc[i]

    portfolio.loc[date, 'position_mstr'] = position_mstr
    portfolio.loc[date, 'position_btc'] = position_btc

# Final PnL
portfolio['returns'] = portfolio['capital'].pct_change().fillna(0)
portfolio['cum_returns'] = (1 + portfolio['returns']).cumprod()
portfolio['drawdown'] = portfolio['cum_returns'] / portfolio['cum_returns'].cummax() - 1


# ======================
# 10. PERFORMANCE METRICS
# ======================
total_return = portfolio['capital'].iloc[-1] / INITIAL_CAPITAL - 1
annualized_return = (1 + total_return) ** (252 / len(portfolio)) - 1
sharpe = portfolio['returns'].mean() / portfolio['returns'].std() * np.sqrt(252)
sortino = portfolio['returns'].mean() / portfolio['returns'][portfolio['returns'] < 0].std() * np.sqrt(252)
max_dd = portfolio['drawdown'].min()

print(f"\n=== PAIRS TRADING BACKTEST RESULTS ===")
print(f"Period: {data.index[0].date()} to {data.index[-1].date()}")
print(f"Total Return: {total_return:.2%}")
print(f"Annualized Return: {annualized_return:.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Sortino Ratio: {sortino:.2f}")
print(f"Max Drawdown: {max_dd:.2%}")
print(f"Final Capital: ${portfolio['capital'].iloc[-1]:,.0f}")

# ======================
# 11. PLOTS
# ======================
fig, axes = plt.subplots(3, 1, figsize=(12,8))

# Plot 1: mNAV Ratio & Z-Score
ax1 = axes[0]
ax1.plot(data.index, data['mnav_ratio'], label='mNAV Ratio', color='blue')
ax1.axhline(1.0, color='black', linestyle='--', alpha=0.5)
ax1_twin = ax1.twinx()
ax1_twin.plot(data.index, data['z_score'], color='red', alpha=0.7, label='Z-Score')
ax1_twin.axhline(Z_ENTRY, color='green', linestyle='--')
ax1_twin.axhline(-Z_ENTRY, color='green', linestyle='--')
ax1_twin.axhline(Z_EXIT, color='orange', linestyle=':')
ax1_twin.axhline(-Z_EXIT, color='orange', linestyle=':')
ax1.set_title('MSTR mNAV Ratio & Z-Score')
ax1.set_ylabel('mNAV')
ax1_twin.set_ylabel('Z-Score')
ax1.legend(loc='upper left')
ax1_twin.legend(loc='upper right')

# Plot 2: Signals
ax2 = axes[1]
ax2.plot(data.index, data['signal'], label='Signal (1=Long MSTR, -1=Short MSTR)', drawstyle='steps-post')
ax2.set_title('Trading Signal')
ax2.set_ylabel('Signal')
ax2.legend()

# Plot 3: Equity Curve
ax3 = axes[2]
ax3.plot(portfolio.index, portfolio['cum_returns'], label='Strategy', color='purple')
ax3.set_title('Strategy Equity Curve')
ax3.set_ylabel('Cumulative Return')
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.show()

# # Optional: Save results
# # data.to_csv("mstr_btc_pairs_data.csv")
# # portfolio.to_csv("mstr_btc_pairs_portfolio.csv")