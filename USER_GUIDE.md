# Crypto Trading Bot - User Guide

## Overview

This high-performance crypto trading bot is designed to achieve 5-10% daily returns through an advanced hybrid trading strategy that combines multiple approaches:

1. **Scalping Strategy** - For high-frequency trading on short timeframes
2. **Volatility Breakout Strategy** - Capitalizes on price breakouts after consolidation periods
3. **Momentum Strategy** - Follows strong market trends using momentum indicators
4. **Hybrid Adaptive Strategy** - Dynamically adjusts strategy weights based on market conditions

The bot features AI integration for market analysis, real-time trading capabilities, and a comprehensive dashboard for monitoring performance.

## Installation

### Prerequisites

- Python 3.10 or higher
- Internet connection
- Binance account (optional for live trading)

### Setup

1. Clone the repository:
```
git clone https://github.com/yourusername/crypto_trading_bot.git
cd crypto_trading_bot
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Configure your API keys (for live trading):
   - Create a file named `config.json` in the root directory
   - Add your Binance API keys (or use the default configuration for demo mode)

## Usage

### Running the Bot

#### Demo Mode

To run the bot in demo mode with simulated data:

```
python bot/final_bot.py --demo
```

This will generate simulated trading data and update it in real-time, allowing you to test the dashboard without risking real funds.

#### Optimization Mode

To optimize the trading strategies before starting:

```
python bot/final_bot.py --optimize
```

This will run the strategy optimizer to find the best parameters for achieving 5-10% daily returns.

#### Live Trading

To run the bot with live trading (requires Binance API keys):

```
python bot/final_bot.py
```

**Warning**: Live trading involves financial risk. Start with a small amount of capital.

### Dashboard

To view the trading performance dashboard:

```
python ui/dashboard.py
```

This will start a web server at http://localhost:8050 where you can monitor:

- Daily and total profit
- Win rate and active trades
- Trade history with detailed information
- Market analysis and strategy weights
- Bot settings and configuration

## Features

### AI Integration

The bot integrates advanced AI capabilities:

- **Market Regime Classification**: Identifies current market conditions (uptrend, downtrend, ranging, volatile)
- **Price Movement Prediction**: Forecasts short-term price movements
- **Strategy Optimization**: Dynamically adjusts strategy weights based on market conditions
- **Reinforcement Learning**: Continuously improves trading decisions based on past performance

### Risk Management

The bot implements sophisticated risk management:

- **Stop-Loss**: Automatically limits losses on each trade
- **Take-Profit**: Secures profits when targets are reached
- **Position Sizing**: Calculates optimal position size based on account balance and risk parameters
- **Leverage Control**: Adjusts leverage based on market volatility

### Performance Targeting

The bot is specifically designed to achieve 5-10% daily returns:

- **Aggressive Strategy**: Uses higher leverage and optimized parameters for higher returns
- **Adaptive Approach**: Adjusts strategy based on market conditions
- **Multiple Timeframes**: Trades across different timeframes (5m, 15m, 1h) for more opportunities
- **Multiple Symbols**: Trades multiple crypto pairs simultaneously

## Configuration

The bot can be configured through the `config.json` file:

```json
{
  "exchange": {
    "name": "binance",
    "api_key": "YOUR_API_KEY",
    "api_secret": "YOUR_API_SECRET",
    "testnet": true
  },
  "trading": {
    "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT"],
    "timeframes": ["5m", "15m", "1h"],
    "base_currency": "USDT",
    "leverage": 10
  },
  "risk_management": {
    "max_risk_per_trade": 0.1,
    "max_open_trades": 5,
    "stop_loss_pct": 1.0,
    "take_profit_pct": 7.0
  },
  "initial_balance": 10000
}
```

### Key Configuration Parameters

- **leverage**: Higher values increase potential returns but also increase risk
- **max_risk_per_trade**: Maximum percentage of account balance to risk per trade
- **stop_loss_pct**: Percentage below entry price to set stop loss
- **take_profit_pct**: Percentage above entry price to set take profit

## Performance Optimization

To achieve the target 5-10% daily returns, the bot has been optimized with:

- Shorter EMA periods (3/15) for faster signal generation
- Higher RSI thresholds (20/80) for stronger confirmation
- Increased ATR multiplier (2.5) for better volatility detection
- Higher leverage (10x) for amplified returns
- Tighter stop-loss (1%) and wider take-profit (7%) for better risk-reward ratio
- Strategy weight distribution favoring volatility and momentum (40% each)

## Troubleshooting

### Common Issues

1. **API Connection Errors**:
   - Verify your API keys are correct
   - Check your internet connection
   - Ensure Binance is accessible in your region

2. **Performance Below Target**:
   - Run the optimizer to adjust parameters
   - Check if market conditions are suitable for the strategy
   - Consider adjusting leverage settings

3. **Dashboard Not Loading**:
   - Ensure all dependencies are installed
   - Check if the data directory exists and has permissions
   - Verify the bot is generating performance data

## Disclaimer

This trading bot involves financial risk. Past performance is not indicative of future results. Use at your own risk and never trade with money you cannot afford to lose.

## Support

For questions or issues, please open an issue on the GitHub repository or contact the developer directly.
