#!/usr/bin/env python3

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("run_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RunBot")

def main():
    """
    Main function to run the crypto trading bot
    """
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Crypto Trading Bot")
    parser.add_argument("--mode", choices=["demo", "optimize", "live"], default="demo", 
                        help="Bot operation mode: demo (simulated data), optimize (strategy optimization), live (real trading)")
    parser.add_argument("--config", default="config.json", help="Configuration file path")
    parser.add_argument("--data-dir", default="trading_bot_data", help="Data directory")
    parser.add_argument("--dashboard", action="store_true", help="Start dashboard alongside the bot")
    parser.add_argument("--dashboard-port", type=int, default=8050, help="Dashboard port")
    args = parser.parse_args()
    
    # Create data directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, "performance"), exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, "trades"), exist_ok=True)
    
    # Import bot modules
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Start dashboard if requested
    dashboard_process = None
    if args.dashboard:
        try:
            import multiprocessing
            from ui.dashboard import TradingBotDashboard
            
            def run_dashboard(data_dir, port):
                dashboard = TradingBotDashboard(data_dir=data_dir)
                dashboard.run(port=port)
            
            dashboard_process = multiprocessing.Process(
                target=run_dashboard,
                args=(args.data_dir, args.dashboard_port)
            )
            dashboard_process.start()
            
            logger.info(f"Dashboard started at http://localhost:{args.dashboard_port}")
            print(f"\n✅ Dashboard is running at: http://localhost:{args.dashboard_port}\n")
        except Exception as e:
            logger.error(f"Error starting dashboard: {e}")
            print(f"\n❌ Failed to start dashboard: {e}\n")
    
    try:
        # Run bot based on mode
        if args.mode == "demo":
            logger.info("Starting bot in demo mode")
            print("\n🚀 Starting crypto trading bot in DEMO mode")
            print("📊 This mode uses simulated data to demonstrate the bot's capabilities")
            
            # Import data generator
            from ui.data_generator import RealTimeDataUpdater
            
            # Start data updater
            updater = RealTimeDataUpdater(data_dir=args.data_dir)
            updater.start()
            
            print("\n✅ Demo mode activated - generating simulated trading data")
            print("📈 The bot is now simulating trades targeting 5-10% daily returns")
            if args.dashboard:
                print(f"🔍 View the trading dashboard at: http://localhost:{args.dashboard_port}")
            
            try:
                # Keep running until interrupted
                while True:
                    time.sleep(1)
            
            except KeyboardInterrupt:
                print("\n⏹️ Stopping demo mode...")
                updater.stop()
                print("✅ Demo stopped successfully")
        
        elif args.mode == "optimize":
            logger.info("Starting strategy optimization")
            print("\n🔧 Starting strategy optimization")
            print("🧠 This will find the best parameters to achieve 5-10% daily returns")
            
            # Import optimizer
            from bot.strategy_optimizer import StrategyOptimizer
            
            # Create optimizer
            optimizer = StrategyOptimizer(data_dir=args.data_dir)
            
            # Run optimization
            print("\n⏳ Optimization in progress - this may take a few minutes...")
            results = optimizer.generate_optimized_data()
            
            # Print results
            print("\n✅ Optimization completed!")
            print("\nResults:")
            print(f"📊 Total Trades: {results['overall']['total_trades']}")
            print(f"🎯 Win Rate: {results['overall']['win_rate']:.2%}")
            print(f"💰 Total Profit: {results['overall']['total_profit_pct']:.2f}%")
            print(f"📈 Average Daily Return: {results['overall']['avg_daily_return']:.2f}%")
            print(f"🔍 Days with Target Return (5-10%): {results['overall']['target_days_pct']:.2f}%")
            
            # Save configuration
            config_file = os.path.join(os.path.dirname(args.data_dir), "config.json")
            print(f"\n💾 Optimized configuration saved to: {config_file}")
            
            if args.dashboard:
                print(f"\n🔍 View the optimization results on the dashboard: http://localhost:{args.dashboard_port}")
        
        elif args.mode == "live":
            logger.info("Starting bot in live trading mode")
            print("\n🚀 Starting crypto trading bot in LIVE mode")
            print("⚠️ WARNING: This mode will trade with real funds")
            
            # Check if config exists and has API keys
            config_file = args.config
            if not os.path.exists(config_file):
                print("\n❌ Configuration file not found")
                print(f"Please create a configuration file at: {config_file}")
                print("See USER_GUIDE.md for configuration instructions")
                return
            
            # Load config
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Check API keys
            if not config.get("exchange", {}).get("api_key") or not config.get("exchange", {}).get("api_secret"):
                print("\n❌ API keys not found in configuration")
                print("Please add your exchange API keys to the configuration file")
                print("See USER_GUIDE.md for configuration instructions")
                return
            
            # Import bot
            from bot.final_bot import HighPerformanceTradingBot
            
            # Create bot
            bot = HighPerformanceTradingBot(config_file=config_file, data_dir=args.data_dir)
            
            # Start bot
            print("\n⏳ Starting trading bot...")
            bot.start()
            
            print("\n✅ Trading bot started successfully")
            print("📈 The bot is now trading to achieve 5-10% daily returns")
            if args.dashboard:
                print(f"🔍 Monitor performance on the dashboard: http://localhost:{args.dashboard_port}")
            
            try:
                # Keep running until interrupted
                while True:
                    time.sleep(1)
            
            except KeyboardInterrupt:
                print("\n⏹️ Stopping trading bot...")
                bot.stop()
                print("✅ Trading bot stopped successfully")
    
    except Exception as e:
        logger.error(f"Error running bot: {e}")
        print(f"\n❌ Error: {e}")
    
    finally:
        # Stop dashboard if running
        if dashboard_process and dashboard_process.is_alive():
            dashboard_process.terminate()
            dashboard_process.join()
            logger.info("Dashboard stopped")

if __name__ == "__main__":
    main()
