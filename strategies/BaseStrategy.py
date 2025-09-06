# BaseStrategy.py
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Assuming BaseDownloader is in a file in the same directory or a package
from BaseDownloader import BaseDownloader 

class BaseStrategy(ABC):
    """
    策略基类：提供完整的【回测引擎】和【实盘信号】生成框架。
    信号生成逻辑 (`generate_signals`) 由子类实现。
    """
    def __init__(self, tickers: list, downloader: BaseDownloader, 
                 benchmark_ticker: str = None, commission_rate: float = 7e-5):
        self.tickers = tickers
        self.benchmark_ticker = benchmark_ticker
        self.commission_rate = commission_rate
        self.downloader = downloader
        self.price_data = None
        self.results = None

    def _prepare_data(self, start_date: str, end_date: str, buffer_days: int = 252):
        """
        准备数据，包含一个前置的缓冲期以预热指标。
        """
        effective_start_date = (pd.to_datetime(start_date) - timedelta(days=buffer_days)).strftime('%Y-%m-%d')
        all_tickers = list(set(self.tickers + ([self.benchmark_ticker] if self.benchmark_ticker else [])))
        
        print("Preparing data...")
        self.price_data = self.downloader.download(all_tickers, effective_start_date, end_date)
        
        if self.price_data is None or self.price_data.empty:
            raise ValueError("Data download failed. Cannot run strategy.")
            
        self.price_data.ffill(inplace=True)
        self.price_data.bfill(inplace=True)
        print("Data preparation complete.")

    def get_price_slice(self, field: str) -> pd.DataFrame:
        """
        安全地获取某个字段的 DataFrame (e.g., 'Close', 'Open')。
        """
        try:
            return self.price_data.xs(field, axis=1, level=1)
        except KeyError:
            raise KeyError(f"Field '{field}' not found in price data.")

    @abstractmethod
    def generate_signals(self, date: pd.Timestamp) -> list:
        """
        [子类必须实现] 核心策略逻辑。
        根据指定日期的数据，生成目标持仓列表。
        返回: list[str] - 目标持仓的 ticker 列表。空列表代表空仓。
        """
        raise NotImplementedError

    def run_backtest(self, start_date: str, end_date: str):
        """
        【核心】执行事件驱动的回测。
        交易逻辑：T-1日收盘后产生信号，在T日开盘时调仓。
        收益计算：T日开盘价 -> T+1日开盘价 (Open-to-Open)。
        """
        self._prepare_data(start_date, end_date)
        
        backtest_range = self.price_data.loc[start_date:end_date].index
        open_prices = self.get_price_slice('Open')
        
        portfolio_values = []
        cash = 1.0
        current_holdings = []
        
        print(f"Running backtest from {start_date} to {end_date}...")
        for i in range(len(backtest_range) - 1):
            date_today = backtest_range[i]
            date_tomorrow = backtest_range[i+1]
            
            # 1. 生成信号 (使用今天之前的所有可用数据)
            # 信号是在今天收盘后生成的，用于指导明天的交易
            target_holdings = self.generate_signals(date_today)
            
            # 2. 计算交易成本
            if set(current_holdings) != set(target_holdings):
                turnover = len(set(current_holdings) ^ set(target_holdings))
                # 简化模型：每次换手都收取固定比例的费用
                cost = (turnover / (len(self.tickers) * 2)) * self.commission_rate if self.tickers else 0
                cash *= (1 - cost)

            current_holdings = target_holdings
            
            # 3. 计算当日收益
            daily_return = 0.0
            if current_holdings:
                weight = 1.0 / len(current_holdings)
                for ticker in current_holdings:
                    open_today = open_prices.at[date_today, ticker]
                    open_tomorrow = open_prices.at[date_tomorrow, ticker]
                    if pd.notna(open_today) and open_today > 0:
                        daily_return += (open_tomorrow / open_today - 1) * weight
            
            # 4. 更新投资组合净值
            cash *= (1 + daily_return)
            portfolio_values.append(cash)

        # 5. 整理回测结果
        self.results = pd.DataFrame(index=backtest_range[:-1])
        self.results['strategy'] = portfolio_values
        
        if self.benchmark_ticker:
            benchmark_returns = open_prices[self.benchmark_ticker].pct_change().shift(-1)
            self.results['benchmark'] = (1 + benchmark_returns.loc[self.results.index]).cumprod()
        
        print("Backtest finished.")
        self.display_metrics()

    def generate_live_signal(self):
        """
        为下一个交易日生成实盘信号。
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d') # 获取过去一年的数据
        
        self._prepare_data(start_date, end_date, buffer_days=0) # 实盘不需要额外缓冲
        
        latest_date = self.price_data.index[-1]
        print(f"\n--- Generating Signal for Next Trading Day (based on data up to {latest_date.date()}) ---")
        
        target_holdings = self.generate_signals(latest_date)
        
        if not target_holdings:
            print("Signal: [CASH]. Market conditions suggest holding cash.")
        else:
            print(f"Signal: [HOLD/BUY]. Target portfolio (equal weight):")
            for ticker in target_holdings:
                print(f" - {ticker}")
        print("------------------------------------------------------------------")

    def display_metrics(self):
        """计算并显示关键绩效指标。"""
        if self.results is None:
            print("No backtest results to display.")
            return

        metrics = pd.DataFrame()
        for col in self.results.columns:
            series = self.results[col].dropna()
            days = (series.index[-1] - series.index[0]).days
            
            # 累计收益率
            cum_return = series.iloc[-1] / series.iloc[0] - 1
            # 年化收益率
            ann_return = (1 + cum_return) ** (365.25 / days) - 1
            # 最大回撤
            rolling_max = series.expanding().max()
            drawdown = (series - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            # 夏普比率
            returns = series.pct_change().dropna()
            sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() != 0 else 0.0

            metrics[col] = [f"{cum_return:.2%}", f"{ann_return:.2%}", f"{max_drawdown:.2%}", f"{sharpe:.2f}"]
        
        metrics.index = ["Cumulative Return", "Annualized Return", "Max Drawdown", "Sharpe Ratio"]
        print("\n--- Backtest Performance ---")
        print(metrics)
        print("--------------------------")

    def plot_results(self):
        """可视化回测结果。"""
        if self.results is None:
            print("No backtest results to plot.")
            return
        
        self.results.plot(figsize=(14, 7), grid=True, title="Strategy Performance vs. Benchmark")
        plt.ylabel("Cumulative Value")
        plt.xlabel("Date")
        plt.show()