import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from datetime import datetime, timedelta
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
# 设置中文字体，以便图表能正确显示中文
def set_chinese_font():
    """
    设置matplotlib以支持中文显示。
    在大多数系统中，'SimHei' 是一个常见的中文字体。
    如果您的系统中没有此字体，请替换为其他可用的中文字体，如 'Microsoft YaHei'。
    """
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
    except Exception as e:
        print(f"设置中文字体失败，请确保您的系统安装了'SimHei'或手动指定其他中文字体: {e}")
        # 如果找不到SimHei，可以尝试其他字体
        # plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

class ETFMonthlyRotationStrategy:
    """
    低频行业ETF轮动策略回测框架
    """
    def __init__(self, etf_tickers, etf_names, benchmark_ticker, market_timing_ticker,
                 momentum_lookback=120, sma_lookback=200, top_n=2, 
                 commission_rate=7e-5):
        self.etf_tickers = etf_tickers
        self.etf_names = etf_names
        self.benchmark_ticker = benchmark_ticker
        self.market_timing_ticker = market_timing_ticker
        self.all_tickers = list(set(etf_tickers + [benchmark_ticker, market_timing_ticker]))
        self.momentum_lookback = momentum_lookback
        self.sma_lookback = sma_lookback
        self.top_n = top_n
        self.commission_rate = commission_rate
        self.open_prices = None
        self.close_prices = None
        self.results = None

    def _download_data(self, start_date, end_date):
        """
        MODIFIED: Downloads both Open and Close prices.
        """
        print(f"Downloading historical Open/Close prices from {start_date} to {end_date}...")
        try:
            # Download the full dataset without slicing for 'Close'
            raw_data = yf.download(self.all_tickers, start=start_date, end=end_date, progress=False)
            if raw_data.empty: return False

            self.open_prices = raw_data['Open']
            self.close_prices = raw_data['Close']

            # Forward-fill and back-fill missing values for both datasets
            for df in [self.open_prices, self.close_prices]:
                if df.isnull().values.any():
                    df.fillna(method='ffill', inplace=True)
                    df.fillna(method='bfill', inplace=True)

            print("Data download complete.")
            return True
        except Exception as e:
            print(f"Data download failed: {e}")
            return False


    def _calculate_momentum_score(self, date):
        """Calculates momentum score based on close prices."""
        try:
            lookback_data = self.close_prices.loc[:date].tail(self.momentum_lookback)
            if len(lookback_data) < self.momentum_lookback: return None
            returns = lookback_data.pct_change().dropna()
            total_return = (1 + returns).prod() - 1
            volatility = returns.std() * np.sqrt(252)
            momentum_score = total_return / (volatility + 1e-6)
            return momentum_score[self.etf_tickers]
        except Exception as e:
            # print(f"Error calculating momentum score on {date}: {e}")
            return None

    def _market_timing_signal(self, date):
        """Generates timing signal based on close prices."""
        try:
            market_data = self.close_prices[self.market_timing_ticker].loc[:date].tail(self.sma_lookback)
            if len(market_data) < self.sma_lookback: return False
            current_price = market_data.iloc[-1]
            sma = market_data.mean()
            return current_price > sma
        except Exception as e:
            # print(f"Error calculating market timing signal on {date}: {e}")
            return False

    def get_target_holdings(self, date):
        """
        【新增】策略核心决策函数 (可复用)
        根据指定日期的数据，返回当天的目标持仓列表。
        :param date: pd.Timestamp, 进行决策的日期
        :return: list, 目标持仓的ticker列表
        """
        is_bull_market = self._market_timing_signal(date)
        if not is_bull_market:
            return []
        momentum_scores = self._calculate_momentum_score(date)
        if momentum_scores is None:
            return []
        target_holdings = momentum_scores.nlargest(self.top_n).index.tolist()
        return target_holdings
    
    def generate_trading_signal_for_tomorrow(self):
        """【新增】生成明日交易信号的实盘函数"""
        print("\n" + "="*50)
        print(f"正在为明天生成交易计划...")
        
        # 1. 准备数据
        end_date = datetime.now() - pd.Timedelta(days=1)
        start_date = end_date - pd.Timedelta(days=max(self.momentum_lookback, self.sma_lookback) + 100)
        if not self._download_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')):
            print("无法生成信号。")
            return

        # 2. 获取最新交易日并调用核心决策逻辑
        latest_date = self.close_prices.index[-1]
        print(f"使用截至 {latest_date.date()} 的收盘数据进行分析。")
        target_holdings = self.get_target_holdings(latest_date)
        
        # 3. 格式化输出交易计划
        print("\n--- 明日交易计划 ---")
        if not target_holdings:
            print("\033[91m[清仓/空仓信号]\033[0m 市场处于熊市或数据不足，建议空仓。")
        else:
            print("\033[92m[持仓/调仓信号]\033[0m 市场处于牛市，目标持仓如下:")
            target_names = [self.etf_names.get(ticker, ticker) for ticker in target_holdings]
            for i, name in enumerate(target_names):
                print(f"  {i+1}. {name} ({target_holdings[i]})")
            print(f"\n请将仓位调整为以上 {self.top_n} 只ETF，等权重持有。")
        print("="*50 + "\n")


    def run_backtest(self, start_date, end_date):
        """
        REWRITTEN: Executes the backtest using Open-to-Open returns.
        Signal is generated on day T-1's close, trade is executed on day T's open.
        """
        # 1. Download data
        required_buffer = int(max(self.momentum_lookback, self.sma_lookback) * 1.5 + 50)
        effective_start_date = (pd.to_datetime(start_date) - timedelta(days=required_buffer)).strftime('%Y-%m-%d')
        if not self._download_data(effective_start_date, end_date):
            return

        # 2. Slice data for the actual backtest period
        backtest_opens = self.open_prices.loc[start_date:end_date]
        backtest_closes = self.close_prices.loc[start_date:end_date]
        if backtest_opens.empty:
            print("Error: No data in the specified backtest range.")
            return
        
        print("Starting backtest with Open-to-Open logic...")
        
        portfolio_values = []
        current_holdings = []
        cash = 1.0
        
        # --- Main Backtest Loop ---
        # We loop up to the second to last day because we need the next day's open for the return calculation
        for i in range(len(backtest_opens) - 1):
            date_today = backtest_opens.index[i]
            date_tomorrow = backtest_opens.index[i+1]
            
            # --- Signal & Rebalancing Logic ---
            # Signal is based on the CLOSE of the PREVIOUS day.
            # We use i-1 index on the full close_prices dataframe.
            previous_day_close_date = backtest_closes.index[i-1] if i > 0 else None
            
            if previous_day_close_date:
                last_holdings = current_holdings.copy()
                # Get today's target holdings based on yesterday's close
                current_holdings = self.get_target_holdings(previous_day_close_date)

                # Apply commission on turnover
                turnover_tickers = set(last_holdings) ^ set(current_holdings)
                if turnover_tickers and self.top_n > 0:
                    turnover_fraction = len(turnover_tickers) / (self.top_n * 2)
                    commission = turnover_fraction * self.commission_rate
                    cash *= (1 - commission)

            # --- Return Calculation Logic (Open of Today -> Open of Tomorrow) ---
            daily_return = 0.0
            if current_holdings:
                weights = 1.0 / len(current_holdings)
                open_today = backtest_opens.loc[date_today]
                open_tomorrow = backtest_opens.loc[date_tomorrow]
                
                for ticker in current_holdings:
                    # Ensure prices are valid
                    if not pd.isna(open_today[ticker]) and not pd.isna(open_tomorrow[ticker]) and open_today[ticker] > 0:
                        daily_return += weights * (open_tomorrow[ticker] / open_today[ticker] - 1)
            
            # Update portfolio value
            cash *= (1 + daily_return)
            portfolio_values.append(cash)

        # Append the final cash value for the last day
        if portfolio_values:
            portfolio_values.append(cash)

        # --- Benchmark Calculation (Open-to-Open) ---
        benchmark_returns = (backtest_opens[self.benchmark_ticker].shift(-1) / backtest_opens[self.benchmark_ticker] -1)
        benchmark_values = (1 + benchmark_returns).cumprod()
        # Align the start to 1.0
        benchmark_values.iloc[0] = 1.0
        benchmark_values.fillna(method='ffill', inplace=True)
        
        self.results = pd.DataFrame({
            'strategy': portfolio_values
        }, index=backtest_opens.index)
        self.results['benchmark'] = benchmark_values
        
        print("Backtest finished.")

    def plot_results(self):
        """可视化回测结果"""
        if self.results is None:
            print("请先运行回测。")
            return

        print("正在绘制结果图表...")
        set_chinese_font() # 设置中文环境

        plt.figure(figsize=(14, 7))
        plt.plot(self.results.index, self.results['strategy'], label='行业轮动策略')
        plt.plot(self.results.index, self.results['benchmark'], label=f'基准 ({self.benchmark_ticker})', alpha=0.7)
        plt.title('行业轮动策略 vs. 基准指数', fontsize=16)
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('累计净值', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.show()

    def display_metrics(self):
        """显示关键绩效指标"""
        if self.results is None:
            print("请先运行回测。")
            return
            
        metrics = pd.DataFrame()
        
        # 计算年化收益率
        days = (self.results.index[-1] - self.results.index[0]).days
        metrics['年化收益率'] = [
            (self.results['strategy'].iloc[-1] ** (365.25 / days)) - 1,
            (self.results['benchmark'].iloc[-1] ** (365.25 / days)) - 1
        ]

        # 计算累计收益率
        metrics['累计收益率'] = [
            self.results['strategy'].iloc[-1] - 1,
            self.results['benchmark'].iloc[-1] - 1
        ]

        # 计算最大回撤
        def max_drawdown(series):
            roll_max = series.expanding().max()
            daily_drawdown = series / roll_max - 1.0
            return daily_drawdown.min()

        metrics['最大回撤'] = [
            max_drawdown(self.results['strategy']),
            max_drawdown(self.results['benchmark'])
        ]
        
        # 计算夏普比率 (假设无风险利率为0)
        def sharpe_ratio(series):
            returns = series.pct_change().dropna()
            return (returns.mean() * 252) / (returns.std() * np.sqrt(252))

        metrics['夏普比率'] = [
            sharpe_ratio(self.results['strategy']),
            sharpe_ratio(self.results['benchmark'])
        ]

        metrics.index = ['策略', '基准']
        
        # 格式化输出
        for col in metrics.columns:
            if '收益率' in col or '回撤' in col:
                metrics[col] = metrics[col].apply(lambda x: f"{x:.2%}")
            else:
                metrics[col] = metrics[col].apply(lambda x: f"{x:.2f}")

        print("\n--- 策略绩效评估 ---")
        print(metrics)
        print("--------------------")


if __name__ == '__main__':
    # --- 策略参数配置 ---

    # 示例1：中国A股市场ETF配置 (需要您的环境能访问.ss和.sz后缀的数据)
    # 请注意：yfinance对A股数据的支持可能不稳定，建议使用国内数据源如Tushare, Akshare等进行实盘研究
    a_share_etfs = {
        '科技': '515790.SS',
        '消费': '510630.SS',
        '医疗': '512170.SS',
        '金融': '512800.SS',
        '券商': '512000.SS',
        '新能源': '515700.SS',
        '半导体':'512480.SS',
        '军工': '512680.SS',
        '黄金': '518680.SS',
        '云计算': '159890.SZ',
        '电力': '159611.SZ',
        '房地产': '512200.SS',
        '原材料': '512400.SS',
        '传统能源': '515220.SS',
        '基础设施': '516950.SS',
        '保险': '515630.SS'
    }
    a_share_benchmark = '000300.SS' # 沪深300指数

    # 示例2：为了确保代码可直接运行，我们使用美国市场的ETF作为演示
    us_sector_etfs = {
        '科技': 'XLK',
        '医疗': 'XLV',
        '金融': 'XLF',
        '消费': 'XLY',
        '能源': 'XLE',
        '工业': 'XLI',
        '公用事业': 'XLU'
    }
    us_benchmark = 'SPY' # 标普500 ETF

    # --- 选择并运行策略 ---
    # 您可以在这里切换使用 a_share_etfs 或 us_sector_etfs
    selected_etfs = a_share_etfs 
    selected_benchmark = a_share_benchmark

    strategy = ETFMonthlyRotationStrategy(
        etf_tickers=list(selected_etfs.values()),
        etf_names=a_share_etfs,
        benchmark_ticker=selected_benchmark,
        market_timing_ticker=selected_benchmark, # 使用大盘本身做择时
        momentum_lookback=60,    # 动量回看期
        sma_lookback=100,
        top_n=4
    )

    strategy.generate_trading_signal_for_tomorrow()

    # strategy.run_backtest(start_date='2025-01-01', end_date='2025-09-03')
    # strategy.display_metrics()
    # strategy.plot_results()
