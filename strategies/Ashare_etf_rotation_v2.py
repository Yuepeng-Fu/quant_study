import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from datetime import datetime, timedelta
# import pandas_ta as ta # NEW: Import for ADX calculation

from akshare_downloader import AkshareDownloader
from BaseStrategy import BaseStrategy

# Function to set Chinese font (unchanged)
def set_chinese_font():
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f"设置中文字体失败: {e}")

class ETFMonthlyRotationStrategyV2(BaseStrategy):
    """
    MODIFIED: Upgraded ETF Rotation Strategy with Multi-Factor Model,
    Market Regime Detection, and Delayed Exit Filter.
    """
    def __init__(self, etf_tickers, downloader, etf_names, benchmark_ticker, market_timing_ticker,
                 momentum_lookback=60, sma_lookback=100, top_n=4, commission_rate=7e-5,
                 # NEW parameters for V2 improvements
                 adx_lookback=14, adx_threshold=20,          # For market regime detection
                 volume_short_window=20, volume_long_window=60, # For volume factor
                 momentum_weight=0.6, volume_weight=0.4,       # Factor weights
                 exit_filter_sma=10):                         # For delayed exit filter
        
        super().__init__(
            tickers=etf_tickers,
            downloader=downloader,
            benchmark_ticker=benchmark_ticker,
            commission_rate=commission_rate
        )
        self.etf_tickers = etf_tickers
        self.etf_names = etf_names
        self.benchmark_ticker = benchmark_ticker
        self.market_timing_ticker = market_timing_ticker
        self.all_tickers = list(set(etf_tickers + [benchmark_ticker, market_timing_ticker]))
        
        # MODIFIED: Assign all new parameters to self
        self.momentum_lookback = momentum_lookback
        self.sma_lookback = sma_lookback
        self.top_n = top_n
        self.adx_lookback = adx_lookback
        self.adx_threshold = adx_threshold
        self.volume_short_window = volume_short_window
        self.volume_long_window = volume_long_window
        self.momentum_weight = momentum_weight
        self.volume_weight = volume_weight
        self.exit_filter_sma = exit_filter_sma
        
        self.open_prices = None
        self.close_prices = None
        self.volume = None # NEW: Store volume data
        self.hloc_data = None # NEW: Store HLOC for ADX
        self.results = None

    def _prepare_data(self, start_date, end_date):
        """
        MODIFIED: Downloads Open, High, Low, Close, and Volume prices.
        """
        print(f"Downloading historical data from {start_date} to {end_date}...")
        try:
            # Download the full HLOCV dataset
            raw_data = self.downloader.download(
                self.all_tickers,
                start_date=start_date,
                end_date=end_date
            )
            if raw_data.empty: return False

            self.open_prices = raw_data['Open']
            self.close_prices = raw_data['Close']
            self.volume = raw_data['Volume']
            self.hloc_data = raw_data # Store all data for ADX calculation

            for df in [self.open_prices, self.close_prices, self.volume]:
                if df.isnull().values.any():
                    df.fillna(method='ffill', inplace=True)
                    df.fillna(method='bfill', inplace=True)

            print("Data download complete.")
            return True
        except Exception as e:
            print(f"Data download failed: {e}")
            return False
        
    def _calculate_adx(self, high, low, close, window):
        """
        Calculates the Average Directional Index (ADX) manually.
        """
        plus_dm = high.diff()
        minus_dm = low.diff().mul(-1)
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        plus_dm[minus_dm > plus_dm] = 0
        minus_dm[plus_dm > minus_dm] = 0

        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift(1)))
        tr3 = pd.DataFrame(abs(low - close.shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
        
        # Using Exponential Moving Average (EMA) for smoothing
        atr = tr.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/window, min_periods=window, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/window, min_periods=window, adjust=False).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
        
        return adx


    # --- Improvement 1: Multi-Factor Model ---
    def _calculate_composite_score(self, date):
        """
        NEW: Calculates a composite score based on multiple factors.
        This implements the "构建多因子行业选择模型" idea.
        """
        # Factor 1: Risk-Adjusted Momentum (Original logic)
        try:
            lookback_data = self.close_prices.loc[:date].tail(self.momentum_lookback)
            if len(lookback_data) < self.momentum_lookback: return None
            returns = lookback_data.pct_change().dropna()
            total_return = (1 + returns).prod() - 1
            volatility = returns.std() * np.sqrt(252)
            momentum_score = total_return / (volatility + 1e-6)
        except Exception:
            return None

        # Factor 2: Volume Factor (Funds Flow Proxy)
        try:
            volume_data = self.volume.loc[:date].tail(self.volume_long_window)
            if len(volume_data) < self.volume_long_window: return None
            short_vol = volume_data.tail(self.volume_short_window).mean()
            long_vol = volume_data.mean()
            volume_score = short_vol / (long_vol + 1e-6)
        except Exception:
            volume_score = pd.Series(0, index=momentum_score.index)

        # Combine factors using ranking and weights
        momentum_rank = momentum_score.rank(pct=True)
        volume_rank = volume_score.rank(pct=True)
        
        composite_score = (self.momentum_weight * momentum_rank) + (self.volume_weight * volume_rank)
        
        return composite_score[self.etf_tickers]

    # --- Improvement 2: Market Regime Detection ---
    def _get_market_regime(self, date):
        """
        MODIFIED: Now uses the internal _calculate_adx method.
        """
        try:
            # 1. Trend Direction (SMA)
            market_close = self.close_prices[self.market_timing_ticker].loc[:date]
            if len(market_close) < self.sma_lookback: return 'BEAR'
            sma = market_close.tail(self.sma_lookback).mean()
            
            if market_close.iloc[-1] < sma:
                return 'BEAR'

            # 2. Trend Strength (ADX)
            market_hloc = self.hloc_data.loc[:date, (slice(None), self.market_timing_ticker)]
            market_hloc.columns = market_hloc.columns.droplevel(1)
            
            # Call the new manual ADX function
            adx = self._calculate_adx(market_hloc['High'], market_hloc['Low'], market_hloc['Close'], window=self.adx_lookback)
            
            if adx.empty or pd.isna(adx.iloc[-1]): return 'SIDEWAYS'
            
            current_adx = adx.iloc[-1]
            
            if current_adx < self.adx_threshold:
                return 'SIDEWAYS'
            else:
                return 'BULL'
                
        except Exception:
            return 'BEAR'

    # MODIFIED: Core logic now uses the regime and composite score
    def generate_signals(self, date):
        market_regime = self._get_market_regime(date)
        
        if market_regime in ['BEAR', 'SIDEWAYS']:
            return [] # Go to cash
        
        composite_scores = self._calculate_composite_score(date)
        if composite_scores is None:
            return []
            
        return composite_scores.nlargest(self.top_n).index.tolist()
    
    # MODIFIED: run_backtest to include the Delayed Exit Filter
    def run_backtest(self, start_date, end_date):
        """
        MODIFIED: Executes the backtest with the Delayed Exit Filter.
        This implements the "增加短期趋势确认层" idea.
        """
        required_buffer = int(max(self.momentum_lookback, self.sma_lookback, self.volume_long_window) * 1.5)
        effective_start_date = (pd.to_datetime(start_date) - timedelta(days=required_buffer)).strftime('%Y-%m-%d')
        if not self._prepare_data(effective_start_date, end_date):
            return

        backtest_opens = self.open_prices.loc[start_date:end_date]
        backtest_closes = self.close_prices.loc[start_date:end_date]
        if backtest_opens.empty: return
        
        print("Starting V2 backtest with Multi-Factor, Regime Detection, and Delayed Exit...")
        
        portfolio_values = []
        current_holdings = []
        cash = 1.0
        
        for i in range(len(backtest_opens) - 1):
            date_today = backtest_opens.index[i]
            date_tomorrow = backtest_opens.index[i+1]
            previous_day_close_date = backtest_closes.index[i-1] if i > 0 else None
            
            if previous_day_close_date:
                last_holdings = current_holdings.copy()
                
                # 1. Get ideal target holdings from core logic
                ideal_target_holdings = self.generate_signals(previous_day_close_date)

                # --- Improvement 3: Delayed Exit Filter ---
                final_holdings = list(ideal_target_holdings)
                sell_candidates = set(last_holdings) - set(ideal_target_holdings)
                
                for ticker in sell_candidates:
                    try:
                        short_term_data = self.close_prices[ticker].loc[:previous_day_close_date]
                        if len(short_term_data) < self.exit_filter_sma: continue
                        
                        short_sma = short_term_data.tail(self.exit_filter_sma).mean()
                        current_price = short_term_data.iloc[-1]
                        
                        if current_price > short_sma:
                            final_holdings.append(ticker) # Override sell signal
                    except Exception:
                        continue
                
                current_holdings = sorted(list(set(final_holdings)))
                # ----------------------------------------

                # Apply commission
                turnover_tickers = set(last_holdings) ^ set(current_holdings)
                if turnover_tickers and len(last_holdings) + len(current_holdings) > 0:
                    turnover_value = len(turnover_tickers)
                    max_positions = self.top_n * 2
                    turnover_fraction = turnover_value / max_positions if max_positions > 0 else 0
                    commission = turnover_fraction * self.commission_rate
                    cash *= (1 - commission)

            # --- Return Calculation ---
            daily_return = 0.0
            if current_holdings:
                weights = 1.0 / len(current_holdings)
                open_today = backtest_opens.loc[date_today]
                open_tomorrow = backtest_opens.loc[date_tomorrow]
                
                for ticker in current_holdings:
                    if not pd.isna(open_today[ticker]) and not pd.isna(open_tomorrow[ticker]) and open_today[ticker] > 0:
                        daily_return += weights * (open_tomorrow[ticker] / open_today[ticker] - 1)
            
            cash *= (1 + daily_return)
            portfolio_values.append(cash)

        if not portfolio_values:
            print("Backtest did not generate any portfolio values.")
            return

        portfolio_values.append(cash)
        self.results = pd.DataFrame({'strategy': portfolio_values}, index=backtest_opens.index)
        
        # Benchmark
        benchmark_returns = (backtest_opens[self.benchmark_ticker].shift(-1) / backtest_opens[self.benchmark_ticker] - 1)
        benchmark_values = (1 + benchmark_returns).cumprod().fillna(method='ffill')
        if not benchmark_values.empty:
            benchmark_values.iloc[0] = 1.0
            self.results['benchmark'] = benchmark_values
        
        print("Backtest finished.")

    

if __name__ == '__main__':
    a_share_etfs = {
        '科技': '515790.SS', '消费': '510630.SS', '医疗': '512170.SS',
        '金融': '512800.SS', '券商': '512000.SS', '新能源': '515700.SS',
        '半导体':'512480.SS', '军工': '512680.SS', '黄金': '518680.SS',
        '云计算': '159890.SZ', '电力': '159611.SZ', '房地产': '512200.SS',
        '原材料': '512400.SS', '传统能源': '515220.SS', '基础设施': '516950.SS',
        '保险': '515630.SS', '电池': '159755.SZ', '化工': '159870.SZ',
    }
    a_share_benchmark = '000300.SS'

    # MODIFIED: Initialize the new V2 strategy class with the new parameters
    strategy_v2 = ETFMonthlyRotationStrategyV2(
        etf_tickers=list(a_share_etfs.values()),
        downloader=AkshareDownloader(),
        etf_names=a_share_etfs,
        benchmark_ticker=a_share_benchmark,
        market_timing_ticker=a_share_benchmark,
        
        # --- Base Parameters ---
        momentum_lookback=60,
        sma_lookback=100,
        top_n=4,
        
        # --- NEW V2 Parameters (Tunable) ---
        adx_lookback=14,
        adx_threshold=20,
        volume_short_window=20,
        volume_long_window=60,
        momentum_weight=0.6,
        volume_weight=0.4,
        exit_filter_sma=10
    )

    # To run the backtest (you'll need to add display_metrics and plot_results to the class)
    strategy_v2.run_backtest(start_date='2023-01-01', end_date='2025-09-11')
    strategy_v2.display_metrics()
    strategy_v2.plot_results()

    # To generate a live signal
    strategy_v2.generate_live_signal()