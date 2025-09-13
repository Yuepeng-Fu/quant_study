# AkshareDownloader.py
import pandas as pd
import akshare as ak
from collections import defaultdict
from BaseDownloader import BaseDownloader

# --- Configuration Map for different Akshare products ---
PRODUCT_MAP = {
    'stock': {
        'api': ak.stock_zh_a_hist,
        'valid_params': ['period', 'adjust'],
        'col_map': {'日期': 'Date', '开盘': 'Open', '收盘': 'Close', '最高': 'High', '最低': 'Low', '成交量': 'Volume'}
    },
    'etf': {
        'api': ak.fund_etf_hist_em,
        'valid_params': ['adjust'],
        'col_map': {'日期': 'Date', '开盘': 'Open', '收盘': 'Close', '最高': 'High', '最低': 'Low', '成交量': 'Volume'}
    },
    'index': {
        'api': ak.index_zh_a_hist,
        'valid_params': ['period'],
        'col_map': {'日期': 'Date', '开盘': 'Open', '收盘': 'Close', '最高': 'High', '最低': 'Low', '成交量': 'Volume'}
    }
}

class AkshareDownloader(BaseDownloader):
    """
    A robust downloader that uses symbol set lookups for accurate product type
    identification of stocks, ETFs, and indexes.
    """
    def __init__(self):
        self._build_symbol_maps()

    def _build_symbol_maps(self):
        """
        Fetches all valid symbols for each asset type on initialization
        to ensure accurate lookups.
        """
        print("Initializing AkshareDownloader: building symbol maps...")
        try:
            # Fetch all A-share stocks
            stock_df = ak.stock_info_a_code_name()
            self.stock_symbols = set(stock_df['code'])
            
            # Fetch all ETFs
            etf_df = ak.fund_etf_spot_em()
            self.etf_symbols = set(etf_df['代码'])

            # Fetch all Indexes
            index_df = ak.index_stock_info()
            self.index_symbols = set(index_df['index_code'])
            
            print("Symbol maps built successfully.")
        except Exception as e:
            print(f"Error building symbol maps: {e}")
            print("Downloader may not be able to identify all ticker types.")
            self.stock_symbols = set()
            self.etf_symbols = set()
            self.index_symbols = set()
            
    def _transform_tickers(self, tickers: list) -> list:
        return [t.split('.')[0] for t in tickers]

    def _get_product_type(self, ticker: str) -> str | None:
        """
        Accurately identifies the product type by checking for membership
        in the pre-fetched symbol sets.
        """
        # Order is important for symbols that might overlap (unlikely but possible)
        if ticker in self.stock_symbols:
            return 'stock'
        elif ticker in self.etf_symbols:
            return 'etf'
        elif ticker in self.index_symbols:
            return 'index'
        else:
            print(f"Warning: Ticker '{ticker}' not found in any symbol map. Skipping.")
            return None

    def download(self, tickers: list, start_date: str = None, end_date: str = None, **kwargs) -> pd.DataFrame:
        # Create a map from the API-specific ticker back to the user's original ticker.
        api_tickers = self._transform_tickers(tickers)
        canonical_map = dict(zip(api_tickers, tickers))

        grouped_tickers = defaultdict(list)
        for ticker in api_tickers:
            product_type = self._get_product_type(ticker)
            if product_type:
                grouped_tickers[product_type].append(ticker)

        if not grouped_tickers: return None

        dfs_to_combine = []
        api_tickers_in_order = []
        ak_start_date = start_date.replace('-', '') if start_date else ''
        ak_end_date = end_date.replace('-', '') if end_date else ''
        
        for p_type, t_list in grouped_tickers.items():
            config = PRODUCT_MAP[p_type]
            api_func, col_map = config['api'], config['col_map']
            api_params = {k: v for k, v in kwargs.items() if k in config['valid_params']}
            
            print(f"\n--- Downloading {p_type.upper()} data for: {t_list} with params: {api_params} ---")
            
            for ticker in t_list:
                try:
                    if p_type == 'index':
                        df = api_func(symbol=ticker, period=api_params.get('period', 'daily'), start_date=ak_start_date, end_date=ak_end_date)
                    else:
                        df = api_func(symbol=ticker, start_date=ak_start_date, end_date=ak_end_date, **api_params)

                    if not df.empty:
                        df.rename(columns=col_map, inplace=True)
                        df['Date'] = pd.to_datetime(df['Date'])
                        df.set_index('Date', inplace=True)
                        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                        dfs_to_combine.append(df)
                        api_tickers_in_order.append(ticker)
                except Exception as e:
                    print(f"Error downloading {ticker}: {e}")
        
        if not dfs_to_combine: return None
        
        # Use the canonical map to create keys for the final DataFrame.
        # This ensures the output columns match the user's input.
        canonical_keys = [canonical_map[t] for t in api_tickers_in_order]
        
        final_df = pd.concat(dfs_to_combine, axis=1, keys=canonical_keys)
        final_df = final_df.swaplevel(0, 1, axis=1)
        final_df.sort_index(axis=1, level=0, inplace=True)
        return final_df

if __name__ == "__main__":
    print("\n--- Running Tests for AkshareDownloader with Symbol Set Lookups ---")
    downloader = AkshareDownloader()
    
    # Using a mixed list including a stock, an ETF, and an Index
    canonical_tickers = [
        '600519.SS',  # 贵州茅台 (Stock)
        '510300.SS',  # 沪深300 ETF (ETF)
        '000300.SH'  # 沪深300 指数 (Index), akshare uses .SH for Shanghai
    ]
    
    start = '2025-09-01'
    end = '2025-09-12'
    
    final_data = downloader.download(canonical_tickers, start, end, period='daily', adjust='hfq')

    if final_data is not None:
        print("\n--- Test Verification ---")
        try:
            final_tickers = final_data.columns.get_level_values(1)
            assert '600519' in final_tickers
            assert '510300' in final_tickers
            assert '000300' in final_tickers
            print("[PASS] All asset types (Stock, ETF, Index) downloaded successfully.")
            print("Final DataFrame 'Close' data:\n", final_data['Close'].head())
        except AssertionError as e:
            print(f"[FAIL] Verification failed: {e}")