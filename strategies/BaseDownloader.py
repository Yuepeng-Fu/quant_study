# BaseDownloader.py
from abc import ABC, abstractmethod
import pandas as pd
import yfinance as yf

class BaseDownloader(ABC):
    """
    数据下载器抽象基类
    负责提供 OHLCV 数据，供策略回测与实盘使用。
    """
    @abstractmethod
    def download(self, tickers: list, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        下载指定标的的历史数据。
        返回: pandas.DataFrame - MultiIndex 格式
        """
        pass

class YFinanceDownloader(BaseDownloader):
    """
    使用 yfinance 库实现的具体数据下载器。
    """
    def __init__(self) -> None:
        super().__init__()
        import os
        os.environ["HTTP_PROXY"] = "https://127.0.0.1:7890"
        os.environ["HTTPS_PROXY"] = "https://127.0.0.1:7890"

    def download(self, tickers: list, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        print(f"Downloading data for {tickers} from {start_date} to {end_date}...")
        try:
            # yfinance returns a DataFrame with a MultiIndex header if multiple tickers are passed
            data = yf.download(tickers, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                print("Warning: No data downloaded. Check tickers and date range.")
                return None
            
            # The format from yfinance is already (Field, Ticker). We need to swap levels.
            # And ensure our standard column names are used.
            data.columns = data.columns.swaplevel(0, 1)
            data.sort_index(axis=1, level=0, inplace=True)
            return data
        except Exception as e:
            print(f"An error occurred during download: {e}")
            return None