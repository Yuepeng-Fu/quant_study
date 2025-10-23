import akshare as ak
import pandas as pd
from datetime import datetime, time, timedelta
import pytz  # For timezone handling
import chinese_calendar # For holiday checks
from concurrent.futures import ThreadPoolExecutor, as_completed
import time as os_time # Renamed to avoid conflict with datetime.time
import random
import requests
from tqdm import tqdm
import os
import sys
import math

# ==============================
# CONFIG SECTION
# ==============================
config = {
    "csv_path": "/root/quant_study/data",
    "threads": 20,
    "market_close_time": time(15, 30, 00) # 3:30 PM China time
}

# Cookies are required for the historical data fetch function
cookie_str = "qgqp_b_id=4778b180c0a11c4091155ac08407f275; st_nvi=XEI7BwZs9h6am0h0mdM3xe8c7; nid=08400045b30bb4ed0fcfd05a884abfa0; nid_create_time=1756723409946; gvi=y7PysGxNbwKsPqEbQ7iON49d9; gvi_create_time=1756723409946; mtp=1; ct=XWf0B7Xf2HTas5EPxLmGC5vVPHnugz2Cag83ZOMCMJUK_ItXAdexTWVLjLVzkmObGsQaRCBVF4uxqMWhryT8OGy576C14iNTR_Z2wPZyrVsKU7LUMjL0z3xo7sMQSLMTk3u-xCc9TPrL723RxO54cT61asZQWbuBRoR5bUOajQg; ut=FobyicMgeV5FJnFT189SwJ0Kokjjt5PZVUwO_2TvRBsFhLd1nZLhKCKpPGGcQIGrq2CAAsWncDVAyKv3o5CmcClDZ-hw3aub32-ORrLur6WRmqJVTUqYCLE80c74qw2odSmQGwpvKiyQcHhKHrVSaSxcSo8HWfKpYIwurzyUZw7USvDNFZwhtPyLZj_6sN1JYVFbZU47gShJIwxbxDHU0E4dx1e4oPI3FbiS6areGFpsjrrqQdgGeweZmxTCGK9L3z11CfweVM5JPrcGDdPzZnl9aRom_2LIVcfLtn4_JqUcfvVGKVmvbdIU7WDcIg9fxk3otcE4ajNSsixoOaLTxg; pi=9873057608764166%3Bt9873057608764166%3B%E8%82%A1%E5%8F%8B1e125Q1089%3BlgHhCpEK3mmnRXcdsT1Hze109xbpvH6hY79REfrkJWP%2B5whuoSChgqmUnkFVHB9zbe1%2FpmOAuVp1FbbmtWgZiGj%2Fpn3iIPc%2BBUi1UiuOYBailLFdEZ6qOR67Om%2FytpWW3ByAziRv1Kh4AE6qd5DgYIe99AHRtQrTLQlJTGGU30PnyOSQtcvJA9D5Yl69ASm6dacjw51L%3BruvmprO1jpa9ChkH0aRn141ux%2Fb0Dtb%2FYOlSQopQeZKEZDhZSsoq7BP8LR6L10TeS198Z7eE6CVriOlATQ3DZ7H8m5U9NP2Pi2vvCfoxIWPGt0jEyCY4b3PgNhlGsyOihRysaV9Yey7KmASUMIASmOG6%2FxT6LQ%3D%3D; uidal=9873057608764166%e8%82%a1%e5%8f%8b1e125Q1089; _adsame_fullscreen_19333=1; st_si=70015697094549; st_pvi=52957124213199; st_sp=2025-04-08%2010%3A25%3A13; st_inirUrl=https%3A%2F%2Fcn.bing.com%2F; st_sn=1; st_psi=2025102315393453-111000300841-1954095690; st_asi=delete; fullscreengg=1; fullscreengg2=1; wsc_checkuser_ok=1"

cookies_dict = {}
for pair in cookie_str.split("; "):
    if "=" in pair:
        key, value = pair.split("=", 1)
        cookies_dict[key] = value

# This function is needed to fetch full history for new stocks
def fixed_stock_zh_a_hist(
    symbol: str = "000001",
    period: str = "daily",
    start_date: str = "19700101",
    end_date: str = "20500101",
    adjust: str = "",
    timeout: float = None,
) -> pd.DataFrame:
    """
    东方财富网-行情首页-沪深京 A 股-每日行情 (Used for fetching full history)
    """
    market_code = 1 if symbol.startswith("6") else 0
    adjust_dict = {"qfq": "1", "hfq": "2", "": "0"}
    period_dict = {"daily": "101", "weekly": "102", "monthly": "103"}
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "klt": period_dict[period],
        "fqt": adjust_dict[adjust],
        "secid": f"{market_code}.{symbol}",
        "beg": start_date,
        "end": end_date,
    }
    r = requests.get(url, params=params, timeout=timeout, cookies=cookies_dict)
    data_json = r.json()
    if not (data_json["data"] and data_json["data"]["klines"]):
        return pd.DataFrame()
    temp_df = pd.DataFrame([item.split(",") for item in data_json["data"]["klines"]])
    temp_df["股票代码"] = symbol
    temp_df.columns = [
        "日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额",
        "振幅", "涨跌幅", "涨跌额", "换手率", "股票代码",
    ]
    
    # Reorder columns to match our standard CSV format
    temp_df = temp_df[
        [
            "日期", "股票代码", "开盘", "收盘", "最高", "最低",
            "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率",
        ]
    ]
    # Convert types
    for col in ["开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"]:
         temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')

    temp_df["日期"] = pd.to_datetime(temp_df["日期"], errors="coerce").dt.date
    return temp_df

# ===================================================================
# CUSTOM SPOT DATA FUNCTIONS (REFACTORED)
# ===================================================================

def fetch_paginated_data(url: str, base_params: dict, timeout: int = 15):
    """
    东方财富-分页获取数据并合并结果 (Custom version with cookies)
    """
    # 复制参数以避免修改原始参数
    params = base_params.copy()
    
    # Try to get the first page to determine total pages
    r = requests.get(url, params=params, timeout=timeout, cookies=cookies_dict)
    r.raise_for_status() # Raise HTTPError for bad responses
    data_json = r.json()
    
    if not data_json.get("data") or "diff" not in data_json["data"]:
            print(f"Warning: Spot data fetch for params {params.get('fs')} returned no data or 'diff' key missing.")
            return pd.DataFrame()
             

    # Calculate paging information
    per_page_num = len(data_json["data"]["diff"])
    if per_page_num == 0:
        print(f"Warning: Spot data fetch for params {params.get('fs')} returned 0 results on first page.")
        return pd.DataFrame()
        
    total_num = data_json["data"].get("total", 0)
    total_page = math.ceil(total_num / per_page_num)
    
    # Store all page data
    temp_list = []
    # Add first page data
    temp_list.append(pd.DataFrame(data_json["data"]["diff"]))
    
    if total_page > 1:
        # Get remaining pages
        # Use the imported tqdm module
        for page in tqdm(range(2, total_page + 1), leave=False, desc=f"Fetching {params.get('fs', '')} pages"):
            params.update({"pn": str(page)}) # Ensure pn is string
            r = requests.get(url, params=params, timeout=timeout, cookies=cookies_dict)
            r.raise_for_status()
            data_json = r.json()
            if not data_json.get("data") or "diff" not in data_json["data"]:
                break # Stop if data is missing
            
            inner_temp_df = pd.DataFrame(data_json["data"]["diff"])
            temp_list.append(inner_temp_df)

                
    # Merge all data
    if not temp_list:
        return pd.DataFrame()
        
    temp_df = pd.concat(temp_list, ignore_index=True)
    
    # This sorting logic was in your function, keeping it.
    if "f3" in temp_df.columns:
        temp_df["f3"] = pd.to_numeric(temp_df["f3"], errors="coerce")
        temp_df.sort_values(by=["f3"], ascending=False, inplace=True, ignore_index=True)
        
    temp_df.reset_index(inplace=True)
    temp_df["index"] = temp_df["index"].astype(int) + 1
    return temp_df

def custom_stock_all_a_spot_em() -> pd.DataFrame:
    """
    东方财富网 - 沪深京 A 股 - 实时行情 (Merged)
    Fetches SH, SZ, and KC boards in a single API call.
    """
    url = "https://82.push2.eastmoney.com/api/qt/clist/get"
    params = {
        "pn": "1", "pz": "100", "po": "1", "np": "1",
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": "2", "invt": "2", "fid": "f12",
        # fs: m:1 t:2  (Shanghai Main)
        #     m:1 t:23 (Shanghai KC)
        #     m:0 t:6  (Shenzhen Main)
        #     m:0 t:80 (Shenzhen ChiNext)
        "fs": "m:1 t:2,m:1 t:23,m:0 t:6,m:0 t:80", 
        "fields": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,"
                  "f24,f25,f22,f11,f62,f128,f136,f115,f152",
    }
    temp_df = fetch_paginated_data(url, params)
    if temp_df.empty:
        return pd.DataFrame()

    # Define all possible columns (based on the longest list from the original functions)
    temp_df.columns = [
        "序号", "_", "最新价", "涨跌幅", "涨跌额", "成交量", "成交额", "振幅",
        "换手率", "市盈率-动态", "量比", "5分钟涨跌", "代码", "_", "名称",
        "最高", "最低", "今开", "昨收", "总市值", "流通市值", "涨速",
        "市净率", "60日涨跌幅", "年初至今涨跌幅", "-", "-", "-", "-", "-",
        "-", "-",
    ]
    
    # Define the columns we want to keep
    final_columns = [
        "序号", "代码", "名称", "最新价", "涨跌幅", "涨跌额", "成交量",
        "成交额", "振幅", "最高", "最低", "今开", "昨收", "量比",
        "换手率", "市盈率-动态", "市净率", "总市值", "流通市值", "涨速",
        "5分钟涨跌", "60日涨跌幅", "年初至今涨跌幅",
    ]
    temp_df = temp_df[final_columns]
    
    # Define columns to convert to numeric
    numeric_cols = [
        "最新价", "涨跌幅", "涨跌额", "成交量", "成交额", "振幅", "最高", 
        "最低", "今开", "昨收", "量比", "换手率", "市盈率-动态", "市净率",
        "总市值", "流通市值", "涨速", "5分钟涨跌", "60日涨跌幅", "年初至今涨跌幅"
    ]
    
    for col in numeric_cols:
        if col in temp_df.columns:
            temp_df[col] = pd.to_numeric(temp_df[col], errors="coerce")
            
    return temp_df


# ===================================================================
# END OF CUSTOM SPOT DATA FUNCTIONS
# ===================================================================


# This function is needed to get the complete list of stocks
def get_base_stocks():
    """Fetches the complete list of A-share stocks."""
    print("Fetching full A-share list ...")
    try:
        sh_df = ak.stock_info_sh_name_code(symbol="主板A股").rename(
            columns={"证券代码": "code", "证券简称": "name", "上市日期": "上市日期"}
        )
        sz_df = ak.stock_info_sz_name_code(symbol="A股列表").rename(
            columns={"A股代码": "code", "A股简称": "name", "上市日期": "上市日期"}
        )
        kc_df = ak.stock_info_sh_name_code(symbol="科创板").rename(
            columns={"证券代码": "code", "证券简称": "name", "上市日期": "上市日期"}
        )
        
        merged = pd.concat([sh_df, sz_df, kc_df], ignore_index=True)
        merged = merged.dropna(subset=['code'])
        merged['上市日期'] = pd.to_datetime(merged['上市日期'], errors='coerce')
        print(f"Found {len(merged)} total stocks in base list.")
        return merged[['code', 'name', '上市日期']]
    except Exception as e:
        print(f"Error fetching base stock list: {e}")
        return pd.DataFrame()

def get_today_spot_data(today_date_str: str):
    """Fetches today's spot data for all stocks."""
    print("Fetching today's spot data from eastmoney...")
    try:
        # Call our new single, merged function
        print("Fetching all SH, SZ, KC stocks...")
        spot_df = custom_stock_all_a_spot_em()
        
        # Remove potential duplicates (still good practice)
        spot_df = spot_df.drop_duplicates(subset=['代码'])

        spot_df = spot_df.rename(columns={
            '代码': '股票代码',
            '今开': '开盘',
            '最新价': '收盘',
            '最高': '最高',
            '最低': '最低',
            '成交量': '成交量',
            '成交额': '成交额',
            '振幅': '振幅',
            '涨跌幅': '涨跌幅',
            '涨跌额': '涨跌额',
            '换手率': '换手率'
        })
        
        # Use the provided China date
        spot_df['日期'] = today_date_str
        
        # Select and reorder columns
        csv_columns = [
            "日期", "股票代码", "开盘", "收盘", "最高", "最低",
            "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率",
        ]
        spot_df = spot_df[[col for col in csv_columns if col in spot_df.columns]]
        
        # Create a dictionary for fast lookup: {code: pd.DataFrame_row}
        spot_data_map = {
            row['股票代码']: pd.DataFrame([row]) for index, row in spot_df.iterrows()
        }
        print(f"Fetched spot data for {len(spot_data_map)} traded stocks.")
        return spot_data_map
        
    except Exception as e:
        print(f"Error fetching today's spot data: {e}")
        return {}

def process_stock_update(code, spot_data_map, today_date_str):
    """
    Worker function to update a single stock's CSV file.
    """
    filepath = f"{config['csv_path']}/{code}.csv"
    
    try:
        if os.path.exists(filepath):
            # File exists, check if update is needed
            try:
                # Fast check: read only the last line's date
                with open(filepath, 'rb') as f:
                    try:
                        f.seek(-100, os.SEEK_END) # Seek near the end
                        last_line = f.readlines()[-1].decode('utf-8')
                        last_date = last_line.split(',')[0]
                    except (OSError, IndexError):
                         # File is too small or empty, fallback
                         f.seek(0)
                         header = f.readline().decode('utf-8')
                         if '日期' not in header:
                             raise ValueError("Missing '日期' header")
                         last_line = f.readlines()[-1].decode('utf-8')
                         last_date = last_line.split(',')[0]

            except Exception:
                # Fallback: read the date column if fast check fails
                try:
                    last_date = pd.read_csv(filepath, usecols=['日期']).iloc[-1, 0]
                except Exception as read_e:
                    return f"{code}: ERROR reading existing file {filepath} - {read_e}"

            if last_date == today_date_str:
                return f"{code}: Already up-to-date."
            
            # Not updated, check if we have spot data for it
            if code in spot_data_map:
                new_row_df = spot_data_map[code]
                new_row_df.to_csv(filepath, mode='a', header=False, index=False, encoding='utf-8-sig')
                return f"{code}: Appended today's data."
            else:
                return f"{code}: No spot data (not traded?)."
                
        else:
            # File does not exist, fetch full history
            os_time.sleep(random.uniform(0.5, 1.5)) # Be nice to the server
            hist_df = fixed_stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")
            
            if not hist_df.empty:
                hist_df.to_csv(filepath, index=False, encoding='utf-8-sig')
                return f"{code}: New file created with full history."
            else:
                return f"{code}: Failed to fetch full history."
                
    except Exception as e:
        return f"{code}: ERROR - {e}"

# ==============================
# MAIN EXECUTION
# ==============================
if __name__ == "__main__":
    
    # 1. --- PRE-RUN CHECKS ---
    # Set the correct timezone
    china_tz = pytz.timezone('Asia/Shanghai')
    now_china = datetime.now(china_tz)
    today_date_obj = now_china.date()
    today_date_str = today_date_obj.strftime("%Y-%m-%d")
    
    print(f"Current China time: {now_china.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check 1: Is today a trading day?
    if not chinese_calendar.is_workday(today_date_obj):
        print(f"Today ({today_date_str}) is a weekend or holiday. No new data. Exiting.")
        sys.exit()
        
    # Check 2: Is the market closed?
    if now_china.time() < config["market_close_time"]:
        print(f"Market is not closed yet (before {config['market_close_time']}). Exiting.")
        sys.exit()
    
    print(f"Checks passed: Today is a trading day and the market is closed.")
    # --- END PRE-RUN CHECKS ---
    
    
    # 2. Ensure data directory exists
    os.makedirs(config["csv_path"], exist_ok=True)
    
    # 3. Get today's data for all traded stocks
    today_spot_data_map = get_today_spot_data(today_date_str)
    if not today_spot_data_map:
        print("Could not fetch spot data. Exiting.")
        sys.exit()
        
    # 4. Get the complete list of all stocks
    all_stocks_df = get_base_stocks()
    if all_stocks_df.empty:
        print("Could not fetch base stock list. Exiting.")
        sys.exit()

    all_codes = all_stocks_df['code'].unique()
    
    # 5. Run multi-threaded update
    print(f"\nProcessing {len(all_codes)} stocks with {config['threads']} threads...")
    
    with ThreadPoolExecutor(max_workers=config["threads"]) as executor:
        futures = {
            executor.submit(process_stock_update, code, today_spot_data_map, today_date_str): code
            for code in all_codes
        }
        
        for future in tqdm(as_completed(futures), total=len(all_codes), desc="Updating local data"):
            result = future.result()
            # Optional: print detailed results for errors or new files
            if "ERROR" in result or "New file" in result or "failed" in result:
                 print(result)
    
    print("\nData update complete.")