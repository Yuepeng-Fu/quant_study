import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random

# ==============================
# CONFIG SECTION
# ==============================
config = {
    "filters": {
        "exclude_st": True,
        "exclude_beijiao": True,
        "exclude_new": True,
        "new_cutoff_days": 365,     # how many days since listing to keep
        "max_price": 100,           # filter: price < 100
    },
    "ranking": {
        "lookback_days": 10,        # past N days
        "top_k": 20,                # top 20 stocks
    },
    "threads": 10,                  # for concurrent requests
}


# ==============================
# STEP 1: Fetch A-share stock list
# ==============================
def get_base_stocks():
    print("Fetching A-share list ...")

    # 上证主板A股
    sh_df = ak.stock_info_sh_name_code(symbol="主板A股")
    sh_df = sh_df.rename(columns={
        "证券代码": "code",
        "证券简称": "name",
        "上市日期": "上市日期"
    })

    # 深证A股列表
    sz_df = ak.stock_info_sz_name_code(symbol="A股列表")
    sz_df = sz_df.rename(columns={
        "A股代码": "code",
        "A股简称": "name",
        "上市日期": "上市日期"
    })

    # 科创板
    kc_df = ak.stock_info_sh_name_code(symbol="科创板")
    kc_df = kc_df.rename(columns={
        "证券代码": "code",
        "证券简称": "name",
        "上市日期": "上市日期"
    })

    # 合并两市数据
    merged = pd.concat([sh_df, sz_df, kc_df], ignore_index=True)
    merged['上市日期'] = pd.to_datetime(merged['上市日期'], errors='coerce')
    return merged


# ==============================
# STEP 2: Apply filters
# ==============================
def apply_filters(df, cfg):
    print("Applying filters ...")

    # (1) Exclude ST
    if cfg["exclude_st"]:
        df = df[~df['name'].str.contains('ST')]

    # (3) Exclude new/次新股
    if cfg["exclude_new"]:
        cutoff = datetime.now() - timedelta(days=cfg["new_cutoff_days"])
        df = df[df['上市日期'] < cutoff]

    df = df.dropna(subset=['code'])
    df = df.reset_index(drop=True)
    print(f"Remaining stocks after filtering: {len(df)}")
    return df


# ==============================
# STEP 3: Compute 过去N天涨幅
# ==============================
def get_pct_change(code, days=10):
    """Return (pct_change, latest_price)"""
    try:
        time.sleep(random.uniform(0.1, 0.5)) 
        
        data = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")
        data['日期'] = pd.to_datetime(data['日期'])
        data = data.sort_values('日期', ascending=False)
        
        if len(data) <= days: 
            return None, None

        start_price = data.iloc[days]['收盘']
        end_price = data.iloc[0]['收盘']
        pct = (end_price - start_price) / start_price * 100
        return pct, end_price
        
    except Exception as e:
        print(f"Failed to process {code}: {e}")
        return None, None


# ==============================
# STEP 4: Run multi-threaded scanning
# ==============================
def run_scan(df, cfg):
    print(f"Scanning {len(df)} stocks over last {cfg['lookback_days']} days ...")

    results = []
    with ThreadPoolExecutor(max_workers=config["threads"]) as executor:
        futures = {
            executor.submit(get_pct_change, code, cfg['lookback_days']): code
            for code in df['code']
        }
        for future in as_completed(futures):
            code = futures[future]
            try:
                pct, price = future.result()
                # if pct is not None and price < config["filters"]["max_price"]:
                name = df.loc[df['code'] == code, 'name'].values[0]
                results.append({"code": code, "name": name, "pct_change": pct, "price": price})
            except Exception:
                continue

    result_df = pd.DataFrame(results)
    return result_df


# ==============================
# STEP 5: Rank and display top stocks
# ==============================
def show_top(df, cfg):
    sort = df.sort_values("pct_change", ascending=False)
    ranked = sort.head(cfg["top_k"])
    print("\n=== Top Stocks ===")
    print(ranked[['code', 'name', 'price', 'pct_change']].to_string(index=False))
    return sort, ranked


# ==============================
# MAIN EXECUTION
# ==============================
if __name__ == "__main__":
    all_stocks = get_base_stocks()
    filtered_stocks = apply_filters(all_stocks, config["filters"])
    results = run_scan(filtered_stocks, config["ranking"])
    all_stocks, top_stocks = show_top(results, config["ranking"])

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"scan_result_{timestamp}.csv"
    all_stocks.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nResults saved to: {output_file}")
