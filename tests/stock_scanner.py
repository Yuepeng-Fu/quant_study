import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os

# ==============================
# CONFIG SECTION
# ==============================
config = {
    # --- NEW ---
    # Set a specific date for analysis, format "YYYY-MM-DD".
    # If set to None, it will use the current date (today).
    "analysis_date": "2025-10-21",  # e.g., "2024-01-01"
    # -----------
    "csv_path": "/root/quant_study/data",
    "filters": {
        "exclude_st": True,
        "exclude_beijiao": True,  # Filter Beijing Stock Exchange (4, 8 prefix)
        "exclude_new": True,
        "new_cutoff_days": 365,     # how many days since listing to keep
        "max_price": 100,           # filter: price < 100
    },
    "ranking": {
        "lookback_days": 10,        # past N days
        "top_k": 20,                # top 20 stocks
    },
    "threads": 20,                  # for concurrent file io
}

# ==============================
# STEP 1: Fetch A-share stock list (Metadata)
# ==============================
def get_base_stocks():
    print("Fetching A-share metadata list (names, listing dates) ...")

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
        "A股日期": "上市日期" 
    })
    
    if "上市日期" not in sz_df.columns and "A股上市日期" in sz_df.columns:
         sz_df = sz_df.rename(columns={"A股上市日期": "上市日期"})
    elif "上市日期" not in sz_df.columns and "A股日期" in sz_df.columns:
         sz_df = sz_df.rename(columns={"A股日期": "上市日期"})

    # 科创板
    kc_df = ak.stock_info_sh_name_code(symbol="科创板")
    kc_df = kc_df.rename(columns={
        "证券代码": "code",
        "证券简称": "name",
        "上市日期": "上市日期"
    })

    merged = pd.concat([sh_df, sz_df, kc_df], ignore_index=True)
    merged['上市日期'] = pd.to_datetime(merged['上市日期'], errors='coerce')
    merged['code'] = merged['code'].astype(str)
    
    return merged


# ==============================
# STEP 2: Apply filters
# ==============================
# MODIFIED: Added analysis_date parameter
def apply_filters(df, cfg, analysis_date):
    print("Applying filters ...")
    start_count = len(df)

    # (1) Exclude ST
    if cfg["exclude_st"]:
        df = df[~df['name'].str.contains('ST', na=False)]

    # (2) Exclude Beijing Stock Exchange
    if cfg["exclude_beijiao"]:
        df = df[~df['code'].str.startswith(('4', '8'))]

    # (3) Exclude new/次新股 (Now enabled based on config)
    if cfg["exclude_new"]:
        # MODIFIED: Use analysis_date instead of datetime.now()
        cutoff = analysis_date - timedelta(days=cfg["new_cutoff_days"])
        # Compare date portion only
        df = df[df['上市日期'].dt.date < cutoff]

    df = df.dropna(subset=['code', 'name', '上市日期'])
    df = df.reset_index(drop=True)
    print(f"Filtered from {start_count} to {len(df)} stocks.")
    return df


# ==============================
# STEP 3: Compute 过去N天涨幅 (Read-Only)
# ==============================
# MODIFIED: Added end_date parameter
def get_pct_change(code, days=10, end_date=None):
    """
    Return (pct_change, latest_price) by reading from local CSV up to end_date.
    If file not found or data is insufficient, return (None, None).
    """
    try:
        file_path = f"{config['csv_path']}/{code}.csv"
        
        if not os.path.exists(file_path):
            return None, None
            
        data = pd.read_csv(file_path)
        
        if '日期' not in data.columns or '收盘' not in data.columns:
            print(f"Skipping {code}: CSV missing '日期' or '收盘' column.")
            return None, None

        # MODIFIED: Convert to date objects for comparison
        data['日期'] = pd.to_datetime(data['日期']).dt.date
        
        # MODIFIED: Filter data to be on or before the analysis_date
        if end_date:
            data = data[data['日期'] <= end_date]

        data = data.sort_values('日期', ascending=False)
        
        if len(data) <= days: 
            return None, None

        start_price = data.iloc[days]['收盘']
        end_price = data.iloc[0]['收盘']
        
        if pd.isna(start_price) or pd.isna(end_price) or start_price == 0:
            return None, None
            
        pct = (end_price - start_price) / start_price * 100
        return pct, end_price
        
    except Exception as e:
        print(f"Failed to process local file for {code}: {e}")
        return None, None


# ==============================
# STEP 4: Run multi-threaded scanning
# ==============================
# MODIFIED: Added analysis_date parameter
def run_scan(df, cfg, analysis_date):
    print(f"Scanning {len(df)} stocks from local files (lookback: {cfg['lookback_days']} days) ...")

    results = []
    with ThreadPoolExecutor(max_workers=config["threads"]) as executor:
        # MODIFIED: Pass analysis_date to get_pct_change
        futures = {
            executor.submit(get_pct_change, code, cfg['lookback_days'], analysis_date): code
            for code in df['code']
        }
        
        for future in tqdm(as_completed(futures), total=len(df), desc="Reading local files"):
            code = futures[future]
            try:
                pct, price = future.result()
                
                if pct is not None and price is not None and price < config["filters"]["max_price"]:
                    name = df.loc[df['code'] == code, 'name'].values[0]
                    results.append({"code": code, "name": name, "pct_change": pct, "price": price})
            except Exception as e:
                print(f"Error processing result for {code}: {e}")
                continue

    result_df = pd.DataFrame(results)
    return result_df


# ==============================
# STEP 5: Rank and display top stocks
# ==============================
def show_top(df, cfg):
    if df.empty:
        print("\n=== No stocks matched all criteria ===")
        return df, pd.DataFrame()
        
    sort = df.sort_values("pct_change", ascending=False)
    ranked = sort.head(cfg["top_k"])
    print("\n=== Top Stocks ===")
    
    ranked_display = ranked.copy()
    ranked_display['price'] = ranked_display['price'].round(2)
    ranked_display['pct_change'] = ranked_display['pct_change'].round(2).astype(str) + '%'
    
    print(ranked_display[['code', 'name', 'price', 'pct_change']].to_string(index=False))
    return sort, ranked


# ==============================
# MAIN EXECUTION
# ==============================
if __name__ == "__main__":
    if not os.path.isdir(config["csv_path"]):
        print("Error: 'data' directory not found.")
        print("Please make sure your CSV files are in a folder named 'data'.")
    else:
        start_time = datetime.now()
        
        # --- NEW: Process analysis_date ---
        if config["analysis_date"]:
            analysis_date = pd.to_datetime(config["analysis_date"]).date()
            print(f"--- Running analysis as of date: {analysis_date} ---")
        else:
            analysis_date = datetime.now().date()
            print(f"--- Running analysis as of current date: {analysis_date} ---")
        # ----------------------------------

        # Step 1: Get metadata (names, dates)
        all_stocks_meta = get_base_stocks()
        
        # Step 2: Filter metadata (MODIFIED: pass analysis_date)
        filtered_stocks_meta = apply_filters(all_stocks_meta, config["filters"], analysis_date)
        
        # Step 3 & 4: Scan local files (MODIFIED: pass analysis_date)
        results = run_scan(filtered_stocks_meta, config["ranking"], analysis_date)
        
        # Step 5: Rank and show
        all_results, top_results = show_top(results, config["ranking"])

        if not all_results.empty:
            # MODIFIED: Add analysis_date to the output filename for clarity
            date_str = analysis_date.strftime("%Y%m%d")
            output_file = f"scan_result_{date_str}.csv"
            all_results.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"\nAll results saved to: {output_file}")
        
        end_time = datetime.now()
        print(f"Total execution time: {end_time - start_time}")