import pandas as pd
import glob
import os

# --- Konfigurationsparameter (您可以调整这些阈值) ---

# !! 核心功能: 模拟 "今天"
# 设置您想 "站立" 的日期 (格式: YYYY-MM-DD)。
SIMULATION_DATE = "2025-10-23" # <-- 在这里设置您想要的日期

# --- "爆发中" (Bursting) 的定义 ---

# 1. 爆发持续时间 (Burst Duration)
# 我们将检查过去多少天内的爆发? (例如: 2天或3天前开始的爆发)
BURST_LOOKBACK_DAYS = [1, 2] # 检查2日和3日爆发

# 2. 爆发涨幅 (Burst Price Increase)
BURST_PRICE_INCREASE = 0.08

# --- "爆发前" (Pre-Burst Setup) 的定义 ---
# (检查爆发开始前一天, 即 "setup_day")

# 3. 压缩阈值 (Setup Compression)
# 爆发 *前* 的那一天, 5日EMA和34日EMA必须非常接近 (例如: 相差 3% 以内)
SETUP_COMPRESSION_THRESHOLD = 0.03

# 4. 价格接近度 (Setup Price Proximity)
# 爆发 *前* 的那一天, 收盘价必须在压缩区内 (例如: 离34日EMA不超过 4%)
SETUP_PRICE_PROXIMITY_THRESHOLD = 0.04

# 5. 均线平坦 (Setup Flat Slope)
# 爆发 *前* 的那一天, 34日EMA必须是平的 (例如: 5天内变化 < 1.5%)
FLAT_SLOPE_PERIOD = 5
SETUP_FLAT_SLOPE_THRESHOLD = 0.015

# --- 新功能: "长期" 调整 (Long Adjustment) ---
# (检查 "setup_day" 之前的 X 天)

# 6. 长期调整期
# 爆发前必须经历多长时间的盘整? (我们将检查 setup_day 之前的 20 天)
ADJUSTMENT_PERIOD_DAYS = 20

# 7. 调整期一致性
# 在这 20 天内, 至少有多少比例的日子必须处于 "压缩" 且 "价格接近" 状态?
ADJUSTMENT_CONSISTENCY_PCT = 0.80 # 70% 的日子 (即 14 天)

# ----------------------------------------------------

def calculate_indicators(df):
    """
    计算所有需要的技术指标
    """
    try:
        df['EMA5'] = df['收盘'].ewm(span=5, adjust=False).mean()
        df['EMA13'] = df['收盘'].ewm(span=13, adjust=False).mean()
        df['EMA21'] = df['收盘'].ewm(span=21, adjust=False).mean()
        df['EMA34'] = df['收盘'].ewm(span=34, adjust=False).mean()
        
        # 为 "长期调整" 检查预先计算指标
        df['MA_Spread'] = abs((df['EMA5'] - df['EMA34'])) / df['EMA34']
        df['Price_Proximity'] = abs((df['收盘'] - df['EMA34'])) / df['EMA34']
        
        return df
    except Exception as e:
        print(f"  [错误] 计算指标时出错: {e}")
        return None

def check_active_burst_conditions(df, ticker):
    """
    检查 "今天" (df的最后一行) 是否处于一个 "刚刚开始的爆发" 状态
    """
    
    # 检查每一种爆发持续时间 (例如: 2天前, 3天前)
    for duration in BURST_LOOKBACK_DAYS:
        # 我们需要足够的数据来检查: 爆发持续时间 + EMA周期 + 斜率周期 + 长期调整期
        required_data_len = max(34, (duration + 1) + FLAT_SLOPE_PERIOD, (duration + 1) + ADJUSTMENT_PERIOD_DAYS)
        
        if len(df) < required_data_len:
            continue # 数据不足, 无法检查此持续时间

        # --- 1. 定义日期和索引 ---
        today = df.iloc[-1]
        
        setup_day_index = -(duration + 1)
        setup_day = df.iloc[setup_day_index]
        
        slope_check_day_index = setup_day_index - FLAT_SLOPE_PERIOD
        slope_check_day = df.iloc[slope_check_day_index]

        # --- 2. 检查 "爆发前" (Setup Day) 的条件 ---
        # (检查爆发前 *最后一天* 的状态)
        
        is_compressed = setup_day['MA_Spread'] < SETUP_COMPRESSION_THRESHOLD
        is_proximate = setup_day['Price_Proximity'] < SETUP_PRICE_PROXIMITY_THRESHOLD
        
        setup_slope_34 = abs((setup_day['EMA34'] - slope_check_day['EMA34'])) / slope_check_day['EMA34']
        is_flat = setup_slope_34 < SETUP_FLAT_SLOPE_THRESHOLD

        # --- 3. 新功能: 检查 "长期" 调整 ---
        # (检查 setup_day 之前的 ADJUSTMENT_PERIOD_DAYS 天)
        
        adj_period_start_index = setup_day_index - ADJUSTMENT_PERIOD_DAYS
        adj_period_end_index = setup_day_index # 不包括 setup_day 本身
        
        adjustment_period_df = df.iloc[adj_period_start_index : adj_period_end_index]
        
        # 检查是否大部分时间都处于压缩和接近状态
        compressed_days = adjustment_period_df['MA_Spread'] < SETUP_COMPRESSION_THRESHOLD
        proximate_days = adjustment_period_df['Price_Proximity'] < SETUP_PRICE_PROXIMITY_THRESHOLD
        
        consistent_adjustment_days = (compressed_days & proximate_days).sum()
        required_adjustment_days = int(ADJUSTMENT_PERIOD_DAYS * ADJUSTMENT_CONSISTENCY_PCT)
        
        is_long_adjustment = consistent_adjustment_days >= required_adjustment_days

        # 必须满足所有 "爆发前" 的条件 (包括 "长期" 检查)
        if not (is_compressed and is_proximate and is_flat and is_long_adjustment):
            continue 

        # --- 4. 检查 "爆发" (Burst) 本身 ---
        
        # A. 检查价格涨幅: 从 "爆发前" 到 "今天" 涨幅是否足够?
        price_increase = (today['收盘'] - setup_day['收盘']) / setup_day['收盘']
        is_bursting = price_increase > BURST_PRICE_INCREASE

        # B. 检查 "今天" 的均线是否已开始对齐?
        is_aligning = (today['EMA5'] > today['EMA13']) and (today['EMA13'] > today['EMA21'])
        
        # 必须同时满足爆发涨幅和均线开始对齐
        if is_bursting and is_aligning:
            return {
                'Ticker': ticker,
                'Date': today['日期'].strftime('%Y-%m-%d'),
                'Close': today['收盘'],
                'Burst_Days': duration,
                'Burst_Increase': price_increase,
                'Setup_Adj_Days': f"{consistent_adjustment_days}/{ADJUSTMENT_PERIOD_DAYS}"
            }
    
    return None

def main():
    """
    主函数: 扫描 data/ 目录, 分析所有CSV文件, 并打印结果
    """
    if not SIMULATION_DATE:
        print("错误: 请在脚本顶部设置 'SIMULATION_DATE'。")
        return

    data_folder = 'data'
    csv_files = glob.glob(os.path.join(data_folder, '*.csv'))
    
    print(f"--- 模拟日期 (今天): {SIMULATION_DATE} ---")
    print(f"--- 在 {len(csv_files)} 个文件中搜索... ---")
    
    simulation_dt = pd.to_datetime(SIMULATION_DATE)
    bursting_stocks = []

    for file_path in csv_files:
        ticker = os.path.basename(file_path).replace('.csv', '')
        # print(f"\n正在分析: {ticker}") # 调试时取消注释

        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            
            # --- 日期处理 ---
            try:
                df['日期'] = pd.to_datetime(df['日期'])
            except Exception as e:
                print(f"  [警告] {ticker} 的 '日期' 列格式错误: {e}")
                continue
            
            # 筛选数据: 只保留 "今天" 及之前的数据
            df = df[df['日期'] <= simulation_dt].copy()
            
            if df.empty:
                continue
            
            # 检查 "今天" 是否有数据
            if df.iloc[-1]['日期'] != simulation_dt:
                continue

            # --- 计算和检查 ---
            df_with_indicators = calculate_indicators(df.sort_values(by='日期'))
            
            if df_with_indicators is not None:
                result = check_active_burst_conditions(df_with_indicators, ticker)
                if result:
                    print(f"  [!!] 发现: {ticker} ({result['Burst_Days']}日爆发 {result['Burst_Increase']*100:.1f}%)")
                    bursting_stocks.append(result)
            
        except pd.errors.EmptyDataError:
            pass # 忽略空文件
        except Exception as e:
            print(f"  [严重错误] 处理 {ticker} 时发生意外错误: {e}")

    # --- 打印最终报告 ---
    print("\n\n" + "="*50)
    print(f"     筛选报告: {SIMULATION_DATE} 当天 '爆发中' 的股票")
    print("="*50)

    if not bursting_stocks:
        print("未找到任何符合 '长期盘整后爆发' 条件的股票。")
        print("提示: 您可以尝试放宽脚本顶部的阈值 (例如, 降低 ADJUSTMENT_CONSISTENCY_PCT)。")
    else:
        print(f"总共找到 {len(bursting_stocks)} 只符合条件的股票:")
        print_format = "{:<12} | {:<10} | {:<10} | {:<12} | {:<15}"
        print(print_format.format("股票代码", "收盘价", "爆发天数", "N日涨幅", "盘整期天数"))
        print("-" * 65)
        
        # 按爆发涨幅排序, 最强的在最前面
        bursting_stocks.sort(key=lambda x: x['Burst_Increase'], reverse=True)
        
        for stock in bursting_stocks:
            print(print_format.format(
                stock['Ticker'],
                f"{stock['Close']:.2f}",
                f"{stock['Burst_Days']} 天",
                f"+{stock['Burst_Increase']*100:.1f}%",
                f"{stock['Setup_Adj_Days']} (一致性)"
            ))
    print("="*50)


if __name__ == "__main__":
    main()

