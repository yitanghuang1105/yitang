import pandas as pd
import numpy as np

# --- 指標計算函數 ---
def rsi_score(series, window=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window).mean()
    avg_loss = pd.Series(loss).rolling(window).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def kdj_score(df, n=9, k_window=3, d_window=3):
    low_min = df['low'].rolling(n).min()
    high_max = df['high'].rolling(n).max()
    rsv = (df['close'] - low_min) / (high_max - low_min + 1e-9) * 100
    k = rsv.ewm(com=k_window-1, adjust=False).mean()
    d = k.ewm(com=d_window-1, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j

def obv_slope(series, window=10):
    x = np.arange(window)
    slope = series.rolling(window).apply(lambda y: np.polyfit(x, y, 1)[0], raw=True)
    return slope

# --- 進出場策略主體 ---
def run_strategy(df_1min, reverse=False, reverse2=False):
    """
    Run multi-timeframe strategy with reverse options
    
    Parameters:
    - reverse: Original reverse mode (reverses both entry and exit logic)
    - reverse2: New reverse mode (separately reverses entry and exit signals)
    
    reverse2 logic:
    - reverse_entry: High scores → Sell signals, Low scores → Buy signals
    - reverse_exit: Exit conditions are reversed (e.g., score_exit becomes opposite)
    """
    df_5min = df_1min.resample('5T').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    df_30min = df_1min.resample('30T').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()

    df_30min['rsi_30min'] = rsi_score(df_30min['close'], window=14)
    df_30min['obv_30min'] = (np.sign(df_30min['close'].diff()) * df_30min['volume']).fillna(0).cumsum()
    df_30min['obv_slope_30min'] = obv_slope(df_30min['obv_30min'], window=10)

    df_5min['rsi_5min'] = rsi_score(df_5min['close'], window=14)
    k, d, j = kdj_score(df_5min)
    df_5min['kdj_k'] = k
    df_5min['kdj_d'] = d
    df_5min['kdj_j'] = j
    df_5min['kdj_score_5min'] = (k + d + j) / 3
    df_5min['rsi_score_5min'] = df_5min['rsi_5min']
    df_5min['entry_score'] = df_5min['rsi_score_5min'] + df_5min['kdj_score_5min']
    df_5min['kdj_cross'] = np.where((df_5min['kdj_k'] < df_5min['kdj_d']) & (df_5min['kdj_k'].shift(1) >= df_5min['kdj_d'].shift(1)), 'dead',
                                    np.where((df_5min['kdj_k'] > df_5min['kdj_d']) & (df_5min['kdj_k'].shift(1) <= df_5min['kdj_d'].shift(1)), 'gold', None))

    df_1min = df_1min.copy()
    df_1min['rsi_30min'] = df_30min['rsi_30min'].reindex(df_1min.index, method='ffill')
    df_1min['obv_slope_30min'] = df_30min['obv_slope_30min'].reindex(df_1min.index, method='ffill')
    df_1min['rsi_score_5min'] = df_5min['rsi_score_5min'].reindex(df_1min.index, method='ffill')
    df_1min['kdj_score_5min'] = df_5min['kdj_score_5min'].reindex(df_1min.index, method='ffill')
    df_1min['entry_score'] = df_5min['entry_score'].reindex(df_1min.index, method='ffill')
    df_1min['kdj_cross'] = df_5min['kdj_cross'].reindex(df_1min.index, method='ffill')
    df_1min['kdj_k'] = df_5min['kdj_k'].reindex(df_1min.index, method='ffill')
    df_1min['kdj_d'] = df_5min['kdj_d'].reindex(df_1min.index, method='ffill')
    df_1min['rsi_5min'] = df_5min['rsi_5min'].reindex(df_1min.index, method='ffill')

    position = 0
    entry_price = 0
    entry_time = None
    entry_row = None
    details = []
    
    for idx, row in df_1min.iterrows():
        pnl = (row['close'] - entry_price) / entry_price if position == 1 and entry_price > 0 else 0
        
        # 基礎條件（不受 reverse2 影響）
        base_trend_ok = (row['rsi_30min'] > 50) and (row['obv_slope_30min'] > 0)
        base_score_threshold = 160
        current_score = row['rsi_score_5min'] + row['kdj_score_5min']
        
        # 決定最終的進出場邏輯
        if reverse:
            # 原始 reverse 模式：完全反向
            trend_ok = not base_trend_ok
            entry_cond = trend_ok or (current_score <= base_score_threshold)
            exit_reasons = [
                ('score_exit', row['entry_score'] > base_score_threshold),
                ('kdj_exit', row['kdj_cross'] == 'gold'),
                ('stop_loss', pnl > 0.05),
                ('take_profit', pnl < -0.03)
            ]
        else:
            # 正常模式
            trend_ok = base_trend_ok
            entry_cond = trend_ok and (current_score > base_score_threshold)
            exit_reasons = [
                ('score_exit', row['entry_score'] < 40),
                ('kdj_exit', row['kdj_cross'] == 'dead'),
                ('stop_loss', pnl < -0.03),
                ('take_profit', pnl > 0.05)
            ]
        
        # reverse2 模式：分別反向進場和出場訊號
        if reverse2:
            # 反向進場訊號：原本高分數買入，現在高分數賣出
            if reverse:
                # Reverse + Reverse2: 低分數時買入
                entry_cond = trend_ok and (current_score <= base_score_threshold)
            else:
                # Normal + Reverse2: 高分數時賣出（反向進場）
                entry_cond = trend_ok and (current_score > base_score_threshold)
            
            # 反向出場訊號：出場條件相反
            if reverse:
                # Reverse + Reverse2: 高分數時出場
                exit_reasons = [
                    ('score_exit', row['entry_score'] > base_score_threshold),
                    ('kdj_exit', row['kdj_cross'] == 'gold'),
                    ('stop_loss', pnl < -0.03),
                    ('take_profit', pnl > 0.05)
                ]
            else:
                # Normal + Reverse2: 低分數時出場
                exit_reasons = [
                    ('score_exit', row['entry_score'] < 40),
                    ('kdj_exit', row['kdj_cross'] == 'dead'),
                    ('stop_loss', pnl > 0.05),
                    ('take_profit', pnl < -0.03)
                ]
        
        # 進場
        if position == 0 and entry_cond:
            position = 1
            entry_price = row['close']
            entry_time = idx
            entry_row = row.copy()
        
        # 出場
        elif position == 1:
            reason = None
            for r, cond in exit_reasons:
                if cond:
                    reason = r
                    break
            if reason:
                hold_minutes = int((idx - entry_time).total_seconds() // 60) if entry_time else 0
                
                # 決定模式標籤
                mode_parts = []
                if reverse:
                    mode_parts.append('reverse')
                if reverse2:
                    mode_parts.append('reverse2')
                if not mode_parts:
                    mode_parts.append('normal')
                mode_label = '+'.join(mode_parts)
                
                details.append({
                    'entry_time': entry_time,
                    'exit_time': idx,
                    'entry_price': entry_price,
                    'exit_price': row['close'],
                    'hold_minutes': hold_minutes,
                    'exit_reason': reason,
                    'entry_score': entry_row['entry_score'] if entry_row is not None else np.nan,
                    'exit_score': row['entry_score'],
                    'entry_kdj': entry_row['kdj_cross'] if entry_row is not None else None,
                    'exit_kdj': row['kdj_cross'],
                    'entry_rsi': entry_row['rsi_5min'] if entry_row is not None else np.nan,
                    'exit_rsi': row['rsi_5min'],
                    'pnl': pnl,
                    'mode': mode_label,
                    'reverse': reverse,
                    'reverse2': reverse2
                })
                position = 0
                entry_price = 0
                entry_time = None
                entry_row = None
    
    details_df = pd.DataFrame(details)
    return details_df

# 匯出到 Excel

def export_to_excel(details_df, filename):
    # 統計總表
    if details_df.empty:
        summary_df = pd.DataFrame()
    else:
        summary_df = details_df.groupby(['mode','exit_reason']).agg(
            count=('pnl', 'count'),
            avg_pnl=('pnl', 'mean'),
            max_pnl=('pnl', 'max'),
            min_pnl=('pnl', 'min'),
            avg_hold_minutes=('hold_minutes', 'mean')
        ).reset_index()
    with pd.ExcelWriter(filename) as writer:
        details_df.to_excel(writer, sheet_name='details', index=False)
        summary_df.to_excel(writer, sheet_name='summary', index=False) 