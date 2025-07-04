# 出場策略分析器

這個目錄包含了選擇權交易平台的出場策略分析工具，專注於高勝率交易策略的風險管理和出場時機判斷。

## 檔案說明

### 1. 出場.py
完整的出場策略分析器，包含多種出場條件和風險管理功能。

**主要功能：**
- 多種技術指標計算（RSI、MACD、布林通道、隨機指標等）
- 五種出場條件：
  - 止損 (Stop Loss)
  - 止盈 (Take Profit)
  - 追蹤止損 (Trailing Stop)
  - 時間出場 (Time-based Exit)
  - 信號出場 (Signal-based Exit)
- 詳細的表現分析和視覺化圖表

### 2. simple_exit_analyzer.py
簡化的出場分析器，專注於高勝率交易策略的核心出場條件。

**主要功能：**
- 簡化的技術指標（RSI、MACD、布林通道、成交量）
- 優化的出場參數設定：
  - 止損：1.5%
  - 止盈：3%
  - 追蹤止損：0.8%
  - 最大持倉期數：20期
- 按出場類型分類的表現分析

## 出場策略參數

### 風險管理參數
```python
params = {
    'stop_loss_pct': 0.015,      # 1.5% 止損
    'take_profit_pct': 0.03,     # 3% 止盈
    'trailing_stop_pct': 0.008,  # 0.8% 追蹤止損
    'max_hold_periods': 20,      # 最大持倉期數
}
```

### 技術指標出場條件
```python
params = {
    'rsi_exit_oversold': 25,     # RSI 超賣出場
    'rsi_exit_overbought': 75,   # RSI 超買出場
    'volume_exit_threshold': 0.7, # 成交量出場閾值
}
```

## 出場條件優先級

1. **止損** (最高優先級) - 保護資金安全
2. **止盈** - 鎖定獲利
3. **追蹤止損** - 保護既有獲利
4. **時間出場** - 避免過度持倉
5. **信號出場** - 基於技術指標

## 使用方法

### 執行完整分析
```python
python 出場.py
```

### 執行簡化分析
```python
python simple_exit_analyzer.py
```

### 自定義參數
```python
from simple_exit_analyzer import SimpleExitAnalyzer

analyzer = SimpleExitAnalyzer()
custom_params = {
    'stop_loss_pct': 0.02,       # 2% 止損
    'take_profit_pct': 0.04,     # 4% 止盈
    'trailing_stop_pct': 0.01,   # 1% 追蹤止損
    'max_hold_periods': 15,      # 15期最大持倉
    'rsi_exit_oversold': 30,     # RSI 30 出場
    'rsi_exit_overbought': 70,   # RSI 70 出場
}

df, exit_signals = analyzer.run_simple_analysis(
    timeframe='4H', 
    lookback_days=30
)
```

## 輸出結果

### 分析報告
- 出場信號總數
- 各類型出場的分布
- 勝率和平均報酬率
- 按出場類型的表現分析

### 視覺化圖表
- 價格和出場信號圖（按類型著色）
- RSI 指標圖
- MACD 指標圖

## 與進場策略的整合

出場策略設計為與進場策略無縫整合：

1. **進場策略** (`RUN/進場/`) 負責識別進場時機
2. **出場策略** (`RUN/出場/`) 負責風險管理和出場時機
3. 兩者可以獨立運行，也可以整合到完整的交易系統中

## 注意事項

1. **風險管理優先**：止損條件具有最高優先級，確保資金安全
2. **參數調優**：根據市場條件和交易品種調整出場參數
3. **回測驗證**：在實盤交易前，務必進行充分的回測驗證
4. **市場適應性**：不同市場環境可能需要不同的出場策略

## 未來擴展

- 動態參數調整
- 機器學習優化
- 多時間框架分析
- 實時監控系統 