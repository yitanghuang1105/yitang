# 多策略分數加總投資決策系統

## 📦 專案概述

這是一個模組化的多策略投資決策系統，將多種技術分析策略（RSI、布林通道、OBV）轉換為0-100的做多意願分數，並通過加權平均來決定進場/出場時機。

## 🏗️ 系統架構

```
multi_strategy_system/
├── __init__.py                 # 系統初始化
├── strategies/                 # 策略模組目錄
│   ├── __init__.py            # 策略模組初始化
│   ├── strategy_rsi.py        # RSI策略
│   ├── strategy_bb.py         # 布林通道策略
│   └── strategy_obv.py        # OBV策略
├── strategy_combiner.py       # 策略整合模組
├── demo.py                    # 演示腳本
└── README.md                  # 說明文件
```

## 🎯 核心功能

### 1. 策略模組（Strategy Modules）

每個策略模組都實現統一的介面：

```python
def compute_score(df: pd.DataFrame, params: dict) -> pd.Series
```

- **RSI策略** (`strategy_rsi.py`)
  - 基於RSI指標計算做多意願分數
  - RSI < 30 (超賣) → 高分數 (80-100)
  - RSI > 70 (超買) → 低分數 (0-20)
  - RSI 30-70 → 線性插值 (20-80)

- **布林通道策略** (`strategy_bb.py`)
  - 基於價格在布林通道中的位置
  - 接近下軌 → 高分數 (強烈做多信號)
  - 接近上軌 → 低分數 (微弱做多信號)

- **OBV策略** (`strategy_obv.py`)
  - 基於OBV動量和成交量比率
  - 正OBV動量 + 高成交量 → 高分數
  - 負OBV動量 + 低成交量 → 低分數

### 2. 分數統整模組（Strategy Combiner）

```python
def aggregate_scores(score_list: List[pd.Series], weights: List[float]) -> pd.Series
def decision_from_score(score: pd.Series, buy_threshold=70, sell_threshold=30) -> pd.Series
```

- 將多個策略分數進行加權平均
- 轉換為交易決策：
  - Score ≥ 70 → Buy
  - Score ≤ 30 → Sell
  - 30 < Score < 70 → Hold

## 📊 使用範例

### 基本使用

```python
import pandas as pd
from multi_strategy_system.strategy_combiner import run_multi_strategy_analysis

# 載入數據
df = pd.read_csv('your_data.csv')

# 執行分析
results = run_multi_strategy_analysis(df)

# 取得結果
combined_score = results['combined_score']
decisions = results['decisions']
individual_scores = results['individual_scores']
```

### 自定義參數

```python
# 自定義策略參數
params = {
    'rsi_window': 14,
    'rsi_oversold': 25,
    'rsi_overbought': 75,
    'bb_window': 20,
    'bb_std': 2.0,
    'obv_window': 10,
    'obv_threshold': 1.5,
    'buy_threshold': 75,
    'sell_threshold': 25
}

# 自定義權重
weights = {
    'rsi': 0.5,
    'bollinger_bands': 0.3,
    'obv': 0.2
}

# 執行分析
results = run_multi_strategy_analysis(df, params, weights)
```

### 運行演示

```bash
cd RUN/multi_strategy_system
python demo.py
```

## 🔧 預設參數

### 策略參數
```python
{
    'rsi_window': 14,           # RSI計算週期
    'rsi_oversold': 30,         # RSI超賣閾值
    'rsi_overbought': 70,       # RSI超買閾值
    'bb_window': 20,            # 布林通道移動平均週期
    'bb_std': 2.0,              # 布林通道標準差倍數
    'obv_window': 10,           # OBV移動平均週期
    'obv_threshold': 1.2,       # 成交量閾值倍數
    'buy_threshold': 70,        # 買入分數閾值
    'sell_threshold': 30        # 賣出分數閾值
}
```

### 策略權重
```python
{
    'rsi': 0.4,                 # RSI權重 40%
    'bollinger_bands': 0.35,    # 布林通道權重 35%
    'obv': 0.25                 # OBV權重 25%
}
```

## 📈 輸出結果

系統會輸出以下結果：

1. **個別策略分數** (`individual_scores`)
   - 每個策略的0-100分數

2. **綜合分數** (`combined_score`)
   - 加權平均後的綜合分數

3. **交易決策** (`decisions`)
   - Buy/Sell/Hold 決策

4. **統計資訊**
   - 分數統計（均值、標準差、最大最小值）
   - 決策統計（各決策的數量和百分比）

## 🎨 視覺化

演示腳本會生成包含以下圖表的分析報告：

1. **價格和交易信號圖**
   - 收盤價走勢
   - 買入/賣出信號標記

2. **個別策略分數圖**
   - 各策略的分數變化
   - 買入/賣出閾值線

3. **綜合分數圖**
   - 加權平均分數
   - 買入/賣出區域標示

4. **成交量圖**
   - 成交量變化

## 🔄 擴展新策略

要添加新策略，請按照以下步驟：

1. 在 `strategies/` 目錄下創建新的策略文件
2. 實現 `compute_score(df, params)` 函數
3. 在 `strategies/__init__.py` 中導入新策略
4. 在 `strategy_combiner.py` 的 `compute_all_strategies()` 中添加新策略

範例：
```python
# strategies/strategy_macd.py
def compute_score(df: pd.DataFrame, params: dict) -> pd.Series:
    # 實現MACD策略邏輯
    # 返回0-100的分數
    pass
```

## 📝 注意事項

1. **數據格式**：輸入數據需要包含 `open`, `high`, `low`, `close`, `volume` 欄位
2. **依賴套件**：需要安裝 `pandas`, `numpy`, `talib`, `matplotlib`
3. **參數調整**：根據不同市場和時間框架調整參數
4. **回測驗證**：在實際交易前進行充分的回測驗證

## 🚀 未來發展

- [ ] 添加更多技術指標策略
- [ ] 實現動態權重調整
- [ ] 添加機器學習策略
- [ ] 實現實時交易介面
- [ ] 添加風險管理模組 