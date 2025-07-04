# 量化交易系統 - 參數最佳化模組

## 系統概述

本系統實現了一個基於布林通道（Bollinger Bands）+ RSI + OBV的量化交易策略，並提供完整的參數最佳化功能。

## 核心功能

### 1. 技術指標計算
- **布林通道（Bollinger Bands）**: 計算中軌、上軌、下軌、通道寬度和價格位置
- **RSI（相對強弱指標）**: 計算超買超賣信號
- **OBV（能量潮指標）**: 計算成交量趨勢確認

### 2. 交易信號生成
- **買入信號**: 價格觸及下軌 + RSI超賣 + OBV確認
- **賣出信號**: 價格觸及上軌 + RSI超買 + OBV確認

### 3. 參數最佳化
- **網格搜索**: 系統性測試所有參數組合
- **績效評估**: 計算夏普比率、最大回撤、勝率等指標
- **交易成本**: 考慮手續費和滑點成本

## 檔案結構

```
RUN/1/
├── 1                    # 主要程式碼檔案
├── test_optimization.py # 測試腳本
└── README.md           # 說明文件
```

## 使用方法

### 1. 執行完整最佳化
```python
from RUN.1.1 import main
best_params, best_metrics = main()
```

### 2. 執行測試
```python
python RUN/1/test_optimization.py
```

### 3. 自定義參數範圍
```python
from RUN.1.1 import optimize_parameters, load_txf_data

# 載入資料
df = load_txf_data("TXF1_Minute_2020-01-01_2025-06-16.txt")

# 定義參數範圍
param_ranges = {
    'bb_window': [15, 20, 25],
    'bb_std': [2.0, 2.5],
    'rsi_window': [14],
    'rsi_oversold': [25, 30],
    'rsi_overbought': [70, 75],
    'obv_threshold': [1.2, 1.3]
}

# 執行最佳化
best_params, best_metrics = optimize_parameters(df, param_ranges)
```

## 參數說明

### 布林通道參數
- `bb_window`: 移動平均線週期（預設20）
- `bb_std`: 標準差倍數（預設2.0）

### RSI參數
- `rsi_window`: RSI計算週期（預設14）
- `rsi_oversold`: 超賣閾值（預設30）
- `rsi_overbought`: 超買閾值（預設70）

### OBV參數
- `obv_threshold`: OBV確認閾值（預設1.2）

### 交易成本
- `transaction_cost`: 每筆交易成本（預設0.01%）

## 績效指標

系統計算以下績效指標：
- **總報酬率**: 策略總收益
- **年化報酬率**: 年化收益率
- **波動率**: 年化波動率
- **夏普比率**: 風險調整後報酬
- **最大回撤**: 最大虧損幅度
- **勝率**: 獲利交易比例
- **交易次數**: 總交易筆數

## 最佳化目標

系統以**夏普比率**作為主要最佳化目標，同時考慮：
- 交易成本影響
- 參數穩定性
- 樣本外表現

## 注意事項

1. **資料格式**: 確保資料檔案包含 Date, Time, Open, High, Low, Close, TotalVolume 欄位
2. **記憶體使用**: 大量參數組合可能消耗較多記憶體
3. **執行時間**: 完整最佳化可能需要數小時，建議先用小範圍測試
4. **過度擬合**: 避免使用過多參數組合，可能導致過度擬合

## 下一步發展

1. **風險管理**: 加入停損停利機制
2. **訊號過濾**: 減少假訊號
3. **多時間框架**: 結合不同週期分析
4. **樣本外驗證**: 分離訓練和測試期間
5. **多策略組合**: 整合多個交易策略 