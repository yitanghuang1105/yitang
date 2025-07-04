# TA-Lib 使用指南

## 什麼是 TA-Lib？

TA-Lib (Technical Analysis Library) 是一個強大的技術分析函數庫，提供了超過 150 個技術指標函數。它被廣泛用於量化交易、金融分析和技術分析。

## 主要特點

- **高效能**: 使用 C 語言編寫，執行速度快
- **豐富的指標**: 包含移動平均、動量指標、波動率指標、成交量指標等
- **標準化**: 遵循業界標準的技術分析指標計算方法
- **多語言支援**: 支援 Python、R、Java、C# 等多種語言

## 安裝方法

### Windows 安裝

1. **下載預編譯版本** (推薦):
   ```bash
   pip install TA-Lib
   ```

2. **如果上述方法失敗，手動安裝**:
   - 下載對應的 wheel 檔案: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
   - 安裝下載的 wheel 檔案:
   ```bash
   pip install TA_Lib-0.4.24-cp39-cp39-win_amd64.whl
   ```

### Linux/Mac 安裝

```bash
# Ubuntu/Debian
sudo apt-get install ta-lib
pip install TA-Lib

# CentOS/RHEL
sudo yum install ta-lib
pip install TA-Lib

# macOS
brew install ta-lib
pip install TA-Lib
```

## 基本使用方法

### 1. 導入和準備數據

```python
import talib
import pandas as pd
import numpy as np

# 準備 OHLCV 數據
df = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
})

# 轉換為 numpy 陣列 (TA-Lib 需要)
close = df['close'].values
high = df['high'].values
low = df['low'].values
volume = df['volume'].values
```

### 2. 常用指標計算

#### 移動平均線
```python
# 簡單移動平均線
sma_20 = talib.SMA(close, timeperiod=20)

# 指數移動平均線
ema_12 = talib.EMA(close, timeperiod=12)

# 加權移動平均線
wma_10 = talib.WMA(close, timeperiod=10)
```

#### 布林通道
```python
# 布林通道 (返回上軌、中軌、下軌)
bb_upper, bb_middle, bb_lower = talib.BBANDS(
    close, 
    timeperiod=20, 
    nbdevup=2, 
    nbdevdn=2, 
    matype=0
)
```

#### RSI
```python
# 相對強弱指數
rsi = talib.RSI(close, timeperiod=14)
```

#### MACD
```python
# MACD (返回 MACD 線、信號線、柱狀圖)
macd, macd_signal, macd_hist = talib.MACD(
    close, 
    fastperiod=12, 
    slowperiod=26, 
    signalperiod=9
)
```

#### 隨機指標
```python
# 隨機指標 (返回 %K 和 %D)
slowk, slowd = talib.STOCH(
    high, low, close, 
    fastk_period=5, 
    slowk_period=3, 
    slowk_matype=0, 
    slowd_period=3, 
    slowd_matype=0
)
```

### 3. 波動率指標

```python
# 平均真實波幅
atr = talib.ATR(high, low, close, timeperiod=14)

# 平均方向指數
adx = talib.ADX(high, low, close, timeperiod=14)
```

### 4. 成交量指標

```python
# 能量潮
obv = talib.OBV(close, volume)

# 累積/派發線
ad = talib.AD(high, low, close, volume)
```

## 指標分類

### 重疊研究 (Overlap Studies)
- SMA, EMA, WMA - 移動平均線
- BBANDS - 布林通道
- DEMA, TEMA - 雙重/三重指數移動平均線
- KAMA - 考夫曼自適應移動平均線

### 動量指標 (Momentum Indicators)
- RSI - 相對強弱指數
- STOCH - 隨機指標
- MACD - 移動平均收斂發散
- MOM - 動量
- ROC - 變動率

### 成交量指標 (Volume Indicators)
- OBV - 能量潮
- AD - 累積/派發線
- ADOSC - 蔡金振盪器

### 波動率指標 (Volatility Indicators)
- ATR - 平均真實波幅
- NATR - 標準化平均真實波幅
- TRANGE - 真實波幅

### 價格轉換 (Price Transform)
- AVGPRICE - 平均價格
- MEDPRICE - 中間價格
- TYPPRICE - 典型價格
- WCLPRICE - 加權收盤價

### 週期指標 (Cycle Indicators)
- HT_DCPERIOD - 希爾伯特變換主導週期
- HT_DCPHASE - 希爾伯特變換主導相位
- HT_SINE - 希爾伯特變換正弦波

### 統計函數 (Statistic Functions)
- BETA - 貝塔係數
- CORREL - 相關係數
- LINEARREG - 線性回歸
- STDDEV - 標準差
- VAR - 變異數

## 使用技巧

### 1. 數據處理
```python
# 處理 NaN 值
df['sma_20'] = talib.SMA(close, timeperiod=20)
df['sma_20'] = df['sma_20'].fillna(method='bfill')
```

### 2. 多指標組合
```python
# 結合多個指標進行信號生成
buy_signal = (
    (close > sma_20) &  # 價格在移動平均線之上
    (rsi < 70) &        # RSI 未超買
    (macd > macd_signal)  # MACD 金叉
)
```

### 3. 動態參數
```python
# 根據市場條件調整參數
if volatility_high:
    rsi_period = 21  # 較長週期減少噪音
else:
    rsi_period = 14  # 標準週期

rsi = talib.RSI(close, timeperiod=rsi_period)
```

## 常見問題

### Q: 為什麼有些指標返回 NaN？
A: 許多指標需要一定的歷史數據才能計算，例如 20 期移動平均線需要至少 20 個數據點。

### Q: 如何選擇合適的參數？
A: 參數選擇取決於：
- 交易時間框架 (日線、小時線、分鐘線)
- 市場特性 (趨勢型、震盪型)
- 個人交易風格

### Q: TA-Lib 和 pandas 的技術指標有什麼區別？
A: TA-Lib 執行速度更快，計算更精確，且提供更多指標。pandas 的技術指標功能較為有限。

## 進階應用

### 1. 自定義指標組合
```python
def custom_indicator(close, volume, period=14):
    """自定義指標範例"""
    # 結合價格和成交量
    price_volume = close * volume
    return talib.SMA(price_volume, timeperiod=period)
```

### 2. 信號系統
```python
def generate_signals(df):
    """生成交易信號"""
    signals = pd.DataFrame(index=df.index)
    
    # 計算指標
    signals['rsi'] = talib.RSI(df['close'].values, timeperiod=14)
    signals['macd'], signals['macd_signal'], _ = talib.MACD(df['close'].values)
    
    # 生成信號
    signals['buy'] = (
        (signals['rsi'] < 30) & 
        (signals['macd'] > signals['macd_signal'])
    )
    
    signals['sell'] = (
        (signals['rsi'] > 70) & 
        (signals['macd'] < signals['macd_signal'])
    )
    
    return signals
```

## 參考資源

- [TA-Lib 官方文檔](http://ta-lib.org/)
- [TA-Lib Python 綁定](https://github.com/mrjbq7/ta-lib)
- [技術分析指標百科](https://www.investopedia.com/terms/t/technicalindicator.asp)

## 注意事項

1. **回測驗證**: 任何技術指標都應該在歷史數據上進行回測驗證
2. **風險管理**: 技術指標只是工具，不能替代風險管理
3. **市場適應性**: 不同市場條件下，指標的有效性可能不同
4. **過度擬合**: 避免使用過多指標或過度優化參數 