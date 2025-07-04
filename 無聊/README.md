# 📡 台指期 Discord 機器人即時預警系統

## 🎯 專案目標

每 **1 分鐘即時監控台指期（TXF）大台** 價格走勢，偵測：
- ✅ **大漲、大跌**（如 ±1.5% or ±200 點）
- ✅ **急漲、急跌**（如 3 分鐘內跳動超過 100 點）
- ✅ **技術指標共振**：RSI、MACD 趨勢出現強烈變化
- ✅ **成交量異常**：成交量突然放大

並透過 **Discord 機器人即時推播警示訊息**。

## 🏗️ 系統架構

```
無聊/
├── main.py                  # 主程序，定時呼叫模組並處理通知
├── data_source/
│   ├── __init__.py
│   ├── mc_file_reader.py    # MultiCharts 匯出資料監控
│   └── ib_api_reader.py     # IB 接口擷取行情（備用）
├── signal_engine/
│   ├── __init__.py
│   ├── rules.py             # 訊號判斷邏輯
│   └── config.yaml          # 條件參數設定
├── notifier/
│   ├── __init__.py
│   └── discord_bot.py       # Discord webhook 傳送
├── logs/                    # 日誌檔案目錄
├── requirements.txt         # 依賴套件清單
└── README.md               # 專案說明
```

## 🚀 快速開始

### 1. 安裝依賴套件

```bash
pip install -r requirements.txt
```

### 2. 設定 Discord Webhook

在 `notifier/discord_bot.py` 中設定你的 Discord Webhook URL：

```python
# 方法 1：使用 Webhook（推薦）
discord_notifier = DiscordNotifier(webhook_url="YOUR_WEBHOOK_URL")

# 方法 2：使用 Bot Token
discord_notifier = DiscordNotifier(
    bot_token="YOUR_BOT_TOKEN", 
    channel_id="YOUR_CHANNEL_ID"
)
```

### 3. 設定資料來源

確保你的台指期資料檔案路徑正確：

```python
# 在 data_source/mc_file_reader.py 中
data_reader = MultiChartsReader(data_file_path="TXF1_Minute_2020-01-01_2025-06-16.txt")
```

### 4. 調整訊號參數

在 `signal_engine/config.yaml` 中調整各種閾值：

```yaml
price_change:
  large_move_threshold: 200      # 大漲大跌閾值（點）
  large_move_pct_threshold: 1.5  # 大漲大跌百分比閾值
  rapid_move_threshold: 100      # 急漲急跌閾值（3分鐘內）

technical_indicators:
  rsi_overbought: 80             # RSI 超買閾值
  rsi_oversold: 20               # RSI 超賣閾值
```

### 5. 運行系統

```bash
python main.py
```

## 📊 訊號類型

### 1. 大漲大跌訊號
- **觸發條件**：價格變化超過 ±200 點或 ±1.5%
- **訊息範例**：`🚀 Dude Wake Up！台指期大漲！！！+218 點 (1.2%)`

### 2. 急漲急跌訊號
- **觸發條件**：3 分鐘內價格變化超過 ±100 點
- **訊息範例**：`⚡ 台指期急漲！3分鐘內+150 點`

### 3. 技術指標訊號
- **RSI 超買/超賣**：RSI > 80 或 < 20
- **MACD 金叉/死叉**：MACD 線與信號線交叉
- **訊息範例**：`📈 RSI超買警告！RSI=86.5 (閾值:80)`

### 4. 成交量異常訊號
- **觸發條件**：成交量超過平均值的 2 倍
- **訊息範例**：`📊 成交量異常放大！2.5倍於平均`

## 🔧 配置說明

### Discord 設定

#### 方法 1：Webhook（推薦）
1. 在 Discord 頻道中右鍵 → 編輯頻道 → 整合 → Webhook
2. 建立新的 Webhook 並複製 URL
3. 在程式碼中設定 `webhook_url`

#### 方法 2：Bot Token
1. 到 [Discord Developer Portal](https://discord.com/developers/applications) 建立應用程式
2. 建立 Bot 並複製 Token
3. 邀請 Bot 到你的伺服器
4. 在程式碼中設定 `bot_token` 和 `channel_id`

### 資料來源設定

#### MultiCharts 檔案監控
- 設定 MultiCharts 每分鐘匯出資料到指定檔案
- 系統會自動監控檔案變化並讀取最新資料

#### IB API 連接（可選）
- 安裝 `ib_insync` 套件：`pip install ib_insync`
- 確保 IB Gateway 或 TWS 正在運行
- 在 `ib_api_reader.py` 中取消註解相關程式碼

## 📝 日誌記錄

系統會自動記錄所有活動到 `logs/signal_log.txt`：

```
2024-01-15 14:30:15 - INFO - Starting Trading Alert System...
2024-01-15 14:30:16 - INFO - Reading updated data file: TXF1_Minute_2020-01-01_2025-06-16.txt
2024-01-15 14:30:17 - INFO - Discord webhook message sent successfully
```

## 🔮 未來擴充功能

- ⛓️ 結合 Flowise / LangChain，自動解讀新聞與技術面訊號
- 📈 產生技術圖 + 價格圖，上傳 Discord
- 🧠 建立交易策略勝率統計系統
- 🪙 多商品支援（加權指數、ETF、匯率、加密貨幣等）
- 🤖 更智能的訊號過濾和優先級排序
- 📱 手機 App 推送通知

## 🛠️ 故障排除

### 常見問題

1. **Discord 訊息發送失敗**
   - 檢查 Webhook URL 是否正確
   - 確認 Bot 有發送訊息的權限

2. **資料檔案讀取錯誤**
   - 確認檔案路徑正確
   - 檢查檔案格式是否符合預期

3. **訊號觸發過於頻繁**
   - 調整 `config.yaml` 中的閾值
   - 檢查冷卻時間設定

### 除錯模式

在 `main.py` 中啟用詳細日誌：

```python
logging.basicConfig(level=logging.DEBUG)
```

## 📄 授權

本專案僅供學習和研究使用，請勿用於實際交易決策。 

## 使用方式

1. **重新啟動 Flask 控制頁面**  
   （如果已經在執行，請先 Ctrl+C 停止，再重新執行）
   ```bash
   python manual_sender.py
   ```

2. **打開瀏覽器**，進入  
   ```
   http://localhost:5000
   ```

3. **點選「參數設定」分頁**  
   - 你會看到 `config.yaml` 的內容
   - 可以直接編輯參數（如 `large_move_threshold`）
   - 按下「儲存參數」即可

4. **參數儲存後，請重新啟動主程式**  
   讓新參數生效！

---

有任何問題或想要更進階的設定頁面，隨時告訴我！ 