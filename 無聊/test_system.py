"""
台指期 Discord 機器人即時預警系統 - 測試程式
用於驗證系統各模組功能
"""
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from data_source.mc_file_reader import MultiChartsReader
from signal_engine.rules import SignalEngine
from notifier.discord_bot import DiscordNotifier

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_data():
    """建立測試用的台指期資料"""
    # 建立模擬的台指期資料
    dates = pd.date_range(start='2024-01-15 09:00:00', periods=100, freq='1min')
    
    # 模擬價格走勢（包含大漲大跌情況）
    base_price = 18200
    prices = []
    
    for i in range(100):
        if i == 50:  # 在第50分鐘模擬大漲
            price = base_price + 250
        elif i == 70:  # 在第70分鐘模擬大跌
            price = base_price - 180
        else:
            # 正常波動
            change = np.random.normal(0, 10)
            price = base_price + change
        
        prices.append(price)
        base_price = price
    
    # 建立 DataFrame
    df = pd.DataFrame({
        'Date': [d.strftime('%Y-%m-%d') for d in dates],
        'Time': [d.strftime('%H:%M:%S') for d in dates],
        'Open': prices,
        'High': [p + np.random.uniform(0, 5) for p in prices],
        'Low': [p - np.random.uniform(0, 5) for p in prices],
        'Close': prices,
        'Volume': [np.random.randint(100, 1000) for _ in prices]
    })
    
    return df

async def test_data_reader():
    """測試資料讀取器"""
    logger.info("Testing data reader...")
    
    # 建立測試資料
    test_df = create_test_data()
    
    # 儲存測試資料
    test_file = "test_txf_data.txt"
    test_df.to_csv(test_file, index=False)
    
    # 測試資料讀取器
    reader = MultiChartsReader(test_file)
    data = reader.get_latest_data()
    
    if data is not None:
        logger.info(f"Data reader test passed. Loaded {len(data)} rows")
        logger.info(f"Latest price: {data.iloc[-1]['Close']}")
        return data
    else:
        logger.error("Data reader test failed")
        return None

def test_signal_engine(df):
    """測試訊號引擎"""
    logger.info("Testing signal engine...")
    
    if df is None:
        logger.error("No data available for signal testing")
        return []
    
    # 建立訊號引擎
    engine = SignalEngine()
    
    # 分析訊號
    signals = engine.analyze_signals(df)
    
    if signals:
        logger.info(f"Signal engine test passed. Found {len(signals)} signals:")
        for signal in signals:
            logger.info(f"  - {signal['signal']}: {signal['message']}")
    else:
        logger.info("Signal engine test passed. No signals detected.")
    
    return signals

async def test_discord_notifier(signals):
    """測試 Discord 通知器"""
    logger.info("Testing Discord notifier...")
    
    if not signals:
        logger.info("No signals to test Discord notification")
        return
    
    # 建立 Discord 通知器（使用模擬模式）
    notifier = DiscordNotifier()
    
    try:
        # 發送測試訊息
        await notifier.send_alerts(signals)
        logger.info("Discord notifier test passed (simulation mode)")
    except Exception as e:
        logger.error(f"Discord notifier test failed: {e}")

async def test_full_system():
    """測試完整系統"""
    logger.info("Starting full system test...")
    
    # 1. 測試資料讀取
    df = await test_data_reader()
    
    # 2. 測試訊號引擎
    signals = test_signal_engine(df)
    
    # 3. 測試 Discord 通知
    await test_discord_notifier(signals)
    
    logger.info("Full system test completed!")

def test_config_loading():
    """測試設定檔載入"""
    logger.info("Testing config loading...")
    
    try:
        engine = SignalEngine()
        config = engine.config
        
        logger.info("Config loaded successfully:")
        logger.info(f"  - Large move threshold: {config['price_change']['large_move_threshold']}")
        logger.info(f"  - RSI overbought: {config['technical_indicators']['rsi_overbought']}")
        
    except Exception as e:
        logger.error(f"Config loading test failed: {e}")

async def main():
    """主測試函數"""
    logger.info("=== 台指期 Discord 機器人即時預警系統測試 ===")
    
    # 測試設定檔載入
    test_config_loading()
    
    # 測試完整系統
    await test_full_system()
    
    logger.info("=== 測試完成 ===")

if __name__ == "__main__":
    asyncio.run(main()) 