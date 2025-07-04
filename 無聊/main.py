"""
台指期 Discord 機器人即時預警系統 - 主程序
負責定時呼叫模組並處理通知
"""
import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path

from data_source.mc_file_reader import MultiChartsReader
from signal_engine.rules import SignalEngine
from notifier.discord_bot import DiscordNotifier

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/signal_log.txt', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingAlertSystem:
    def __init__(self):
        self.data_reader = MultiChartsReader()
        self.signal_engine = SignalEngine()
        self.discord_notifier = DiscordNotifier()
        self.last_check_time = None
        
    async def run(self):
        """主運行循環"""
        logger.info("Starting Trading Alert System...")
        
        while True:
            try:
                # 檢查是否有新資料
                current_data = self.data_reader.get_latest_data()
                
                if current_data is not None and current_data != self.last_check_time:
                    logger.info(f"Processing new data: {current_data}")
                    
                    # 分析訊號
                    signals = self.signal_engine.analyze_signals(current_data)
                    
                    # 發送通知
                    if signals:
                        await self.discord_notifier.send_alerts(signals)
                    
                    self.last_check_time = current_data
                
                # 等待1分鐘
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(60)

async def main():
    """主函數"""
    # 確保日誌目錄存在
    Path("logs").mkdir(exist_ok=True)
    
    # 建立並運行系統
    system = TradingAlertSystem()
    await system.run()

if __name__ == "__main__":
    asyncio.run(main()) 