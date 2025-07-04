"""
簡單的 Discord 連接測試程式
"""
import asyncio
import json
from notifier.discord_bot import DiscordNotifier

async def test_discord():
    """測試 Discord 連接"""
    print("🧪 測試 Discord 連接...")
    
    # 載入設定
    with open('discord_config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 建立通知器
    notifier = DiscordNotifier(webhook_url=config['webhook_url'])
    
    # 發送測試訊息
    from datetime import datetime
    test_signal = {
        'type': 'test',
        'signal': '測試訊號',
        'message': '🧪 台指期 Discord 機器人連接測試成功！\n🎯 系統已準備好監控台指期走勢！',
        'timestamp': datetime.now()
    }
    
    try:
        async with notifier:
            await notifier.send_alerts([test_signal])
        print("✅ Discord 連接測試成功！")
        print("📱 請檢查你的 Discord 頻道，應該會看到測試訊息")
    except Exception as e:
        print(f"❌ Discord 連接測試失敗: {e}")

if __name__ == "__main__":
    asyncio.run(test_discord()) 