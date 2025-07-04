"""
Discord 設定輔助程式
幫助用戶快速設定 Discord Webhook 或 Bot
"""
import os
import json
from pathlib import Path

def setup_discord_webhook():
    """設定 Discord Webhook"""
    print("=== Discord Webhook 設定 ===")
    print("1. 在 Discord 頻道中右鍵 → 編輯頻道 → 整合 → Webhook")
    print("2. 點擊 '新增 Webhook'")
    print("3. 設定 Webhook 名稱（例如：台指期預警機器人）")
    print("4. 複製 Webhook URL")
    print()
    
    webhook_url = input("請貼上你的 Discord Webhook URL: ").strip()
    
    if webhook_url.startswith("https://discord.com/api/webhooks/"):
        # 儲存設定
        config = {
            "discord_type": "webhook",
            "webhook_url": webhook_url
        }
        
        save_config(config)
        print("✅ Webhook 設定已儲存！")
        return webhook_url
    else:
        print("❌ 無效的 Webhook URL，請檢查後重試")
        return None

def setup_discord_bot():
    """設定 Discord Bot"""
    print("=== Discord Bot 設定 ===")
    print("1. 前往 https://discord.com/developers/applications")
    print("2. 點擊 'New Application' 建立新應用程式")
    print("3. 在左側選單選擇 'Bot'")
    print("4. 點擊 'Add Bot'")
    print("5. 複製 Bot Token")
    print("6. 在 'OAuth2' → 'URL Generator' 中選擇 'bot' 權限")
    print("7. 選擇 'Send Messages' 權限")
    print("8. 使用生成的 URL 邀請 Bot 到你的伺服器")
    print("9. 在 Discord 中右鍵點擊頻道 → 複製頻道 ID")
    print()
    
    bot_token = input("請貼上你的 Bot Token: ").strip()
    channel_id = input("請貼上頻道 ID: ").strip()
    
    if bot_token and channel_id:
        config = {
            "discord_type": "bot",
            "bot_token": bot_token,
            "channel_id": channel_id
        }
        
        save_config(config)
        print("✅ Bot 設定已儲存！")
        return bot_token, channel_id
    else:
        print("❌ 請提供完整的 Bot Token 和頻道 ID")
        return None, None

def save_config(config):
    """儲存設定到檔案"""
    config_file = Path("discord_config.json")
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"設定已儲存到: {config_file}")

def load_config():
    """載入設定"""
    config_file = Path("discord_config.json")
    
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def test_discord_connection():
    """測試 Discord 連接"""
    config = load_config()
    
    if not config:
        print("❌ 未找到 Discord 設定，請先執行設定")
        return False
    
    print("=== 測試 Discord 連接 ===")
    
    try:
        import asyncio
        from notifier.discord_bot import DiscordNotifier
        
        async def test():
            if config["discord_type"] == "webhook":
                notifier = DiscordNotifier(webhook_url=config["webhook_url"])
            else:
                notifier = DiscordNotifier(
                    bot_token=config["bot_token"],
                    channel_id=config["channel_id"]
                )
            
            # 發送測試訊息
            test_signal = {
                'type': 'test',
                'signal': '測試訊號',
                'message': '🧪 這是一則測試訊息，確認 Discord 連接正常！',
                'timestamp': '2024-01-15T14:30:00'
            }
            
            await notifier.send_alerts([test_signal])
            print("✅ Discord 連接測試成功！")
        
        asyncio.run(test())
        return True
        
    except Exception as e:
        print(f"❌ Discord 連接測試失敗: {e}")
        return False

def main():
    """主函數"""
    print("🎯 台指期 Discord 機器人即時預警系統 - 設定程式")
    print()
    
    while True:
        print("請選擇操作：")
        print("1. 設定 Discord Webhook（推薦）")
        print("2. 設定 Discord Bot")
        print("3. 測試 Discord 連接")
        print("4. 查看當前設定")
        print("5. 退出")
        print()
        
        choice = input("請輸入選項 (1-5): ").strip()
        
        if choice == "1":
            setup_discord_webhook()
        elif choice == "2":
            setup_discord_bot()
        elif choice == "3":
            test_discord_connection()
        elif choice == "4":
            config = load_config()
            if config:
                print("當前設定：")
                print(json.dumps(config, indent=2, ensure_ascii=False))
            else:
                print("未找到設定檔案")
        elif choice == "5":
            print("再見！")
            break
        else:
            print("❌ 無效選項，請重新選擇")
        
        print()

if __name__ == "__main__":
    main() 