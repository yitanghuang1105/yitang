"""
Discord 機器人通知模組
負責發送台指期警示訊息到 Discord 頻道
"""
import aiohttp
import asyncio
import json
import logging
from datetime import datetime
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class DiscordNotifier:
    def __init__(self, webhook_url=None, channel_id=None, bot_token=None):
        self.webhook_url = webhook_url or "YOUR_DISCORD_WEBHOOK_URL"
        self.channel_id = channel_id
        self.bot_token = bot_token
        self.session = None
        self.last_sent_time = {}  # 用於防止重複發送
        
    async def __aenter__(self):
        """非同步上下文管理器入口"""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """非同步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    async def send_alerts(self, signals: List[Dict[str, Any]]):
        """發送警示訊息"""
        if not signals:
            return
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            for signal in signals:
                await self._send_single_alert(signal)
                # 避免發送過於頻繁
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error sending Discord alerts: {e}")
    
    async def _send_single_alert(self, signal: Dict[str, Any]):
        """發送單個警示訊息"""
        try:
            # 檢查是否在冷卻期內
            signal_key = f"{signal['type']}_{signal['signal']}"
            current_time = datetime.now()
            
            if signal_key in self.last_sent_time:
                time_diff = (current_time - self.last_sent_time[signal_key]).total_seconds()
                if time_diff < 300:  # 5分鐘冷卻
                    logger.info(f"Signal {signal_key} is in cooldown period")
                    return
            
            # 建立訊息內容
            message_content = self._format_message_content(signal)
            embed = self._create_embed(signal)
            
            # 發送訊息
            if self.webhook_url and self.webhook_url != "YOUR_DISCORD_WEBHOOK_URL":
                await self._send_webhook_message(message_content, embed)
            elif self.bot_token and self.channel_id:
                await self._send_bot_message(message_content, embed)
            else:
                logger.warning("No Discord configuration found. Message would be:")
                logger.info(f"Content: {message_content}")
                logger.info(f"Embed: {embed}")
            
            # 更新發送時間
            self.last_sent_time[signal_key] = current_time
            
        except Exception as e:
            logger.error(f"Error sending single alert: {e}")
    
    def _format_message_content(self, signal: Dict[str, Any]) -> str:
        """格式化訊息內容"""
        content = signal.get('message', '')
        
        # 添加 @用戶 功能
        if signal.get('type') in ['large_price_move', 'rapid_price_move']:
            content += "\n@TraderTeam 快看！"
        
        return content
    
    def _create_embed(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """建立 Discord Embed 訊息"""
        # 根據訊號類型設定顏色
        color_map = {
            'large_price_move': 0x00ff00 if '大漲' in signal['signal'] else 0xff0000,
            'rapid_price_move': 0x00ff00 if '急漲' in signal['signal'] else 0xff0000,
            'technical_rsi': 0xff0000 if '超買' in signal['signal'] else 0x00ff00,
            'technical_macd': 0x00ff00 if '金叉' in signal['signal'] else 0xff0000,
            'volume_anomaly': 0xffa500
        }
        
        embed = {
            "title": f"YO BRO WAKE UP!!!🚨 台指期預警!!! - {signal['signal']}",
            "description": signal.get('message', ''),
            "color": color_map.get(signal['type'], 0x808080),
            "timestamp": signal.get('timestamp', datetime.now()).isoformat(),
            "footer": {
                "text": "台指期 Discord 機器人即時預警系統"
            },
            "fields": []
        }
        
        # 添加詳細資訊欄位
        if signal['type'] == 'large_price_move':
            embed["fields"].extend([
                {
                    "name": "價格變化",
                    "value": f"{signal['price_change']:+.0f} 點",
                    "inline": True
                },
                {
                    "name": "百分比變化",
                    "value": f"{signal['price_change_pct']:+.2f}%",
                    "inline": True
                },
                {
                    "name": "當前價格",
                    "value": f"{signal['current_price']:,.0f}",
                    "inline": True
                }
            ])
        elif signal['type'] == 'rapid_price_move':
            embed["fields"].extend([
                {
                    "name": "3分鐘變化",
                    "value": f"{signal['price_change']:+.0f} 點",
                    "inline": True
                },
                {
                    "name": "當前價格",
                    "value": f"{signal['current_price']:,.0f}",
                    "inline": True
                }
            ])
        elif signal['type'] == 'technical_rsi':
            embed["fields"].append({
                "name": "RSI 數值",
                "value": f"{signal['rsi']:.1f}",
                "inline": True
            })
        elif signal['type'] == 'volume_anomaly':
            embed["fields"].extend([
                {
                    "name": "成交量倍數",
                    "value": f"{signal['volume_ratio']:.1f}x",
                    "inline": True
                },
                {
                    "name": "當前成交量",
                    "value": f"{signal['current_volume']:,.0f}",
                    "inline": True
                }
            ])
        
        return embed
    
    async def _send_webhook_message(self, content: str, embed: Dict[str, Any]):
        """透過 Webhook 發送訊息"""
        payload = {
            "content": content,
            "embeds": [embed]
        }
        
        async with self.session.post(self.webhook_url, json=payload) as response:
            if response.status == 204:
                logger.info("Discord webhook message sent successfully")
            else:
                logger.error(f"Failed to send webhook message: {response.status}")
    
    async def _send_bot_message(self, content: str, embed: Dict[str, Any]):
        """透過 Bot 發送訊息"""
        if not self.bot_token or not self.channel_id:
            return
        
        url = f"https://discord.com/api/v10/channels/{self.channel_id}/messages"
        headers = {
            "Authorization": f"Bot {self.bot_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "content": content,
            "embeds": [embed]
        }
        
        async with self.session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                logger.info("Discord bot message sent successfully")
            else:
                logger.error(f"Failed to send bot message: {response.status}")
    
    def set_webhook_url(self, webhook_url: str):
        """設定 Webhook URL"""
        self.webhook_url = webhook_url
    
    def set_bot_config(self, bot_token: str, channel_id: str):
        """設定 Bot 配置"""
        self.bot_token = bot_token
        self.channel_id = channel_id 