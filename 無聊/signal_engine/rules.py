"""
訊號判斷邏輯引擎
包含漲跌幅、技術指標等條件判斷
"""
import pandas as pd
import numpy as np
import yaml
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class SignalEngine:
    def __init__(self, config_file="config.yaml"):
        self.config = self._load_config(config_file)
        self.last_signals = set()  # 避免重複發送相同訊號
        
    def _load_config(self, config_file):
        """載入設定檔"""
        config_path = Path(__file__).parent / config_file
        
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            # 預設設定
            return {
                'price_change': {
                    'large_move_threshold': 200,  # 大漲大跌閾值（點）
                    'large_move_pct_threshold': 1.5,  # 大漲大跌百分比閾值
                    'rapid_move_threshold': 100,  # 急漲急跌閾值（3分鐘內）
                },
                'technical_indicators': {
                    'rsi_overbought': 80,
                    'rsi_oversold': 20,
                    'macd_threshold': 0.5,
                },
                'volume': {
                    'volume_surge_threshold': 2.0,  # 成交量放大倍數
                }
            }
    
    def analyze_signals(self, df):
        """分析訊號"""
        if df is None or len(df) < 20:
            return []
        
        signals = []
        current_row = df.iloc[-1]
        
        # 1. 檢查大漲大跌
        large_move_signal = self._check_large_price_move(df)
        if large_move_signal:
            signals.append(large_move_signal)
        
        # 2. 檢查急漲急跌
        rapid_move_signal = self._check_rapid_price_move(df)
        if rapid_move_signal:
            signals.append(rapid_move_signal)
        
        # 3. 檢查技術指標
        technical_signals = self._check_technical_indicators(df)
        signals.extend(technical_signals)
        
        # 4. 檢查成交量異常
        volume_signal = self._check_volume_anomaly(df)
        if volume_signal:
            signals.append(volume_signal)
        
        # 避免重複發送相同訊號
        current_signal_key = self._generate_signal_key(signals)
        if current_signal_key not in self.last_signals:
            self.last_signals.add(current_signal_key)
            return signals
        else:
            return []
    
    def _check_large_price_move(self, df):
        """檢查大漲大跌"""
        current_row = df.iloc[-1]
        
        # 計算與昨日收盤價的差異
        if len(df) >= 2:
            yesterday_close = df.iloc[-2]['Close']
            price_change = current_row['Close'] - yesterday_close
            price_change_pct = (price_change / yesterday_close) * 100
            
            threshold = self.config['price_change']['large_move_threshold']
            pct_threshold = self.config['price_change']['large_move_pct_threshold']
            
            if abs(price_change) >= threshold or abs(price_change_pct) >= pct_threshold:
                signal_type = "大漲" if price_change > 0 else "大跌"
                emoji = "🚀" if price_change > 0 else "💥"
                
                return {
                    'type': 'large_price_move',
                    'signal': signal_type,
                    'price_change': price_change,
                    'price_change_pct': price_change_pct,
                    'current_price': current_row['Close'],
                    'emoji': emoji,
                    'timestamp': current_row.get('DateTime', datetime.now()),
                    'message': f"{emoji} Dude Wake Up！台指期{signal_type}！！！{price_change:+.0f} 點 ({price_change_pct:+.2f}%)"
                }
        
        return None
    
    def _check_rapid_price_move(self, df):
        """檢查急漲急跌（3分鐘內）"""
        if len(df) < 4:
            return None
        
        current_price = df.iloc[-1]['Close']
        price_3min_ago = df.iloc[-4]['Close']  # 3分鐘前
        price_change = current_price - price_3min_ago
        
        threshold = self.config['price_change']['rapid_move_threshold']
        
        if abs(price_change) >= threshold:
            signal_type = "急漲" if price_change > 0 else "急跌"
            emoji = "⚡" if price_change > 0 else "🔥"
            
            return {
                'type': 'rapid_price_move',
                'signal': signal_type,
                'price_change': price_change,
                'current_price': current_price,
                'emoji': emoji,
                'timestamp': df.iloc[-1].get('DateTime', datetime.now()),
                'message': f"{emoji} 台指期{signal_type}！3分鐘內{price_change:+.0f} 點"
            }
        
        return None
    
    def _check_technical_indicators(self, df):
        """檢查技術指標"""
        signals = []
        current_row = df.iloc[-1]
        
        # RSI 檢查
        if 'RSI' in current_row and not pd.isna(current_row['RSI']):
            rsi = current_row['RSI']
            overbought = self.config['technical_indicators']['rsi_overbought']
            oversold = self.config['technical_indicators']['rsi_oversold']
            
            if rsi >= overbought:
                signals.append({
                    'type': 'technical_rsi',
                    'signal': 'RSI超買',
                    'rsi': rsi,
                    'emoji': '📈',
                    'timestamp': current_row.get('DateTime', datetime.now()),
                    'message': f"📈 RSI超買警告！RSI={rsi:.1f} (閾值:{overbought})"
                })
            elif rsi <= oversold:
                signals.append({
                    'type': 'technical_rsi',
                    'signal': 'RSI超賣',
                    'rsi': rsi,
                    'emoji': '📉',
                    'timestamp': current_row.get('DateTime', datetime.now()),
                    'message': f"📉 RSI超賣警告！RSI={rsi:.1f} (閾值:{oversold})"
                })
        
        # MACD 檢查（簡化版）
        if len(df) >= 26:
            ema12 = df['Close'].ewm(span=12).mean()
            ema26 = df['Close'].ewm(span=26).mean()
            macd = ema12 - ema26
            signal_line = macd.ewm(span=9).mean()
            
            current_macd = macd.iloc[-1]
            current_signal = signal_line.iloc[-1]
            prev_macd = macd.iloc[-2]
            prev_signal = signal_line.iloc[-2]
            
            # MACD 金叉/死叉
            if current_macd > current_signal and prev_macd <= prev_signal:
                signals.append({
                    'type': 'technical_macd',
                    'signal': 'MACD金叉',
                    'macd': current_macd,
                    'emoji': '🟢',
                    'timestamp': current_row.get('DateTime', datetime.now()),
                    'message': f"🟢 MACD金叉！看漲訊號"
                })
            elif current_macd < current_signal and prev_macd >= prev_signal:
                signals.append({
                    'type': 'technical_macd',
                    'signal': 'MACD死叉',
                    'macd': current_macd,
                    'emoji': '🔴',
                    'timestamp': current_row.get('DateTime', datetime.now()),
                    'message': f"🔴 MACD死叉！看跌訊號"
                })
        
        return signals
    
    def _check_volume_anomaly(self, df):
        """檢查成交量異常"""
        if len(df) < 20:
            return None
        
        current_volume = df.iloc[-1]['Volume']
        avg_volume = df['Volume'].rolling(window=20).mean().iloc[-1]
        
        if avg_volume > 0:
            volume_ratio = current_volume / avg_volume
            threshold = self.config['volume']['volume_surge_threshold']
            
            if volume_ratio >= threshold:
                return {
                    'type': 'volume_anomaly',
                    'signal': '成交量放大',
                    'volume_ratio': volume_ratio,
                    'current_volume': current_volume,
                    'avg_volume': avg_volume,
                    'emoji': '📊',
                    'timestamp': df.iloc[-1].get('DateTime', datetime.now()),
                    'message': f"📊 成交量異常放大！{volume_ratio:.1f}倍於平均"
                }
        
        return None
    
    def _generate_signal_key(self, signals):
        """生成訊號鍵值，用於避免重複發送"""
        if not signals:
            return ""
        
        # 將所有訊號類型組合為鍵值
        signal_types = [s['type'] for s in signals]
        return "_".join(sorted(signal_types)) 