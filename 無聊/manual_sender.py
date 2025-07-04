"""
手動控制訊息發送頁面 + 參數設定頁（修正版，無 extends）
"""
from flask import Flask, render_template_string, request, redirect, url_for, flash
import json
import asyncio
from notifier.discord_bot import DiscordNotifier
from datetime import datetime
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # 用於 flash 訊息

# 載入 Discord 設定
with open('discord_config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'signal_engine', 'config.yaml')

# ========== 發送訊息功能 ==========
def send_discord_message(content, signal_type='manual', signal_name='手動訊息'):
    async def send():
        notifier = DiscordNotifier(webhook_url=config['webhook_url'])
        test_signal = {
            'type': signal_type,
            'signal': signal_name,
            'message': content,
            'timestamp': datetime.now()
        }
        async with notifier:
            await notifier.send_alerts([test_signal])
    asyncio.run(send())

HTML_BASE = '''
<!DOCTYPE html>
<html lang="zh-Hant">
<head>
    <meta charset="UTF-8">
    <title>Discord 控制台</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f5f5f5; }
        .container { max-width: 600px; margin: 60px auto; background: #fff; padding: 30px; border-radius: 10px; box-shadow: 0 2px 8px #ccc; }
        h2 { text-align: center; }
        textarea { width: 100%; height: 100px; margin-bottom: 15px; }
        select, button, input[type=text] { width: 100%; padding: 10px; margin-bottom: 10px; }
        .msg { color: green; text-align: center; margin-bottom: 10px; }
        .nav { text-align: center; margin-bottom: 20px; }
        .nav a { margin: 0 10px; text-decoration: none; color: #007bff; }
        .nav a.active { font-weight: bold; color: #0056b3; }
        .yaml-area { width: 100%; height: 350px; font-family: monospace; }
    </style>
</head>
<body>
    <div class="container">
        <div class="nav">
            <a href="/" class="{{ 'active' if page=='send' else '' }}">手動發送</a>
            <a href="/config" class="{{ 'active' if page=='config' else '' }}">參數設定</a>
        </div>
        {{ content|safe }}
    </div>
</body>
</html>
'''

SEND_FORM = '''
    <h2>Discord 手動訊息發送</h2>
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        <div class="msg">{{ messages[0] }}</div>
      {% endif %}
    {% endwith %}
    <form method="post">
        <label for="content">訊息內容：</label>
        <textarea name="content" id="content" required></textarea>
        <label for="signal_type">訊息類型：</label>
        <select name="signal_type" id="signal_type">
            <option value="manual">手動訊息</option>
            <option value="large_price_move">大漲/大跌</option>
            <option value="rapid_price_move">急漲/急跌</option>
            <option value="technical_rsi">RSI</option>
            <option value="technical_macd">MACD</option>
            <option value="volume_anomaly">成交量異常</option>
        </select>
        <label for="signal_name">訊號名稱（可自訂）：</label>
        <input type="text" name="signal_name" id="signal_name" value="手動訊息">
        <button type="submit">發送到 Discord</button>
    </form>
'''

CONFIG_FORM = '''
    <h2>參數設定 (config.yaml)</h2>
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        <div class="msg">{{ messages[0] }}</div>
      {% endif %}
    {% endwith %}
    <form method="post">
        <textarea name="yaml_content" class="yaml-area" required>{{ yaml_content }}</textarea>
        <button type="submit">儲存參數</button>
    </form>
'''

from flask import render_template_string as rts

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        content = request.form['content']
        signal_type = request.form['signal_type']
        signal_name = request.form['signal_name'] or '手動訊息'
        try:
            send_discord_message(content, signal_type, signal_name)
            flash('訊息已發送到 Discord！')
        except Exception as e:
            flash(f'發送失敗: {e}')
        return redirect(url_for('index'))
    content = rts(SEND_FORM)
    return rts(HTML_BASE, page='send', content=content)

@app.route('/config', methods=['GET', 'POST'])
def config_page():
    if request.method == 'POST':
        yaml_content = request.form['yaml_content']
        try:
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                f.write(yaml_content)
            flash('參數已儲存！')
        except Exception as e:
            flash(f'儲存失敗: {e}')
        return redirect(url_for('config_page'))
    # 讀取現有 YAML
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            yaml_content = f.read()
    except Exception as e:
        yaml_content = f'# 讀取失敗: {e}'
    content = rts(CONFIG_FORM, yaml_content=yaml_content)
    return rts(HTML_BASE, page='config', content=content)

if __name__ == '__main__':
    app.run(debug=True) 