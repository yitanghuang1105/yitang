import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import talib
from datetime import datetime, timedelta
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGridLayout, QLabel, QSlider, 
                             QSpinBox, QDoubleSpinBox, QPushButton, QTabWidget,
                             QTextEdit, QGroupBox, QCheckBox, QComboBox,
                             QProgressBar, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QPalette, QColor
import warnings
warnings.filterwarnings('ignore')

class StrategyParameterAdjuster(QMainWindow):
    """策略參數調整器 - 提供GUI介面調整RSI、布林通道、OBV等參數"""
    
    def __init__(self):
        super().__init__()
        self.data_file = "TXF1_Minute_2020-01-01_2025-06-16.txt"
        self.df = None
        self.current_params = {}
        self.init_ui()
        self.load_data()
        
    def init_ui(self):
        """初始化使用者介面"""
        self.setWindowTitle("Strategy Parameter Adjustment Tool")
        self.setGeometry(100, 100, 1400, 900)
        
        # 設定樣式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #ffffff;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4CAF50;
                border: 1px solid #5c6b7a;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
        """)
        
        # 創建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主佈局
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # 左側參數控制面板
        left_panel = self.create_parameter_panel()
        main_layout.addWidget(left_panel, 1)
        
        # 右側圖表顯示區域
        right_panel = self.create_chart_panel()
        main_layout.addWidget(right_panel, 2)
        
    def create_parameter_panel(self):
        """創建參數控制面板"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # 標題
        title = QLabel("Strategy Parameter Adjustment")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # 創建標籤頁
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)
        
        # RSI參數頁面
        rsi_tab = self.create_rsi_tab()
        tab_widget.addTab(rsi_tab, "RSI Parameters")
        
        # 布林通道參數頁面
        bb_tab = self.create_bollinger_tab()
        tab_widget.addTab(bb_tab, "Bollinger Bands")
        
        # OBV參數頁面
        obv_tab = self.create_obv_tab()
        tab_widget.addTab(obv_tab, "OBV Parameters")
        
        # 風險管理參數頁面
        risk_tab = self.create_risk_tab()
        tab_widget.addTab(risk_tab, "Risk Management")
        
        # 執行按鈕
        button_layout = QHBoxLayout()
        
        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.clicked.connect(self.run_analysis)
        button_layout.addWidget(self.run_btn)
        
        self.reset_btn = QPushButton("Reset Parameters")
        self.reset_btn.clicked.connect(self.reset_parameters)
        button_layout.addWidget(self.reset_btn)
        
        layout.addLayout(button_layout)
        
        # 結果顯示區域
        self.result_text = QTextEdit()
        self.result_text.setMaximumHeight(150)
        layout.addWidget(self.result_text)
        
        return panel
    
    def create_rsi_tab(self):
        """創建RSI參數頁面"""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        # RSI週期
        rsi_period_group = QGroupBox("RSI Period")
        rsi_layout = QGridLayout()
        
        self.rsi_period_slider = QSlider(Qt.Orientation.Horizontal)
        self.rsi_period_slider.setMinimum(5)
        self.rsi_period_slider.setMaximum(30)
        self.rsi_period_slider.setValue(14)
        self.rsi_period_slider.valueChanged.connect(self.update_rsi_period_label)
        
        self.rsi_period_label = QLabel("14")
        self.rsi_period_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        rsi_layout.addWidget(QLabel("Period:"), 0, 0)
        rsi_layout.addWidget(self.rsi_period_slider, 0, 1)
        rsi_layout.addWidget(self.rsi_period_label, 0, 2)
        rsi_period_group.setLayout(rsi_layout)
        layout.addWidget(rsi_period_group)
        
        # RSI超買超賣閾值
        rsi_threshold_group = QGroupBox("RSI Thresholds")
        threshold_layout = QGridLayout()
        
        # 超賣閾值
        self.rsi_oversold_slider = QSlider(Qt.Orientation.Horizontal)
        self.rsi_oversold_slider.setMinimum(10)
        self.rsi_oversold_slider.setMaximum(40)
        self.rsi_oversold_slider.setValue(30)
        self.rsi_oversold_slider.valueChanged.connect(self.update_rsi_oversold_label)
        
        self.rsi_oversold_label = QLabel("30")
        self.rsi_oversold_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        threshold_layout.addWidget(QLabel("Oversold:"), 0, 0)
        threshold_layout.addWidget(self.rsi_oversold_slider, 0, 1)
        threshold_layout.addWidget(self.rsi_oversold_label, 0, 2)
        
        # 超買閾值
        self.rsi_overbought_slider = QSlider(Qt.Orientation.Horizontal)
        self.rsi_overbought_slider.setMinimum(60)
        self.rsi_overbought_slider.setMaximum(90)
        self.rsi_overbought_slider.setValue(70)
        self.rsi_overbought_slider.valueChanged.connect(self.update_rsi_overbought_label)
        
        self.rsi_overbought_label = QLabel("70")
        self.rsi_overbought_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        threshold_layout.addWidget(QLabel("Overbought:"), 1, 0)
        threshold_layout.addWidget(self.rsi_overbought_slider, 1, 1)
        threshold_layout.addWidget(self.rsi_overbought_label, 1, 2)
        
        rsi_threshold_group.setLayout(threshold_layout)
        layout.addWidget(rsi_threshold_group)
        
        layout.addStretch()
        return widget
    
    def create_bollinger_tab(self):
        """創建布林通道參數頁面"""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        # 布林通道週期
        bb_period_group = QGroupBox("Bollinger Bands Period")
        bb_layout = QGridLayout()
        
        self.bb_period_slider = QSlider(Qt.Orientation.Horizontal)
        self.bb_period_slider.setMinimum(10)
        self.bb_period_slider.setMaximum(50)
        self.bb_period_slider.setValue(20)
        self.bb_period_slider.valueChanged.connect(self.update_bb_period_label)
        
        self.bb_period_label = QLabel("20")
        self.bb_period_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        bb_layout.addWidget(QLabel("Period:"), 0, 0)
        bb_layout.addWidget(self.bb_period_slider, 0, 1)
        bb_layout.addWidget(self.bb_period_label, 0, 2)
        bb_period_group.setLayout(bb_layout)
        layout.addWidget(bb_period_group)
        
        # 標準差倍數
        bb_std_group = QGroupBox("Standard Deviation Multiplier")
        std_layout = QGridLayout()
        
        self.bb_std_slider = QSlider(Qt.Orientation.Horizontal)
        self.bb_std_slider.setMinimum(10)
        self.bb_std_slider.setMaximum(50)
        self.bb_std_slider.setValue(20)
        self.bb_std_slider.valueChanged.connect(self.update_bb_std_label)
        
        self.bb_std_label = QLabel("2.0")
        self.bb_std_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        std_layout.addWidget(QLabel("Multiplier:"), 0, 0)
        std_layout.addWidget(self.bb_std_slider, 0, 1)
        std_layout.addWidget(self.bb_std_label, 0, 2)
        bb_std_group.setLayout(std_layout)
        layout.addWidget(bb_std_group)
        
        # 布林通道擠壓閾值
        bb_squeeze_group = QGroupBox("BB Squeeze Threshold")
        squeeze_layout = QGridLayout()
        
        self.bb_squeeze_spinbox = QDoubleSpinBox()
        self.bb_squeeze_spinbox.setMinimum(0.01)
        self.bb_squeeze_spinbox.setMaximum(0.50)
        self.bb_squeeze_spinbox.setValue(0.10)
        self.bb_squeeze_spinbox.setSingleStep(0.01)
        self.bb_squeeze_spinbox.setDecimals(2)
        
        squeeze_layout.addWidget(QLabel("Threshold:"), 0, 0)
        squeeze_layout.addWidget(self.bb_squeeze_spinbox, 0, 1)
        bb_squeeze_group.setLayout(squeeze_layout)
        layout.addWidget(bb_squeeze_group)
        
        layout.addStretch()
        return widget
    
    def create_obv_tab(self):
        """創建OBV參數頁面"""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        # OBV比率閾值
        obv_ratio_group = QGroupBox("OBV Ratio Threshold")
        obv_layout = QGridLayout()
        
        self.obv_ratio_slider = QSlider(Qt.Orientation.Horizontal)
        self.obv_ratio_slider.setMinimum(50)
        self.obv_ratio_slider.setMaximum(200)
        self.obv_ratio_slider.setValue(120)
        self.obv_ratio_slider.valueChanged.connect(self.update_obv_ratio_label)
        
        self.obv_ratio_label = QLabel("1.20")
        self.obv_ratio_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        obv_layout.addWidget(QLabel("Ratio:"), 0, 0)
        obv_layout.addWidget(self.obv_ratio_slider, 0, 1)
        obv_layout.addWidget(self.obv_ratio_label, 0, 2)
        obv_ratio_group.setLayout(obv_layout)
        layout.addWidget(obv_ratio_group)
        
        # OBV移動平均週期
        obv_ma_group = QGroupBox("OBV Moving Average Period")
        obv_ma_layout = QGridLayout()
        
        self.obv_ma_slider = QSlider(Qt.Orientation.Horizontal)
        self.obv_ma_slider.setMinimum(10)
        self.obv_ma_slider.setMaximum(50)
        self.obv_ma_slider.setValue(20)
        self.obv_ma_slider.valueChanged.connect(self.update_obv_ma_label)
        
        self.obv_ma_label = QLabel("20")
        self.obv_ma_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        obv_ma_layout.addWidget(QLabel("Period:"), 0, 0)
        obv_ma_layout.addWidget(self.obv_ma_slider, 0, 1)
        obv_ma_layout.addWidget(self.obv_ma_label, 0, 2)
        obv_ma_group.setLayout(obv_ma_layout)
        layout.addWidget(obv_ma_group)
        
        layout.addStretch()
        return widget
    
    def create_risk_tab(self):
        """創建風險管理參數頁面"""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        # 止損參數
        stop_loss_group = QGroupBox("Stop Loss")
        sl_layout = QGridLayout()
        
        self.stop_loss_spinbox = QDoubleSpinBox()
        self.stop_loss_spinbox.setMinimum(0.005)
        self.stop_loss_spinbox.setMaximum(0.10)
        self.stop_loss_spinbox.setValue(0.015)
        self.stop_loss_spinbox.setSingleStep(0.001)
        self.stop_loss_spinbox.setDecimals(3)
        self.stop_loss_spinbox.setSuffix("%")
        
        sl_layout.addWidget(QLabel("Percentage:"), 0, 0)
        sl_layout.addWidget(self.stop_loss_spinbox, 0, 1)
        stop_loss_group.setLayout(sl_layout)
        layout.addWidget(stop_loss_group)
        
        # 止盈參數
        take_profit_group = QGroupBox("Take Profit")
        tp_layout = QGridLayout()
        
        self.take_profit_spinbox = QDoubleSpinBox()
        self.take_profit_spinbox.setMinimum(0.01)
        self.take_profit_spinbox.setMaximum(0.20)
        self.take_profit_spinbox.setValue(0.03)
        self.take_profit_spinbox.setSingleStep(0.001)
        self.take_profit_spinbox.setDecimals(3)
        self.take_profit_spinbox.setSuffix("%")
        
        tp_layout.addWidget(QLabel("Percentage:"), 0, 0)
        tp_layout.addWidget(self.take_profit_spinbox, 0, 1)
        take_profit_group.setLayout(tp_layout)
        layout.addWidget(take_profit_group)
        
        # 追蹤止損參數
        trailing_stop_group = QGroupBox("Trailing Stop")
        ts_layout = QGridLayout()
        
        self.trailing_stop_spinbox = QDoubleSpinBox()
        self.trailing_stop_spinbox.setMinimum(0.002)
        self.trailing_stop_spinbox.setMaximum(0.05)
        self.trailing_stop_spinbox.setValue(0.008)
        self.trailing_stop_spinbox.setSingleStep(0.001)
        self.trailing_stop_spinbox.setDecimals(3)
        self.trailing_stop_spinbox.setSuffix("%")
        
        ts_layout.addWidget(QLabel("Percentage:"), 0, 0)
        ts_layout.addWidget(self.trailing_stop_spinbox, 0, 1)
        trailing_stop_group.setLayout(ts_layout)
        layout.addWidget(trailing_stop_group)
        
        # 最大持倉期數
        max_hold_group = QGroupBox("Maximum Hold Periods")
        mh_layout = QGridLayout()
        
        self.max_hold_spinbox = QSpinBox()
        self.max_hold_spinbox.setMinimum(5)
        self.max_hold_spinbox.setMaximum(100)
        self.max_hold_spinbox.setValue(20)
        
        mh_layout.addWidget(QLabel("Periods:"), 0, 0)
        mh_layout.addWidget(self.max_hold_spinbox, 0, 1)
        max_hold_group.setLayout(mh_layout)
        layout.addWidget(max_hold_group)
        
        layout.addStretch()
        return widget
    
    def create_chart_panel(self):
        """創建圖表顯示面板"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # 圖表標題
        self.chart_title = QLabel("Strategy Analysis Chart")
        self.chart_title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.chart_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.chart_title)
        
        # 圖表區域
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # 進度條
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        return panel
    
    def update_rsi_period_label(self):
        """更新RSI週期標籤"""
        value = self.rsi_period_slider.value()
        self.rsi_period_label.setText(str(value))
    
    def update_rsi_oversold_label(self):
        """更新RSI超賣標籤"""
        value = self.rsi_oversold_slider.value()
        self.rsi_oversold_label.setText(str(value))
    
    def update_rsi_overbought_label(self):
        """更新RSI超買標籤"""
        value = self.rsi_overbought_slider.value()
        self.rsi_overbought_label.setText(str(value))
    
    def update_bb_period_label(self):
        """更新布林通道週期標籤"""
        value = self.bb_period_slider.value()
        self.bb_period_label.setText(str(value))
    
    def update_bb_std_label(self):
        """更新布林通道標準差標籤"""
        value = self.bb_std_slider.value() / 10
        self.bb_std_label.setText(f"{value:.1f}")
    
    def update_obv_ratio_label(self):
        """更新OBV比率標籤"""
        value = self.obv_ratio_slider.value() / 100
        self.obv_ratio_label.setText(f"{value:.2f}")
    
    def update_obv_ma_label(self):
        """更新OBV移動平均標籤"""
        value = self.obv_ma_slider.value()
        self.obv_ma_label.setText(str(value))
    
    def get_current_parameters(self):
        """獲取當前參數設定"""
        params = {
            # RSI參數
            'rsi_period': self.rsi_period_slider.value(),
            'rsi_oversold': self.rsi_oversold_slider.value(),
            'rsi_overbought': self.rsi_overbought_slider.value(),
            
            # 布林通道參數
            'bb_period': self.bb_period_slider.value(),
            'bb_std_multiplier': self.bb_std_slider.value() / 10,
            'bb_squeeze_threshold': self.bb_squeeze_spinbox.value(),
            
            # OBV參數
            'obv_ratio_threshold': self.obv_ratio_slider.value() / 100,
            'obv_ma_period': self.obv_ma_slider.value(),
            
            # 風險管理參數
            'stop_loss_pct': self.stop_loss_spinbox.value() / 100,
            'take_profit_pct': self.take_profit_spinbox.value() / 100,
            'trailing_stop_pct': self.trailing_stop_spinbox.value() / 100,
            'max_hold_periods': self.max_hold_spinbox.value()
        }
        return params
    
    def load_data(self):
        """載入數據"""
        try:
            print(f"Loading data from {self.data_file}...")
            
            # Read the data file
            df = pd.read_csv(self.data_file, sep='\t')
            
            # Parse the data properly
            first_col = df.columns[0]
            data_rows = df[first_col].str.split(',', expand=True)
            
            # Set proper column names
            data_rows.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'TotalVolume']
            
            # Combine date and time
            data_rows['timestamp'] = pd.to_datetime(data_rows['Date'] + ' ' + data_rows['Time'])
            
            # Convert price columns to numeric
            for col in ['Open', 'High', 'Low', 'Close', 'TotalVolume']:
                data_rows[col] = pd.to_numeric(data_rows[col], errors='coerce')
            
            # Set timestamp as index
            data_rows.set_index('timestamp', inplace=True)
            
            # Sort by timestamp
            data_rows.sort_index(inplace=True)
            
            # Drop rows with NaN values
            data_rows = data_rows.dropna()
            
            print(f"Data loaded: {len(data_rows)} records")
            self.df = data_rows
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")
    
    def run_analysis(self):
        """執行分析"""
        if self.df is None:
            QMessageBox.warning(self, "Warning", "Please load data first!")
            return
        
        # 獲取當前參數
        params = self.get_current_parameters()
        
        # 顯示進度條
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # 執行分析
        try:
            # 轉換時間框架
            self.progress_bar.setValue(20)
            df_4h = self.df.resample('4H').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'TotalVolume': 'sum'
            }).dropna()
            
            # 計算指標
            self.progress_bar.setValue(40)
            df_with_indicators = self.calculate_indicators(df_4h, params)
            
            # 生成信號
            self.progress_bar.setValue(60)
            df_with_signals = self.generate_signals(df_with_indicators, params)
            
            # 分析結果
            self.progress_bar.setValue(80)
            results = self.analyze_results(df_with_signals)
            
            # 繪製圖表
            self.progress_bar.setValue(90)
            self.plot_results(df_with_signals, results)
            
            # 顯示結果
            self.progress_bar.setValue(100)
            self.display_results(results, params)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Analysis failed: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)
    
    def calculate_indicators(self, df, params):
        """計算技術指標"""
        df = df.copy()
        
        # Convert to numpy arrays for TA-Lib
        close = df['Close'].values.astype(float)
        high = df['High'].values.astype(float)
        low = df['Low'].values.astype(float)
        volume = df['TotalVolume'].values.astype(float)
        
        # RSI
        df['rsi'] = talib.RSI(close, timeperiod=params['rsi_period'])
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            close, 
            timeperiod=params['bb_period'], 
            nbdevup=params['bb_std_multiplier'], 
            nbdevdn=params['bb_std_multiplier'], 
            matype=0
        )
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        
        # OBV
        df['obv'] = talib.OBV(close, volume)
        df['obv_ratio'] = df['obv'] / df['obv'].rolling(window=params['obv_ma_period']).mean()
        
        return df
    
    def generate_signals(self, df, params):
        """生成交易信號"""
        df = df.copy()
        
        # Initialize signal columns
        df['buy_signal'] = 0
        df['sell_signal'] = 0
        df['signal_strength'] = 0
        df['signal_reason'] = ''
        
        for i in range(max(params['rsi_period'], params['bb_period']), len(df)):
            buy_conditions = []
            sell_conditions = []
            signal_strength = 0
            reasons = []
            
            # RSI conditions
            if df.iloc[i]['rsi'] <= params['rsi_oversold']:
                buy_conditions.append(True)
                signal_strength += 1
                reasons.append('RSI_Oversold')
            elif df.iloc[i]['rsi'] >= params['rsi_overbought']:
                sell_conditions.append(True)
                signal_strength += 1
                reasons.append('RSI_Overbought')
            
            # Bollinger Bands conditions
            if df.iloc[i]['Close'] <= df.iloc[i]['bb_lower']:
                buy_conditions.append(True)
                signal_strength += 1
                reasons.append('BB_Lower')
            elif df.iloc[i]['Close'] >= df.iloc[i]['bb_upper']:
                sell_conditions.append(True)
                signal_strength += 1
                reasons.append('BB_Upper')
            
            # OBV conditions
            if df.iloc[i]['obv_ratio'] >= params['obv_ratio_threshold']:
                buy_conditions.append(True)
                signal_strength += 0.5
                reasons.append('OBV_Strong')
            elif df.iloc[i]['obv_ratio'] <= 1/params['obv_ratio_threshold']:
                sell_conditions.append(True)
                signal_strength += 0.5
                reasons.append('OBV_Weak')
            
            # BB Squeeze
            if df.iloc[i]['bb_width'] <= params['bb_squeeze_threshold']:
                signal_strength += 0.5
                reasons.append('BB_Squeeze')
            
            # Generate signals
            if len(buy_conditions) >= 2 and signal_strength >= 2:
                df.iloc[i, df.columns.get_loc('buy_signal')] = 1
                df.iloc[i, df.columns.get_loc('signal_strength')] = signal_strength
                df.iloc[i, df.columns.get_loc('signal_reason')] = ', '.join(reasons)
            elif len(sell_conditions) >= 2 and signal_strength >= 2:
                df.iloc[i, df.columns.get_loc('sell_signal')] = 1
                df.iloc[i, df.columns.get_loc('signal_strength')] = signal_strength
                df.iloc[i, df.columns.get_loc('signal_reason')] = ', '.join(reasons)
        
        return df
    
    def analyze_results(self, df):
        """分析結果"""
        # Get recent data (last 30 days)
        recent_date = df.index.max() - timedelta(days=30)
        recent_df = df[df.index >= recent_date].copy()
        
        # Find signals
        buy_signals = recent_df[recent_df['buy_signal'] == 1]
        sell_signals = recent_df[recent_df['sell_signal'] == 1]
        
        results = {
            'total_buy_signals': len(buy_signals),
            'total_sell_signals': len(sell_signals),
            'total_signals': len(buy_signals) + len(sell_signals),
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'recent_df': recent_df
        }
        
        return results
    
    def plot_results(self, df, results):
        """繪製結果圖表"""
        self.figure.clear()
        
        # Get recent data for plotting
        recent_df = results['recent_df']
        
        # Create subplots
        gs = self.figure.add_gridspec(3, 1, height_ratios=[2, 1, 1])
        
        # Price and signals plot
        ax1 = self.figure.add_subplot(gs[0])
        ax1.plot(recent_df.index, recent_df['Close'], label='Close Price', alpha=0.7)
        
        # Plot buy signals
        buy_points = recent_df[recent_df['buy_signal'] == 1]
        if len(buy_points) > 0:
            ax1.scatter(buy_points.index, buy_points['Close'], 
                       color='green', s=100, marker='^', label='Buy Signals')
        
        # Plot sell signals
        sell_points = recent_df[recent_df['sell_signal'] == 1]
        if len(sell_points) > 0:
            ax1.scatter(sell_points.index, sell_points['Close'], 
                       color='red', s=100, marker='v', label='Sell Signals')
        
        # Plot Bollinger Bands
        ax1.plot(recent_df.index, recent_df['bb_upper'], '--', alpha=0.5, label='BB Upper')
        ax1.plot(recent_df.index, recent_df['bb_lower'], '--', alpha=0.5, label='BB Lower')
        
        ax1.set_title('Price and Trading Signals')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # RSI plot
        ax2 = self.figure.add_subplot(gs[1])
        ax2.plot(recent_df.index, recent_df['rsi'], label='RSI', color='purple')
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
        ax2.set_title('RSI')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # OBV plot
        ax3 = self.figure.add_subplot(gs[2])
        ax3.plot(recent_df.index, recent_df['obv_ratio'], label='OBV Ratio', color='orange')
        ax3.axhline(y=1, color='black', linestyle='-', alpha=0.5, label='Baseline')
        ax3.set_title('OBV Ratio')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def display_results(self, results, params):
        """顯示分析結果"""
        result_text = f"""
=== STRATEGY ANALYSIS RESULTS ===

Parameters Used:
- RSI Period: {params['rsi_period']}
- RSI Oversold: {params['rsi_oversold']}
- RSI Overbought: {params['rsi_overbought']}
- BB Period: {params['bb_period']}
- BB Std Multiplier: {params['bb_std_multiplier']:.1f}
- BB Squeeze Threshold: {params['bb_squeeze_threshold']:.2f}
- OBV Ratio Threshold: {params['obv_ratio_threshold']:.2f}
- OBV MA Period: {params['obv_ma_period']}
- Stop Loss: {params['stop_loss_pct']:.1%}
- Take Profit: {params['take_profit_pct']:.1%}
- Trailing Stop: {params['trailing_stop_pct']:.1%}
- Max Hold Periods: {params['max_hold_periods']}

Signal Analysis (Last 30 Days):
- Total Buy Signals: {results['total_buy_signals']}
- Total Sell Signals: {results['total_sell_signals']}
- Total Signals: {results['total_signals']}

Signal Details:
"""
        
        # Add buy signal details
        if len(results['buy_signals']) > 0:
            result_text += "\nBuy Signals:\n"
            for idx, row in results['buy_signals'].iterrows():
                result_text += f"- {idx}: Price={row['Close']:.2f}, RSI={row['rsi']:.1f}, Strength={row['signal_strength']:.1f}\n"
        
        # Add sell signal details
        if len(results['sell_signals']) > 0:
            result_text += "\nSell Signals:\n"
            for idx, row in results['sell_signals'].iterrows():
                result_text += f"- {idx}: Price={row['Close']:.2f}, RSI={row['rsi']:.1f}, Strength={row['signal_strength']:.1f}\n"
        
        self.result_text.setText(result_text)
    
    def reset_parameters(self):
        """重置參數到預設值"""
        # RSI parameters
        self.rsi_period_slider.setValue(14)
        self.rsi_oversold_slider.setValue(30)
        self.rsi_overbought_slider.setValue(70)
        
        # Bollinger Bands parameters
        self.bb_period_slider.setValue(20)
        self.bb_std_slider.setValue(20)
        self.bb_squeeze_spinbox.setValue(0.10)
        
        # OBV parameters
        self.obv_ratio_slider.setValue(120)
        self.obv_ma_slider.setValue(20)
        
        # Risk management parameters
        self.stop_loss_spinbox.setValue(1.5)
        self.take_profit_spinbox.setValue(3.0)
        self.trailing_stop_spinbox.setValue(0.8)
        self.max_hold_spinbox.setValue(20)
        
        QMessageBox.information(self, "Reset", "Parameters have been reset to default values!")

def main():
    """主函數"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show the main window
    window = StrategyParameterAdjuster()
    window.show()
    
    # Start the event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 