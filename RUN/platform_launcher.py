#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Platform Launcher
多策略交易平台啟動器
整合所有可用的平台選項
"""

import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import sys
import os
import threading
from datetime import datetime

class PlatformLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("多策略交易平台啟動器")
        self.root.geometry("800x600")
        
        # Get current file base path dynamically
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        
        # Platform options with dynamic base path
        self.platforms = {
            "多策略參數平台": {
                "description": "調整多策略參數和權重的GUI平台",
                "file": os.path.join(self.base_path, "param_test/multi_strategy_parameter_platform.py"),
                "category": "參數調整"
            },
            "簡單測試工具": {
                "description": "基本功能測試和驗證",
                "file": os.path.join(self.base_path, "multi_strategy_system/simple_test.py"),
                "category": "測試工具"
            }
        }
        
        self.create_widgets()
        
    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置網格權重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # 標題
        title_label = ttk.Label(main_frame, text="🚀 多策略交易平台啟動器", 
                               font=("Arial", 18, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # 副標題
        subtitle_label = ttk.Label(main_frame, text="選擇要啟動的平台", 
                                  font=("Arial", 12))
        subtitle_label.grid(row=1, column=0, columnspan=3, pady=(0, 20))
        
        # 平台選擇框架
        platform_frame = ttk.LabelFrame(main_frame, text="可用平台", padding="10")
        platform_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        platform_frame.columnconfigure(0, weight=1)
        
        # 創建平台選項
        self.platform_vars = {}
        row = 0
        
        for platform_name, platform_info in self.platforms.items():
            # 平台選擇框
            var = tk.BooleanVar()
            self.platform_vars[platform_name] = var
            
            # 平台框架
            p_frame = ttk.Frame(platform_frame)
            p_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=5)
            p_frame.columnconfigure(1, weight=1)
            
            # 選擇框
            ttk.Checkbutton(p_frame, text="", variable=var).grid(row=0, column=0, padx=(0, 10))
            
            # 平台名稱
            name_label = ttk.Label(p_frame, text=platform_name, font=("Arial", 11, "bold"))
            name_label.grid(row=0, column=1, sticky=tk.W)
            
            # 分類標籤
            category_label = ttk.Label(p_frame, text=f"[{platform_info['category']}]", 
                                      foreground="blue", font=("Arial", 9))
            category_label.grid(row=0, column=2, padx=(10, 0))
            
            # 描述
            desc_label = ttk.Label(p_frame, text=platform_info['description'], 
                                  foreground="gray", font=("Arial", 9))
            desc_label.grid(row=1, column=1, columnspan=2, sticky=tk.W, pady=(2, 0))
            
            # 文件路徑
            path_label = ttk.Label(p_frame, text=f"文件: {platform_info['file']}", 
                                  foreground="darkgreen", font=("Arial", 8))
            path_label.grid(row=2, column=1, columnspan=2, sticky=tk.W, pady=(2, 0))
            
            row += 1
        
        # 按鈕框架
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=3, pady=20)
        
        # 按鈕
        ttk.Button(button_frame, text="🚀 啟動選中平台", 
                  command=self.launch_selected_platforms).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="📋 全選", 
                  command=self.select_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="❌ 取消全選", 
                  command=self.deselect_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="ℹ️ 平台資訊", 
                  command=self.show_platform_info).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="🔧 系統檢查", 
                  command=self.system_check).pack(side=tk.LEFT, padx=5)
        
        # 狀態列
        self.status_var = tk.StringVar(value="就緒 - 選擇要啟動的平台")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
    def launch_selected_platforms(self):
        """啟動選中的平台"""
        selected_platforms = [name for name, var in self.platform_vars.items() if var.get()]
        
        if not selected_platforms:
            messagebox.showwarning("警告", "請至少選擇一個平台")
            return
        
        self.status_var.set(f"正在啟動 {len(selected_platforms)} 個平台...")
        
        # 在新線程中啟動平台
        def launch_thread():
            try:
                for platform_name in selected_platforms:
                    platform_info = self.platforms[platform_name]
                    file_path = platform_info['file']
                    
                    if not os.path.exists(file_path):
                        self.root.after(0, lambda: messagebox.showerror("錯誤", f"找不到文件: {file_path}"))
                        continue
                    
                    # 啟動平台
                    try:
                        subprocess.Popen([sys.executable, file_path], 
                                       cwd=self.base_path)
                        self.root.after(0, lambda: self.status_var.set(f"已啟動: {platform_name}"))
                    except Exception as e:
                        self.root.after(0, lambda: messagebox.showerror("啟動失敗", f"啟動 {platform_name} 失敗: {str(e)}"))
                
                self.root.after(0, lambda: self.status_var.set("平台啟動完成"))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("錯誤", f"啟動過程發生錯誤: {str(e)}"))
        
        threading.Thread(target=launch_thread, daemon=True).start()
    
    def select_all(self):
        """全選所有平台"""
        for var in self.platform_vars.values():
            var.set(True)
        self.status_var.set("已全選所有平台")
    
    def deselect_all(self):
        """取消全選"""
        for var in self.platform_vars.values():
            var.set(False)
        self.status_var.set("已取消全選")
    
    def show_platform_info(self):
        """顯示平台資訊"""
        info_text = "📊 平台資訊\n\n"
        
        for platform_name, platform_info in self.platforms.items():
            info_text += f"🔹 {platform_name}\n"
            info_text += f"   分類: {platform_info['category']}\n"
            info_text += f"   描述: {platform_info['description']}\n"
            info_text += f"   文件: {platform_info['file']}\n\n"
        
        info_text += f"📁 當前目錄: {os.getcwd()}\n"
        info_text += f"📁 基礎路徑: {self.base_path}\n"
        info_text += f"🐍 Python版本: {sys.version}\n"
        info_text += f"⏰ 啟動時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # 創建新窗口顯示資訊
        info_window = tk.Toplevel(self.root)
        info_window.title("平台資訊")
        info_window.geometry("600x500")
        
        text_widget = tk.Text(info_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert(tk.END, info_text)
        text_widget.config(state=tk.DISABLED)
    
    def system_check(self):
        """系統檢查"""
        self.status_var.set("正在進行系統檢查...")
        
        def check_thread():
            try:
                issues = []
                
                # 檢查必要文件
                for platform_name, platform_info in self.platforms.items():
                    if not os.path.exists(platform_info['file']):
                        issues.append(f"❌ {platform_name}: 文件不存在 ({platform_info['file']})")
                
                # 檢查Python模組
                required_modules = ['pandas', 'numpy', 'matplotlib', 'tkinter', 'talib']
                for module in required_modules:
                    try:
                        __import__(module)
                    except ImportError:
                        issues.append(f"❌ 缺少模組: {module}")
                
                # 檢查目錄結構
                required_dirs = ['platforms', 'param_test', 'multi_strategy_system', 'single_strategy', 'excel', 'talib']
                for dir_name in required_dirs:
                    dir_path = os.path.join(self.base_path, dir_name)
                    if not os.path.exists(dir_path):
                        issues.append(f"❌ 缺少目錄: {dir_path}")
                
                if not issues:
                    self.root.after(0, lambda: messagebox.showinfo("系統檢查", "✅ 系統檢查通過，所有組件正常"))
                    self.root.after(0, lambda: self.status_var.set("系統檢查完成 - 一切正常"))
                else:
                    issue_text = "發現以下問題:\n\n" + "\n".join(issues)
                    self.root.after(0, lambda: messagebox.showwarning("系統檢查", issue_text))
                    self.root.after(0, lambda: self.status_var.set("系統檢查完成 - 發現問題"))
                    
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("系統檢查錯誤", f"檢查過程發生錯誤: {str(e)}"))
        
        threading.Thread(target=check_thread, daemon=True).start()

def main():
    """主函數"""
    root = tk.Tk()
    app = PlatformLauncher(root)
    root.mainloop()

if __name__ == "__main__":
    main() 