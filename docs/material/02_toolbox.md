# 第二章：我們的工具箱 (The "What with")

工欲善其事，必先利其器。在我們的量化交易旅程中，有幾樣強大的工具需要先認識一下。

### 2.1 核心程式庫

- **Python:** 我們的核心語言。因為語法簡單、開源，且擁有地表上最強大的數據科學生態系，成為量化分析的首選。
- **Pandas:** 數據分析的基石。可以把它想像成一個超強大的 Python 版 Excel。我們用它來讀取、整理、篩選、計算我們的 K 線資料。
- **Matplotlib / Seaborn:** 數據的眼睛。我們用它來繪製圖表，將枯燥的數字轉化為直觀的趨勢圖和分佈圖，幫助我們發現數據中的秘密。
- **Jupyter Notebook:** 我們的數位實驗室。一個互動式的開發環境，可以讓我們一邊寫程式、一邊看結果，非常適合進行探索性數據分析。

### 2.2 強力輔助模組

- **TA-Lib (Technical Analysis Library):**
    - **這是什麼：** 一個非常知名的技術分析指標庫。你想得到的常用指標（如均線 MA、相對強弱指標 RSI、布林通道 Bollinger Bands 等），裡面幾乎都有。
    - **為什麼要用：** 它底層由 C 語言寫成，計算速度極快。更重要的是，它提供了上百種已經被驗證過的指標算法，讓我們不必自己造輪子，可以專注在策略開發上。
    - **注意：** 在 Windows 上安裝有時較為麻煩，需要特別注意安裝步驟。

- **VectorBT:**
    - **這是什麼：** 一個現代、高效的向量化回測框架。
    - **為什麼要用：** 傳統的回測通常使用 `for` 迴圈來遍歷每一根 K 棒，當資料量大或參數組合多時，會非常緩慢。VectorBT 利用 `numpy` 和 `numba` 的威力，將整個時間序列當作一個「向量」來處理，一次性完成計算。這讓它在執行參數優化、大規模測試時，速度比傳統方法快上百甚至上千倍。

### 2.3 開發與文檔工具

- **Git (版本控制系統):**
    - **這是什麼：** 一個用來追蹤程式碼變更的系統，就像是程式碼的「時光機」。
    - **為什麼要用：** 在開發過程中，我們難免會不小心把程式改壞。有了 Git，你可以隨時回到任何一個過去儲存過的版本。它也是多人協作開發的基礎。
    - **基本指令：**
        - `git add <file>`: 將檔案的變更加入暫存區。
        - `git commit -m "你的更新說明"`: 將暫存區的內容建立一個新的「存檔點」。
        - `git status`: 檢查目前的檔案狀態。

- **Markdown (輕量級標記語言):**
    - **這是什麼：** 一種非常簡單的純文字格式化語法。你現在正在閱讀的這份文件，就是用 Markdown 寫的。
    - **為什麼要用：** 用來撰寫清晰、易讀的說明文件、報告和筆記。在 Jupyter Notebook 和 GitHub `README` 檔案中被廣泛使用。
    - **基礎語法：**
        - `# 標題一`, `## 標題二`
        - `*斜體*`, `**粗體**`
        - `[這是一個連結](https://www.google.com)`
        - `` `print("Hello")` `` 用來標示單行程式碼。 