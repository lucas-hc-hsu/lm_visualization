---
active: true
iteration: 7
max_iterations: 100
completion_promise: null
started_at: "2026-01-10T15:09:56Z"
---

# Transformer 比較動畫 - 精簡開發指令

## Phase 1: 環境與基礎元件

### Task 1.1: 專案初始化
建立 Manim 專案結構，安裝依賴。

**指令**:
- 執行 Defaulting to user installation because normal site-packages is not writeable
Collecting manim
  Downloading manim-0.19.1-py3-none-any.whl (642 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 642.3/642.3 KB 1.2 MB/s eta 0:00:00
Collecting moderngl<6.0.0,>=5.0.0
  Downloading moderngl-5.12.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (291 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 291.4/291.4 KB 9.2 MB/s eta 0:00:00
Collecting skia-pathops>=0.7.0
  Downloading skia_pathops-0.9.1-cp310-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (3.3 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.3/3.3 MB 9.9 MB/s eta 0:00:00
Requirement already satisfied: rich>=12.0.0 in /home/R10946017/.local/lib/python3.10/site-packages (from manim) (13.9.4)
Collecting screeninfo>=0.7
  Downloading screeninfo-0.8.1-py3-none-any.whl (12 kB)
Collecting cloup>=2.0.0
  Downloading cloup-3.0.8-py2.py3-none-any.whl (54 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 54.6/54.6 KB 9.8 MB/s eta 0:00:00
Requirement already satisfied: click>=8.0 in /usr/lib/python3/dist-packages (from manim) (8.0.3)
Collecting moderngl-window>=2.0.0
  Downloading moderngl_window-3.1.1-py3-none-any.whl (382 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 382.4/382.4 KB 41.1 MB/s eta 0:00:00
Collecting mapbox-earcut>=1.0.0
  Downloading mapbox_earcut-2.0.0-cp310-cp310-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl (59 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 59.6/59.6 KB 10.2 MB/s eta 0:00:00
Collecting watchdog>=2.0.0
  Downloading watchdog-6.0.0-py3-none-manylinux2014_x86_64.whl (79 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 79.1/79.1 KB 18.3 MB/s eta 0:00:00
Collecting tqdm>=4.0.0
  Downloading tqdm-4.67.1-py3-none-any.whl (78 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 78.5/78.5 KB 16.9 MB/s eta 0:00:00
Requirement already satisfied: pillow>=9.1 in /home/R10946017/.local/lib/python3.10/site-packages (from manim) (11.1.0)
Requirement already satisfied: scipy>=1.13.0 in /home/R10946017/.local/lib/python3.10/site-packages (from manim) (1.14.1)
Collecting isosurfaces>=0.1.0
  Downloading isosurfaces-0.1.2-py3-none-any.whl (11 kB)
Collecting pycairo<2.0.0,>=1.13
  Downloading pycairo-1.29.0.tar.gz (665 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 665.9/665.9 KB 43.6 MB/s eta 0:00:00
  Installing build dependencies: started
  Installing build dependencies: finished with status 'done'
  Getting requirements to build wheel: started
  Getting requirements to build wheel: finished with status 'done'
  Installing backend dependencies: started
  Installing backend dependencies: finished with status 'done'
  Preparing metadata (pyproject.toml): started
  Preparing metadata (pyproject.toml): finished with status 'error'
- 建立 ,  目錄
- 建立  入口檔案

**完成條件**: 執行  成功，目錄結構存在。

---

### Task 1.2: 建立 TransformerBlock 元件
實現可重用的 Transformer 區塊視覺化元件，必須支援三種類型：
1.  - 雙向 Self-Attention + FFN
2.  - Masked Self-Attention + Cross-Attention + FFN
3.  - Causal Self-Attention + FFN（無 Cross-Attention）

每個區塊需包含:
- Layer Normalization 標示
- 子層之間的殘差連接視覺化

**完成條件**: 執行  可渲染出三種不同的區塊。

---

### Task 1.3: 建立 AttentionMatrix 元件
實現 Attention 矩陣視覺化，支援：
- 雙向 attention（全部可見）
- Causal attention（下三角矩陣）
- 動態遮罩動畫

注意：矩陣格子與標籤的間距必須對齊。

**完成條件**: 渲染出 5x5 的 causal mask 矩陣，標籤與格子正確對齊。

---

## Phase 2: 架構比較場景

### Task 2.1: Encoder-Decoder 架構動畫
展示完整的 Encoder-Decoder 架構：
- Encoder stack（N 層，雙向 Self-Attention）
- Decoder stack（N 層，含 Cross-Attention）
- Encoder 到 Decoder 的連接（Cross-Attention 的 K,V 來源）
- 標註 Positional Encoding 輸入位置

**關鍵視覺化**:
- Cross-Attention 中 Q 來自 Decoder，K/V 來自 Encoder 的箭頭標示

**完成條件**: 渲染完整架構圖，Cross-Attention 連接清晰可見。

---

### Task 2.2: Decoder-Only 架構動畫
展示 Decoder-Only 架構：
- 單一 stack（無 Cross-Attention）
- Causal Self-Attention 遮罩視覺化
- 強調輸入輸出是連續序列

**完成條件**: 渲染架構圖，明確顯示無 Cross-Attention。

---

## Phase 3: 訓練階段比較

### Task 3.1: Encoder-Decoder 訓練流程
範例：翻譯 Hello World → 你好 世界

展示：
1. 輸入序列進入 Encoder（雙向處理，可看前後）
2. Target 序列右移（BOS 開頭）作為 Decoder 輸入
3. Teacher Forcing：訓練時 Decoder 輸入是真實 target tokens
4. 平行計算：一次 forward pass 處理所有位置
5. Cross-Attention 查詢 Encoder 輸出

**關鍵動畫**: 顯示 Decoder 同時接收兩個輸入來源（shifted targets + Encoder states）

**完成條件**: 動畫清楚展示 Teacher Forcing 和 Cross-Attention 機制。

---

### Task 3.2: Decoder-Only 訓練流程
範例：預測 The cat sat on the mat

展示：
1. Causal Mask 動畫（逐步建立下三角矩陣）
2. 每個位置只能 attend 到自己和之前的 tokens
3. 平行訓練：一次 forward 產生所有位置的預測
4. Loss 在每個位置計算（預測下一個 token）

**關鍵動畫**: Causal Mask 矩陣的建立過程

**完成條件**: 動畫展示 causal attention 的限制和平行訓練特性。

---

### Task 3.3: 訓練比較總結
並排比較兩種訓練方式的異同：
- 相同：都使用 Teacher Forcing，都可平行訓練
- 不同：Encoder-Decoder 有雙向編碼 + Cross-Attention

**完成條件**: 並排動畫清楚對比兩種訓練流程。

---

## Phase 4: 推理階段比較

### Task 4.1: Encoder-Decoder 推理流程
展示完整推理過程：

Phase A: Encode (一次性)
Input → Encoder → Hidden States h1 h2 h3
        快取供 Cross-Attention 使用

Phase B: Decode (逐步自迴歸)
Step 1: BOS → Decoder → 你好
        - Self-Attention KV Cache 開始建立
        - Cross-Attention 查詢 h1,h2,h3

Step 2: BOS 你好 → Decoder → 世界
        - Self-Attention 使用 KV Cache
        - Cross-Attention 查詢 h1,h2,h3（相同）

**關鍵**: 必須展示 Decoder 也有自己的 KV Cache（用於 Self-Attention）

**完成條件**: 動畫同時展示 Encoder cache 和 Decoder KV Cache。

---

### Task 4.2: Decoder-Only 推理流程
展示兩階段推理：

Phase A: Prefill (並行處理 prompt)
The cat → Decoder (parallel) → 建立完整 KV Cache
              K: k1 k2
              V: v1 v2

Phase B: Generation (逐 token)
Step 1: 新 token
        - 計算 Q_new
        - Attend to KV Cache k1,k2 + k_new
        - 輸出 sat
        - 更新 Cache

Step 2: 新 token
        - 只計算新 token 的 Q
        - Attend to 完整 Cache
        - 輸出 on

**關鍵動畫**:
1. Prefill 階段的並行處理
2. KV Cache 的增長
3. 新 token 只需計算 Q，複用舊的 K,V

**完成條件**: 動畫明確區分 Prefill 和 Generation 兩階段。

---

### Task 4.3: 推理效率比較
視覺化比較計算量和記憶體使用的差異。

**完成條件**: 動畫展示計算量和記憶體使用的差異。

---

## Phase 5: 整合與總結

### Task 5.1: 總結比較表
動態建立比較表格展示架構、Attention、訓練、推理快取、代表模型的差異。

**完成條件**: 表格動畫逐行顯示。

---

### Task 5.2: 最終整合
將所有場景串接，添加轉場動畫。

**完成條件**:
- manim -pqh main.py TransformerComparison 成功渲染
- 總時長約 5 分鐘
- 無渲染錯誤

---

## 驗收標準

### 技術正確性
- Decoder-Only 架構無 Cross-Attention
- Cross-Attention 的 Q/K/V 來源正確標示
- Encoder-Decoder 推理展示 Decoder KV Cache
- Decoder-Only 推理區分 Prefill 和 Generation 階段
- 架構圖包含 Layer Normalization 和 Positional Encoding

### 視覺品質
- Attention 矩陣標籤正確對齊
- 動畫節奏流暢，無跳幀
- 顏色區分清晰

### 可執行性
- 所有場景可獨立渲染測試
- 完整動畫可成功渲染為 1080p 影片
