# Transformer 比較動畫 - 精簡開發指令

## Phase 0: 專案管理基礎設施

### Task 0.1: 建立專案虛擬環境
為專案建立專屬的 Python 虛擬環境，確保依賴隔離。

**指令**:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows
```

**完成條件**:
- `venv/` 目錄存在
- 執行 `which python` 顯示虛擬環境路徑
- 執行 `pip list` 顯示乾淨的環境

---

### Task 0.2: 建立變更日誌
建立 `CHANGELOG.md` 檔案來記錄專案的所有改動。

**日誌格式要求**:
每次記錄必須包含以下資訊：
1. **日期時間**: ISO 8601 格式 (YYYY-MM-DD HH:MM)
2. **變更類型**: `[新增]`, `[修改]`, `[修復]`, `[刪除]`, `[重構]`
3. **影響範圍**: 受影響的檔案或模組
4. **詳細描述**: 具體做了什麼改動
5. **原因說明**: 為什麼要做這個改動
6. **測試結果**: 改動後的驗證結果

**範例格式**:
```markdown
## [YYYY-MM-DD HH:MM] 變更類型

### 影響範圍
- `path/to/file1.py`
- `path/to/file2.py`

### 詳細描述
具體描述做了什麼改動...

### 原因說明
為什麼需要這個改動...

### 測試結果
- [x] 測試項目 1 通過
- [x] 測試項目 2 通過
```

**完成條件**:
- `CHANGELOG.md` 檔案存在
- 包含初始化記錄

---

## Phase 1: 環境與基礎元件

### Task 1.1: 專案初始化
建立 Manim 專案結構，安裝依賴。

**指令**:
```bash
# 確保在虛擬環境中
source venv/bin/activate

# 安裝系統依賴（如需要）
# sudo apt-get install libcairo2-dev pkg-config python3-dev

# 安裝 manim
pip install manim

# 建立目錄結構
mkdir -p scenes components utils
touch main.py scenes/__init__.py components/__init__.py utils/__init__.py
```

**完成條件**:
- 執行 `manim --version` 成功
- 目錄結構存在
- 在 `CHANGELOG.md` 中記錄此變更

---

### Task 1.2: 建立 TransformerBlock 元件
實現可重用的 Transformer 區塊視覺化元件，必須支援三種類型：
1. `encoder` - 雙向 Self-Attention + FFN
2. `enc_dec_decoder` - Masked Self-Attention + Cross-Attention + FFN
3. `decoder_only` - Causal Self-Attention + FFN（無 Cross-Attention）

每個區塊需包含:
- Layer Normalization 標示
- 子層之間的殘差連接視覺化

**完成條件**: 執行 `manim -pql test_block.py` 可渲染出三種不同的區塊。

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
範例：翻譯 "Hello World" → "你好 世界"

展示：
1. 輸入序列進入 Encoder（雙向處理，可看前後）
2. Target 序列右移（<BOS> 開頭）作為 Decoder 輸入
3. Teacher Forcing：訓練時 Decoder 輸入是真實 target tokens
4. 平行計算：一次 forward pass 處理所有位置
5. Cross-Attention 查詢 Encoder 輸出

**關鍵動畫**: 顯示 Decoder 同時接收兩個輸入來源（shifted targets + Encoder states）

**完成條件**: 動畫清楚展示 Teacher Forcing 和 Cross-Attention 機制。

---

### Task 3.2: Decoder-Only 訓練流程
範例：預測 "The cat sat on the mat"

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

```
Phase A: Encode (一次性)
Input → Encoder → Hidden States [h1, h2, h3]
                  ↓
        快取供 Cross-Attention 使用

Phase B: Decode (逐步自迴歸)
Step 1: [<BOS>] → Decoder → [你好]
        - Self-Attention KV Cache 開始建立
        - Cross-Attention 查詢 [h1,h2,h3]

Step 2: [<BOS>, 你好] → Decoder → [世界]
        - Self-Attention 使用 KV Cache
        - Cross-Attention 查詢 [h1,h2,h3]（相同）
```

**關鍵**: 必須展示 Decoder 也有自己的 KV Cache（用於 Self-Attention）

**完成條件**: 動畫同時展示 Encoder cache 和 Decoder KV Cache。

---

### Task 4.2: Decoder-Only 推理流程
展示兩階段推理：

```
Phase A: Prefill (並行處理 prompt)
[The] [cat] → Decoder (parallel) → 建立完整 KV Cache
              K: [k1, k2]
              V: [v1, v2]

Phase B: Generation (逐 token)
Step 1: 新 token [?]
        - 計算 Q_new
        - Attend to KV Cache [k1,k2] + [k_new]
        - 輸出 [sat]
        - 更新 Cache: K=[k1,k2,k3], V=[v1,v2,v3]

Step 2: 新 token [?]
        - 只計算新 token 的 Q
        - Attend to 完整 Cache
        - 輸出 [on]
```

**關鍵動畫**:
1. Prefill 階段的並行處理
2. KV Cache 的增長
3. 新 token 只需計算 Q，複用舊的 K,V

**完成條件**: 動畫明確區分 Prefill 和 Generation 兩階段。

---

### Task 4.3: 推理效率比較
視覺化比較：

| 操作 | Encoder-Decoder | Decoder-Only |
|------|-----------------|--------------|
| 初始化 | Encode O(n²) | Prefill O(L²) |
| 每步生成 | Self-Attn O(m) + Cross-Attn O(n) | Self-Attn O(L) |
| 記憶體 | Encoder states + Decoder KV | 單一 KV Cache |

**完成條件**: 動畫展示計算量和記憶體使用的差異。

---

## Phase 5: 整合與總結

### Task 5.1: 總結比較表
動態建立比較表格：

| 特性 | Encoder-Decoder | Decoder-Only |
|------|-----------------|--------------|
| 架構 | Encoder + Decoder | 僅 Decoder |
| Attention | 雙向 + Causal + Cross | 僅 Causal |
| 輸入輸出 | 分離的序列 | 連續的序列 |
| 訓練 | 兩階段處理 | 單階段處理 |
| 推理快取 | Encoder states + Decoder KV | 單一 KV Cache |
| 代表模型 | T5, BART, mBART | GPT, LLaMA, Mistral |

**完成條件**: 表格動畫逐行顯示。

---

### Task 5.2: 最終整合
將所有場景串接，添加轉場動畫。

**完成條件**:
- `manim -pqh main.py TransformerComparison` 成功渲染
- 總時長約 5 分鐘
- 無渲染錯誤

---

## 驗收標準

### 專案管理
- [ ] 虛擬環境 `venv/` 已建立且正常運作
- [ ] `CHANGELOG.md` 存在且包含所有變更記錄
- [ ] 每次變更都有完整的日誌記錄（日期、類型、範圍、描述、原因、測試結果）

### 技術正確性
- [ ] Decoder-Only 架構無 Cross-Attention
- [ ] Cross-Attention 的 Q/K/V 來源正確標示
- [ ] Encoder-Decoder 推理展示 Decoder KV Cache
- [ ] Decoder-Only 推理區分 Prefill 和 Generation 階段
- [ ] 架構圖包含 Layer Normalization 和 Positional Encoding

### 視覺品質
- [ ] Attention 矩陣標籤正確對齊
- [ ] 動畫節奏流暢，無跳幀
- [ ] 顏色區分清晰（Encoder/Decoder/Attention 類型）

### 可執行性
- [ ] 所有場景可獨立渲染測試
- [ ] 完整動畫可成功渲染為 1080p 影片

---

## 重要提醒

**每完成一個 Task 後，必須在 `CHANGELOG.md` 中記錄變更！**

記錄應包含：
1. 完成了什麼
2. 修改了哪些檔案
3. 遇到什麼問題及如何解決
4. 測試驗證結果
