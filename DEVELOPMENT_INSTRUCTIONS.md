# Transformer 比較動畫 - 精簡開發指令

## Phase 0: 專案管理基礎設施

### Task 0.1: 建立專案虛擬環境
為專案建立專屬的 Python 虛擬環境，確保依賴隔離。

**指令**:
```bash
uv venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows
```

**完成條件**:
- `.venv/` 目錄存在
- 執行 `which python` 顯示虛擬環境路徑
- 虛擬環境可正常啟動

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
source .venv/bin/activate

# 安裝系統依賴（如需要）
# sudo apt-get install libcairo2-dev pkg-config python3-dev

# 安裝 manim (使用 uv)
uv pip install manim

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

**完成條件**:
- 執行 `manim -pql test_block.py` 可渲染出三種不同的區塊
- **影片驗證**：分析渲染影片，確認三種區塊的內部結構正確：
  - encoder: 只有 Self-Attention + FFN
  - enc_dec_decoder: Masked Self-Attention + Cross-Attention + FFN
  - decoder_only: Causal Self-Attention + FFN（確認無 Cross-Attention）

---

### Task 1.3: 建立 AttentionMatrix 元件
實現 Attention 矩陣視覺化，支援：
- 雙向 attention（全部可見）
- Causal attention（下三角矩陣）
- 動態遮罩動畫

注意：矩陣格子與標籤的間距必須對齊。

**完成條件**:
- 渲染出 5x5 的 causal mask 矩陣
- **影片驗證**：逐幀檢查確認：
  - 標籤與格子正確對齊（無錯位）
  - Causal mask 為下三角矩陣（對角線及以下為可見，對角線以上為遮罩）
  - 雙向 attention 矩陣全部格子都可見

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

**完成條件**:
- 渲染完整架構圖
- **影片驗證**：嚴格檢查以下技術準確性：
  - Cross-Attention 連接清晰可見，箭頭方向正確（K/V 從 Encoder 流向 Decoder）
  - Encoder 標示為雙向 Self-Attention
  - Decoder 包含 Masked Self-Attention + Cross-Attention + FFN
  - Positional Encoding 標示在正確位置（輸入 embedding 之後）
  - Layer Normalization 位置正確

---

### Task 2.2: Decoder-Only 架構動畫
展示 Decoder-Only 架構：
- 單一 stack（無 Cross-Attention）
- Causal Self-Attention 遮罩視覺化
- 強調輸入輸出是連續序列

**完成條件**:
- 渲染架構圖
- **影片驗證**：嚴格檢查以下技術準確性：
  - 確認影片中 **完全沒有** Cross-Attention 相關元素（無標籤、無箭頭、無連接）
  - 只有 Causal Self-Attention + FFN
  - Causal mask 視覺化正確（下三角矩陣）
  - 輸入輸出呈現為連續序列（非分離的兩個序列）

---

## Phase 3: 訓練階段比較

### Task 3.1: Encoder-Decoder 訓練流程
範例：翻譯 'Hello World' → '你好 世界'

展示：
1. 輸入序列進入 Encoder（雙向處理，可看前後）
2. Target 序列右移（<BOS> 開頭）作為 Decoder 輸入
3. Teacher Forcing：訓練時 Decoder 輸入是真實 target tokens
4. 平行計算：一次 forward pass 處理所有位置
5. Cross-Attention 查詢 Encoder 輸出

**關鍵動畫**: 顯示 Decoder 同時接收兩個輸入來源（shifted targets + Encoder states）

**完成條件**:
- 動畫成功渲染
- **影片驗證**：嚴格檢查以下技術準確性：
  - Teacher Forcing 正確呈現：Decoder 輸入是 **真實的 target tokens**（非模型預測）
  - Target 序列正確右移（以 `<BOS>` 開頭）
  - Cross-Attention 的 Q 來自 Decoder 當前層，K/V 來自 Encoder 輸出
  - 所有輸出位置同時計算（平行處理，非逐步）
  - Encoder 處理是雙向的（每個 token 可以看到所有其他 tokens）

---

### Task 3.2: Decoder-Only 訓練流程
範例：預測 'The cat sat on the mat'

展示：
1. Causal Mask 動畫（逐步建立下三角矩陣）
2. 每個位置只能 attend 到自己和之前的 tokens
3. 平行訓練：一次 forward 產生所有位置的預測
4. Loss 在每個位置計算（預測下一個 token）

**關鍵動畫**: Causal Mask 矩陣的建立過程

**完成條件**:
- 動畫成功渲染
- **影片驗證**：嚴格檢查以下技術準確性：
  - Causal Mask 為下三角矩陣（位置 i 只能看到位置 0 到 i）
  - 每個位置的預測目標是 **下一個 token**（非當前 token）
  - 訓練是平行的（所有位置同時計算，一次 forward pass）
  - Loss 在每個位置計算
  - **確認沒有** Cross-Attention（純 Decoder-Only）

---

### Task 3.3: 訓練比較總結
並排比較兩種訓練方式的異同：
- 相同：都使用 Teacher Forcing，都可平行訓練
- 不同：Encoder-Decoder 有雙向編碼 + Cross-Attention

**完成條件**:
- 並排動畫成功渲染
- **影片驗證**：嚴格檢查以下技術準確性：
  - 相同點正確呈現：兩者都使用 Teacher Forcing、都可平行訓練
  - 不同點正確呈現：
    - Encoder-Decoder 有雙向編碼
    - Encoder-Decoder 有 Cross-Attention
    - Decoder-Only 只有 Causal attention
  - 視覺上清楚區分兩種架構

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

**完成條件**:
- 動畫成功渲染
- **影片驗證**：嚴格檢查以下技術準確性：
  - Encoder 只執行一次，輸出被快取
  - Cross-Attention 的 K/V 來自 Encoder cache（每步都相同）
  - Decoder 有自己的 KV Cache（用於 Self-Attention）
  - Decoder KV Cache 隨著生成步驟增長
  - 生成是自迴歸的（逐 token，非平行）
  - 每步只計算新 token 的 Q，複用之前的 K/V

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

**完成條件**:
- 動畫成功渲染
- **影片驗證**：嚴格檢查以下技術準確性：
  - Prefill 階段：prompt 的所有 tokens **平行處理**
  - Prefill 完成後建立完整的 KV Cache
  - Generation 階段：**逐 token** 生成（非平行）
  - 每個新 token 只計算 Q_new
  - 新 token 的 attention 使用完整的 KV Cache
  - KV Cache 在每步增加一個 entry
  - **確認沒有** Cross-Attention（純 Decoder-Only）

---

### Task 4.3: 推理效率比較
視覺化比較：

| 操作 | Encoder-Decoder | Decoder-Only |
|------|-----------------|--------------|
| 初始化 | Encode O(n²) | Prefill O(L²) |
| 每步生成 | Self-Attn O(m) + Cross-Attn O(n) | Self-Attn O(L) |
| 記憶體 | Encoder states + Decoder KV | 單一 KV Cache |

**完成條件**:
- 動畫成功渲染
- **影片驗證**：嚴格檢查以下技術準確性：
  - 複雜度標示正確：
    - Enc-Dec 初始化: O(n²)（n 為輸入長度）
    - Dec-Only 初始化: O(L²)（L 為 prompt 長度）
    - Enc-Dec 每步: Self-Attn O(m) + Cross-Attn O(n)
    - Dec-Only 每步: Self-Attn O(L)
  - 記憶體比較正確：
    - Enc-Dec: 兩個 cache（Encoder states + Decoder KV）
    - Dec-Only: 單一 KV Cache
  - 視覺化清楚呈現差異

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

**完成條件**:
- 表格動畫成功渲染
- **影片驗證**：嚴格檢查以下技術準確性：
  - 每一行的對比內容都正確：
    - 架構：Enc-Dec 有兩個 stack，Dec-Only 只有一個
    - Attention：Enc-Dec 有雙向+Causal+Cross，Dec-Only 僅 Causal
    - 輸入輸出：Enc-Dec 分離序列，Dec-Only 連續序列
    - 訓練：Enc-Dec 兩階段，Dec-Only 單階段
    - 推理快取：Enc-Dec 雙 cache，Dec-Only 單 cache
    - 代表模型正確

---

### Task 5.2: 最終整合
將所有場景串接，添加轉場動畫。

**完成條件**:
- `manim -pqh scenes/summary.py FinalIntegration` 成功渲染
- 無渲染錯誤
- **影片驗證**：完整觀看整部影片，嚴格檢查：
  - 所有前述 Task 的技術準確性在整合後仍然正確
  - 轉場邏輯合理（從架構→訓練→推理→總結）
  - 沒有任何技術性錯誤或誤導性內容
  - 整體敘事準確呈現語言模型的運作機制

---

## 驗收標準

### 核心驗收方法（必須遵守）

**每個 Task 的完成條件**：透過分析影片（或者影片中的 frames）來 **非常嚴格地** 檢查渲染的影片真的呈現了語言模型的運作機制。只要有些微的不準確存在，則要在修改代碼後重新渲染影片。

驗證步驟：
1. 渲染影片後，必須逐幀或逐段分析影片內容
2. 對照語言模型的實際運作原理，檢查每個視覺化元素是否準確
3. 特別注意：
   - 資料流向是否正確（箭頭方向、連接關係）
   - 時序是否合理（哪些操作是並行、哪些是順序）
   - 標籤和說明文字是否準確描述機制
   - 動畫呈現的因果關係是否符合實際模型行為
4. 若發現任何不準確之處（即使是細微的），必須：
   - 記錄問題
   - 修改代碼
   - 重新渲染
   - 再次驗證
5. 只有在影片 100% 準確呈現語言模型機制時，該 Task 才算完成

### 專案管理
- [ ] 虛擬環境 `.venv/` 已建立且正常運作
- [ ] `CHANGELOG.md` 存在且包含所有變更記錄
- [ ] 每次變更都有完整的日誌記錄（日期、類型、範圍、描述、原因、測試結果）

### 技術正確性（必須透過影片分析嚴格驗證）
- [ ] Decoder-Only 架構無 Cross-Attention（影片中不得出現任何 Cross-Attention 相關元素）
- [ ] Cross-Attention 的 Q/K/V 來源正確標示（Q 必須來自 Decoder，K/V 必須來自 Encoder）
- [ ] Encoder-Decoder 推理展示 Decoder KV Cache（必須清楚區分 Encoder cache 和 Decoder KV cache）
- [ ] Decoder-Only 推理區分 Prefill 和 Generation 階段（時序和並行/順序處理必須準確）
- [ ] 架構圖包含 Layer Normalization 和 Positional Encoding（位置必須符合實際架構）
- [ ] Teacher Forcing 機制正確呈現（訓練時 Decoder 輸入是真實 target tokens）
- [ ] Causal Mask 的遮罩模式正確（下三角矩陣，每個位置只能看到自己和之前的 tokens）

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
