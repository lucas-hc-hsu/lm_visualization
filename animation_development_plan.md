# Encoder-Decoder vs Decoder-Only Transformer 動畫開發計畫

## 專案概述

使用 Manim Community Edition 製作一個教育性動畫，詳細比較 Encoder-Decoder 模型（如 T5、BART）和 Decoder-Only 模型（如 GPT）在訓練階段和推理階段的差異。

## 環境設置

```bash
pip install manim
```

## 專案結構

```
lm_visualization/
├── scenes/
│   ├── __init__.py
│   ├── intro.py              # 開場介紹
│   ├── architecture.py       # 架構比較
│   ├── training.py           # 訓練階段比較
│   ├── inference.py          # 推理階段比較
│   └── summary.py            # 總結
├── components/
│   ├── __init__.py
│   ├── transformer_block.py  # Transformer 區塊視覺化
│   ├── attention.py          # Attention 機制視覺化
│   ├── token_flow.py         # Token 流動動畫
│   └── matrix_viz.py         # 矩陣運算視覺化
├── utils/
│   ├── __init__.py
│   ├── colors.py             # 顏色定義
│   └── styles.py             # 樣式設定
├── main.py                   # 主程式入口
└── README.md
```

---

## 動畫場景規劃

### Scene 1: 開場介紹 (30秒)

**目標**: 介紹兩種架構的基本概念

**視覺元素**:
- 標題動畫: "Encoder-Decoder vs Decoder-Only Transformers"
- 兩個並排的簡化模型圖示
- 代表性模型名稱標籤 (T5, BART vs GPT, LLaMA)

**動畫流程**:
1. 標題淡入
2. 左側顯示 Encoder-Decoder 圖示
3. 右側顯示 Decoder-Only 圖示
4. 各自標註代表模型

---

### Scene 2: 架構比較 (60秒)

**目標**: 展示兩種架構的結構差異

#### 2.1 Encoder-Decoder 架構

**視覺元素**:
```
┌─────────────────┐     ┌─────────────────┐
│    Encoder      │     │    Decoder      │
│  ┌───────────┐  │     │  ┌───────────┐  │
│  │Self-Attn  │  │     │  │Masked     │  │
│  │(Bi-dir)   │  │────►│  │Self-Attn  │  │
│  └───────────┘  │     │  └───────────┘  │
│  ┌───────────┐  │     │  ┌───────────┐  │
│  │Feed       │  │     │  │Cross-Attn │  │
│  │Forward    │  │     │  └───────────┘  │
│  └───────────┘  │     │  ┌───────────┐  │
│       x N       │     │  │Feed       │  │
└─────────────────┘     │  │Forward    │  │
                        │  └───────────┘  │
                        │       x N       │
                        └─────────────────┘
```

**關鍵動畫**:
1. Encoder 區塊逐層建構
2. 雙向自注意力箭頭（可看到前後 tokens）
3. Decoder 區塊建構
4. Cross-Attention 連接線（從 Encoder 到 Decoder）
5. Causal Mask 遮罩效果

#### 2.2 Decoder-Only 架構

**視覺元素**:
```
┌─────────────────┐
│    Decoder      │
│  ┌───────────┐  │
│  │Causal     │  │
│  │Self-Attn  │  │
│  └───────────┘  │
│  ┌───────────┐  │
│  │Feed       │  │
│  │Forward    │  │
│  └───────────┘  │
│       x N       │
└─────────────────┘
```

**關鍵動畫**:
1. 單一 Decoder stack 建構
2. Causal attention mask 視覺化（下三角矩陣）
3. 強調無 Cross-Attention

---

### Scene 3: 訓練階段比較 (90秒)

**目標**: 展示 Teacher Forcing 和平行訓練機制

#### 3.1 Encoder-Decoder 訓練

**範例任務**: 翻譯 "Hello World" → "你好 世界"

**視覺流程**:
```
Input: [Hello] [World] [<EOS>]
         ↓       ↓       ↓
      ┌─────────────────────┐
      │      Encoder        │
      │  (Bidirectional)    │
      └──────────┬──────────┘
                 │
                 ▼
      Encoder Hidden States
      [h1]    [h2]    [h3]
         ╲      │      ╱
          ╲     │     ╱
           ▼    ▼    ▼
      ┌─────────────────────┐
      │      Decoder        │
      │  + Teacher Forcing  │
      └─────────────────────┘
               ↑
Target: [<BOS>] [你好] [世界]  ← (Shifted Right)
               ↓
Output: [你好]  [世界] [<EOS>]  ← (Predictions)
```

**動畫重點**:
1. 輸入序列進入 Encoder（雙向處理）
2. Encoder 產生 hidden states
3. Target 序列右移一位（<BOS> 開頭）
4. **Teacher Forcing**: 訓練時使用真實目標 token
5. **平行處理**: 所有位置同時計算 loss
6. Cross-Attention 從 Encoder states 獲取資訊

#### 3.2 Decoder-Only 訓練

**範例任務**: Next Token Prediction "The cat sat on the"

**視覺流程**:
```
Input:  [The] [cat] [sat] [on] [the]
          ↓     ↓     ↓     ↓     ↓
      ┌───────────────────────────────┐
      │         Decoder               │
      │    (Causal Self-Attention)    │
      └───────────────────────────────┘
          ↓     ↓     ↓     ↓     ↓
Target: [cat] [sat] [on] [the] [mat]

Attention Mask (Causal):
        The  cat  sat  on  the
The   [  1    0    0    0    0  ]
cat   [  1    1    0    0    0  ]
sat   [  1    1    1    0    0  ]
on    [  1    1    1    1    0  ]
the   [  1    1    1    1    1  ]
```

**動畫重點**:
1. Causal Mask 矩陣動畫（逐步顯示下三角）
2. 每個位置只能 attend 到之前的 tokens
3. **單向處理**: 從左到右的資訊流動
4. **Teacher Forcing**: 同樣使用真實 tokens
5. **平行訓練**: 一次 forward pass 計算所有位置的 loss

#### 3.3 訓練效率比較

**並排視覺化**:
```
Encoder-Decoder                 Decoder-Only
┌──────────────────┐           ┌──────────────────┐
│ Encoder Forward  │           │                  │
│ (Parallel)       │           │ Single Forward   │
├──────────────────┤           │ Pass (Parallel)  │
│ Decoder Forward  │           │                  │
│ (Parallel)       │           │                  │
└──────────────────┘           └──────────────────┘
     ↓                              ↓
 2 Stacks                      1 Stack
 Cross-Attention               Simpler, Faster
```

---

### Scene 4: 推理階段比較 (90秒)

**目標**: 展示自迴歸生成的差異

#### 4.1 Encoder-Decoder 推理

**視覺流程**:
```
Step 0: Encode Input (一次性)
[Hello] [World] → Encoder → [h1] [h2] [h3]

Step 1: Generate Token 1
[<BOS>] + Cross-Attn([h1,h2,h3]) → [你好]

Step 2: Generate Token 2
[<BOS>] [你好] + Cross-Attn([h1,h2,h3]) → [世界]

Step 3: Generate Token 3
[<BOS>] [你好] [世界] + Cross-Attn([h1,h2,h3]) → [<EOS>]
```

**動畫重點**:
1. Encoder 只執行一次（快取 hidden states）
2. Decoder 逐步生成
3. 每步都進行 Cross-Attention 查詢 Encoder 輸出
4. 自迴歸：前一步輸出成為下一步輸入

#### 4.2 Decoder-Only 推理

**視覺流程**:
```
Step 0: Initial Prompt
"The cat"

Step 1: Generate Token 1
[The] [cat] → Decoder → [sat]
   ↓     ↓
 K,V cached

Step 2: Generate Token 2 (KV Cache)
[sat] + KV_cache([The, cat]) → [on]
  ↓
 K,V appended to cache

Step 3: Generate Token 3
[on] + KV_cache([The, cat, sat]) → [the]

Step 4: Generate Token 4
[the] + KV_cache([The, cat, sat, on]) → [mat]
```

**動畫重點**:
1. **KV Cache 機制**（關鍵優化）
2. 只需對新 token 計算 attention
3. 舊 tokens 的 K, V 矩陣被快取
4. 顯示記憶體使用增長
5. 計算效率提升視覺化

#### 4.3 推理效率比較

**並排動畫**:
```
Encoder-Decoder                 Decoder-Only
┌──────────────────┐           ┌──────────────────┐
│ Encode (1x)      │ O(n)      │                  │
├──────────────────┤           │ KV Cache         │
│ Decode Step 1    │ O(m)      │ Growing          │
│ Decode Step 2    │ O(m)      │ ┌────┐           │
│ ...              │           │ │ K1 │           │
│ Decode Step m    │           │ │ V1 │           │
└──────────────────┘           │ │ K2 │           │
                               │ │ V2 │           │
Cross-Attn: O(n×m)             │ │... │           │
                               └─┴────┴───────────┘
                               Self-Attn: O(L)
```

---

### Scene 5: 總結比較 (30秒)

**比較表格動畫**:

```
┌─────────────────┬──────────────────┬──────────────────┐
│     特性        │  Encoder-Decoder │   Decoder-Only   │
├─────────────────┼──────────────────┼──────────────────┤
│ 架構複雜度      │      較複雜       │       簡單       │
├─────────────────┼──────────────────┼──────────────────┤
│ Attention       │ Bi-dir + Causal  │   Causal Only    │
│                 │ + Cross-Attn     │                  │
├─────────────────┼──────────────────┼──────────────────┤
│ 訓練            │ Teacher Forcing  │  Teacher Forcing │
│                 │ 兩階段處理        │  單階段處理      │
├─────────────────┼──────────────────┼──────────────────┤
│ 推理            │ Encode 1x +      │  KV Cache        │
│                 │ Decode 逐步       │  逐步生成        │
├─────────────────┼──────────────────┼──────────────────┤
│ 適用場景        │ Seq2Seq 任務     │  文本生成        │
│                 │ 翻譯、摘要        │  對話、程式碼    │
└─────────────────┴──────────────────┴──────────────────┘
```

---

## 核心程式碼模組

### 1. Transformer Block 元件 (`components/transformer_block.py`)

```python
from manim import *

class TransformerBlock(VGroup):
    """可重用的 Transformer 區塊視覺化元件"""

    def __init__(self, block_type="encoder", **kwargs):
        super().__init__(**kwargs)
        self.block_type = block_type
        self.create_block()

    def create_block(self):
        # 主要容器
        self.container = RoundedRectangle(
            corner_radius=0.2,
            width=3,
            height=4 if self.block_type == "decoder" else 3,
            stroke_color=WHITE
        )

        # Self-Attention 層
        self.self_attn = self.create_layer(
            "Self-Attention",
            BLUE if self.block_type == "encoder" else ORANGE
        )

        # Feed Forward 層
        self.ffn = self.create_layer("Feed Forward", GREEN)

        if self.block_type == "decoder":
            # Cross-Attention 層 (僅 decoder)
            self.cross_attn = self.create_layer("Cross-Attention", PURPLE)
            layers = VGroup(self.self_attn, self.cross_attn, self.ffn)
        else:
            layers = VGroup(self.self_attn, self.ffn)

        layers.arrange(DOWN, buff=0.3)
        self.add(self.container, layers)

    def create_layer(self, name, color):
        rect = RoundedRectangle(
            corner_radius=0.1,
            width=2.5,
            height=0.6,
            fill_color=color,
            fill_opacity=0.3,
            stroke_color=color
        )
        text = Text(name, font_size=18).move_to(rect)
        return VGroup(rect, text)
```

### 2. Attention 視覺化 (`components/attention.py`)

```python
from manim import *
import numpy as np

class AttentionMatrix(VGroup):
    """Attention 矩陣視覺化"""

    def __init__(self, tokens, causal=False, **kwargs):
        super().__init__(**kwargs)
        self.tokens = tokens
        self.causal = causal
        self.create_matrix()

    def create_matrix(self):
        n = len(self.tokens)

        # 創建網格
        self.cells = VGroup()
        for i in range(n):
            row = VGroup()
            for j in range(n):
                cell = Square(side_length=0.5)
                if self.causal and j > i:
                    cell.set_fill(RED, opacity=0.3)  # Masked
                else:
                    cell.set_fill(BLUE, opacity=0.3)  # Visible
                row.add(cell)
            row.arrange(RIGHT, buff=0)
            self.cells.add(row)

        self.cells.arrange(DOWN, buff=0)

        # Token 標籤
        self.row_labels = VGroup(*[
            Text(t, font_size=14) for t in self.tokens
        ]).arrange(DOWN, buff=0.3)
        self.row_labels.next_to(self.cells, LEFT)

        self.col_labels = VGroup(*[
            Text(t, font_size=14) for t in self.tokens
        ]).arrange(RIGHT, buff=0.3)
        self.col_labels.next_to(self.cells, UP)

        self.add(self.cells, self.row_labels, self.col_labels)

    def animate_causal_mask(self):
        """動畫展示 causal mask 的建立"""
        anims = []
        n = len(self.tokens)
        for i in range(n):
            for j in range(i + 1, n):
                cell = self.cells[i][j]
                anims.append(cell.animate.set_fill(RED, opacity=0.5))
        return AnimationGroup(*anims, lag_ratio=0.1)
```

### 3. Token 流動動畫 (`components/token_flow.py`)

```python
from manim import *

class TokenFlow(VGroup):
    """Token 流動和處理動畫"""

    def __init__(self, tokens, **kwargs):
        super().__init__(**kwargs)
        self.tokens = tokens
        self.create_tokens()

    def create_tokens(self):
        self.token_boxes = VGroup()
        for token in self.tokens:
            box = VGroup(
                RoundedRectangle(
                    corner_radius=0.1,
                    width=1.2,
                    height=0.5,
                    fill_color=BLUE,
                    fill_opacity=0.5
                ),
                Text(token, font_size=16)
            )
            self.token_boxes.add(box)

        self.token_boxes.arrange(RIGHT, buff=0.2)
        self.add(self.token_boxes)

    def animate_flow_to(self, target, shift_right=False):
        """動畫 tokens 流向目標"""
        if shift_right:
            # 右移一位 (Teacher Forcing)
            shifted = VGroup(
                self.create_special_token("<BOS>"),
                *self.token_boxes[:-1].copy()
            )
            shifted.arrange(RIGHT, buff=0.2)
            return shifted.animate.move_to(target)
        return self.token_boxes.animate.move_to(target)

    def create_special_token(self, name):
        return VGroup(
            RoundedRectangle(
                corner_radius=0.1,
                width=1.2,
                height=0.5,
                fill_color=YELLOW,
                fill_opacity=0.5
            ),
            Text(name, font_size=14)
        )
```

### 4. KV Cache 視覺化 (`components/kv_cache.py`)

```python
from manim import *

class KVCache(VGroup):
    """KV Cache 機制視覺化"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cache_entries = VGroup()
        self.create_container()

    def create_container(self):
        self.container = Rectangle(
            width=2,
            height=3,
            stroke_color=GREEN
        )
        self.title = Text("KV Cache", font_size=16)
        self.title.next_to(self.container, UP)
        self.add(self.container, self.title)

    def add_entry(self, token_name):
        """添加新的 cache 條目"""
        entry = VGroup(
            Rectangle(width=1.8, height=0.4, fill_color=GREEN, fill_opacity=0.3),
            Text(f"K,V: {token_name}", font_size=12)
        )
        self.cache_entries.add(entry)
        self.cache_entries.arrange(DOWN, buff=0.1)
        self.cache_entries.move_to(self.container)
        return Create(entry)
```

---

## 主場景實現 (`main.py`)

```python
from manim import *
from scenes.intro import IntroScene
from scenes.architecture import ArchitectureScene
from scenes.training import TrainingScene
from scenes.inference import InferenceScene
from scenes.summary import SummaryScene

class TransformerComparison(Scene):
    def construct(self):
        # Scene 1: 開場
        self.play_intro()

        # Scene 2: 架構比較
        self.play_architecture()

        # Scene 3: 訓練比較
        self.play_training()

        # Scene 4: 推理比較
        self.play_inference()

        # Scene 5: 總結
        self.play_summary()

    def play_intro(self):
        title = Text(
            "Encoder-Decoder vs Decoder-Only\nTransformers",
            font_size=48
        )
        self.play(Write(title))
        self.wait(2)
        self.play(FadeOut(title))

    # ... 其他方法實現
```

---

## 執行命令

```bash
# 渲染完整動畫 (1080p)
manim -pqh main.py TransformerComparison

# 預覽模式 (低品質快速渲染)
manim -pql main.py TransformerComparison

# 渲染特定場景
manim -pqh scenes/training.py TrainingScene
```

---

## 開發階段 (Ralph Loop Iterations)

### Iteration 1: 基礎設施
- [ ] 建立專案結構
- [ ] 實現基本 TransformerBlock 元件
- [ ] 測試基本動畫

### Iteration 2: 架構視覺化
- [ ] 完成 Encoder-Decoder 架構動畫
- [ ] 完成 Decoder-Only 架構動畫
- [ ] 添加 Attention 矩陣視覺化

### Iteration 3: 訓練階段
- [ ] 實現 Teacher Forcing 動畫
- [ ] 實現 Causal Mask 動畫
- [ ] 並排比較訓練流程

### Iteration 4: 推理階段
- [ ] 實現自迴歸生成動畫
- [ ] 實現 KV Cache 機制動畫
- [ ] 效率比較視覺化

### Iteration 5: 整合與優化
- [ ] 整合所有場景
- [ ] 添加過渡動畫
- [ ] 優化時間節奏
- [ ] 最終渲染

---

## 參考資源

- [Manim Community Documentation](https://docs.manim.community/en/stable/)
- [Manim Example Gallery](https://docs.manim.community/en/stable/examples.html)
- [Understanding Encoder And Decoder LLMs](https://magazine.sebastianraschka.com/p/understanding-encoder-and-decoder)
- [Decoder-Only Transformers](https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse)
- [Hugging Face Transformer Architectures](https://huggingface.co/learn/llm-course/chapter1/6)
