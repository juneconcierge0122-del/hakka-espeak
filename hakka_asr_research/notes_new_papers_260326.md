# 新論文筆記：STAR + Vanilla LoRA → 如何助力客語六腔 ASR Model Merging

**日期：** 2026-03-26
**整理：** June 🔬

---

## 論文一覽

### 新論文（今天找到）

| 論文 | 全名 | 出處 | 核心貢獻 |
|------|------|------|----------|
| **STAR** | Spectral Truncation And Rescale for Model Merging | NAACL 2025 (Lee et al.) | 在 spectral space 截斷小奇異值 + 自動 rescale 保持 nuclear norm，減少 merging conflict |
| **Vanilla LoRA** | Learning Rate Matters: Vanilla LoRA May Suffice for LLM Fine-tuning | arXiv 2602.04998 (Lee et al., 2026) | 調好 learning rate 後，vanilla LoRA 跟各種花式 LoRA variant 性能差距 <2%；用 Hessian 最大特徵值解釋最佳 LR 差異 |

### 既有 baseline（我們 proposal 的基礎）

| 論文 | 全名 | 核心貢獻 |
|------|------|----------|
| **LoRS-Merging** | Low-Rank + Sparse Merging (2502.17380) | 將 task vector 分解為 low-rank + sparse 兩部分分別處理，適用於 multilingual speech |
| **Speech-FT** | Stable-FT + Weight Interpolation (2502.12672, 台大 Lee 組) | 穩定微調 + 權重插值，防止 fine-tune 時偏離 pre-trained basin 太遠 |

---

## STAR 論文摘要

### 方法

1. 對每個 task vector $\tau_t = \theta_t - \theta_0$ 做 **SVD 分解**
2. **截斷**小的奇異值（保留 top-k），移除 noise / conflict 分量
3. **Rescale** 使截斷後矩陣的 nuclear norm = 原矩陣的 nuclear norm
4. 用處理過的 task vector 做標準 merging（Task Arithmetic / TIES 等）

### 關鍵特性

- **不需要原始訓練資料**（不像 Fisher-weighted 需要 data inference）
- **對 hyperparameter 魯棒**（截斷比例不太敏感）
- 合併 12 個模型時比 baseline 好 4.2%（Flan-T5）
- 模型規模越大效果越穩定

### 直覺

小奇異值 ≈ task-specific noise + 跨 task conflict 的分量。截掉它們 = 保留「大方向一致」的部分，去掉互相打架的部分。Rescale 確保保留部分的「力度」不被削弱。

---

## Vanilla LoRA 論文摘要

### 核心發現

- PiSSA、DoRA、LoRA-GA 等 LoRA variant 在**固定 hyperparameter** 下看起來比 vanilla LoRA 好很多
- 但做完 **systematic LR search** 後，所有方法的 peak performance 差距 < 1–2%
- 不同方法的最佳 LR 範圍不同 → 公平比較需要各自調 LR

### 二階分析（Hessian 解釋）

- 不同 LoRA 初始化 → loss landscape 局部曲率不同 → **最大 Hessian eigenvalue $\lambda_{\max}$ 不同**
- $\lambda_{\max}$ 大 → 需要小 LR；$\lambda_{\max}$ 小 → 可以用大 LR
- 這解釋了為什麼 PiSSA（用主成分初始化）可以用更大 LR — 它的初始化讓 landscape 更平坦

### 實用結論

> 不需要花式 LoRA variant，vanilla LoRA + 好好調 LR 就夠了。

---

## 如何連結到客語六腔 ASR（我們的 proposal）

### 連結 1：STAR → 改進 LoRS-Merging 的 low-rank 處理

**LoRS-Merging 做什麼：** 把 task vector 分解為 low-rank + sparse，分別 merge

**STAR 加什麼：** 在 LoRS 的 low-rank 部分，不是直接用所有奇異值，而是用 STAR 的 spectral truncation + rescale 先清理 conflict

**具體操作：**
```
原始 LoRS:  τ = τ_LR + τ_sparse → 分別 merge
改良版:     τ = τ_LR + τ_sparse
            τ_LR → SVD → 截斷小奇異值 → rescale (STAR)
            τ_sparse → 原 LoRS 流程（trim + elect sign）
            合併兩部分
```

**為什麼對客語有用：**
- 六腔之間的 task vector 差異大（phonological divergence），conflict 分量比一般 multilingual 更嚴重
- STAR 的 spectral truncation 可以**選擇性移除腔調間 conflict 最嚴重的維度**
- 保留的 top-k 奇異值 ≈ 六腔共享的「客語核心」特徵

**與 proposal 的 Dialect Basin Hypothesis 連結：**
- STAR 截掉的小奇異值 ≈ 我們說的 $\tau_d^{\text{lex}}$（腔調特有的 lexical-phonological 分量）
- 保留的大奇異值 ≈ $\tau_d^{\text{acoustic}} + \tau_d^{\text{lm}}$（共享的聲學 + 語法結構）
- **可驗證：** 對截斷掉的分量做分析，看是否集中在 decoder / LM head 層（lexical mapping 層）

### 連結 2：Vanilla LoRA + Hessian 分析 → 改進 Speech-FT 的 fine-tuning 策略

**Speech-FT 做什麼：** Stable-FT（穩定微調）+ weight interpolation（微調後跟原模型做插值）

**Vanilla LoRA 論文給的啟發：**

1. **LR 調整比方法選擇重要** → 六腔 fine-tune 時，每個腔調可能需要**不同的最佳 LR**
   - 資料量差異大（四縣 > 海陸 >> 其他四腔）
   - Phonological complexity 不同
   - → 統一 LR 可能讓某些腔調 underfitting，某些 overfitting

2. **Hessian $\lambda_{\max}$ 作為 LR 指引** → 可以用每個腔調 fine-tune 初期的 $\lambda_{\max}$ 來**自動設定 per-dialect LR**
   - 這跟 Speech-FT 的 Stable-FT 互補：Stable-FT 控制不要跑太遠，$\lambda_{\max}$-guided LR 控制步伐大小

3. **對 merging 的影響** → 調好 LR 的 checkpoint 在 loss landscape 上更「乾淨」
   - 過高 LR → 跑到 basin 邊緣 → 難 merge
   - 過低 LR → 沒充分學到 dialect-specific 特徵 → merge 後沒用
   - 最佳 LR → basin 中心附近 → 更容易與其他腔調的 checkpoint 做 linear interpolation

### 連結 3：STAR + Hessian → 理論上強化我們的 interference 分析

我們 proposal 的 Section 2.4 定義了 merging interference：

$$\tau_A^\top H \tau_B$$

**STAR 的 spectral truncation** 本質上是在做：

$$\tau_A' = \text{SVD-truncate}(\tau_A), \quad \tau_B' = \text{SVD-truncate}(\tau_B)$$

截斷後：
$$(\tau_A')^\top H (\tau_B') < \tau_A^\top H \tau_B$$

因為小奇異值方向通常是 conflict 最嚴重的方向（不同 task 在這些方向上有相反的更新）。

**Vanilla LoRA 的 Hessian 分析** 則告訴我們：
- $H$ 的結構（特別是 $\lambda_{\max}$）決定了 fine-tune 的行為
- 不同初始化 → 不同的 $H$ spectrum → 不同的最佳訓練策略
- → **fine-tune 策略本身影響 task vector 的 spectral 結構** → 影響 STAR 截斷的效果

**整合建議：** 用 Hessian-aware LR 做 fine-tune → 得到更 clean 的 task vector → STAR truncation 更有效 → merge 效果更好。

### 連結 4：Tool Augmentation × STAR 的協同效應

我們 proposal 的核心是 tool augmentation 減少 $\|\tau_d - \tau_{d'}\|$。

STAR 從另一個角度做同樣的事 — 在 spectral space 截掉 conflict。

**兩者可以疊加：**
1. Tool augmentation 先減少**語義層面**的 conflict（lexical knowledge 外包給字典）
2. STAR 再減少**參數層面**殘留的 conflict（spectral truncation 清理剩餘 noise）

**實驗設計啟發：**
在 Experiment 5 (Model Merging) 中，可以加一組條件：

| Condition | Tool Aug | STAR | 預期 |
|-----------|----------|------|------|
| B3 | ✗ | ✗ | Baseline merge |
| E1 | ✓ | ✗ | Tool-augmented merge |
| **E2** | **✗** | **✓** | **STAR-enhanced merge** |
| **E3** | **✓** | **✓** | **Tool + STAR（最佳）** |

---

## 對 Proposal 的具體修改建議

### 修改 1：Section 6.6 加入 STAR 作為 merging method M6

| ID | Method | Description |
|----|--------|-------------|
| M6 | **STAR + Task Arithmetic** | SVD truncation + rescale + addition |
| M7 | **STAR + TIES** | SVD truncation + TIES sign election |
| M8 | **LoRS + STAR** | LoRS 的 low-rank 部分用 STAR 處理 |

### 修改 2：Section 6.2 加入 per-dialect LR search

基於 Vanilla LoRA 論文，fine-tune 各腔調時：
- 不用統一 LR，做 per-dialect LR sweep（至少 3–5 個值）
- 記錄每個腔調的 $\lambda_{\max}$，驗證是否與最佳 LR 成反比
- 這本身就是一個小 contribution（speech domain 的驗證）

### 修改 3：Section 2 加入 spectral truncation 的理論分析

STAR 的 spectral truncation 可以用我們已有的 Hessian 框架解釋：
- SVD 截斷 = 在奇異值空間做 projection
- 如果 $H$ 的主要特徵方向跟 task vector 的大奇異值方向 align → 截斷小奇異值 ≈ 保留 $H$-important 方向
- → STAR 是一種隱式的 H-metric aware merging

### 修改 4：References 加入這兩篇

```
- Lee, Y.-A., Ko, C.-Y., Pedapati, T., Chung, I.-H., Yeh, M.-Y., & Chen, P.-Y. (2025). 
  "STAR: Spectral Truncation and Rescale for Model Merging." NAACL 2025.
  
- Lee, Y.-A., Ko, C.-Y., Chen, P.-Y., & Yeh, M.-Y. (2026). 
  "Learning Rate Matters: Vanilla LoRA May Suffice for LLM Fine-tuning." arXiv:2602.04998.
```

（注意：兩篇的第一作者都是 **Yu-Ang Lee**，跟 Pin-Yu Chen / Mi-Yen Yeh 是同一個 research group，IBM + 台灣的合作。這個 group 在 model merging + efficient fine-tuning 的交叉領域很活躍，值得追蹤。）

---

## 總結：四篇的關係圖

```
        Fine-tuning 階段                    Merging 階段
        ──────────────                    ──────────────
        
  Vanilla LoRA (LR 調參)              STAR (spectral cleanup)
        │                                     │
        ▼                                     ▼
  Speech-FT (穩定 FT)  ──→  Task Vectors  ──→  LoRS-Merging (LR+sparse)
        │                        │                    │
        │                        ▼                    │
        │               Tool Augmentation             │
        │              (字典外包 lexical)              │
        │                        │                    │
        ▼                        ▼                    ▼
   六腔 checkpoint         reduced τ_d          merged model
   (clean, well-tuned)   (less conflict)      (better performance)
```

**核心 insight：** STAR 和 Vanilla LoRA 分別從 merging 和 fine-tuning 兩端改善 pipeline，跟我們的 tool augmentation 是互補的三個維度：

1. **Training 端：** Vanilla LoRA → per-dialect optimal LR → cleaner checkpoints
2. **Knowledge 端：** Tool augmentation → 字典外包 lexical conflict → smaller task vector distance
3. **Merging 端：** STAR → spectral truncation → remove residual conflict in parameter space

---

*筆記整理：June 🔬 — 2026-03-26*
