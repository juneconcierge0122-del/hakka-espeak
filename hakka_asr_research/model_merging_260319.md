# 客語 ASR × Model Merging 研究提案分析報告

> 作者：Anna Liang / June（研究助理）  
> 日期：2026-03-21  
> 版本：v1.0

---

## 目錄

1. [研究動機與背景](#1-研究動機與背景)
2. [Model Merging 數學基礎](#2-model-merging-數學基礎)
3. [語音領域 Model Merging 最新突破（2024–2026）](#3-語音領域-model-merging-最新突破2024-2026)
4. [客語 ASR 現有研究盤點](#4-客語-asr-現有研究盤點)
5. [提案設計：Stitch/SHANKS 式客語 ASR](#5-提案設計stitchshanks-式客語-asr)
6. [數學框架與理論分析](#6-數學框架與理論分析)
7. [實驗規劃](#7-實驗規劃)
8. [風險評估與挑戰](#8-風險評估與挑戰)
9. [時程規劃](#9-時程規劃)
10. [參考文獻](#10-參考文獻)

---

## 1. 研究動機與背景

### 1.1 問題定義

客語（Hakka）為台灣法定國家語言之一，但在 ASR 資源上屬於**極低資源語言（extremely low-resource）**：

- 標註語音資料量遠低於華語/英語（估計 < 100 小時高品質標註）
- 客語方言多元（四縣、海陸、大埔、饒平、詔安等至少 5 種主要腔調）
- 漢字書寫系統不統一，存在多對多的「音→字」對應問題
- 現有基座模型（如 Whisper）對客語支援極弱或完全沒有

### 1.2 核心假設

> **若我們能利用 Model Merging 技術，將多個已在相關語言/任務上 fine-tune 的模型 "stitch" 成一個客語 ASR 模型，則可以在不需要大量客語標註資料的前提下，獲得可用的客語語音→客語漢字轉寫能力。**

這個方向的核心直覺是：
- 華語 ASR 模型已學到「漢字語言模型」知識
- 客語的音韻結構與粵語、華語有部分重疊
- 透過 model merging，我們可以把「聲學前端對客語音素的理解」和「解碼端對漢字的理解」拼接在一起

---

## 2. Model Merging 數學基礎

### 2.1 Task Arithmetic（任務算術）

**核心定義 (Ilharco et al., 2023)：**

給定預訓練模型權重 $\theta_{\text{pre}}$，在任務 $t$ 上 fine-tune 得到 $\theta_t$，定義**任務向量（task vector）**：

$$\tau_t = \theta_t - \theta_{\text{pre}}$$

多任務合併的模型為：

$$\theta_{\text{merged}} = \theta_{\text{pre}} + \lambda \sum_{t} \tau_t$$

其中 $\lambda \in [0, 1]$ 為 scaling coefficient。

**數學直覺：** 假設各任務的 fine-tuning 移動方向大致正交（低干涉），則線性組合可近似保留各任務能力。

### 2.2 TIES-Merging（Trim, Elect Sign & Merge）

**Yadav et al. (2024)** 提出處理任務向量間的**符號衝突（sign conflict）**問題：

1. **Trim（修剪）：** 對每個任務向量 $\tau_t$，僅保留變化量最大的 top-k% 參數
2. **Elect Sign（選擇符號）：** 對每個參數位置，取多數任務向量的符號方向
3. **Merge（合併）：** 對符號一致的值取平均

$$\tau_{\text{merged}}^{(i)} = \text{sign}_{\text{majority}}^{(i)} \cdot \frac{1}{|A_i|} \sum_{t \in A_i} |\tau_t^{(i)}|$$

其中 $A_i$ 是第 $i$ 個參數位置上符號與 majority sign 一致的任務集合。

### 2.3 DARE（Drop And REscale）

**Yu et al. (2024)** 提出在合併前隨機 drop 任務向量中的部分參數：

$$\tilde{\tau}_t = \frac{m \odot \tau_t}{1 - p}$$

其中 $m \sim \text{Bernoulli}(1-p)$ 為隨機遮罩，$p$ 為 drop rate，除以 $(1-p)$ 做 rescale 保持期望值不變。

**數學意義：** 稀疏化任務向量可降低不同任務間的干涉（類似 dropout 的正則化效果）。

### 2.4 SLERP（Spherical Linear Interpolation）

用於兩個模型間的球面插值：

$$\theta_{\text{merged}} = \frac{\sin((1-t)\Omega)}{\sin \Omega} \theta_A + \frac{\sin(t \cdot \Omega)}{\sin \Omega} \theta_B$$

其中 $\Omega = \arccos\left(\frac{\theta_A \cdot \theta_B}{\|\theta_A\| \|\theta_B\|}\right)$，$t \in [0, 1]$。

**優勢：** 保持權重向量的 norm，避免線性插值可能造成的 norm 衰減問題。

### 2.5 Model Stitching（模型拼接）

不同於上述方法在同一架構的權重空間做合併，**Model Stitching** 是：

- 取模型 A 的前 $l$ 層作為 encoder 前段
- 取模型 B 的後 $L - l$ 層作為 decoder 後段
- 在拼接處插入一個輕量的 **stitching layer**（通常為 1×1 卷積或線性層）來對齊表示空間

$$h_{\text{stitch}} = W_s \cdot h_A^{(l)} + b_s$$

$$y = f_B^{(l+1:L)}(h_{\text{stitch}})$$

**Stitching layer 的訓練成本極低（僅需訓練 $W_s, b_s$）。**

### 2.6 LoRS-Merging（Low-Rank and Sparse Merging）

**Zhao et al. (2025)** 針對語音的最新方法，結合低秩分解與稀疏剪枝：

1. 對每個任務向量做 SVD 分解：$\tau_t \approx U_r \Sigma_r V_r^T$（保留 rank-$r$）
2. 再做稀疏剪枝，移除冗餘參數
3. 合併時減少語言間干涉，達成 > 20% 的正規化性能提升

---

## 3. 語音領域 Model Merging 最新突破（2024–2026）

### 3.1 關鍵論文一覽

| 論文 | 時間 | 核心貢獻 | 與客語 ASR 的關聯 |
|------|------|----------|-----------------|
| **Task Arithmetic with Support Languages for Low-Resource ASR** (Rafkin et al.) | 2026.01 | 將「語言」視為 task，用 task arithmetic 合併高/低資源語言的 Whisper 模型，23 種低資源語言 WER 提升達 10% | **直接可用**——客語即為 low-resource target |
| **LoRS-Merging** (Zhao et al.) | 2025.02 | 低秩+稀疏的多語言語音合併，Whisper 上 10 語言實驗達 20%+ 提升 | **核心方法參考**——語音多語言 merging 的 SOTA |
| **Group-Aware Partial Model Merging for Children's ASR** (Rolland & Abad) | 2025.11 | 部分參數合併（partial merging），針對群體特徵差異 | **方言適配啟發**——客語方言可視為不同 group |
| **Exploring Model Merging for Multi-Domain Adaptation in ASR** (Carvalho et al.) | 2026.03 | 系統性比較多種 merging 方法在 ASR 多域適配的效果 | **方法論基準**——可參考其實驗設計 |
| **Speech-FT: Merging Pre-trained and Fine-Tuned Speech Models** (Lin et al., NTU) | 2025.02 | 合併預訓練與 fine-tune 後的語音表示模型以保持跨任務泛化 | **台灣研究組**——Hung-yi Lee 組，可合作 |
| **Robust Fine-tuning via Model Merging: Disordered Speech** (Ducorroy & Riad) | 2025.05 | 用 model merging 做 ASR robustness 提升 | 魯棒性增強思路 |
| **Selective Attention Merging for Low Resource ASR** (Shankar et al.) | 2025.01 | 針對低資源場景的注意力機制選擇性合併 | **低資源策略直接參考** |
| **Correlation-Permutation for Speech-Music Encoder Merging** (Ritter-Gutierrez et al.) | 2025.06 | 用 correlation + permutation alignment 做跨模態 encoder 合併 | 跨模態 encoder 對齊技術 |
| **Efficient Dialect-Aware Modeling for Taiwanese Hakka** (Peng et al.) | 2026.02 | **直接針對台灣客語**的方言感知建模 | **最直接相關——客語方言 ASR** |
| **Model Merging in the Era of LLMs: Survey** (Song & Zheng) | 2026.03 | 最新 survey，涵蓋所有主流方法 | 方法論總覽 |

### 3.2 重點發現

1. **Task Arithmetic 在語音領域確認有效：** Rafkin et al. (2026) 首次將 task arithmetic 直接應用於低資源 ASR，結果顯示在 Whisper 上用高資源「支持語言」的 task vector 合併可以穩定降低 WER。

2. **低秩稀疏合併優於傳統方法：** LoRS-Merging 顯示低秩分解可以有效降低語言間干涉，這對客語（與華語有重疊又有差異）的場景特別重要。

3. **部分合併（Partial Merging）是關鍵：** 不是所有層都應該合併——底層（acoustic feature）和高層（language model）可能需要不同的合併策略。

4. **台灣已有相關研究基礎：** Peng et al. (2026) 的客語方言感知建模、NTU Hung-yi Lee 組的 Speech-FT 都是可以直接銜接的研究。

---

## 4. 客語 ASR 現有研究盤點

### 4.1 台灣客語語音資源

- **客家委員會語料庫**：最主要的客語語音資源
- **VoxHakka** (2024)：客語 TTS 系統，涵蓋多種腔調
- **Peng et al. (2026)**：方言感知的客語語音處理，使用 dialect embedding 做 conditioning

### 4.2 客語 ASR 的特殊挑戰

1. **聲調系統**：客語有 6-7 個聲調（依方言），比華語更複雜
2. **方言變異**：四縣腔與海陸腔的聲母/韻母/聲調系統差異顯著
3. **漢字對應**：同一客語詞可能對應多種漢字寫法（尚無統一標準）
4. **Code-switching**：客語使用者經常混用華語，產生語碼轉換

---

## 5. 提案設計：Stitch/SHANKS 式客語 ASR

### 5.1 核心架構

我們提出 **Hakka-Stitch** 架構，結合 Model Stitching 與 Task Arithmetic：

```
┌──────────────────────────────────────────────────────┐
│                  Hakka-Stitch Pipeline                │
│                                                      │
│  ┌─────────────┐   ┌──────────┐   ┌──────────────┐  │
│  │  Acoustic    │   │ Stitch   │   │  Language     │  │
│  │  Encoder     │──▶│  Layer   │──▶│  Decoder      │  │
│  │  (merged)    │   │          │   │  (merged)     │  │
│  └─────────────┘   └──────────┘   └──────────────┘  │
│        ▲                                   ▲         │
│        │                                   │         │
│  ┌─────┴──────┐                    ┌───────┴──────┐  │
│  │ τ_mandarin │                    │ τ_mandarin   │  │
│  │ τ_cantonese│                    │ τ_hakka_lm   │  │
│  │ τ_hakka_ph │                    │ τ_cantonese  │  │
│  └────────────┘                    └──────────────┘  │
└──────────────────────────────────────────────────────┘
```

### 5.2 三階段方法

#### Stage 1: Task Vector 萃取

從 Whisper（或其他基座模型）出發，分別 fine-tune 得到多個專家模型：

- $\theta_{\text{mandarin}}$：華語 ASR fine-tune
- $\theta_{\text{cantonese}}$：粵語 ASR fine-tune（音韻相近）
- $\theta_{\text{hakka-small}}$：少量客語資料 fine-tune
- $\theta_{\text{tonal}}$：聲調語言通用 fine-tune

萃取任務向量：$\tau_t = \theta_t - \theta_{\text{pre}}$

#### Stage 2: Selective Merging（選擇性合併）

**分層策略：**

- **Encoder 前段（Layer 1-8）**：以 $\tau_{\text{hakka-small}}$ 為主，加入少量 $\tau_{\text{cantonese}}$（音素相似性）
  $$\theta_{\text{enc}}^{1:8} = \theta_{\text{pre}} + \alpha_1 \tau_{\text{hakka}}^{1:8} + \alpha_2 \tau_{\text{cantonese}}^{1:8}$$

- **Encoder 後段（Layer 9-16）**：用 LoRS-Merging 合併多語言 task vector
  $$\theta_{\text{enc}}^{9:16} = \theta_{\text{pre}} + \text{LoRS}(\tau_{\text{mandarin}}^{9:16}, \tau_{\text{cantonese}}^{9:16}, \tau_{\text{hakka}}^{9:16})$$

- **Decoder**：以 $\tau_{\text{mandarin}}$ 為主（漢字語言模型），加入 $\tau_{\text{hakka-lm}}$
  $$\theta_{\text{dec}} = \theta_{\text{pre}} + \beta_1 \tau_{\text{mandarin}}^{\text{dec}} + \beta_2 \tau_{\text{hakka-lm}}^{\text{dec}}$$

#### Stage 3: Stitching Layer 微調

在合併後的 encoder 和 decoder 之間插入 stitching layer，僅用**少量客語標註資料**（< 10 小時）微調：

$$\mathcal{L}_{\text{stitch}} = \sum_{(x, y) \in \mathcal{D}_{\text{hakka}}} -\log P(y | x; \theta_{\text{merged}}, W_s, b_s)$$

只優化 $W_s, b_s$（參數量 << 全模型），訓練成本極低。

### 5.3 方言適配擴展

利用 **Group-Aware Partial Merging** 的思路：

- 為每種客語腔調維護一個 dialect-specific 的 task vector
- 合併時根據目標腔調調整各 task vector 的權重
- 可用 dialect embedding 或 learned routing 做自動選擇

$$\theta_{\text{dialect}_d} = \theta_{\text{merged}} + \gamma_d \cdot \tau_{\text{dialect}_d}$$

---

## 6. 數學框架與理論分析

### 6.1 干涉分析（Interference Analysis）

定義兩個任務向量間的**干涉度**：

$$I(\tau_i, \tau_j) = \frac{\sum_k \mathbb{1}[\text{sign}(\tau_i^{(k)}) \neq \text{sign}(\tau_j^{(k)})] \cdot |\tau_i^{(k)}| \cdot |\tau_j^{(k)}|}{\sum_k |\tau_i^{(k)}| \cdot |\tau_j^{(k)}|}$$

**假設：** 華語 vs 客語的干涉度在 encoder 淺層較高（音素差異）、在 decoder 較低（共用漢字系統）。

→ 這支持我們「分層合併策略」的合理性。

### 6.2 最優合併係數求解

給定驗證集 $\mathcal{D}_{\text{val}}$，最優合併係數可透過以下優化問題求解：

$$\alpha^* = \arg\min_{\alpha} \text{WER}\left(\theta_{\text{pre}} + \sum_t \alpha_t \tau_t; \mathcal{D}_{\text{val}}\right)$$

由於 WER 不可微分，可使用：
- **Grid search**（如 Rafkin et al. 2026 使用的方法）
- **Bayesian optimization**
- **CMA-ES**（Covariance Matrix Adaptation Evolution Strategy）

### 6.3 LoRS 的理論保證

對任務向量做 rank-$r$ 近似，誤差界為：

$$\|\tau_t - \tau_t^{(r)}\|_F \leq \sigma_{r+1}(\tau_t)$$

其中 $\sigma_{r+1}$ 為第 $r+1$ 個奇異值。只要任務向量的有效秩（effective rank）低，truncation 損失就小。

**語音任務向量通常有低有效秩**（因為 fine-tuning 主要改變少數關鍵方向），這為 LoRS-Merging 在語音上的成功提供了理論依據。

### 6.4 Stitching Layer 的表示對齊

Stitching layer 本質上在做兩個表示空間之間的**仿射對齊**：

$$h_B = W_s h_A + b_s$$

若兩個模型的表示空間存在近似的線性對應關係（linear mode connectivity），則 stitching layer 可以用少量資料高效學習。

**Kornblith et al. (2019)** 的 CKA（Centered Kernel Alignment）可以用來事先評估不同模型在各層的表示相似度，指導 stitching point 的選擇。

---

## 7. 實驗規劃

### 7.1 基線系統

| 系統 | 描述 |
|------|------|
| B1: Whisper-large-v3 zero-shot | 直接用 Whisper 辨識客語（預期很差） |
| B2: Whisper + 客語 full fine-tune | 用所有可得客語資料 fine-tune |
| B3: Whisper + 華語 fine-tune → 客語 fine-tune | Sequential transfer |
| B4: Multi-task training（華語+客語+粵語） | 傳統多任務方式 |

### 7.2 提案系統

| 系統 | 描述 |
|------|------|
| P1: Task Arithmetic（簡單加法） | $\theta = \theta_{\text{pre}} + \lambda(\tau_{\text{mandarin}} + \tau_{\text{hakka}})$ |
| P2: TIES-Merging | 帶符號選擇的 task vector 合併 |
| P3: DARE + Task Arithmetic | 稀疏化後合併 |
| P4: LoRS-Merging | 低秩稀疏合併 |
| P5: Hakka-Stitch（分層合併+stitching） | 本提案的完整方法 |
| P6: Hakka-Stitch + dialect adaptation | P5 + 方言適配 |

### 7.3 評估指標

- **CER（Character Error Rate）**：客語漢字的字元錯誤率（主指標）
- **WER（Word Error Rate）**：詞級錯誤率
- **Tone Accuracy**：聲調辨識準確率（客語特有評估）
- **Cross-dialect transfer**：不同腔調間的遷移效果
- **Inference speed**：推論速度（FLOPs, latency）

### 7.4 資料需求

| 資料 | 用途 | 預估量 |
|------|------|--------|
| 華語 ASR 資料 | Fine-tune 華語模型 | > 1000 hr（公開可得）|
| 粵語 ASR 資料 | Fine-tune 粵語模型 | > 100 hr（Common Voice 等）|
| 客語標註語音 | Fine-tune 客語模型 + 驗證 | 10-50 hr（客委會語料庫）|
| 客語文本 | 訓練客語 LM | 文本量待調查 |

---

## 8. 風險評估與挑戰

### 8.1 高風險項

| 風險 | 嚴重度 | 緩解策略 |
|------|--------|----------|
| 客語與華語/粵語的表示空間差異過大，merging 效果差 | 高 | 先用 CKA 分析表示相似度；退化為 sequential fine-tune |
| 客語漢字不統一，導致 CER 虛高 | 高 | 建立標準化對照表；使用 phoneme-level 評估作為補充 |
| 聲調辨識能力無法透過 merging 遷移 | 中 | 額外加入聲調分類 auxiliary loss |
| 方言差異造成「一個模型無法通吃」 | 中 | 使用 dialect-aware partial merging |

### 8.2 已知限制

- Model merging 是一種近似方法，理論保證有限
- Stitching layer 的容量受限，可能需要多層 stitching
- 客語語料品質參差不齊

---

## 9. 時程規劃

| 階段 | 時間 | 工作內容 |
|------|------|----------|
| Phase 0 | Month 1 | 文獻精讀、語料盤點、環境建置 |
| Phase 1 | Month 2-3 | 基線系統建立（B1-B4）、Task vector 萃取 |
| Phase 2 | Month 3-4 | 干涉度分析（CKA、task vector cosine sim）|
| Phase 3 | Month 4-6 | 各 merging 方法實驗（P1-P4）|
| Phase 4 | Month 6-8 | Hakka-Stitch 完整系統（P5-P6）|
| Phase 5 | Month 8-9 | 方言適配實驗、ablation study |
| Phase 6 | Month 9-10 | 論文撰寫、投稿 |

**目標投稿：** Interspeech 2027 / ICASSP 2027 / ACL 2027

---

## 10. 參考文獻

### Model Merging 核心理論

1. Ilharco, G., et al. "Editing Models with Task Arithmetic." ICLR 2023. — Task Arithmetic 原論文
2. Yadav, P., et al. "TIES-Merging: Resolving Interference When Merging Models." NeurIPS 2024. — TIES-Merging
3. Yu, L., et al. "Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch." ICML 2024. — DARE
4. Song, M. & Zheng, M. "Model Merging in the Era of Large Language Models: Methods, Applications, and Future Directions." arXiv:2603.xxxxx, Mar 2026. — 最新 survey

### 語音領域 Model Merging

5. **Rafkin, E., DeGenaro, D., & Yang, X.** "Task Arithmetic with Support Languages for Low-Resource ASR." arXiv:2601.07038, Jan 2026. ⭐ **最直接相關**
6. **Zhao, Q., Sun, G., & Zhang, C.** "Low-Rank and Sparse Model Merging for Multi-Lingual Speech Recognition and Translation." arXiv:2502.17380, Feb 2025. ⭐ **LoRS-Merging**
7. **Rolland, T. & Abad, A.** "Group-Aware Partial Model Merging for Children's ASR." arXiv:2511.23098, Nov 2025.
8. **Carvalho, C., et al.** "Exploring the potential and limitations of Model Merging for Multi-Domain Adaptation in ASR." Mar 2026.
9. **Lin, T.-Q., Huang, W.-P., Tang, H., & Lee, H.-Y.** "Speech-FT: Merging Pre-trained And Fine-Tuned Speech Representation Models For Cross-Task Generalization." IEEE TASLP, vol. 34, 2026. ⭐ **台大 Hung-yi Lee 組**
10. **Ducorroy, A. & Riad, R.** "Robust fine-tuning of speech recognition models via model merging: application to disordered speech." May 2025.
11. **Shankar, N. B., et al.** "Selective Attention Merging for low resource tasks: A case study of Child ASR." Jan 2025.
12. **Ritter-Gutierrez, F., et al.** "A correlation-permutation approach for speech-music encoders model merging." Jun 2025.

### 客語相關

13. **Peng, A.-C., et al.** "Efficient Dialect-Aware Modeling and Conditioning for Low-Resource Taiwanese Hakka Speech Processing." Feb 2026. ⭐ **直接客語研究**
14. "VoxHakka: A Dialectally Diverse Multi-speaker Text-to-Speech System for Taiwanese Hakka." Sep 2024.

### 其他

15. **Ramesh, P., et al.** "Resolving Interference: Disentangling Models for Improved Model Merging." Mar 2026. — 降低合併干涉的最新方法
16. Kornblith, S., et al. "Similarity of Neural Network Representations Revisited." ICML 2019. — CKA

---

## 附錄 A：數學符號表

| 符號 | 定義 |
|------|------|
| $\theta_{\text{pre}}$ | 預訓練模型權重 |
| $\theta_t$ | 在任務 $t$ 上 fine-tune 後的權重 |
| $\tau_t$ | 任務向量 $= \theta_t - \theta_{\text{pre}}$ |
| $\lambda, \alpha, \beta, \gamma$ | 合併係數 |
| $W_s, b_s$ | Stitching layer 的權重與偏差 |
| CER | Character Error Rate |
| WER | Word Error Rate |
| CKA | Centered Kernel Alignment |

---

## 附錄 B：提案總結（一頁摘要）

**研究題目：** Hakka-Stitch: Model Merging for Low-Resource Taiwanese Hakka Automatic Speech Recognition

**核心問題：** 能否透過合併已有的華語/粵語/少量客語 ASR 模型，在不需大量客語標註資料的前提下建構可用的客語 ASR 系統？

**方法：** 
1. 從 Whisper 基座模型出發，fine-tune 多語言專家模型
2. 萃取 task vectors，用分層選擇性合併策略整合
3. 透過 stitching layer 做最終表示空間對齊

**預期貢獻：**
- 首個將 model merging/stitching 應用於客語 ASR 的研究
- 提出分層合併策略（encoder 前段 vs 後段 vs decoder）
- 方言感知的 partial merging 機制
- 為其他台灣本土語言（閩南語、原住民族語）提供可複製的方法論

**關鍵假設待驗證：**
- 華語/粵語的 task vector 對客語 ASR 有正遷移效果
- 分層合併優於全模型統一合併
- Stitching layer 可用少量資料有效對齊表示空間

---

*報告結束。歡迎 Anna 提出修改建議，我們可以進一步深化任何章節。*
