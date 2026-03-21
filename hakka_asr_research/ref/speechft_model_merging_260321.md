# Speech-FT: Merging Pre-trained And Fine-Tuned Speech Representation Models For Cross-Task Generalization

> **arXiv:** 2502.12672 | **期刊:** IEEE Transactions on Audio, Speech, and Language Processing (TASLP), vol. 34, pp. 70–83, 2026  
> **作者:** Tzu-Quan Lin 等  
> **程式碼:** https://github.com/nervjack2/Speech-FT  
> **分析日期:** 2026-03-21

---

## 一句話核心

提出 Speech-FT 兩階段框架，以「穩定微調 + 預訓練/微調模型權重內插」，讓語音表示模型在針對特定任務微調的同時保留跨任務泛化能力，並在 SUPERB benchmark 上全面超越 LoRA、正規化等基準方法。

---

## 1. 研究缺口（Gap）與動機（Motivation）

### Gap
- 語音基礎模型（HuBERT、wav2vec 2.0、WavLM 等）透過大規模自監督預訓練習得通用語音表示，但針對特定任務微調後，往往**嚴重犧牲跨任務泛化能力**（catastrophic forgetting / representational drift）。
- 現有對策——**權重空間正規化**（weight-space regularization）——在 weight space 限制偏移，但無法保證 feature space 的相似性，導致預訓練習得的通用表示依然流失。
- 同樣地，LoRA、DoRA 等參數高效微調方法雖限制參數更新量，但對特徵相似性的維護同樣不足。

### Motivation
> 「即使權重更新量大，只要微調方式正確，模型仍可保有與預訓練模型高度的特徵相似性。」（Raghavan et al., 2024）

本文以此洞察為起點：**與其限制 weight space 的偏移，不如直接確保 feature space 的相似性**，從而提出結合「穩定微調」與「模型合併（model merging）」的 Speech-FT 框架。

---

## 2. Research Questions

1. 如何在微調語音表示模型以提升特定任務表現的同時，**保留跨任務泛化能力**？
2. Weight-space 正規化為何不足以保持 feature similarity？Speech-FT 的內插策略是否優於正規化？
3. Speech-FT 是否能泛化至：
   - 不同的語音基礎模型（wav2vec 2.0、DeCoAR 2.0、WavLM Base+）？
   - 無監督微調（跨語言繼續預訓練）？
   - 多任務微調場景？

---

## 3. 方法（Method）— 模組化拆解

### 整體架構（3 步驟流程）

```
[預訓練模型 θ₀]
       │
       ▼
  [步驟 1] 穩定微調（Stable Fine-Tuning）
       │   → 先只訓練任務預測頭 D（前 β% 步）
       │   → 凍結下採樣模組（CNN feature extractor）
       │   → 輸出微調後模型 θ'（丟棄 D）
       ▼
  [步驟 2] 權重空間內插（Weight-Space Interpolation）
       │   → θ̂ = (1 − α)·θ₀ + α·θ'
       ▼
  [步驟 3] SUPERB 評估（重新訓練下游模型，凍結 θ̂）
```

### 模組一：穩定微調（Stable-FT）

**目的：** 最小化微調過程中的表示漂移（representational drift）

**設計：**
1. **Warm-up 階段**（前 β% = 10% 步）：僅更新任務預測頭 D，讓 D 先收斂，避免隨機初始化的 D 拉扯表示模型偏移。
2. **凍結下採樣模組**：CNN feature extractor 捕捉低階頻率特徵，凍結以保留通用低階表示。
3. **其後**：全模型（除下採樣模組外）一起訓練。

### 模組二：權重空間內插（Weight-Space Interpolation）

**公式：**

$$\hat{\theta} = (1 - \alpha) \cdot \theta_0 + \alpha \cdot \theta'$$

- $\theta_0$：預訓練模型權重
- $\theta'$：穩定微調後的模型權重
- $\alpha$：內插比例（固定為 0.25，傾向保留預訓練端）

**直覺：** 就像把微調後的「老師」和預訓練的「百科全書」按比例混合，讓模型既有任務知識，又保住通用素養。這是 model merging 技術（如 WiSE-FT, TIES-Merging）在語音領域的應用。

### 模組三：多任務擴展策略

| 策略 | 說明 |
|------|------|
| 多任務微調（MTF） | 同時以多任務訓練一個模型 θ'，再內插 |
| 線性合併（Linear Merge） | 各任務分別微調，取平均後再內插 |
| TIES Merging | 修剪冗餘更新、解決符號衝突後再合併 |
| 序列微調（Sequential） | 逐任務套用 Speech-FT，上一輪輸出作下一輪基礎 |

---

## 4. 實驗（Experiments）

### 主要設定
- **主要模型：** HuBERT Base（94M 參數）
- **評估框架：** SUPERB Benchmark（PR / SID / ER / SF 四任務），以 SUPERB Score（SUPERB_S）為統一指標
- **GPU：** 單張 NVIDIA RTX 3090

### 4-I. 有監督微調（8 個任務）

**Table I 關鍵結果（HuBERT，SUPERB_S）：**

| 微調任務 | 預訓練 | Stable-FT | Speech-FT |
|---------|--------|-----------|-----------|
| ASR (TED-LIUM) | 870.20 | 870.71 | **905.79** |
| SID (VoxCeleb1) | 870.20 | 651.42 | **881.07** |
| PC (TIMIT) | 870.20 | 726.64 | **877.66** |
| ER (IEMOCAP) | 870.20 | 908.86 | **1010.29** |

**關鍵觀察：**
- Stable-FT 在強任務特化任務（SID、PC）下大幅破壞跨任務泛化（SID 微調後 PER 從 5.17% 暴增到 29.08%）
- Speech-FT 在所有 8 個微調任務下均優於 Pre-trained，做到「微調後不退步」

### 4-II. 無監督微調（跨語言繼續預訓練）

| 語言 | 方法 | English SUPERB_S | 目標語言 ASR |
|------|------|-----------------|-------------|
| 中文 | Stable-FT | 789.88 | **23.47% CER** |
| 中文 | Speech-FT | **866.51** | 24.23% CER |
| 中文 | Pre-trained | 870.20 | 24.94% CER |

> Speech-FT 在中文 ASR 幾乎不損失表現的情況下，大幅保住了英文泛化能力（Stable-FT 下跌 80 點）。

### 4-III. 多任務微調（PR + SID）

序列微調在 Stable-FT 下最差（648.96），但套上 Speech-FT 後反而最強（887.50）。原因：反覆內插步驟逐步整合各任務知識，同時每次還原通用表示。

### 4-IV. Ablation Study

去掉 Stable-FT 後純做內插（Table VII）：
- SID 微調時，PR 的 PER 上升 3.27%，SUPERB_S 低於預訓練基準
- 說明 Stable-FT 是讓內插有效的前提條件

### 4-V. 與其他基準的比較（Table VIII）

| 方法 | SID fine-tuning SUPERB_S |
|------|--------------------------|
| **Speech-FT** | **881.07** |
| Weight-Space Reg. | 838.56 |
| LoRA | 785.93 |
| DoRA | 790.12 |
| Early Checkpoint (20%) | 736.06 |
| Pre-trained | 870.20 |

Speech-FT 全面領先，且是唯一能在微調後穩定超越預訓練基準的方法。

---

## 5. Limitation 與 Future Work

### Limitations（論文中隱含，部分為推論）
1. **α 超參數需要調整**：作者固定 α=0.25，雖稱泛化性良好，但實際場景中最佳 α 可能因任務與資料規模而異，缺乏自動化選擇機制。
2. **記憶體開銷：** 需同時保存預訓練模型 θ₀ 與微調模型 θ' 以進行內插，推論時無需，但訓練後合併階段有一定成本。
3. **跨語言場景中仍有目標語言表現小幅損失：** 中文場景的 CER 略差於 Stable-FT（24.23% vs 23.47%），說明泛化保留與目標語言適應存在 trade-off。
4. **僅驗證 base-size 模型：** 未在 large-size 模型（如 HuBERT Large）上驗證，擴展性未知。

### Future Work（論文明示）
- 擴展至更多語言的跨語言場景
- 探索更多任務組合下的多任務效益
- 將 Speech-FT 應用至生成式模型（如語音合成、語音對話）
- 研究內插後模型的 layer-wise feature 分佈特性

---

## 6. 公式與圖表解讀

### 📐 公式 (1)：權重空間內插

$$\hat{\theta} = (1 - \alpha) \cdot \theta_0 + \alpha \cdot \theta'$$

**【直覺】**  
想像你有兩個版本的自己：一個是剛考完 ASR 期末考、但忘了其他科目的「考後你」（θ'），另一個是啥都學過、但考試沒準備的「原版你」（θ₀）。內插就是把這兩個版本混在一起——α=0.25 意思是 25% 靠「考後你」，75% 靠「原版你」，調出一個「既懂 ASR 又不忘其他科目」的平衡體。

**【拆符號】**
- $\theta_0$：預訓練模型的完整參數集（幾十億個浮點數）
- $\theta'$：穩定微調後的模型參數
- $\alpha = 0.25$：內插係數，控制「微調端」的比重
- $\hat{\theta}$：最終合併模型，用於 SUPERB 評估

---

### 📐 公式 (4)(5)：SUPERB Score

$$\Phi_{t,j}(f) = \frac{\phi_{t,j}(f) - \phi_{t,j}(\text{Baseline})}{\phi_{t,j}(\text{SOTA}) - \phi_{t,j}(\text{Baseline})}$$

$$\text{SUPERB}_S(f) = \frac{1000}{|\mathcal{T}|} \sum_{t \in \mathcal{T}} \frac{1}{|\mathcal{M}_t|} \sum_{j \in \mathcal{M}_t} \Phi_{t,j}(f)$$

**【直覺】**  
這就是把各任務的表現「換算成百分比進度條」：0% = FBank baseline，100% = SOTA。再把所有任務的進度條平均，乘以 1000 拿來讀。900 分代表你在每個任務上平均達到 SOTA 的 90%。

**【重要性】**  
這個設計讓「不同量綱的指標」（PER%、ACC%、F1、CER%）可以統一比較，且對「原本就很難的任務」更有辨識力。

---

### 📊 Table I 解讀

- **背景色格子**：微調任務 = 評估任務，是「作弊優勢區」——Stable-FT 在這裡強，但其他格子都爛
- **Speech-FT 的亮點**：每一行、每一列都不輸預訓練，且多數格子都在改善
- **最驚人的數字**：SID on Voxceleb1 微調後，Stable-FT 的 PER 從 5.17% → 29.08%（爛掉了），而 Speech-FT 只到 6.91%，幾乎沒什麼損失

### 📊 Table II 解讀（跨語言場景）

中文繼續預訓練後：
- Stable-FT SUPERB_S: 789.88（-80 分）→ 英文能力嚴重遺忘
- Speech-FT SUPERB_S: 866.51（-4 分）→ 幾乎保住英文能力，同時中文 CER 接近 Stable-FT

這說明 Speech-FT 的內插機制有類似「忘記防護」的效果。

---

## 7. 專業點評

### 主要貢獻
1. **框架設計簡潔有效**：兩個組件（Stable-FT + 內插），無需修改訓練流程，直接在後處理階段執行內插。
2. **首次系統性驗證 model merging 在語音表示學習中的效果**，並提出明確的 feature similarity 視角來解釋跨任務泛化。
3. **多場景驗證完整**：有監督、無監督、多任務、跨語言，涵蓋面廣。
4. 發表於 TASLP（語音領域頂級期刊），品質有保證。

### 技術優點
1. **feature similarity 優於 weight-space 正規化的論點有說服力**：即使 weight deviation 更大，Speech-FT 仍維持更高的 feature similarity（Section V-D 分析），指向「invariant subspace」的存在。
2. **α=0.25 跨任務泛化穩健**：超參數不需針對每個任務調，降低使用門檻。
3. **序列微調 + Speech-FT 的協同效果令人驚喜**：原本最差的策略，配上 Speech-FT 後反而最強。

### 潛在不足
1. **α 的選擇未有理論支撐**：α=0.25 是實驗得出，為何這個值在大多數任務下都有效？論文沒有給出理論解釋，讓人對邊界條件感到不安。
2. **feature similarity 的測量指標未統一**：Section V-D 用 CKA 或餘弦相似度衡量 feature similarity，但這些指標是否完整反映「跨任務泛化能力」仍有討論空間。
3. **計算成本分析不足**：需保存兩個模型進行內插，對大模型（large-size）的記憶體需求未討論。
4. **無討論微調資料量的影響**：用更多/更少資料微調時，α 是否需要調整？
5. **跨語言場景有 trade-off**：Speech-FT 在目標語言 ASR 上略差於 Stable-FT，論文承認但未提出解決方案。

### 與相關工作的定位

```
語音基礎模型微調
├── 正規化方法（EWC, L2-SP）         ← Speech-FT 明確優於此
├── 參數高效微調（LoRA, DoRA）       ← Speech-FT 明確優於此
├── Model Merging（WiSE-FT, TIES）   ← Speech-FT 的「語音版」，引入 Stable-FT 增強效果
└── 多語言/跨語言預訓練              ← 本文新貢獻：無監督場景的 Speech-FT
```

Speech-FT 可視為 WiSE-FT（Wortsman et al., 2022，在 CV 領域的同類工作）在語音領域的延伸，關鍵創新在於加入 Stable-FT 來確保內插有意義的出發點。

### 對客語 ASR 研究的啟發

1. **直接可用**：客語資料稀少，先以 HuBERT/WavLM 預訓練模型為基礎，用 Speech-FT 在有限客語 ASR 資料上微調，可在提升客語 ASR 表現的同時保留對其他語言的泛化能力，適合多語言 ASR 系統。
2. **跨語言場景最相關**：Table II 的中文繼續預訓練設定與客語情境高度類似——都是在英語/普通話預訓練模型上適應低資源語言。Speech-FT 的跨語言保留能力值得在客語上驗證。
3. **多任務潛力**：可嘗試同時微調客語 ASR + 客語音素辨識，利用 Speech-FT 的多任務擴展策略（Sequential 或 Linear Merge）累積任務知識。
4. **值得驗證的問題**：客語語料更少（可能幾十小時），α=0.25 是否仍是最佳值？在極低資源下，Stable-FT 的 warm-up 策略是否仍有效？

---

*分析者：June 🔬 | 2026-03-21*
