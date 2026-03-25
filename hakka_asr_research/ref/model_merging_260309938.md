# 論文筆記：Model Merging in the Era of Large Language Models

> **論文**: Song, M. & Zheng, M. (2026). "Model Merging in the Era of Large Language Models: Methods, Applications, and Future Directions."  
> **來源**: arXiv:2603.09938v1 [cs.CL], 2026-03-10  
> **機構**: Tencent, China  
> **類型**: Survey（53 頁，~100+ 引用）  
> **筆記整理**: June, 2026-03-23  
> **用途**: 客語 ASR model merging 研究的理論基礎與方法論參考

---

## 目錄

1. [論文架構：FUSE 分類法](#1-fuse-分類法)
2. [理論基礎（§2）](#2-理論基礎)
3. [Weight-Space Averaging 方法（§3）](#3-weight-space-averaging)
4. [Task Vector Arithmetic 與稀疏化方法（§4）](#4-task-vector-arithmetic)
5. [結構化與資訊引導方法（§5）](#5-結構化與資訊引導方法)
6. [應用場景（§6）](#6-應用場景)
7. [開放問題與未來方向（§7）](#7-開放問題與未來方向)
8. [對客語 ASR 研究的啟示](#8-對客語-asr-研究的啟示)
9. [關鍵參考文獻索引](#9-關鍵參考文獻索引)

---

## 1. FUSE 分類法

Survey 提出 **FUSE** 作為 model merging 的統一分類框架：

| 維度 | 內容 | 對應章節 |
|------|------|----------|
| **F**oundations | 為什麼 merging 有效：loss landscape geometry、mode connectivity、permutation symmetry | §2 |
| **U**nification Strategies | 怎麼做 merging：weight averaging → task vectors → MoE/routing → search-based | §3–§5 |
| **S**cenarios | 哪裡用 merging：multi-task、safety alignment、federated learning、domain specialization | §6 |
| **E**cosystem | 什麼支援 merging：toolkits (mergekit)、benchmarks (FusionBench)、community platforms | §7 |

---

## 2. 理論基礎

### 2.1 Loss Landscape Geometry

**核心洞見**：現代深度神經網路（尤其是 overparameterized 的）的 loss landscape 並非病態的非凸面，而是存在大面積的連通低損失區域。

- **Loss basin** 形式化定義：$B_\epsilon(\theta^*) = \{\theta : L(\theta) < L(\theta^*) + \epsilon\}$
- Basin 寬度由 Hessian 矩陣 $\nabla^2 L(\theta^*)$ 的特徵譜衡量
- **寬而平的 basin** 容許權重空間中的大幅移動而不顯著降低效能 → 直接支持 weight averaging
- Overparameterized regime 下，全域最小值的流形 $\mathcal{M}$ 成為連通的、通常是凸的子空間，其中任意線性組合仍是最優解

**Flat minima 與泛化的關係**：
- Flat minima（Hessian 特徵譜曲率低）與更好的泛化性能相關（Izmailov et al., 2018）
- 當合併的模型位於平坦區域，averaged 參數更可能留在低損失區域
- **Sharp minima** 則對 merging 不利——即使小擾動也可能推到高損失區域

**⚠️ 對客語 ASR 的意義**：Whisper 是 overparameterized 模型，理論上應有寬 basin，有利於 merging。但客語 fine-tune 可能因為資料量極小而停留在 sharp minimum 附近，需要驗證。

### 2.2 Linear Mode Connectivity（線性模式連通性）

**定義**：兩個解 $\theta_1, \theta_2$ 具有 linear mode connectivity，若沿線性路徑 $\theta(\alpha) = (1-\alpha)\theta_1 + \alpha\theta_2$，$\alpha \in [0,1]$，loss 滿足：

$$L(\theta(\alpha)) \leq \max(L(\theta_1), L(\theta_2)) + \epsilon, \quad \epsilon \geq 0$$

**Loss barrier** 定義：

$$\Delta L = \max_{\alpha \in [0,1]} L(\theta(\alpha)) - \frac{1}{2}(L(\theta_1) + L(\theta_2))$$

**🔑 關鍵定理 — Merging Error Bound（Proposition 1）**：

> 令 $\theta_1, \theta_2$ 為共享預訓練初始化 $\theta_0$ 的模型，$\theta_{\text{avg}} = \frac{1}{2}(\theta_1 + \theta_2)$。在 loss 函數 $L$-smooth 的條件下：
>
> $$L(\theta_{\text{avg}}) - \frac{1}{2}(L(\theta_1) + L(\theta_2)) \leq \frac{L}{8}\|\theta_1 - \theta_2\|^2$$

**這個 bound 的重大意義**：
- **共享初始化是關鍵**——fine-tuned 模型與 pretrained 的距離 $\|\theta_1 - \theta_2\|$ 越小，merging 的性能保證越好
- 解釋了為什麼「從同一個 checkpoint fine-tune」是 merging 成功的最重要前提

**非線性 mode connectivity**：
- 當線性路徑存在 barrier 時，可用 **Bézier curves** 參數化彎曲路徑
- Nobari et al. (2025)：很多表面上的 loss barrier 其實來自 **feature distribution mismatch** 而非真正的幾何分離 → activation renormalization 可以恢復 connectivity

### 2.3 Weight Space Symmetries（權重空間對稱性）

**Permutation Invariance（排列不變性）**：

對一個有 $L$ 個隱藏層、每層 $n$ 個 unit 的網路，等價參數化的數量為 $(n!)^L$。

具體來說：對任意排列矩陣 $P^{(l)}$，變換 $(W^{(l)}, W^{(l+1)}) \mapsto (P^{(l)}W^{(l)}, W^{(l+1)}(P^{(l)})^T)$ 保持網路函數不變。

**對 merging 的致命影響**：
- 兩個獨立訓練的網路，hidden unit 排列基本是隨機且不相關的
- 直接平均 $\frac{1}{2}(\theta_A + \theta_B)$ 會把不對齊的 feature 混在一起 → 語意上無意義 → catastrophic performance degradation
- **解決方案**：Git Re-Basin (Ainsworth et al., 2023) 用 optimal transport / Hungarian method 做 permutation alignment

**共享初始化的自然對稱性破缺**：
- 從同一個 pretrained checkpoint fine-tune → 隱藏單元的對應關係在預訓練時已建立
- Fine-tuned 模型隱式保留了預訓練時的 hidden unit correspondence
- 這解釋了為什麼 fine-tuned 模型的 mergeability 遠優於獨立訓練的模型

### 2.4 Merging 成功的前提條件（Mergeability Conditions）

| 條件 | 說明 | 重要程度 |
|------|------|----------|
| **共享預訓練初始化** | 所有候選模型必須從同一個 pretrained checkpoint 出發 | ⭐⭐⭐ 最關鍵 |
| **架構相容性** | 必須有相同的層配置、激活函數、正規化方案 | ⭐⭐⭐ 必要條件 |
| **訓練程序相似性** | 相似的 hyperparameters (lr, batch size, optimizer) → 更強的 linear mode connectivity | ⭐⭐ 重要 |
| **Task vector 干擾低** | 不同任務的 fine-tuning 修改佔據互補子空間（而非衝突方向）| ⭐⭐ 重要 |

**Transformer 架構的優勢**：
- Residual connections + layer normalization → 更一致的 feature representations across fine-tuning variants
- Residual stream 結構約束了每層修改的幅度，使 fine-tuned 模型離預訓練初始化更近

### 2.5 開放理論問題

1. **為什麼大模型有如此好的 mergeability？** 現有理論在小模型上發展，未必適用於 LLM 規模
2. **模型規模 vs mergeability 的關係**：經驗上越大的模型 merging 效果越好，但缺乏數學解釋
   - 假設 A：更多 overparameterization → loss basin 有效維度更高 → 更多「空間」容納不同任務的適應
   - 假設 B：大模型的表示更 compositional / modular → 自然支持 task-specific modifications 的疊加
3. **Merge compatibility prediction**：能否學到一個函數 $C: (\theta_1, \dots, \theta_k) \to [0,1]$ 在不實際做 merge 的情況下預測效果？
4. **跨架構 merging 的理論基礎**：目前所有方法都假設架構完全一致

---

## 3. Weight-Space Averaging

### 3.1 Uniform Averaging & Model Soups

**基礎公式**：

$$\theta_{\text{merged}} = \frac{1}{N}\sum_{i=1}^{N}\theta_i$$

**Model Soups (Wortsman et al., 2022, ICML)**：
- **Uniform soup**：所有 checkpoints 的算術平均
- **Greedy soup**：迭代式加入，只在驗證集損失改善時才加入新模型：

$$L\left(\frac{k \cdot \theta_{\text{soup}} + \theta_i}{k+1}\right) < L(\theta_{\text{soup}})$$

- 關鍵優勢：增強 distribution shift 的 robustness，且推論成本不變
- **Model Ratatouille** (Ramé et al., 2023a)：回收多元輔助模型 → 改善 OOD 泛化

### 3.2 Fisher-Weighted Averaging（重要性加權）

**Fisher Information Matrix (FIM)**：

$$F = \mathbb{E}[\nabla_\theta \log p(x|\theta) \nabla_\theta \log p(x|\theta)^\top]$$

**Fisher-weighted merging**：

$$\theta_{\text{merged}} = \left(\sum_i F_i\right)^{-1} \sum_i F_i \theta_i$$

- 高 Fisher information 的參數 = 小擾動就能顯著影響模型預測 → 對任務更重要
- 實際上用 **diagonal approximation**：$\theta_{\text{merged},j} = \frac{\sum_i F_{i,jj} \theta_{i,j}}{\sum_i F_{i,jj}}$
- 複雜度從 $O(d^2)$ 降至 $O(d)$

**局限**：
- 需要每個模型的代表性資料樣本來估算 Fisher → 外部模型可能無法取得
- Diagonal approximation 在高度 overparameterized 的網路中會退化

### 3.3 Trajectory-Based Averaging

- **SWA (Stochastic Weight Averaging)**：訓練後期 cyclic lr 收集 checkpoints 後取平均 → 找到 basin centroid → 更平的最小值 → 更好的泛化
- **EMA (Exponential Moving Average)**：$\theta_{\text{EMA}}^{(t)} = \alpha \theta_{\text{EMA}}^{(t-1)} + (1-\alpha)\theta^{(t)}$，持續更新
- **SWAG**：SWA + 低秩 covariance 近似 → uncertainty estimation

### 3.4 SLERP（Spherical Linear Interpolation）

$$\text{SLERP}(\theta_A, \theta_B; \alpha) = \frac{\sin((1-\alpha)\Omega)}{\sin(\Omega)}\theta_A + \frac{\sin(\alpha\Omega)}{\sin(\Omega)}\theta_B$$

其中 $\Omega = \arccos\left(\frac{\theta_A \cdot \theta_B}{\|\theta_A\|\|\theta_B\|}\right)$

- **核心優勢**：保持權重向量的 norm，避免線性插值的 magnitude shrinkage
- 實務上 **layer-wise SLERP** 效果優於全域 SLERP
- 當模型在 semantically distant tasks 上 fine-tune 時，SLERP 優於 linear averaging
- **限制**：只能做 pairwise merging

**⚠️ Variance preservation problem**：平均多個獨立參數向量時，結果的 variance 會按模型數量成比例下降，可能導致 learned features 坍縮到無資訊的均值。

---

## 4. Task Vector Arithmetic

### 4.1 核心定義

**Task vector**：$\tau_t = \theta_t - \theta_{\text{pre}}$（fine-tuned 權重與預訓練權重的差）

**四種基本運算**：

| 運算 | 公式 | 用途 |
|------|------|------|
| **加法** | $\theta_{\text{multi}} = \theta_{\text{pre}} + \sum_i \lambda_i \tau_i$ | 多任務融合 |
| **減法** | $\theta_{\text{forget}} = \theta_{\text{pre}} - \lambda\tau$ | 遺忘/移除能力（去毒化、去偏差）|
| **縮放** | $\theta_{\text{scaled}} = \theta_{\text{pre}} + \alpha\tau$ | 調節特定能力的強度 |
| **類比** | $\theta_{\text{analogy}} = \theta_{\text{pre}} + \tau_C + (\tau_B - \tau_A)$ | 跨任務關係遷移（類似 word analogy）|

**關鍵觀察**：
- Task vectors 具有相當的 **稀疏性** ——fine-tuning 時只有相對少數的參數發生實質改變
- 加法成功的前提：不同任務的 task vectors 佔據 **近似正交的子空間**

### 4.2 參數干擾（Parameter Interference）— 三種機制

1. **符號衝突（Sign Conflicts）**：不同任務向量在同一參數位置有相反方向的更新
   - $\exists i,k: \text{sign}(\tau_i^{(j)}) \neq \text{sign}(\tau_k^{(j)})$
   - 這是最嚴重的干擾形式
   - Attention projection matrices 和 FFN intermediate layers 的衝突率最高

2. **幅度差異（Magnitude Disparities）**：不同任務的更新幅度差距過大
   - 大幅度任務「淹沒」小幅度任務的貢獻
   - 常見於 learning rate、dataset size、training duration 不同的情況

3. **冗餘參數修改（Redundant Modifications）**：低幅度、不編碼有意義任務知識的更新
   - 累積噪聲，降低信噪比
   - 大部分參數更新實際上對任務效能貢獻極小

### 4.3 TIES-Merging（Yadav et al., 2023, NeurIPS）

三階段管線：

```
Step 1: TRIM — 對每個 τᵢ，僅保留 magnitude top-k% 的參數（其餘歸零）
Step 2: ELECT SIGN — 對每個參數位置 j，取多數任務向量的符號方向：γⱼ = sign(Σ τ̃ᵢ,ⱼ)
Step 3: MERGE — 只平均符號與 majority sign 一致的值：τ̄ⱼ = (1/|Sⱼ|) Σᵢ∈Sⱼ τ̃ᵢ,ⱼ
最終：θ_merged = θ_pre + τ̄
```

- 在 merging > 3 個 task vectors 時效果顯著優於 naive task arithmetic

### 4.4 DARE（Yu et al., 2023 → ICML 2024）

**Drop And REscale**：對 task vector 隨機 dropout 後 rescale

$$\hat{\tau}_i = \frac{1}{1-p} \cdot \tau_i \odot m_i, \quad m_{i,j} \sim \text{Bernoulli}(1-p)$$

- 理論依據：fine-tuning 主要產生 **低秩修改** → 大量冗餘 → 隨機 drop 不會損失關鍵資訊
- 隱含 ensemble 效果
- 擴展：**DELLA-Merging** (Deep et al., 2024) — 基於 magnitude 的 sampling 策略

### 4.5 Tangent Space Methods（進階）

標準 task arithmetic 假設參數空間是線性的，但實際上是高度非線性的。

**Linearized model**：$f_{\text{lin}}(x;\theta) = f(x;\theta_0) + \nabla_\theta f(x;\theta_0)^\top (\theta - \theta_0)$

- 在切線空間上做 task vector 操作 → 更好的線性疊加性質
- 對 negation 操作特別有效（標準方法常產生不穩定結果）
- 對大幅度 task vectors 或大量 fine-tuning 的情況改善最明顯

---

## 5. 結構化與資訊引導方法

### 5.1 MoE（Mixture-of-Experts）風格 Merging

**基本公式**：

$$y = \sum_{i=1}^{K} g_i(x) \cdot E_i(x)$$

其中 $g_i(x) = \frac{\exp(w_i^\top h(x))}{\sum_{j=1}^K \exp(w_j^\top h(x))}$ 為 softmax routing

**Top-k sparse routing**（效率關鍵）：

$$g_i^{\text{top-k}}(x) = \begin{cases} g_i(x) & \text{if } i \in \text{TopK}(\{g_j(x)\}_{j=1}^K) \\ 0 & \text{otherwise} \end{cases}$$

推論成本 $\sim O(k/K)$

**重要方法**：

| 方法 | Expert 類型 | 記憶體開銷 | Router 訓練 | 重點 |
|------|-------------|-----------|-------------|------|
| Branch-Train-Merge | Full model | $K \times$ model | 從頭訓練 | 分支訓練 + 專家合併 |
| PHATGOOSE | Full model | $K \times$ model | Per-expert | 後置輕量 gating |
| LoRAHub | LoRA adapter | $K \times$ LoRA | Few-shot | Gradient-free Bayesian optimization |
| MoLE | LoRA adapter | $K \times$ LoRA | Learned gating | MoE over LoRA |
| WEMoE | Weight diff | $K \times$ delta | Task-aware | Weight-ensembling from task vectors |

**Load balancing loss**（防止 expert collapse）：

$$L_{\text{balance}} = \alpha \cdot K \cdot \sum_{i=1}^K f_i \cdot p_i$$

- $f_i$ = 路由到 expert $i$ 的 token 比例
- $p_i$ = batch 中 expert $i$ 的平均路由概率

**⚠️ MoE vs Parameter Averaging 的抉擇**：
- Parameter averaging：一個模型、無額外推論成本，但有能力干擾
- MoE：保留專家能力，但記憶體 $O(K \cdot |E|)$，且需要 router 訓練資料

### 5.2 Activation-Informed Merging

- 用 **前向傳播的 activation patterns** 判斷參數重要性，比 gradient-based 更能捕捉功能性重要度
- **CKA（Centered Kernel Alignment）** 用於量化不同模型各層的表示對應程度
- 當 linear mode connectivity 很強時（共享初始化 + 相近 fine-tuning），activation-informed 方法的邊際收益不大

### 5.3 Evolutionary & Search-Based Optimization

**Merging 配置空間**：
- 模型子集選擇 $S \subseteq M$
- Per-layer mixing coefficients $\alpha^{(l)} \in \Delta^{|S|-1}$
- 層排列/重複的拓撲決策

**搜索範式比較**：

| 範式 | 樣本效率 | 並行化 | 變數處理 | 最佳場景 |
|------|----------|--------|----------|----------|
| **Evolutionary** | 低 | 高 | 極好（離散+連續）| 發現新穎的層架構組合 |
| **Bayesian Optimization** | 高 | 低 | 有限 | 微調 mixing coefficients |
| **Random Search** | 極低 | 極高 | 好 | Baseline benchmarking |

**Akiba et al. (2024) 的里程碑結果**：
- 用 evolutionary 搜索同時優化 Parameter Space (PS) 和 Data Flow Space (DFS)
- 合併日語 LLM + 數學 LLM → 日語數學推理 SOTA（**兩個 parent model 都不具備的能力**）
- 產生了 "Franken-merge"——不同模型的層以非直覺方式交錯

**CMA-ES 的實際效率**：在 FusionBench 上，CMA-ES 搜索通常恢復 oracle merge 效能的 85–95%，計算成本僅為 exhaustive grid search 的 ~1/10。

### 5.4 Representation-Level Alignment

- **Optimal Transport (OT)**：最小化 $\|W_A - PW_B\|_F^2$ 找到對齊矩陣 $P$
- **Activation-based correspondence**：同一輸入在兩模型各層的 activation correlation → 找到跨層對應
- **Neural network stitching**：保留各模型的組件完整，學習最小的連接層

### 5.5 Structural Preservation vs Parameter Unification 的抉擇

| 面向 | Parameter Unification | Structural Preservation (MoE) |
|------|----------------------|------------------------------|
| 推論成本 | $= N$ parameters | $\sim kN + R$（sparse）或 $KN + R$（dense）|
| 能力保留 | 有干擾風險 | 近乎完整 |
| 靈活性 | 容易繼續 fine-tune/merge | 加新 expert 需重訓 router |
| 適用場景 | 資源受限、需要寬廣能力 | 需要峰值專業效能、資源較寬裕 |

---

## 6. 應用場景

### 6.1 多任務 & 多語言能力合併

- Task vector addition 在 LLM 上：merged model 的平均準確度在個別 fine-tuned 模型的 **2–3% 以內**
- DARE：成功合併多達 **6 個** specialized LLMs，保留超過 90% 的個別任務效能
- 多語言遷移：插值 English-specialized 和 target-language 模型 → 低資源語言在 XNLI 上提升 **15–20%**
- **Chat Vector** (Huang et al., 2023b, NTU)：透過 task vector 操作讓 LLM 在新語言獲得 instruction following 能力

**⚠️ 失敗模式**：
- 任務不相似（gradient conflict 度量高）→ 最多 25% 效能下降
- 訓練規模不平衡 → 表示偏向主導任務
- 合併 "簡潔摘要" 和 "詳細闡述" 模型 → 兩個目標都退化

### 6.2 Alignment & Safety

- **Task vector negation**：$\theta_{\text{debiased}} = \theta_{\text{base}} - \alpha \cdot \tau_{\text{bias}}$ → 可手術式移除毒性/偏見
- **RLHF model 插值**：$\theta_{\text{merged}} = (1-\lambda)\theta_{\text{helpful}} + \lambda\theta_{\text{harmless}}$ → 中間 $\lambda$ 值常優於兩端
- **WARM** (Ramé et al., 2024)：平均獨立訓練的 reward models → 減少 reward hacking 4–7%
- **SafeMERGE** (Djuhera et al., 2025)：selective layer-wise merging 保持 safety alignment
- ⚠️ **雙刃劍**：同樣的算術也能「移除」safety alignment → misuse 風險

### 6.3 Federated Learning

**FedAvg**：$\theta_{\text{global}} = \sum_{k=1}^K \frac{n_k}{n}\theta_k$

- 直接是 §3 uniform averaging 的分散式實例化
- 壓縮版：傳輸稀疏 task vectors（利用 TIES/DARE 的稀疏化原理）→ 大幅降低頻寬需求

### 6.4 Domain Specialization

**公式**：$\theta_{\text{merged}} = (1-\alpha)\theta_{\text{base}} + \alpha\theta_{\text{domain}}$

- 醫療：$\alpha \in [0.6, 0.8]$（臨床決策支援）；$\alpha \in [0.3, 0.5]$（患者對話介面）
- 層級式 merging：general → scientific → field-specific（化學/物理/生物）

### 6.5 Evaluation — 評估方法論

**Task Retention Rate (TRR)**：

$$\text{TRR} = \frac{1}{K}\sum_{i=1}^K \frac{\text{Perf}_{\text{merged},i}}{\text{Perf}_{\text{source},i}}$$

- 進階方法（TIES, task arithmetic 等）在 NLP 分類上可達 TRR > 0.95
- 知識密集型任務、generative benchmarks → TRR 掉到 0.80–0.90

**干擾矩陣**：$I_{ij} = P_i^{\text{merged}(i,j)} - P_i^{\text{merged}(i)}$（加入任務 j 後任務 i 的效能差異）

---

## 7. 開放問題與未來方向

### 7.1 當前四大挑戰

1. **理論缺口**：為什麼大型預訓練模型有如此好的 mergeability，缺乏嚴格證明
2. **可擴展性**：模型規模到數千億參數時，alignment 和 conflict resolution 成本超線性增長
3. **標準化評估**：缺乏共識基準同時測量個別任務保留和跨任務遷移
4. **最佳實踐缺失**：如何選擇 merge 配置（哪些模型、什麼粒度、哪種演算法）缺乏原則性指導

### 7.2 未來研究方向

**自動預測系統**：

$$C(\theta_1, \dots, \theta_k) = f_\phi(\text{Sim}(\theta_i, \theta_j), \text{Task}(\theta_i), \text{Arch}(\theta_i)) \to \mathbb{E}[\text{Perf}(\text{Merge}(\theta_1, \dots, \theta_k))]$$

**跨架構 merging**：
- Representation-level translation mechanisms
- Architecture-agnostic knowledge distillation
- Learned correspondence mappings

**動態/持續 merging**：

$$\theta_{\text{merged}}^{(t+1)} = \text{Update}(\theta_{\text{merged}}^{(t)}, \theta_{\text{new}}, D_{\text{val}})$$

**頻域 merging**：
- **FREE-Merging** (Zheng & Wang, 2025, ICCV)：在 Fourier transform domain 做 merging → 分解 task vectors 為頻率成分

**理論保證**：
- PAC-style learning bounds for merged model generalization？
- 能力保留的 fundamental limits？
- Optimal merging coefficients 的 sample complexity？

---

## 8. 對客語 ASR 研究的啟示

### 直接可用的理論工具

| Survey 內容 | 客語 ASR 的對應 | 怎麼用 |
|-------------|---------------|--------|
| Merging Error Bound (Prop. 1) | 驗證 Whisper 為基底的各語言 fine-tune 模型之間的 $\|\theta_1 - \theta_2\|$ | 距離越近 → merging 越安全 |
| Linear mode connectivity | 測量華語/粵語/客語 fine-tuned Whisper 之間的 loss barrier | 可直接測量，指導是否適合 merging |
| Loss landscape flatness | 比較不同資料量 fine-tune 後的 Hessian 特徵譜 | 客語資料少 → 可能是 sharp minimum → 需要策略應對 |
| CKA for representation alignment | 衡量各語言模型各層的表示相似度 | 指導 stitching layer 的位置選擇 |
| Sign conflict analysis | 分析華語 vs 客語 task vectors 在各層的符號衝突率 | 支持分層合併策略（encoder 前段 vs 後段 vs decoder）|

### 方法論選擇建議

基於 survey 的方法論比較，對客語 ASR 的 model merging 建議路線：

1. **第一輪**：用 **Task Arithmetic** 做 baseline → 簡單、data-free、直覺性強
2. **第二輪**：加入 **TIES-Merging** 或 **DARE** → 處理符號衝突和冗餘參數
3. **第三輪**：嘗試 **分層策略**（不同層用不同方法/係數）→ 這是 survey 中未充分探索但極具潛力的方向
4. **進階**：如果效果好，用 **CMA-ES** 搜索最優 layer-wise coefficients
5. **方言適配**：用 **MoE routing** 處理客語各腔調 → 但要注意記憶體開銷

### 論文 novelty 空間

根據 survey 盤點，以下方向有明確的研究空白：

1. **Model merging 在語音 ASR 上的系統性研究**——survey 幾乎完全聚焦 NLP/LLM，語音是未覆蓋的領域
2. **低資源語言的 merging 理論**——survey 認為共享初始化是最關鍵前提，但低資源下 fine-tune 可能 underfitting，task vector 的性質可能不同
3. **泛函分析框架**——survey §2.5 明確指出 "theoretical explanations for observed scaling behaviors remain speculative"，你的 functional analysis 框架可以填補這個空白
4. **方言子流形結構**——survey 沒有討論 dialect/accent 相關的 merging
5. **跨模態 merging（acoustic → text）**——encoder-decoder 的分層 merging 在 survey 中完全沒有出現

---

## 9. 關鍵參考文獻索引

### 必讀（理論基礎）

| 文獻 | Venue | 引用名 |
|------|-------|--------|
| Izmailov et al. (2018) — SWA | UAI 2018 | Weight averaging → wider optima |
| Garipov et al. (2018) — Mode Connectivity | NeurIPS 2018 | Loss surfaces, mode connectivity |
| Draxler et al. (2018) | NeurIPS 2018 | No barriers in energy landscape |
| Frankle et al. (2020) — Linear Mode Connectivity | ICML 2020 | Shared init → LMC |
| Singh & Jaggi (2020) — OT Fusion | NeurIPS 2020 | Model fusion via optimal transport |
| Kornblith et al. (2019) — CKA | ICML 2019 | Representation similarity |
| Ferbach et al. (2024) — OT + LMC | AISTATS 2024 | Proving LMC via optimal transport |

### 必讀（核心方法）

| 文獻 | Venue | 方法 |
|------|-------|------|
| Wortsman et al. (2022) — Model Soups | ICML 2022 | Weight averaging of fine-tuned models |
| Matena & Raffel (2022) — Fisher Merging | NeurIPS 2022 | Fisher-weighted averaging |
| Ilharco et al. (2023) — Task Arithmetic | ICLR 2023 | Task vector 定義 + 四種運算 |
| Ainsworth et al. (2023) — Git Re-Basin | ICLR 2023 | Permutation alignment |
| Yadav et al. (2023) — TIES-Merging | NeurIPS 2023 | Trim + Elect Sign + Merge |
| Yu et al. (2023) — DARE | arXiv → ICML 2024 | Drop And REscale |
| Yang et al. (2024d) — AdaMerging | ICLR 2024 | Adaptive merging coefficients |
| Tang et al. (2024d) — WEMoE | ICML 2024 | Weight-ensembling MoE |
| Huang et al. (2024) — EMR-Merging | NeurIPS 2024 | Tuning-free high-performance merging |
| Akiba et al. (2024) — Evolutionary Merging | arXiv | 進化搜索 merge recipes |

### 應用相關

| 文獻 | 重點 |
|------|------|
| Huang et al. (2023b) — Chat Vector | 跨語言 instruction following |
| Ramé et al. (2024) — WARM | Weight averaged reward models |
| Djuhera et al. (2025) — SafeMERGE | Safety-preserving layer-wise merging |
| Zheng & Wang (2025) — FREE-Merging | 頻域 merging (ICCV 2025) |
| Tang et al. (2024b) — FusionBench | 標準化 evaluation suite |

### Survey 論文（方法論總覽）

| 文獻 | 焦點 |
|------|------|
| Li et al. (2023a) — Deep Model Fusion Survey | 廣泛的 model fusion 技術 |
| Yang et al. (2024b) — Model Merging in LLMs/MLLMs | LLM/多模態 merging |
| Yadav et al. (2024a) — Model MoErging Survey | MoE 風格 composition |
| **Song & Zheng (2026) — 本篇** | 最全面的 FUSE 分類法 survey |

---

## 附錄：方法一覽表（快速查找）

| 方法 | 類型 | Data-Free? | 干擾處理 | 需共享初始化? | 擴展性 |
|------|------|-----------|----------|-------------|--------|
| Simple Averaging | Weight-space | ✅ | ❌ | ✅ | 高 |
| Model Soups | Weight-space | ✅ | ❌（greedy selection 間接處理）| ✅ | 高 |
| Fisher Merging | Importance-weighted | ❌ | 部分 | ✅ | 中 |
| SLERP | Geometric | ✅ | ❌ | ✅ | 高（僅 pairwise）|
| Task Arithmetic | Task vector | ✅ | ❌ | ✅ | 高 |
| TIES-Merging | Sparsification | ✅ | ✅（符號選擇）| ✅ | 高 |
| DARE | Sparsification | ✅ | ✅（隨機 drop）| ✅ | 高 |
| AdaMerging | Adaptive | ❌ | ✅ | ✅ | 中 |
| Git Re-Basin | Alignment | ❌ | N/A | ❌ | 低 |
| ZipIt! | Feature matching | ❌ | N/A | ❌ | 低 |
| MoE Routing | Structural | ❌ | N/A（保留專家）| ✅ | 高 |
| Evolutionary | Search-based | ❌ | 隱式 | ✅ | 中 |

---

*筆記結束。後續可依需要補充語音 ASR 特定的 merging 論文（Rafkin et al. 2026, Zhao et al. 2025 等），待 arXiv 驗證後加入。*
