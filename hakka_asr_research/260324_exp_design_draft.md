# Tool Calling Task Vector 的 Functional Decomposition 與 Selective Orthogonalization

**日期：** 2026-03-24  
**背景：** 從 NTK linearization + weight disentanglement framework 出發，將 tool calling fine-tuning 的 task vector 做 functional decomposition，推導出與其他 task vector（如 ASR）merge 時的干擾結構，並設計 selective orthogonalization 實驗。

---

## 1. Notation & Setup

| Symbol | Definition |
|--------|-----------|
| $\theta_0$ | Pre-trained base model parameters |
| $\theta_t = \theta_0 + \tau_t$ | Model fine-tuned on task $t$ |
| $\tau_t = \theta_t - \theta_0$ | Task vector for task $t$ |
| $f(x; \theta)$ | Model output (logits) for input $x$ |
| $J(x) = \nabla_\theta f(x; \theta_0)$ | Jacobian at base model |
| $K(x, x') = J(x)^\top J(x')$ | Neural Tangent Kernel (NTK) |
| $H = \nabla^2 L(\theta_0)$ | Hessian of loss at base model |
| $\Lambda$ | Diagonal matrix from softmax: $\Lambda = \text{diag}(p_i(1 - p_i))$ |
| $\mathcal{D}_t$ | Data support for task $t$ |
| $f_{\text{dict}}: \text{query} \to \text{definition}$ | External dictionary tool |

**重要區分：** 對於 cross-entropy loss，Hessian 的 Gauss-Newton 近似為：

$$
H \approx J^\top \Lambda J \neq J^\top J
$$

因此 H-metric inner product $\tau_A^\top H \tau_B$ 與 NTK-metric $\tau_A^\top (J^\top J) \tau_B$ **不等價**。以下統一使用 H-metric，除非特別標注。

---

## 2. NTK Linearization Framework（回顧）

### 2.1 Linearized Model

$$
f_{\text{lin}}(x; \theta) = f(x; \theta_0) + J(x)^\top (\theta - \theta_0)
$$

### 2.2 Task Vector 作為 Kernel Regression 解

在 linearized regime + squared loss 下：

$$
\tau_t \approx J \cdot (J^\top J)^{-1} \cdot r_t
$$

其中 $r_t = y_t - f(x_t; \theta_0)$ 為 residual vector。

對 cross-entropy loss，精確形式為：

$$
\tau_t \approx (J^\top \Lambda J)^{-1} J^\top \Lambda \cdot r_t
$$

### 2.3 Weight Disentanglement 條件

$$
J(x)^\top \tau_t \approx 0, \quad \forall x \notin \mathcal{D}_t
$$

Task vector $\tau_t$ 只在自己的 data support 上改變 function value。

### 2.4 Merge 的二階展開

$$
L(\theta_0 + \tau_A + \tau_B) \approx L(\theta_0) + \nabla L^\top (\tau_A + \tau_B) + \frac{1}{2}\tau_A^\top H \tau_A + \frac{1}{2}\tau_B^\top H \tau_B + \underbrace{\tau_A^\top H \tau_B}_{\text{interference term}}
$$

**Merging 成功的充分條件：** $\tau_A^\top H \tau_B \approx 0$

---

## 3. Tool Calling Task Vector 的三項分解

### 3.1 Tool Calling 的訓練序列結構

一個 tool calling 訓練 sample 的 token sequence：

$$
\tilde{x} = (\underbrace{x_{\text{query}}}_{\text{user input}}, \underbrace{[\text{CALL}], \text{args}}_{\text{tool invocation}}, \underbrace{[\text{RESULT}], r_{\text{dict}}}_{\text{tool return}}, \underbrace{x_{\text{continue}}}_{\text{final response}})
$$

### 3.2 Loss 分解

總 loss 可以依 token position 分成三個 component：

$$
L_{\text{tool}} = \underbrace{L_{\text{trigger}}}_{\substack{\text{學會判斷} \\ \text{何時呼叫工具}}} + \underbrace{L_{\text{args}}}_{\substack{\text{學會生成} \\ \text{正確的 API call}}} + \underbrace{L_{\text{continue}}}_{\substack{\text{學會整合} \\ \text{tool return 後的回答}}}
$$

其中：

- $L_{\text{trigger}}$：cross-entropy on token positions $\{i : x_i \in [\text{CALL}]\}$，即模型需要在正確時機產生 `[CALL]` token
- $L_{\text{args}}$：cross-entropy on token positions within `args`
- $L_{\text{continue}}$：cross-entropy on token positions within $x_{\text{continue}}$（tool return 之後的自然語言生成）

### 3.3 Task Vector 分解

在 NTK linearized regime 下，gradient 的 linearity 保證：

$$
\tau_{\text{tool}} = \tau_{\text{trigger}} + \tau_{\text{args}} + \tau_{\text{continue}}
$$

其中：

$$
\tau_{\text{trigger}} \approx (J^\top \Lambda J)^{-1} J^\top \Lambda \cdot r_{\text{trigger}}
$$

$$
\tau_{\text{args}} \approx (J^\top \Lambda J)^{-1} J^\top \Lambda \cdot r_{\text{args}}
$$

$$
\tau_{\text{continue}} \approx (J^\top \Lambda J)^{-1} J^\top \Lambda \cdot r_{\text{continue}}
$$

### 3.4 各 Sub-vector 的 Data Support

| Sub-vector | Support set | 描述 |
|-----------|-------------|------|
| $\tau_{\text{trigger}}$ | $\mathcal{D}_{\text{trigger}} = \{x : \text{model 需要判斷是否呼叫工具的 context}\}$ | 包含高 uncertainty 的 factual query，語言理解 component |
| $\tau_{\text{args}}$ | $\mathcal{D}_{\text{args}} = \{x : \text{token position} \in \text{API call format}\}$ | **混合型**：structural tokens (`[CALL]`, `dict_lookup(`) 與 content tokens (`query="deprecate"`) |
| $\tau_{\text{continue}}$ | $\mathcal{D}_{\text{continue}} = \{x : \text{token position} \in \text{post-return generation}\}$ | 自然語言生成，與其他語言任務有顯著 overlap |

---

## 4. 與 ASR Task Vector 的干擾分析

### 4.1 展開 Interference Term

$$
\tau_{\text{tool}}^\top H \tau_{\text{asr}} = \underbrace{\tau_{\text{trigger}}^\top H \tau_{\text{asr}}}_{I_1} + \underbrace{\tau_{\text{args}}^\top H \tau_{\text{asr}}}_{I_2} + \underbrace{\tau_{\text{continue}}^\top H \tau_{\text{asr}}}_{I_3}
$$

第一行展開由 inner product 的 bilinearity 保證，是精確的。

### 4.2 各項的量級分析

**$I_1 = \tau_{\text{trigger}}^\top H \tau_{\text{asr}}$：DOMINANT interference**

- $\mathcal{D}_{\text{trigger}}$ 包含需要語言理解的自然語言 query
- $\mathcal{D}_{\text{asr}}$ 包含語音轉文字的自然語言 output
- 兩者在 **language understanding / generation** 的 NTK eigenfunction 上有顯著 overlap
- 特別是：ASR 轉錄出的句子若包含模型不確定的事實內容，$\mathcal{D}_{\text{trigger}} \cap \mathcal{D}_{\text{asr}} \neq \emptyset$

$$
\boxed{|I_1| = O(\|\tau_{\text{trigger}}\| \cdot \|\tau_{\text{asr}}\| \cdot \cos\angle_H(\tau_{\text{trigger}}, \tau_{\text{asr}}))}
$$

**$I_2 = \tau_{\text{args}}^\top H \tau_{\text{asr}}$：SMALL but non-zero**

- Structural tokens (`[CALL]`, `)`, `query=`) 不出現在 ASR data → 這部分貢獻 $\approx 0$
- Content tokens 在 args 內（如 `"deprecate"`）與 ASR vocabulary 有 overlap → **殘餘干擾**
- 估計：$|I_2| \ll |I_1|$，但 $I_2 \neq 0$

**$I_3 = \tau_{\text{continue}}^\top H \tau_{\text{asr}}$：NON-NEGLIGIBLE**

- Tool return 後的自然語言回答與 ASR 的語言生成 component 共享大量 NTK eigenfunctions
- 這是之前分析中被 **低估** 的項
- $\mathcal{D}_{\text{continue}}$ 的 output distribution 與 ASR 的 language model component 幾乎完全重疊

$$
\boxed{|I_1| > |I_3| \gg |I_2| \approx 0}
$$

### 4.3 修正後的結論

干擾結構不是「只有 trigger」，而是：

$$
\tau_{\text{tool}}^\top H \tau_{\text{asr}} \approx \underbrace{\tau_{\text{trigger}}^\top H \tau_{\text{asr}}}_{\text{dominant}} + \underbrace{\tau_{\text{continue}}^\top H \tau_{\text{asr}}}_{\text{secondary, non-negligible}} + \underbrace{\tau_{\text{args}}^\top H \tau_{\text{asr}}}_{\approx 0}
$$

---

## 5. Selective Orthogonalization Algorithm

### 5.1 目標

不對整個 $\tau_{\text{tool}}$ 做 orthogonalization（會損失 tool calling 能力），而是只處理有干擾的 sub-vectors。

### 5.2 Algorithm: Selective Gram-Schmidt on Sub-vectors

**Input：**
- $\tau_{\text{tool}}$：tool calling task vector
- $\tau_{\text{asr}}$：ASR task vector
- $H$：Hessian（或其近似，見 §5.3）
- Tool calling 訓練資料（用於分離 sub-vectors）

**Step 1：提取 Sub-vectors**

用 token position mask 從 tool calling fine-tuning 的 gradient 中分離三個 component：

$$
g_{\text{trigger}}^{(k)} = \nabla_\theta L_{\text{trigger}}(\theta^{(k)}), \quad g_{\text{args}}^{(k)} = \nabla_\theta L_{\text{args}}(\theta^{(k)}), \quad g_{\text{continue}}^{(k)} = \nabla_\theta L_{\text{continue}}(\theta^{(k)})
$$

對每個 training step $k$，分別累積，得到：

$$
\hat{\tau}_{\text{trigger}} = \sum_k \eta_k \cdot g_{\text{trigger}}^{(k)}, \quad \hat{\tau}_{\text{args}} = \sum_k \eta_k \cdot g_{\text{args}}^{(k)}, \quad \hat{\tau}_{\text{continue}} = \sum_k \eta_k \cdot g_{\text{continue}}^{(k)}
$$

**注意：** 這是近似，因為 optimizer（如 AdamW）的 momentum 和 weight decay 會讓實際的 $\tau$ 不完全等於 gradient 累積。精確做法見 Step 1b。

**Step 1b（替代方案）：Projection-based 分離**

如果無法修改 training loop，可以用 projection 近似：

$$
\hat{\tau}_{\text{trigger}} = P_{\text{trigger}} \cdot \tau_{\text{tool}}
$$

其中 $P_{\text{trigger}}$ 是到 trigger-relevant parameter subspace 的投影。實作上可以用 Fisher Information Matrix 的 top-k eigenvectors（對應 trigger tokens）來構建。

**Step 2：對 $\tau_{\text{trigger}}$ 做 H-orthogonalization**

$$
\tau_{\text{trigger}}^{\perp} = \tau_{\text{trigger}} - \frac{\tau_{\text{trigger}}^\top H \tau_{\text{asr}}}{\tau_{\text{asr}}^\top H \tau_{\text{asr}}} \cdot \tau_{\text{asr}}
$$

**Step 3：對 $\tau_{\text{continue}}$ 做 H-orthogonalization**

$$
\tau_{\text{continue}}^{\perp} = \tau_{\text{continue}} - \frac{\tau_{\text{continue}}^\top H \tau_{\text{asr}}}{\tau_{\text{asr}}^\top H \tau_{\text{asr}}} \cdot \tau_{\text{asr}}
$$

**Step 4：重組 Merged Model**

$$
\theta_{\text{merged}} = \theta_0 + \tau_{\text{asr}} + \tau_{\text{trigger}}^{\perp} + \tau_{\text{args}} + \tau_{\text{continue}}^{\perp}
$$

$\tau_{\text{args}}$ 不做處理（interference $\approx 0$）。

### 5.3 Hessian 近似的實作選擇

精確的 $H$ 不可計算（$O(d^2)$ 空間）。常用近似：

| Method | Formula | Pros | Cons |
|--------|---------|------|------|
| Fisher Information | $F = \mathbb{E}[g g^\top]$ | 等價於 GN approx | 需要大量 sample |
| Diagonal Fisher | $F_{\text{diag}} = \text{diag}(\mathbb{E}[g \odot g])$ | 可計算 | 丟失 off-diagonal |
| KFAC | Block-diagonal Kronecker | 平衡精度與效率 | 實作複雜 |
| **Identity（baseline）** | $H \approx I$ | 最簡單 | 退化為 Euclidean inner product |

**建議：** 先用 diagonal Fisher 做 baseline，再比較 KFAC。Identity 作為 ablation control。

### 5.4 Pseudocode

```python
def selective_orthogonalize(
    tau_tool_components: dict,  # {'trigger': tensor, 'args': tensor, 'continue': tensor}
    tau_asr: torch.Tensor,
    H_approx: str = 'diagonal_fisher',  # 'identity' | 'diagonal_fisher' | 'kfac'
    fisher_data: DataLoader = None,
    base_model: nn.Module = None,
    orthogonalize_components: list = ['trigger', 'continue'],  # which sub-vectors to process
) -> torch.Tensor:
    """
    Selective Gram-Schmidt orthogonalization of tool calling sub-vectors
    against ASR task vector under H-metric.
    
    Returns: merged task vector tau_tool_orthogonalized
    """
    
    # Step 0: Compute H-metric quantities
    if H_approx == 'identity':
        def h_inner(a, b):
            return torch.dot(a.flatten(), b.flatten())
    elif H_approx == 'diagonal_fisher':
        F_diag = compute_diagonal_fisher(base_model, fisher_data)
        def h_inner(a, b):
            return torch.dot(a.flatten() * F_diag.flatten(), b.flatten())
    elif H_approx == 'kfac':
        kfac_factors = compute_kfac(base_model, fisher_data)
        def h_inner(a, b):
            return kfac_inner_product(a, b, kfac_factors)
    
    # Step 1: H-norm of tau_asr (denominator, computed once)
    asr_h_norm_sq = h_inner(tau_asr, tau_asr)
    
    # Step 2: Orthogonalize selected components
    result_components = {}
    for name, tau_comp in tau_tool_components.items():
        if name in orthogonalize_components:
            # Gram-Schmidt projection
            proj_coeff = h_inner(tau_comp, tau_asr) / asr_h_norm_sq
            tau_comp_orth = tau_comp - proj_coeff * tau_asr
            result_components[name] = tau_comp_orth
            
            # Log interference reduction
            before = h_inner(tau_comp, tau_asr).item()
            after = h_inner(tau_comp_orth, tau_asr).item()
            print(f"  {name}: interference {before:.4e} -> {after:.4e}")
        else:
            result_components[name] = tau_comp  # keep as-is
    
    # Step 3: Recombine
    tau_tool_orth = sum(result_components.values())
    return tau_tool_orth


def extract_sub_vectors_by_gradient(
    base_model: nn.Module,
    tool_data: DataLoader,
    optimizer_cls = torch.optim.SGD,
    lr: float = 1e-5,
    epochs: int = 3,
) -> dict:
    """
    Fine-tune with gradient accumulation separated by token position.
    
    Requires tool_data to include token-level role labels:
      - 'trigger': positions where model decides to call tool
      - 'args': positions within the API call
      - 'continue': positions after tool return
    
    Returns dict of {'trigger': tau_trigger, 'args': tau_args, 'continue': tau_continue}
    """
    import copy
    
    # Three separate accumulators
    models = {
        'trigger': copy.deepcopy(base_model),
        'args': copy.deepcopy(base_model),
        'continue': copy.deepcopy(base_model),
    }
    optimizers = {
        name: optimizer_cls(m.parameters(), lr=lr)
        for name, m in models.items()
    }
    
    for epoch in range(epochs):
        for batch in tool_data:
            input_ids = batch['input_ids']
            labels = batch['labels']
            role_mask = batch['role_mask']  # tensor: 0=trigger, 1=args, 2=continue
            
            for role_idx, name in enumerate(['trigger', 'args', 'continue']):
                optimizers[name].zero_grad()
                
                # Mask labels to only compute loss on this role's tokens
                masked_labels = labels.clone()
                masked_labels[role_mask != role_idx] = -100  # ignore
                
                logits = models[name](input_ids).logits
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    masked_labels.view(-1),
                    ignore_index=-100,
                )
                loss.backward()
                optimizers[name].step()
    
    # Extract task vectors
    base_params = {n: p for n, p in base_model.named_parameters()}
    sub_vectors = {}
    for name, model in models.items():
        tau = torch.cat([
            (p - base_params[n]).flatten()
            for n, p in model.named_parameters()
        ])
        sub_vectors[name] = tau
    
    return sub_vectors


def compute_diagonal_fisher(
    model: nn.Module,
    data: DataLoader,
    n_samples: int = 1000,
) -> torch.Tensor:
    """Compute diagonal Fisher Information Matrix."""
    fisher_diag = torch.zeros(
        sum(p.numel() for p in model.parameters())
    ).to(next(model.parameters()).device)
    
    model.eval()
    count = 0
    for batch in data:
        if count >= n_samples:
            break
        
        model.zero_grad()
        logits = model(batch['input_ids']).logits
        
        # Sample from model's own distribution
        dist = torch.distributions.Categorical(logits=logits[:, -1, :])
        sampled = dist.sample()
        log_prob = dist.log_prob(sampled)
        log_prob.sum().backward()
        
        grad = torch.cat([p.grad.flatten() for p in model.parameters()])
        fisher_diag += grad ** 2
        count += batch['input_ids'].size(0)
    
    fisher_diag /= count
    return fisher_diag
```

---

## 6. 實驗設計

### 6.1 實驗矩陣

| Experiment | Merge Method | 預期結果 |
|-----------|-------------|----------|
| E0 | Direct merge: $\theta_0 + \tau_{\text{tool}} + \tau_{\text{asr}}$ | Baseline，有干擾 |
| E1 | Full orthogonalization: 整個 $\tau_{\text{tool}}$ 對 $\tau_{\text{asr}}$ 做 GS | 干擾減少但 tool calling 能力可能受損 |
| E2 | Selective (trigger only): 只對 $\tau_{\text{trigger}}$ 做 GS | 測試「trigger 是 dominant」的假設 |
| E3 | Selective (trigger + continue): 對 $\tau_{\text{trigger}}$ 和 $\tau_{\text{continue}}$ 做 GS | **主要提案**，預期最佳 |
| E4 | H-metric ablation: E3 但用 $H = I$ | 測試 H-metric vs Euclidean 的差異 |
| E5 | TIES-Merging baseline | 現有 SOTA 比較 |
| E6 | DARE + Task Arithmetic baseline | 現有方法比較 |

### 6.2 Evaluation Metrics

| Metric | 測什麼 | 怎麼算 |
|--------|--------|--------|
| ASR-CER | ASR 保持能力 | Character Error Rate on Hakka test set |
| ASR-WER | ASR 保持能力 | Word Error Rate on Hakka test set |
| Tool-Trigger-F1 | 工具觸發準確率 | F1 on tool call detection |
| Tool-Args-EM | 工具參數準確率 | Exact match on generated args |
| Tool-Response-BLEU | 工具回答品質 | BLEU on post-tool response |
| Interference-Ratio | 干擾量化 | $\frac{|\tau_{\text{tool}}^\top H \tau_{\text{asr}}|}{\|\tau_{\text{tool}}\|_H \cdot \|\tau_{\text{asr}}\|_H}$ (cosine in H-metric) |

### 6.3 Sanity Checks

在跑主實驗之前，需要驗證的假設：

1. **Linearization 品質：** 計算 $\|f(\theta_0 + \tau) - f_{\text{lin}}(\theta_0 + \tau)\| / \|f(\theta_0 + \tau)\|$ for $\tau \in \{\tau_{\text{tool}}, \tau_{\text{asr}}\}$。若 > 10%，linearization 不可信。

2. **分解可加性：** 驗證 $\|\tau_{\text{tool}} - (\hat{\tau}_{\text{trigger}} + \hat{\tau}_{\text{args}} + \hat{\tau}_{\text{continue}})\| / \|\tau_{\text{tool}}\|$。若 > 5%，gradient 分離方法有問題。

3. **Sub-vector 量級：** 報告 $\|\tau_{\text{trigger}}\|$, $\|\tau_{\text{args}}\|$, $\|\tau_{\text{continue}}\|$ 的相對比例，確認 decomposition 是有意義的（不是某一項 dominate 99%）。

4. **CKA similarity：** 計算 base model vs fine-tuned model 在各層的 CKA，確認 fine-tuning 確實只改了部分 representation。

---

## 7. Open Questions

1. **$\mathcal{D}_{\text{trigger}}$ 的操作化定義：** 用 base model 的 output entropy $H(P(y|x)) > \epsilon$ 做 threshold？但 ASR output 的 entropy 本來就高，需要更精細的定義——可能需要限定為 factual uncertainty（而非 acoustic uncertainty）。

2. **Optimizer 效應：** AdamW 的 momentum 和 adaptive learning rate 會讓 $\tau \neq \sum_k \eta_k g^{(k)}$。Gradient 分離法得到的 sub-vectors 與真實的 functional decomposition 之間的 gap 有多大？

3. **超過兩個 task 的推廣：** 如果要同時 merge $\tau_{\text{tool}} + \tau_{\text{asr}} + \tau_{\text{translation}}$，selective GS 要做 sequential（先對 $\tau_{\text{asr}}$ 再對 $\tau_{\text{translation}}$）還是 simultaneous？

4. **Layer-wise 異質性：** 不同 layer 的 $H$ 結構不同，interference 可能集中在特定層。是否應該做 layer-wise selective orthogonalization？

---

## References

- Ortiz-Jimenez, G., Favero, A., & Frossard, P. (2023). Task Arithmetic in the Tangent Space: Improved Editing of Pre-Trained Models. *NeurIPS 2023*. arXiv:2305.12827
- Ilharco, G., et al. (2023). Editing Models with Task Arithmetic. *ICLR 2023*. arXiv:2212.04089
- Yadav, P., et al. (2023). TIES-Merging: Resolving Interference When Merging Models. *NeurIPS 2023*. arXiv:2306.01708
- Yu, L., et al. (2024). Language Model is Secretly a DARE Merging Expert. *ICML 2024*. arXiv:2311.03099
- Schick, T., et al. (2023). Toolformer: Language Models Can Teach Themselves to Use Tools. *NeurIPS 2023*. arXiv:2302.04761
