# 客語 Radio 資料清理：實驗設計討論筆記

**日期：** 2026-03-25
**脈絡：** Anna 正在清理客語 Radio 語料，該語料為人工標注但含大量雜訊（以 `_` 表示空白/未標注位置），並已用 Hakka Whisper 生成 pseudo-label，目標是建立更乾淨的訓練資料。

---

## 1. 問題定義

### 1.1 資料來源

| 資料類型 | 說明 |
|---------|------|
| Target（human annotation） | 人工標注，有真實語意，但含大量 `_`（標注者放棄或聽不清楚的位置） |
| Predict（pseudo-label） | Hakka Whisper 輸出，內容較完整，但不保證正確 |
| Alignment | 兩者的 edit-distance alignment，標記 `s`（substitution）或 `i`（insertion） |

### 1.2 典型樣本

```
Target : 台 _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _灣股票今晡日起
Predict: 鄉親大自家好歡迎收聽今晡日三分鐘个新聞𠊎係西湖个蘇有信首先來關心臺

Target : 價超過 130 _ _ _點台積電起價百分之 1 8 _个消息美國股票道瓊工業指數那斯達克指數 5
Predict: 一百三十 臺 一點八 拓強 五
```

### 1.3 核心困境

- Human annotation 有 ground truth 語意，但 gap 太多
- Whisper pseudo-label 補全了 gap，但錯誤無法直接判斷
- 需要一個**系統化策略**決定每個 token 的信任來源

---

## 2. 初步分析：為何需要「聰明」的合併策略

### 2.1 直接用 pseudo-label 的問題

Whisper 輸出雖完整，但在以下情況容易出錯：
- 數字（「130 點」vs「一百三十」）
- 機構名稱、人名（「那斯達克」vs「拓強」）
- 語碼混用（客語夾華語、英文）

### 2.2 直接用 human annotation 的問題

大量 `_` 導致：
- 語句不完整，模型訓練時 context 斷裂
- Blank ratio 高的句子幾乎無法使用

---

## 3. 方法論澄清：draft.md 能不能解這個問題？

### 3.1 Draft.md（260324）的定位

`260324_exp_design_draft.md` 描述的是：

> 在 **model parameter space** 中，將 ASR task vector 和 tool-calling task vector merge 時的 interference 問題，並設計 selective orthogonalization 解法。

這是一個 **upstream 建模問題**（怎麼把兩顆 model 合起來）。

### 3.2 當前問題的定位

當前問題是：

> 給定兩個 text sequence（human annotation + pseudo-label），如何在 **token 層級**決定信任哪一方，生成更乾淨的 transcription。

這是一個 **downstream 資料處理問題**（annotation merging）。

### 3.3 結論

| | Draft.md | 當前問題 |
|--|--|--|
| 操作對象 | Model parameter space（task vector τ） | Text token sequence |
| 核心問題 | Merge 時的 parameter interference | Noisy annotation 的 token-level 信任決策 |
| 解法核心 | Selective Gram-Schmidt orthogonalization | Alignment-aware token fusion |

**兩者是 upstream（建模）和 downstream（應用）的關係，不是同一個問題空間。**

---

## 4. 初步構想：Hakka LLM + Dictionary as Tool

### 4.1 Anna 的想法

用 Hakka-finetuned LLM + dictionary as tool，作為兩個 transcription 之間的 **arbitrator**——給它看 target 和 pseudo-label，讓它判斷哪個更對，或直接生成更乾淨的版本。

### 4.2 可行性評估

**思路正確，但需要釐清層次：**

- Draft.md 說的是「**怎麼 build** 這顆 merged model」
- 這個應用說的是「**build 完之後**的 model 去解 downstream task」

**Tool-calling 對這個問題的實際價值：**

Tool-calling LLM 的優勢在於**推論時查字典補充知識**（例如罕見詞、數字寫法、人名確認）。但 annotation cleaning 的核心是「哪個 transcription 更對」，這是**對齊與選擇問題**，不是知識查找問題。

**結論：** Tool-calling 在這個 task 不是必要的。妳現有的 Hakka ASR model 就夠做核心實驗。

### 4.3 現有資源確認

| 資源 | 狀態 |
|------|------|
| Hakka Whisper ASR | ✅ 有 |
| Human annotation（有 noise/gaps） | ✅ 有 |
| Alignment 資訊 | ✅ 有 |
| Tool-calling LLM | ❌ 暫無 |

---

## 5. 務實實驗設計（以現有資源為基礎）

### Stage 1：Alignment-guided Merging（Rule-based Baseline）

**目標：** 建立 merged pseudo-clean dataset

**策略：**

```
if target_token == '_':
    use predict_token  # Whisper 填空
elif alignment == 's' and target_token != '_':
    use target_token  # 兩者都有字但不同，信 human annotation
elif alignment == 'i' (predict-only token):
    # 最複雜的 case：Whisper 多出來的字，human 沒標
    → 用 blank_ratio 決定
```

**Blank ratio gating：**

```python
blank_ratio = count('_') / total_tokens

if blank_ratio > 0.7:
    # Human annotation 品質太差，幾乎全信 Whisper
    strategy = 'pseudo_first'
elif blank_ratio < 0.2:
    # Human annotation 可靠，只補缺口
    strategy = 'human_first'
else:
    # 中間地帶，做 alignment-aware fusion
    strategy = 'fusion'
```

### Stage 2：Fine-tune Hakka ASR on Merged Data

**目標：** 驗證 merged data 是否比原始 noisy annotation 更好

**評估指標：**

| Metric | 說明 |
|--------|------|
| CER（Character Error Rate） | 字元層級錯誤率 |
| WER（Word Error Rate） | 詞層級錯誤率 |

**比較基線：**

| Condition | 說明 |
|-----------|------|
| B1 | Fine-tune on original noisy human annotation |
| B2 | Fine-tune on raw Whisper pseudo-label |
| E1 | Fine-tune on merged data（Stage 1 輸出） |

### Stage 3（Future Work）：Tool-calling LLM as Second-pass Cleaner

**前提：** 需要先完成 draft.md 的建模工作

**用途：** 對 Stage 1 輸出中仍有疑問的 segment 做 re-ranking 或二次清理，尤其是：
- 數字格式不一致（「130」vs「一百三十」）
- 罕見詞 / 人名查字典確認

**這是 future work，不是當前實驗的一部分。**

---

## 6. 對齊 Draft.md 的位置

```
[當前工作]
  └─ Stage 1-2：annotation cleaning pipeline
       └─ 輸出：乾淨的 Hakka ASR training data

[Draft.md 的位置]
  └─ Stage 3 的前置作業：build merged model（ASR + tool-calling）
       └─ 輸出：能呼叫字典的 Hakka ASR/LLM
            └─ 用途：Stage 3 的 second-pass cleaning
```

Draft.md 在這個研究中是有意義的，但它在 pipeline 的更後面，不是現在要做的事。

---

## 7. 待釐清問題

1. **Alignment 工具是什麼？** 目前用 edit distance 還是有其他 alignment 方法？
2. **Test set 怎麼建？** 清理完的資料要評估，需要一個乾淨的 held-out set
3. **Blank 的類型有幾種？** `_` 是因為聽不清楚，還是有其他原因（語速太快、背景噪音）？釐清原因有助於設計更好的 merging 策略

---

*本筆記整理自 June × Anna 的討論，2026-03-25*
