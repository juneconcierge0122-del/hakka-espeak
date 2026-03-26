# ML HW2 Snapshot — 2026-03-26

## 作業
- **課程：** Machine Learning 2026 Spring HW2
- **任務：** 10-class anime idol image classification（150×150 PNG）
- **Classes：** tomori, anon, soyo, taki, rana, sakiko, nyamuchi, umiri, uika, mutsumi
- **方法：** AIDE（AI-Driven Exploration）— 用 local LLM 自動生成 / debug / improve PyTorch code
- **LLM：** GLM-4.7-Flash Q4_K_XL（29.9B params, 16.3GB GGUF）on RTX 3090
- **Training model：** ResNet18 pretrained → fine-tune

## 檔案位置
- **Repo：** `juneconcierge0122-del/hakka_asr_research`（master branch）
- **主要檔案：** `codex_AIDE_v1_0325_fixed.py`
- **資料集：** `./hw2_data/`（train/val/test + JSON）
- **輸出：** `./pred.json`（list of `{"id": int, "pred": str}`）

## 已修正的 Bugs（3/26）

### Fix 1 — temperature 沒傳進去
- `generate_response` 裡 `create_chat_completion` hardcode `temperature=0.0`
- 改成 `temperature=temperature`，讓 draft(0.7) / improve(0.5) / debug(0.1) 生效

### Fix 2 — 合併兩個 `__main__` block
- 原本有兩個 `if __name__ == "__main__"`，合併成一個

### Fix 3 — Timeout 太長
- 原本 3600s (1hr) per step → 改成 cap 600s (10min)
- ResNet18 + 150×150 + 10 epochs 不需要 1 小時

### Fix 4 — 移除慢速 LLM analysis calls
- `parse_exec_result` 裡原本用 LLM 做 error analysis 和 training summary
- 每次多花 3-5 分鐘但對 debug prompt 幫助有限
- 改成直接用字串拼接，LLM calls 從 ~5 次/step 降到 1 次/step

### Fix 5 — GLM-4 thinking mode 卡住（最關鍵）
- GLM-4.7 會生成 `<think>...</think>` 長思考，幾千 tokens
- 加 `</think>` 到 stop tokens → 一結束思考就停
- 後處理 strip 掉所有 thinking content
- 加 `sys.stdout.flush()` 確保 print 即時輸出

## 預估時間
- 改前：每 step 20-65 分鐘，5 steps ≈ 2-5 小時（可能卡死）
- 改後：每 step 5-15 分鐘，5 steps ≈ 25-75 分鐘

## 尚未驗證
- [ ] 改後版本完整跑完 5 steps
- [ ] 確認 `pred.json` 輸出正確
- [ ] 確認 val accuracy 數字合理（ResNet18 fine-tune 預期 70-90%）
- [ ] 如果 accuracy 不夠高，可能需要手動寫 training code 而不靠 AIDE

## 備註
- GPU 記憶體分配：LLM ~14GB + KV cache ~680MB + compute ~414MB ≈ 15.2GB，剩 ~8.8GB 給子進程跑 PyTorch training，ResNet18 夠用
- 如果 AIDE 生成的 code 用更大 model（ResNet50 之類），可能 OOM
- `set_seed(531)` 已設好 reproducibility
