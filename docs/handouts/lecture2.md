# 第 2 堂：設計並訓練你的參賽模型

## 課堂

- 逐段讀 `train/llama2.c/model.py`：RMSNorm、RoPE、attention（含 GQA）、FFN。
- 參數都藏在哪裡：每層 7 個矩陣 + embedding + 一些 norm 向量。
- KV cache：生成為什麼需要它、它吃多少記憶體。
  **公式在白板上推導，不印在講義裡——作業要你自己重建它。**

## 作業 2a：預算計算機

補完 `student/budget_calc.py` 裡的三個 TODO 函式（參數量、KV cache、量化後檔案大小），然後：

```bash
uv run python student/budget_calc.py --check
```

`--check` 用 baseline 模型的**實測數字**當 ground truth——三個函式都對了才會全部 PASS。
這支程式就是你的部署資格預檢器，之後每個 config 都要先過它。

## 作業 2b：設計你的 config

以下是 benchmark **規格**（不是提示）：

| 規格 | 值 |
|---|---|
| 部署 context 長度 | T = 128 |
| KV cache（fp32）預算 | ≤ 160 KB |
| 量化後模型大小 | ≤ 2 MB |
| 訓練 `--max_seq_len` | ≤ 256（部署端按模型檔配置 KV cache，開太大板子直接 OOM） |

你的設計空間：`dim / n_layers / n_heads / n_kv_heads`。config 必須通過你自己的 budget_calc。

## 作業 2c：訓練（Colab 或實驗室 GPU）

在 `train/llama2.c/` 下：

```bash
pip install -r requirements.txt
python tinystories.py download            # ~1.6GB，一次性

# 全班統一 tokenizer：用 repo 提供的 tok512，不要自己 train_vocab
mkdir -p data && cp <repo>/models/tok512.model data/tok512.model
python tinystories.py pretokenize --vocab_size=512

python train.py \
  --vocab_source=custom --vocab_size=512 \
  --dim=?? --n_layers=?? --n_heads=?? --n_kv_heads=?? \
  --max_seq_len=256 --batch_size=32 --max_iters=20000 \
  --compile=False --out_dir=out_mymodel
# Colab T4 上建議加 --dtype=float16
```

## 建議實驗（品質榜的分數就從這裡來）

- 同樣的參數量：深而窄 vs 淺而寬，BPB 誰比較好？
- `n_kv_heads` 減半會發生什麼？（先用 budget_calc 算算看它省了什麼，再實測品質掉多少）
- 同樣的訓練時間：模型大一點 vs 步數多一點，哪個划算？

## 思考題

`train.py` log 裡的 val loss 單位是 nats/token。換算成 BPB 大概是多少？
跟第 1 堂量到的數字對得上嗎？差異可能來自哪裡？
