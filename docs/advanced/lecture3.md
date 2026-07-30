# 第 3 堂：量化與推論系統

> 本堂所有指令都可以繼續在第 1 堂的 Colab 環境跑（torch 已預裝；`uv run python` 換成 `python` 即可），不需要本機安裝。

## 課堂示範

```bash
# 量化 baseline：1.03 MB → ? KB
uv run python tools/quantize.py models/stories260K.pt models/stories260K_q80.bin

# 品質代價：fp32 vs Q8_0，各掃一次 BPB
./bin/eval_bpb_f32.exe models/stories260K.bin     -z models/tok512.bin -f eval/validation_100.txt -w 128
./bin/eval_bpb_q.exe   models/stories260K_q80.bin -z models/tok512.bin -f eval/validation_100.txt -w 128

# 速度：host 上 run vs runq 各跑一次，記下 tok/s
```

課堂討論：Q8_0 是怎麼把 4 bytes 壓成 ~1 byte 的？「幾乎零品質損失」是免費的嗎？什麼情況下不免費？

## 作業 3a：量化你自己的模型

```bash
uv run python tools/quantize.py <你的>/ckpt.pt models/mymodel_q80.bin
```

**把工具印出的每一行都讀完。** 如果出現 WARNING：
1. 解釋它在說什麼；
2. 說明它對模型檔大小、以及（將來在板子上的）速度有什麼影響；
3. 決定要不要因此修改 config 重練，寫下你的理由。

以上放進報告。沒看到 WARNING 的人：說明為什麼你的 config 沒觸發它。

## 作業 3b：部署成績

用 `eval_bpb_q.exe` 在 `eval/validation_100.txt`、`-w 128` 下算你量化模型的 BPB。
**這就是你品質榜的正式成績**（競賽規則：一律用部署的那份量化權重計分）。

## 作業 3c：紙上 roofline（為第 4 堂鋪路）

讀 `train/llama2.c/runq.c` 的 `matmul()`，回答：

1. 生成 1 個 token，需要從權重讀進多少 bytes？（提示：答案幾乎等於模型檔大小——為什麼？）
2. 如果權重放在有效頻寬 20 MB/s 的 SPI flash 上，你的模型速度上限是幾 tok/s？
3. 這個上限和參數量是什麼關係？這對你的 config 設計有什麼啟示？

## 加分

host 上 `runq` 比 `run` 快還是慢？差多少？用 memory bandwidth 的角度解釋你量到的結果。
