# 第 1 堂：語言模型在做什麼？

## 課前準備

照 [README](../../README.md) 把 gcc 與 uv 裝好，跑通 `uv sync` 和 `bash tools/build.sh`，並下載 baseline 模型到 `models/`。

## 課堂示範（跟著做）

```bash
# 1. 生成一個故事
./bin/run.exe models/stories260K.bin -z models/tok512.bin -t 0.8 -i "Once upon a time"

# 2. temperature 實驗：-t 0 跑兩次（結果一模一樣——為什麼？），再試 -t 1.4（會發生什麼？）
./bin/run.exe models/stories260K.bin -z models/tok512.bin -t 0 -i "Once upon a time"

# 3. 評分：這個模型有多「懂」英文故事？
./bin/eval_bpb_f32.exe models/stories260K.bin -z models/tok512.bin -f eval/validation_100.txt -w 128
```

## 課堂概念（細節上課講，這裡只放骨架）

- 語言模型 = next-token prediction；訓練目標 = cross-entropy。
- Tokenizer：本課程全班統一使用 512-vocab 的 `tok512`（一個單字會被切成好幾塊）。
- 評估指標：loss → perplexity → **bits-per-byte（BPB）**。
  課堂提問：為什麼跨 tokenizer 比較 per-token perplexity 不公平？
- 品質榜的計分指標就是 BPB，驗證集就是 `eval/validation_100.txt`（已凍結，開賽後不會動）。

## 作業 1（交一頁 markdown）

1. 跑通上面全部指令，貼一段你最喜歡的生成結果。
2. 重現 baseline 的 BPB = 0.8132（w=128）。
3. 把 `-w` 換成 64 和 256 各跑一次。記錄三個數字，**解釋變化的方向與原因**。
4. 思考題：這個 1MB 的模型能把英文故事「壓縮」到每個 byte 只剩 0.81 bits 的資訊量。
   這句話為什麼成立？它跟 zip 這類壓縮器是什麼關係？

## 這堂課刻意不講的

模型內部長什麼樣。第 2 堂之前，把它當黑盒子玩夠。
