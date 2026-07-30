# 第 1 堂：語言模型在做什麼？

## 課前準備

有 Google 帳號就好。**這堂課不需要在自己電腦安裝任何東西。**

打開課程 notebook：
[在 Colab 開啟 lecture1.ipynb](https://colab.research.google.com/github/Ricky610329/tiny_llm_edge_deploied/blob/main/notebooks/lecture1.ipynb)，
`Run all` 一次，確認每格都能跑。

## 課堂示範（都在 notebook 裡）

1. **生成故事**：26 萬參數、1MB 的模型寫英文童話。
2. **temperature 實驗**：`-t 0` 跑兩次（結果一模一樣——為什麼？）、`-t 1.4`（會發生什麼？）。
3. **評分**：BPB = 0.8132 是什麼意思。

## 課堂概念（細節上課講，這裡只放骨架）

- 語言模型 = next-token prediction；訓練目標 = cross-entropy。
- Tokenizer：本課程全班統一使用 512-vocab 的 `tok512`（一個單字會被切成好幾塊）。
- 評估指標：loss → perplexity → **bits-per-byte（BPB）**。
  課堂提問：為什麼跨 tokenizer 比較 per-token perplexity 不公平？
- 品質榜的計分指標就是 BPB，驗證集就是 `eval/validation_100.txt`（已凍結，開賽後不會動）。

## 作業 1（交一頁 markdown）

1. 跑通整本 notebook，貼一段你最喜歡的生成結果。
2. 重現 baseline 的 BPB = 0.8132（w=128）。
3. 把 `-w` 換成 64 和 256 各跑一次。記錄三個數字，**解釋變化的方向與原因**。
4. 思考題：這個 1MB 的模型能把英文故事「壓縮」到每個 byte 只剩 0.81 bits 的資訊量。
   這句話為什麼成立？它跟 zip 這類壓縮器是什麼關係？

## 這堂課刻意不講的

模型內部長什麼樣。第 2 堂之前，把它當黑盒子玩夠。

## 附錄：想在自己電腦跑？（選配，不是作業）

裝 gcc（Windows 用 MSYS2 或 w64devkit）與 [uv](https://docs.astral.sh/uv/)，
然後照 README「快速開始」。所有 notebook 裡的指令都能在本機重現。
