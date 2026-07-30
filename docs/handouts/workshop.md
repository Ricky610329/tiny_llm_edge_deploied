# TinyLLM Workshop 講義（2 小時）

> 定位：實驗室新生熟悉課。目標是讓每個人**親手摸過**一輪：資料 → tokenizer → 訓練 →
> 量化 → 部署到 ESP32。全程 code 提供、順跑一遍，不出作業；優化方向只給名字當懸念。
> 課程的動手主體是 `notebooks/workshop.ipynb`（Colab）。

## 課前準備

**學生**：Google 帳號。就這樣。（想跟著燒板子的人加裝 Arduino IDE + esp32 core。）

**講師**：
- 板子 + USB 線 + 裝好 esp32 core（3.x）的 Arduino IDE，投影 Serial Monitor。
- 開課前 24 小時內自己把 notebook 完整 Run all 一次（Colab 環境偶爾變動）。
- repo 必須是 public，notebook 的 clone 才會通。

## 時間軸（120 分鐘）

| 時間 | 段落 | 內容 | notebook 節 |
|---|---|---|---|
| 0:00–0:10 | 終點預告 | 課程一句話 + 跑預訓練模型看它寫故事 | §1 |
| 0:10–0:35 | Tokenizer | 資料哪來的、BPE 怎麼訓練（合併準則：最高頻相鄰對）、**親手練一顆 512 詞彙的**、玩它 | §2 |
| 0:35–0:50 | 模型 | 架構就講一張圖（注意力=回頭看前文、前饋=查知識），**發射訓練** | §3 |
| 0:50–1:10 | 量化 | 訓練跑著的時候講 Q8_0：分組、int8、scale、為什麼幾乎零損失 | §4 |
| 1:10–1:30 | 收成 | 量化自己的模型 → 看它寫故事 → BPB 打分 → 產 `model_data.h` 下載 | §4–5 |
| 1:30–1:50 | 上板 | 講師 demo 燒錄（學生有板子的跟做）；Serial Monitor 看自己的模型跑 ~20 tok/s | §5 |
| 1:50–2:00 | 懸念收尾 | 「這是故意寫慢的版本」+ 優化方向表 + advanced 指路 | §6 |

只有 1 小時的話：§2 的親手訓練改成講師 demo、§3 的訓練改用預訓練 checkpoint（notebook 備案格），全程變演示課。

## 燒錄步驟（§5 的板端部分）

1. 把 notebook 下載的 `model_data.h` 放進 repo 的 `firmware/tinyllm_arduino/`。
2. Arduino IDE 開 `firmware/tinyllm_arduino/tinyllm_arduino.ino`，Tools 設定：

| 選項 | 值 |
|---|---|
| Board | ESP32 Dev Module（classic 板）/ 依實際板子 |
| **Partition Scheme** | **Huge APP (3MB No OTA/1MB SPIFFS)** ← 必改 |
| Port | 你的 COM port |
| 其他 | 預設 |

3. Upload。若卡在 `Connecting......`：**按住板上的 BOOT 鍵**直到開始寫入（很多板子的自動 reset 不靈）。
4. Serial Monitor、鮑率 **115200** → 看你的模型逐字寫故事 + `achieved tok/s`。

## 懸念收尾（照著講）

板上的引擎是「能跑就好」的版本：單核心（第二顆核心閒著）、純 scalar、flash 半速模式——
即使這樣也有 ~20 tok/s，已超過人類朗讀速度。保守估計還有 **4 倍以上**的空間：

| 方向 | 提示（code 不提供，自己挖） |
|---|---|
| Flash 模式 | Tools 選單裡藏著一個兩倍頻寬的選項 |
| 雙核心 | 矩陣乘法的每一列彼此獨立 |
| fp16 KV cache | RAM 省一半，能換更長的記憶 |
| 訓練更久／更大 | 今天只練了 3000 步 |

想深入：`docs/advanced/` 是完整深度版（4 堂課講義、記憶體預算推導、量化陷阱、作業骨架）。
想比賽：挑戰兩個紀錄——速度 **19.67 tok/s**、品質 **BPB 0.8135**（規則見 advanced/lecture4）。

## 講師筆記

- **保底機制**：notebook §5 有備案格——訓練或 tokenizer 任何一步出狀況，改用預訓練
  stories260K + 官方 tok512 產 header，燒錄環節照常進行，課不會卡死。
- 常見卡點：Colab 忘了開 GPU（訓練會自動降到 800 步的 CPU 模式，故事較粗糙但能跑）；
  `train_vocab` 的互動式提問已用 `echo n |` 處理掉，不要拿掉。
- 學生模型的 BPB 大約落在 1.0–1.4（只練 3000 步），比 baseline 0.8135 差是正常的——
  順勢帶出「訓練更久=更好」的懸念。
- 深度版的解答與陷阱設計都在 `docs/course_design.md`（助教版），改教材前先讀它的 §6。
