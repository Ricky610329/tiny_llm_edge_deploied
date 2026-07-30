# 第 4 堂：部署、優化、競賽

## 課前準備

裝 **Arduino IDE**，Boards Manager 加裝 esp32 core（版本以助教公布為準，全班一致）。
這是本課程唯一需要在自己電腦安裝的東西。

## 部署流程（Arduino 主線）

```bash
# 1. 把你的量化模型轉成 C header（會寫進 firmware/tinyllm_arduino/model_data.h）
uv run python tools/export_header.py models/mymodel_q80.bin models/tok512.bin
```

2. Arduino IDE 開 `firmware/tinyllm_arduino/`，Tools 設定：

| 選項 | 值 |
|---|---|
| Board | 依你的板子（ESP32 Dev Module / ESP32S3 Dev Module） |
| Partition Scheme | **Huge APP (3MB)** |
| CPU Frequency | 240 MHz |
| Flash Mode / Frequency | 依講堂公布的基準設定 |
| PSRAM（S3/WROVER） | 依板子 |

3. Upload，開 Serial Monitor（115200）——你的模型正在一顆微控制器上寫故事。

原理（課堂講）：`const` 陣列放在 flash 的 rodata、經 flash cache 自動 memory-map——
權重完全不佔 RAM，這就是為什麼 1MB 的模型能塞進 RAM 只有幾百 KB 的晶片。

## 你拿到的 skeleton 是什麼

`llm_engine.h` 是「能跑就好」的版本：**單核心、scalar matmul、fp32 KV cache**。
它每一個慢的地方都是故意留的。你們的工作是讓它變快。

## 優化菜單（只給方向，不給 code）

| 方向 | 提示 | 難度 |
|---|---|---|
| 時脈與 flash 設定 | Tools 選單裡每一項都值得懷疑 | ★ |
| 雙核心 | `llm_matmul` 的輸出列彼此獨立；`xTaskCreatePinnedToCore` | ★★ |
| fp16 KV cache | RAM 省一半；省下來的預算能換什麼？ | ★★ |
| SIMD（限 S3） | ESP-DSP 函式庫的 dot product | ★★★ |
| 記憶體布局 | 權重的排列順序 vs flash cache 的行為 | ★★★ |

先做哪個？第 3 堂的 roofline 作業已經告訴你答案了：先判斷你卡在頻寬還是算力。

改完 engine 先在電腦上驗證再上板（快非常多）：

```bash
bash tools/build.sh   # 會多編出 bin/host_engine
./bin/host_engine -t 0.8 -s 42 -n 100 -i "Once upon a time"
# 輸出應該和 bin/runq 用同參數跑的結果一致（優化不該改變數學）
```

## 側路線：ESP-IDF（加分，選配，實驗性）

`firmware/tinyllm_idf/` 是同一顆 engine 的 ESP-IDF 專案（`idf.py set-target esp32 && idf.py flash monitor`）。
想學業界正規嵌入式工作流（menuconfig、sdkconfig、CMake components）的人走這條，
完成者在結報加分。兩條路線的成績同榜計算——比的是模型和優化，不是框架。

注意：ESP-IDF 工具鏈安裝門檻高（數 GB），且**此路線助教未上板驗證過**——engine 本體與
Arduino 路線共用（已驗證），但 IDF 專案殼層可能有小問題要自己排。敢當第一個吃螃蟹的人，
排錯過程本身就值得寫進結報。

## 競賽規則

- **參賽門檻**：在裁判板上燒錄成功、seq_len=128 完整生成 100 tokens 不 crash、≥ 1 tok/s。
- **速度榜**：固定 prompt、greedy、生成 200 tokens，量 tok/s。按晶片型號分組。
  裁判板設定用課程基準（上表），只有你的 code 和模型跟著走。
- **品質榜**：`eval/validation_100.txt` 的 BPB（第 3 堂已算出），通過門檻者取最低。
- 允許：一切軟體手段。不允許：超出規格的超頻、換硬體。

## 結報（每組 5 分鐘）

1. 你做了哪些優化、每一項各帶來多少 tok/s；
2. 一個**失敗的嘗試**，以及你從中學到什麼；
3. 你的 Pareto 位置：tok/s vs BPB，你選擇站在哪、為什麼。
