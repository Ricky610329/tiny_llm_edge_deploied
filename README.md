# TinyLLM on ESP32 — 實驗室迷你課程

四堂課，完整走一遍：**訓練極小 LLM → Q8_0 量化 → 部署到 ESP32 → 優化與競賽**。
兩個排行榜：解碼速度（tok/s）、可部署前提下的最低 bits-per-byte。

## 課程

| 堂 | 主題 | 講義 |
|---|---|---|
| 1 | 語言模型在做什麼？（黑盒子玩法 + 評估指標）［[開啟 Colab notebook](https://colab.research.google.com/github/Ricky610329/tiny_llm_edge_deploied/blob/main/notebooks/lecture1.ipynb)］ | [lecture1](docs/handouts/lecture1.md) |
| 2 | 設計並訓練你的參賽模型（架構 + 預算） | [lecture2](docs/handouts/lecture2.md) |
| 3 | 量化與推論系統 | [lecture3](docs/handouts/lecture3.md) |
| 4 | 部署、優化、競賽 | [lecture4](docs/handouts/lecture4.md) |

作業與競賽規則都在講義裡。助教請看 `docs/course_design.md`（含解答——先自己做完再翻）。

## 目錄結構

```
docs/handouts/  四堂課講義（學生入口）
docs/           課程設計文件（助教版）
notebooks/      lecture1.ipynb（Colab，第 1 堂零安裝路線）
train/          vendored llama2.c（訓練 + host 端推論參考實作）
student/        作業骨架（budget_calc.py：補完 TODO 並通過 --check）
tools/          quantize.py、export_header.py（模型→C header）、make_valset.py、build.sh
eval/           eval_bpb.c（BPB 評分器，與 runq.c 同一份推論碼）、validation_100.txt（凍結驗證集）
models/         模型檔（gitignore；統一 tokenizer tok512 除外）
bin/            編譯產物（gitignore）
firmware/       tinyllm_arduino/（主線 sketch + llm_engine.h）、tinyllm_idf/（ESP-IDF 側線，共用 engine）、host_test.c
```

## 環境需求

**學生**：
- 第 1–3 堂：瀏覽器 + Google 帳號（全部在 Colab 跑，零安裝）
- 第 4 堂：Arduino IDE + esp32 board core（版本以助教公布為準）

**助教／本機開發**：
- gcc（Windows 用 [MSYS2](https://www.msys2.org/) ucrt64 或 w64devkit）
- [uv](https://docs.astral.sh/uv/)：`uv sync` 一鍵裝好（torch CPU 版即可）
- ESP-IDF（僅側路線需要）

## 快速開始（本機路線；學生請直接開 `notebooks/lecture1.ipynb`）

```bash
# 0. Python 環境
uv sync

# 1. 編譯（Windows Git Bash 先：export PATH="/c/msys64/ucrt64/bin:$PATH"）
bash tools/build.sh

# 2. 下載 baseline 模型放到 models/
#    https://huggingface.co/karpathy/tinyllamas/tree/main/stories260K
#    需要：stories260K.bin、stories260K.pt（tok512 已隨 repo 提供）

# 3. 生成一個故事
./bin/run.exe models/stories260K.bin -z models/tok512.bin -t 0.8 -n 200 -i "Once upon a time"
```

## Baseline（等你來打敗）

stories260K（Karpathy 預訓練，260K 參數），凍結驗證集 `eval/validation_100.txt`、w=128：

| 模型 | 大小 | BPB |
|---|---|---|
| fp32 | 1,056,540 B | 0.8132 |
| Q8_0 | 521,728 B | 0.8135 |

品質榜的目標：**在通過部署門檻的前提下，BPB 低於 0.8135**。

**板上 baseline（2026-07-30 實測）**：ESP32-D0WD-V3（WROOM、無 PSRAM、4MB flash）、
naive engine（單核 / scalar / fp32 KV）：**19.67 tok/s**，init 後剩餘 heap 147KB。
速度榜的目標：超過它。優化空間實測存在（提示：把你的 boot log 和 Tools 選單看仔細）。

## 競賽規格（摘要）

- **門檻**：裁判板上燒錄成功、seq_len=128 完整生成 100 tokens 不 crash、≥ 1 tok/s。
- **速度榜**：固定 prompt、greedy、200 tokens，量 tok/s（按晶片型號分組）。
- **品質榜**：凍結驗證集 BPB，一律用**部署的那份量化權重**計分，最低者勝。

完整規則見 [lecture4](docs/handouts/lecture4.md)。

## 狀態

- [x] Host 端 pipeline（編譯、量化、推論、BPB 評分、凍結驗證集）
- [x] 四堂課講義 + 作業骨架 + 第 1 堂 Colab notebook
- [x] Firmware engine（`llm_engine.h`）：host 端已驗證與 runq 輸出逐字一致
- [x] Arduino sketch + ESP-IDF 專案骨架（兩線共用 engine；**尚未上板實測**）
- [x] Repo 上 GitHub、notebook 已填 URL（**repo 目前 private：開課前需轉 public，Colab 的 clone 才會通**）
- [x] Arduino 路線板上實測（19.67 tok/s baseline）；速度門檻定案 ≥1 tok/s（形式門檻，實質約束為 flash/RAM 預算）
- ESP-IDF 側路線：保留為實驗性選配，不排程驗證（engine 與 Arduino 路線共用、已板上驗證；IDF 殼層由走此線的學生自行排錯）
