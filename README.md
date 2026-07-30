# TinyLLM on ESP32 — 兩小時 Workshop

實驗室新生熟悉課：**親手走一遍「資料 → 練 tokenizer → 訓練 LLM → 量化 → 部署到 ESP32」**，
最後在一塊幾十塊錢的晶片上看自己的模型寫故事（~20 tok/s，超過人類朗讀速度）。

## 上課入口

| 東西 | 位置 |
|---|---|
| 動手主體（Colab，零安裝） | [notebooks/workshop.ipynb](https://colab.research.google.com/github/Ricky610329/tiny_llm_edge_deploied/blob/main/notebooks/workshop.ipynb) |
| 講義 + 時間軸 + 燒錄步驟 | [docs/handouts/workshop.md](docs/handouts/workshop.md) |
| 深度版（4 堂課 + 作業 + 競賽，延伸挑戰） | [docs/advanced/](docs/advanced/) |

學生只需要 Google 帳號；想跟著燒板子再裝 Arduino IDE。

## 兩小時流程

```
終點預告（跑預訓練模型）           10 min
Tokenizer：親手訓練 BPE           25 min
模型：一張圖 + 發射訓練            15 min
量化 Q8_0（訓練跑著的時候講）      20 min
收成：量化、打分、產 header        20 min
上板：燒錄 + Serial Monitor       20 min
懸念收尾：優化方向（code 不提供）  10 min
```

## 目錄結構

```
notebooks/      workshop.ipynb（主線）、lecture1.ipynb（深度版第 1 堂）
docs/handouts/  workshop.md（主線講義）
docs/advanced/  深度版 4 堂講義（延伸挑戰）
docs/           course_design.md（助教版：解答、陷阱、節奏設計）
train/          vendored llama2.c（tokenizer 訓練、模型訓練、host 推論）
tools/          fetch_shards.py、quantize.py、export_header.py、build.sh 等
eval/           eval_bpb.c（BPB 評分器）、validation_100.txt（凍結驗證集，勿動）
student/        budget_calc.py（深度版作業骨架）
firmware/       tinyllm_arduino/（主線 sketch + 引擎）、tinyllm_idf/（實驗性側線）
models/         模型檔（gitignore；tok512 tokenizer 除外）
```

## 本機開發（助教）

```bash
export PATH="/c/msys64/ucrt64/bin:$PATH"   # Windows (MSYS2)
bash tools/build.sh                         # 編出 bin/{run,runq,eval_bpb_*,host_engine}
uv sync                                     # Python 環境（torch CPU 即可）
```

完整 pipeline 指令見 `CLAUDE.md`；改 firmware 引擎後必跑 `bin/host_engine` 與 `bin/runq` 的逐字一致驗證。

## 目前紀錄（等人挑戰）

| 項目 | 紀錄 | 條件 |
|---|---|---|
| 板上速度 | **19.67 tok/s** | ESP32-D0WD-V3、naive 引擎、Tools 全預設 |
| 品質 | **BPB 0.8135** | stories260K Q8_0、凍結驗證集、w=128 |

優化方向的懸念清單在 workshop 講義結尾；競賽規則在 docs/advanced/lecture4。

## 狀態

- [x] Host pipeline、firmware（板上實測）、Colab notebook 全部驗證通過
- [x] 2 小時 workshop 版教材（2026-07-31 重整；迷你資料路線本機實測通過）
- [x] 深度版教材保留於 docs/advanced/
- [ ] Workshop notebook 在 Colab 實際 Run all 一次（repo 轉 public 後）
