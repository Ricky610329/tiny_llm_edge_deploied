# CLAUDE.md

實驗室新生教材：CS336 極短版——訓練極小 LLM（llama2.c + TinyStories）、Q8_0 量化、部署到 ESP32，
比兩個 benchmark（tok/s 速度榜、可部署前提下最低 BPB 品質榜）。

## 教材鐵律

- **學生版（README、docs/handouts/、student/）不放答案**：公式、坑的解釋、優化 code 只放
  `docs/course_design.md`（助教版）。改學生文件前先看助教版 §6 的「給／不給」表。
- `eval/validation_100.txt` 是凍結的 benchmark，**永遠不要重生成或修改**。
- 評分規則：品質榜一律用部署的量化權重計分；`eval/eval_bpb.c` 靠 `#include runq.c`
  保持與部署端逐位元一致，不要改寫成獨立實作。

## 建置與指令

```bash
# Windows：gcc 在 MSYS2，必須先加 PATH（DLL 相依，直接呼叫完整路徑會靜默失敗）
export PATH="/c/msys64/ucrt64/bin:$PATH"
bash tools/build.sh          # bin/{run,runq,eval_bpb_q,eval_bpb_f32}(+host_engine 若有 model_data.h)

uv sync                      # Python 環境（torch CPU）
uv run python tools/quantize.py models/xxx.pt models/xxx_q80.bin
uv run python tools/export_header.py models/xxx_q80.bin models/tok512.bin  # → firmware 的 model_data.h
./bin/eval_bpb_q.exe models/xxx_q80.bin -z models/tok512.bin -f eval/validation_100.txt -w 128

# firmware 改動後的驗證：host_engine 與 runq 同參數輸出必須逐字一致
./bin/host_engine.exe -t 0.8 -s 42 -n 100 -i "Once upon a time"
```

## 關鍵技術決策

- **Worst-case 硬體設計**：假設無 PSRAM 的 ESP32-WROOM。權重不進 RAM——模型經
  `tools/export_header.py` 轉成 const 陣列編進 app（rodata 即 flash memory-map，
  與 esp_partition_mmap 同一條 cache 路徑）；RAM 只放 KV cache（部署 seq_len clamp 128）。
- **Firmware 兩線共用一份 engine**：單一來源 `firmware/tinyllm_arduino/llm_engine.h`
  （Arduino 主線；`tinyllm_idf/` 用 INCLUDE_DIRS 引用它）。engine **故意 naive**
  （單核、scalar、fp32 KV）——優化是學生第 4 堂作業，不要幫他們做。
  相對 runq.c 的四個刻意修改見檔頭註解（embedding 即時反量化省 131KB、memcpy 對齊等）。
- **改 engine 後必驗**：重跑 `tools/build.sh` 後 `bin/host_engine` 與 `bin/runq` 同參數
  輸出須逐字一致。注意逐字一致要求相同浮點旗標（FMA 融合會讓取樣在 ~55 token 後分岔）。
- **量化陷阱**：`runq.c` 的分組 matmul 在維度非 group size 倍數時**靜默算錯**
  （stories260K 的 hidden_dim=172 + GS=64 → 輸出全是 "there"）。一律用 `tools/quantize.py`
  （自動退 GS），不要直接用 `train/llama2.c/export.py`。
- `train/llama2.c` 是 vendored（非 submodule），已修 `win.c` 的 VirtualProtect 型別；升級 upstream 要重套。
- Ground truth 數字（budget_calc `--check` 依據）：stories260K params 260,032、
  KV@T=128 = 163,840 B、Q8_0@GS=64 = 278,608 B。改 Config 或格式時要同步驗證。

## 待辦

- 板上實測（使用者自己測 Arduino 與 IDF 兩路線）；速度數字出來後定案競賽 Gate 門檻。
- 使用者的 ESP32 型號未確認（可能無 PSRAM）；接上後用 `esptool.py flash_id` 認板子。
- Repo：https://github.com/Ricky610329/tiny_llm_edge_deploied ——**目前 private**，
  開課前要轉 public（Colab notebook 的 `git clone` 才會通），或改發 token。
