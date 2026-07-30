# TinyLLM on ESP32 — 實驗室新生迷你課程設計

> **本文件是助教版**：含作業解答、刻意設下的坑、以及每堂課「講什麼／不講什麼」的節奏設計。
> 學生入口是 [README](../README.md) 與 [docs/handouts/](handouts/)。學生翻到這份也無妨——先自己做完再看。

> 定位：CS336（Stanford, Language Modeling from Scratch）的極短濃縮版。
> 目標：新生親手「訓練一個極小的 LLM → 量化 → 部署到 ESP32 → 跑 benchmark 比賽」。
> 基礎程式碼：Andrej Karpathy 的 `llama2.c`（MIT 授權，訓練端 `train.py` + 純 C 推論 `run.c` / `runq.c`）。

---

## 1. 第一步：先搞清楚手上的 ESP32 是哪一款

「ESP32」是一整個家族，RAM 差異巨大。課程的記憶體預算取決於這一步。

### 1.1 各型號規格與對本課程的意義

| 型號 | CPU | 內部 SRAM | PSRAM | SIMD 向量指令 | 對本課程的意義 |
|---|---|---|---|---|---|
| ESP32（WROOM-32 模組） | 2× LX6 @240MHz | 520 KB | **無** | 無 | worst case；權重必須用 flash mmap，不能複製進 RAM |
| ESP32（WROVER 模組） | 2× LX6 @240MHz | 520 KB | 4–8 MB | 無 | 影片作者的流程可行，但沒有 SIMD 加速 |
| ESP32-S3（WROOM-1 NxRy） | 2× LX7 @240MHz | 512 KB | R2=2MB / R8=8MB | **有**（ESP-DSP） | 影片同款，速度天花板最高（19 tok/s 等級） |
| ESP32-C3 | 1× RISC-V @160MHz | 400 KB | 無 | 無 | 勉強可行，模型要再縮小，速度慢 |
| ESP32-S2 | 1× LX7 | 320 KB | 視模組 | 無 | 不建議用於本課程 |

### 1.2 辨識方法（三選一，由簡到準）

1. **看模組絲印**：金屬屏蔽罩上印的字，例如 `ESP32-WROOM-32`、`ESP32-WROVER-E`、`ESP32-S3-WROOM-1 N16R8`。
   S3 模組的尾碼：`NxRy` = x MB Flash + y MB PSRAM（如 N8R2 = 8MB Flash + 2MB PSRAM）。
2. **esptool 讀晶片資訊**（接上 USB 後）：
   ```
   pip install esptool
   esptool.py --port COM3 flash_id
   ```
   會顯示晶片型號（ESP32-D0WDQ6 / ESP32-S3 …）與 Flash 大小。注意：esptool 讀不到 PSRAM，PSRAM 要靠下一步。
3. **燒一個 hello world 看開機 log**：有 PSRAM 的板子開機時會印 `esp_psram: Found XMB PSRAM device`；
   程式內可用 `heap_caps_get_total_size(MALLOC_CAP_SPIRAM)` 確認可用量。

> 課程設計原則：**以 worst case（WROOM-32、無 PSRAM、4MB Flash）為預算基準**。
> 這樣不管新生拿到哪塊板子都能完成課程；拿到 S3 的人只是在「速度榜」有硬體優勢（見 §5 的分組建議）。

---

## 2. 記憶體預算（本課程的核心工程課題）

### 2.1 與影片作法的關鍵差異：權重不進 RAM

影片作者的路徑：SPIFFS 讀檔 → `malloc` 到 PSRAM → 從 PSRAM 執行。這需要 PSRAM ≥ 模型大小。

本課程改用：**把權重放在一個 raw flash partition，用 `esp_partition_mmap()` 直接映射成唯讀指標**。
- 權重佔 RAM = 0，任何 ESP32 都能跑（classic ESP32 的資料 mmap 視窗約 4MB，足夠）。
- 代價：讀權重走 QSPI flash + cache，比 SRAM/PSRAM 慢，這正是「速度榜」要優化的東西。
- 這也是很好的教學點：對照 Linux 上 llama2.c 的 `mmap()`，概念一模一樣，只是換成 MCU 的 API。

### 2.2 RAM 裡剩下什麼？KV cache 是大頭

推論時 RAM 只需要放：KV cache + activation buffers + tokenizer 表 + stack。其中 KV cache 佔絕對大頭：

```
KV cache bytes = 2 × n_layers × seq_len × kv_dim × bytes_per_element
kv_dim = dim × n_kv_heads / n_heads
```

以 Karpathy 的 `stories260K` 參考設定（dim=64, n_layers=5, n_heads=8, n_kv_heads=4 → kv_dim=32, vocab=512）試算：

| seq_len | KV fp32 | KV fp16 |
|---|---|---|
| 512（原設定） | 640 KB ✗ | 320 KB ✗ |
| 256 | 320 KB ✗ | 160 KB △ |
| 128 | 160 KB ✓ | 80 KB ✓ |

WROOM-32 不開 WiFi 時實際可用 heap 約 250–300 KB。**建議課程統一：部署時 seq_len=128（fp32 KV）**，這在 worst case 板子上留有餘裕；行有餘力的學生可以實作 fp16 KV cache 換更長的 context（本身就是一個好作業）。

### 2.3 Flash 預算

- 4MB Flash ≈ app（~1MB）+ model partition（~2.5MB 可用）。
- fp32 權重 = 4 bytes/參數 → 上限約 60 萬參數。
- Q8_0 量化（`llama2.c` 的 `runq.c`，group-wise int8）≈ 1.06 bytes/參數 → 26 萬參數的模型只要 ~280 KB。
- 量化同時是**速度**優化：mmap 路徑的瓶頸是 flash 頻寬，權重小 4 倍 ≈ 每個 token 少讀 4 倍資料。

### 2.4 給學生的設計預算（競賽規則的雛形）

| 資源 | 預算（worst case 板） |
|---|---|
| 模型檔（量化後） | ≤ 2 MB（塞進 model partition） |
| KV cache + activations | ≤ 160 KB @ 部署 seq_len=128 |
| 換算約束 | `n_layers × kv_dim ≤ 150`（fp32 KV, T=128） |

另外三條**硬性設計規則**（2026-07 host 端實測定案）：

1. **`dim` 與 `hidden_dim` 必須是 64 的倍數**（訓練時 `--multiple_of=64`）。`runq.c` 的分組 matmul 在列長度非 group size 倍數時會**靜默算錯**（尾端元素被丟棄、scale 索引錯位）——stories260K 的 hidden_dim=172 用 GS=64 匯出後輸出直接變垃圾。`tools/quantize.py` 會自動退到能整除的 group size，但 GS 變小 = scale 開銷變大（GS=4 時模型膨脹近一倍），所以從架構設計端就遵守 64 倍數。
2. **`--max_seq_len ≤ 256`**：推論端按模型檔內的 seq_len 配置 KV cache，訓練時開 512 會讓板端 KV 需求翻倍（韌體會 clamp 到 128，但訓練端就收斂比較省事）。
3. **品質計分一律用部署的那份量化權重**：`eval/eval_bpb.c` 直接 `#include` runq.c 的推論程式碼，保證與板端逐位元一致。

學生的取捨空間：參數量 P、深度 vs 寬度、n_kv_heads（GQA 直接省 KV！）、量化精度。**這個預算表就是「可部署性」的定義**——實際數字等板子確認後由助教定案。

Baseline 參考（stories260K，凍結驗證集 `eval/validation_100.txt`，w=128）：fp32 BPB **0.8132** / Q8_0 BPB **0.8135**——Q8_0 幾乎零品質損失，這是第 3 堂課的現成示範素材。

---

### 2.5 部署實作路線（2026-07 定案，取代 2.1 的 partition 方案）

最終做法比 `esp_partition_mmap()` 更簡單：**模型用 `tools/export_header.py` 轉成 C header、
以 `const` 陣列直接編進 app**。ESP32 的 rodata 本來就放 flash、經 flash cache 自動 memory-map——
與 partition mmap 走同一條硬體路徑、同樣效能，但 partition table / esptool 燒錄 / menuconfig 全部消失。

- **主線 = Arduino**（`firmware/tinyllm_arduino/`）：新生友善，Tools 選單即所有設定；
  `build_opt.h` 已放 `-O3`（Arduino 預設 `-Os`）。Partition Scheme 選 Huge APP。
- **側線 = ESP-IDF**（`firmware/tinyllm_idf/`）：加分選配，教正規工作流。
  兩線共用同一份 `llm_engine.h`（單一來源放 Arduino sketch 資料夾，IDF 用 INCLUDE_DIRS 引用）。
- **工具鏈分階段**：第 1–3 堂全在 Colab（gcc、torch 都預裝，學生零安裝），
  第 4 堂才裝 Arduino IDE。本機 gcc 只有助教需要。

`llm_engine.h` 相對 runq.c 的四個刻意修改（都有教學意義，詳見檔頭註解）：
1. **embedding 表不預先反量化**——runq.c 會把整張表反量化到 RAM（stories260K = 131KB！），
   單這一項就會炸掉 WROOM 的 heap。改成 per-token 即時反量化，數學完全相同。
2. seq_len clamp 到 128（部署規格），KV cache 依 clamp 後的值配置。
3. 不對齊的欄位一律 memcpy——x86 上未對齊讀取只是慢，Xtensa 上直接 crash，host 測不出來。
4. I/O 抽象成 `LLM_PUTS/LLM_MILLIS/LLM_LOGF`，同一份檔案跑 Arduino / IDF / host。

**驗證方法**：`bin/host_engine`（`firmware/host_test.c`）在 PC 上編譯「將燒進板子的同一份
engine + model_data.h」，與 runq 同參數（`-t 0.8 -s 42 -n 100`）輸出**逐字一致**（2026-07-24 實測）。
注意：逐字一致要求相同的浮點 codegen——`-O2` 與 `-O3 -march=native` 會因 FMA 融合在
~55 token 後取樣分岔（數值差 <1e-6，BPB 無感，但文字會不同）。build.sh 已用相同旗標。
板上 Xtensa FPU 的捨入也可能有最後一位差異：品質榜計分以 host 端 `eval_bpb_q` 為準，速度榜在板上量。

### 2.6 板上實測結果（2026-07-30，Arduino 路線）

硬體：ESP32-D0WD-V3（WROOM 級、無 PSRAM、4MB flash、CP210x @COM7，
**燒錄需手按 BOOT**——這塊板自動 reset 不靈，開課時要提醒學生）。

- **速度：19.67 tok/s**（stories260K Q8_0 GS=4 = 509KB、naive engine、Tools 全預設）。
  與參考影片在 S3 + 雙核 + SIMD + PSRAM 的 19 tok/s 相同——Q8_0 省下的 flash 流量
  抵掉了對方全部優化。第 4 堂開場的好素材：「選對格式贏過拚命優化」。
- **記憶體帳目**：free heap 336,728 → 147,380（Δ≈189KB = KV 163,840 + 緩衝/tokenizer ~25KB），
  與 budget_calc 完全吻合；剩餘 147KB。開機 log 無 PSRAM 行 → worst case 確認。
- **數值忠實度**：板上 seed 42 輸出與 host 非-FMA 編譯變體逐字一致（Xtensa 無 x86 式 FMA 融合，符合預期）。
- **助教掌握、學生自己挖的優化空間**：boot log 顯示 `mode:DIO`（2-bit flash 匯流排）→
  Tools 改 QIO/80MHz 頻寬翻倍；GS=64 的正規模型比這顆 GS=4 小一半 → 流量再省一半。
  兩項合計 ~4× 空間，之後才輪到雙核與演算法層。實測有效頻寬 ≈ 10MB/s（509KB × 19.67）。
- **Gate 定案：≥1 tok/s 維持**。塞得進 2MB flash 預算的最大模型估計仍有 ~4-5 tok/s，
  所以速度門檻實質上不會刷人——真正的可部署性約束是 flash/RAM 預算，符合設計意圖。

## 3. 課程結構（4 堂課，CS336 極短版）

每堂 2–3 小時（半講半做），課後各一份作業。假設新生會 Python、修過基礎 ML。

### 第 1 堂：語言模型基礎 + Tokenizer + 評估指標
- LM 的機率視角：next-token prediction、cross-entropy loss。
- Tokenizer：BPE 概念；本課程**統一使用助教提供的 tokenizer**（如 llama2.c 的 512-vocab tokenizer），原因見 §4。
- 評估指標：loss → perplexity → **bits-per-byte (BPB)**，講清楚為什麼不同 tokenizer 之間 per-token PPL 不可比。
- 資料集：TinyStories（HuggingFace 直接抓）。
- **作業 1**：跑通 baseline 訓練（`stories260K` 等級 config），算出 validation BPB。

### 第 2 堂：Transformer 解剖 + 訓練自己的模型
- 逐行讀 llama2.c 的模型定義（幾百行 Python）：RMSNorm、RoPE、GQA、SwiGLU。
- 超參數與 scaling 直覺：同樣參數預算下深 vs 寬、head 數怎麼選。
- 算力需求：26 萬參數等級在單張消費級 GPU 幾分鐘～半小時可收斂，免費 Colab 可行，實驗室 GPU 更快——這是課程能「極短」的前提。
- **作業 2**：在 §2.4 預算內設計自己的 config 並訓練，這就是之後的參賽模型。

### 第 3 堂：推論系統 + 移植到 ESP32
- llama2.c 的 `run.c` 逐塊解讀：權重載入、KV cache、sampling。
- Q8_0 量化：`export.py --version 2` + `runq.c`，量化前後 BPB 對比（讓學生親眼看量化的品質代價）。
- ESP-IDF 移植（助教提供 firmware skeleton，學生只換模型檔）：
  - `mmap` → `esp_partition_mmap()`
  - partition table 加 model 分割區、`esptool` 燒權重
  - menuconfig：CPU 240MHz、flash 80MHz QIO
- **作業 3**：把自己的模型部署到板子上，貼出第一次生成的截圖。

### 第 4 堂：優化 + Benchmark + 競賽日
- 優化選單（講原理，學生自選實作）：
  - 雙核心：把 matmul 切給第二顆核心（FreeRTOS task + 同步）。
  - fp16 KV cache（換 context 長度）。
  - S3 限定：ESP-DSP 的向量 MAC 指令。
- 現場跑兩個 benchmark（規則見 §4），排 leaderboard，得獎的講 5 分鐘做法。

---

## 4. Benchmark 設計（兩榜 + 一道門檻）

### 4.0 可部署性門檻（Gate）——兩榜共同的參賽資格
在**裁判板**（同一塊板子，避免個人板子差異）上：
1. 韌體 + 模型燒得進去（符合 §2.4 flash 預算）；
2. 用固定 prompt 以 seq_len=128 完整生成 100 tokens，不 crash、不 OOM；
3. 速度 ≥ 1 tok/s（下限門檻，避免「能跑但不能用」的模型上品質榜）。

### 4.1 速度榜：解碼 tok/s
- 固定 prompt、固定生成 200 tokens、temperature=0（greedy，保證可重現）。
- 計時：第一個 token 之後起算（排除載入），`tok/s = 199 / 牆鐘時間`。llama2.c 本身就會印這個數字。
- 允許改：一切軟體手段（量化、雙核、SIMD、cache 友善的權重排列、menuconfig 時脈）。
- 不允許：超頻超出規格、換硬體。
- 註：模型愈小愈快，所以速度榜和品質榜天然互相拉扯——這正是設計意圖。

### 4.2 品質榜：可部署前提下的最低困惑度
- 指標：**固定 validation set 上的 bits-per-byte（BPB）**，越低越好。
  - Validation set：從 TinyStories validation split 凍結抽樣（如 100 篇），開課時公布、訓練集禁止包含。
  - 為什麼用 BPB 不用 PPL：BPB 對 tokenizer 中立。本課程雖然統一 tokenizer（此時 PPL 也可比），但用 BPB 可以留下日後開放自訂 tokenizer 的擴充空間，也是 CS336 式的正確習慣。
- **必須用部署的那份量化權重算分**（否則量化爛掉也沒懲罰）。實務上：在 host 用 `runq.c` 同款的反量化邏輯跑 eval script（助教提供），與板上 bit-exact。
- 排名：通過 Gate 的模型中 BPB 最低者勝。可以加一張 Pareto 圖（x=tok/s, y=BPB）當總結視覺。

### 4.3 公平性備註
- 若新生板子型號混雜（有人 S3 有人 classic），**速度榜按晶片分組**，品質榜不用分（BPB 與硬體無關，Gate 用統一裁判板）。
- 統一 tokenizer + 統一部署 seq_len，把變因收斂到「架構設計 × 訓練 × 量化 × 系統優化」——這正是想教的四件事。

---

## 5. Repo 規劃與下一步

### 建議目錄結構
```
docs/            課程講義（本文件 + 每堂 slides/notes）
train/           vendored llama2.c 訓練端（train.py, export.py, tokenizer）
firmware/        ESP-IDF 專案（skeleton：mmap 載入、partition table、benchmark 輸出）
eval/            BPB eval script + 凍結的 validation set
benchmark/       規則 + leaderboard（一個 markdown 表就夠）
```

### 下一步（依序）
1. **確認手上板子的型號**（§1.2 的三個方法）→ 定案 §2.4 的預算數字。
2. Vendor llama2.c，跑通 `stories260K` 等級的訓練 + 量化 + host 端推論（不碰硬體先打通 pipeline）。
3. 寫 firmware skeleton：`esp_partition_mmap` 載權重 + `runq.c` 移植 + tok/s 輸出。在自己的板子上量出 baseline 速度（classic ESP32 上估個位數到十幾 tok/s，需實測——這個數字決定 Gate 門檻怎麼定）。
4. 做 eval script + 凍結 validation set。
5. 回頭寫每堂課的講義與作業說明。

## 6. 教學節奏與留白設計（助教必讀）

原則：**藏的不是文件，是答案。** 規格公開、工具提供、公式和優化留給學生。每堂課的「給／不給」：

| 堂 | 提供（示範 code / 工具） | 刻意留白（學生要自己拿到的） |
|---|---|---|
| 1 | 完整 pipeline 二進位、baseline 模型、eval 工具 | BPB 隨 window 變化的解釋（他們第一次親手看到 context 的價值）；LM=壓縮器的直覺 |
| 2 | `student/budget_calc.py` 骨架 + `--check` 自動對答案 | 參數量／KV cache／量化檔大小三條公式（白板推導，講義不印）；深瘦 vs 淺胖、GQA 省 KV 等設計直覺（用實驗換） |
| 3 | `tools/quantize.py`（含 group size 安全網） | **維度非 64 倍數的坑**：quantize.py 會自動退 GS 並 WARNING，模型變大但不會壞——後果（模型膨脹→flash 頻寬→變慢）留給學生自己追。作業 3a 強迫他們面對這個 WARNING |
| 4 | naive firmware skeleton（單核、scalar、fp32 KV） | 全部優化：雙核、fp16 KV、SIMD、記憶體布局。只給方向表，不給 code |

安全網 vs 留白的界線：**會浪費一整天又學不到東西的**（例如 runq.c 靜默算錯導致輸出全是 "there there there"）用工具擋掉；**有明確線索可以追的**（WARNING、變慢、預算爆掉）留給學生。

`--max_seq_len ≤ 256` 這類「踩了就浪費一次訓練」的規則直接寫進規格，不當坑用。

### 作業 2a 參考解（budget_calc.py 三個 TODO）

```python
kv_dim = dim * n_kv_heads // n_heads

count_params = vocab*dim                                  # embedding（共享分類頭不另計）
             + n_layers * (2*dim*dim + 2*kv_dim*dim       # wq, wo / wk, wv
                           + 3*dim*hidden_dim + 2*dim)    # w1,w2,w3 / 2 rmsnorm
             + dim                                        # final rmsnorm

kv_cache_bytes = 2 * n_layers * seq_len * kv_dim * 4      # K 和 V、每層、每個位置

q80_file_bytes: norm = dim*(2*n_layers+1)（維持 fp32）；q = params - norm
             = q + (q//GS)*4 + norm*4 + 256
```

Ground truth（stories260K 實測，`--check` 用的就是這三個數）：params 260,032；KV@T=128 = 163,840 B；Q8_0@GS=64 = 278,608 B（與實際匯出檔逐 byte 一致）。

### 各堂思考題要點

- 1-3（window 掃描）：**注意，直覺答案是錯的，實測方向相反**（2026-07-30 實測：
  w=64 → 0.7283、w=128 → 0.8132、w=256 → 0.8768，BPB 隨 w **上升**）。
  機制：w 不改變任何單一位置的難度（每個位置都從故事開頭有完整前文），只決定平均算到故事多深。
  對 260K 模型 × TinyStories，「難度-位置」曲線是上升的：開頭公式化（模型背熟了，超便宜）、
  越深情節越難猜、而迷你模型從長 context 撈到的增益太小蓋不過去。大模型上通常相反
  （per-position loss 下降 = in-context learning 曲線）——曲線方向是模型容量的指紋。
  改作業時：學生若答「context 越長越好猜所以 BPB 下降」= 沒跑實驗或沒看數據；
  能答到「w 只改變評分覆蓋的位置範圍 + 開頭被背熟」才算懂。
  另注意：品質榜固定 w=128 的理由是「與部署 context 一致」，不是「越長越準」。
- 1-4：LM 的 loss 就是壓縮率下界（arithmetic coding），BPB 0.81 ≈ 每 byte 壓到 0.1 byte。
- 2 思考題：val loss（nats/token）× 1/ln2 = bits/token，再除以 bytes/token（≈2.19，from eval 輸出 27821/12685）≈ BPB。對不上的主因：訓練 val 是隨機 window、eval 是每篇開頭 + w=128。
- 3c（roofline）：每 token 讀整份權重（每個矩陣都參與一次 matmul）→ 上限 = 頻寬/模型大小。20MB/s ÷ 272KB ≈ 75 tok/s；GS=4 膨脹到 509KB 就只剩 ~40——這是量化坑的速度後果。

### 風險與備案
- **板子 RAM 比預期更小**（如只有 C3）：預算表整體下修（模型 ~10 萬參數、seq_len=64），課程結構不變——這正是 worst-case 設計的好處。
- **新生沒有 GPU**：Colab 免費版足夠訓練這個量級；助教也可提供實驗室 GPU 排程。
- ~~19 tok/s 的期望管理~~ **已實測推翻**：classic ESP32 上 naive engine 就有 19.67 tok/s（見 §2.6）——
  Q8_0 的頻寬優勢比預想大。期望管理反轉：要提醒學生 baseline 不慢，優化要動真格的才拉得開差距。
