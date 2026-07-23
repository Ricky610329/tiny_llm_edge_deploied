# 第 4 堂：部署、優化、競賽

> firmware skeleton 待硬體到位後補完；本講義先定規則與優化方向。

## 你會拿到什麼

`firmware/` 提供一個「能跑就好」的 naive 移植：

- `esp_partition_mmap()` 從 flash 直接讀權重（權重不佔 RAM）
- 單核心、純 scalar 的 matmul
- fp32 KV cache、seq_len clamp 到 128

**它每一個慢的地方都是故意留的。** 你們的工作是讓它變快。

## 優化菜單（只給方向，不給 code）

| 方向 | 提示 | 難度 |
|---|---|---|
| 時脈設定 | `menuconfig` 裡 CPU 與 flash 的頻率選項 | ★ |
| 雙核心 | matmul 的輸出列彼此獨立——可以切給兩顆核心 | ★★ |
| fp16 KV cache | RAM 省一半；省下來的預算能換什麼？ | ★★ |
| SIMD（限 S3） | ESP-DSP 函式庫的 dot product | ★★★ |
| 記憶體布局 | 權重的排列順序 vs flash cache 的行為 | ★★★ |

先做哪個？第 3 堂的 roofline 作業已經告訴你答案了：先判斷你卡在頻寬還是算力。

## 競賽規則

- **參賽門檻**：在裁判板上燒錄成功、seq_len=128 完整生成 100 tokens 不 crash、≥ 1 tok/s。
- **速度榜**：固定 prompt、greedy、生成 200 tokens，量 tok/s。按晶片型號分組。
- **品質榜**：`eval/validation_100.txt` 的 BPB（第 3 堂已算出），通過門檻者取最低。
- 允許：一切軟體手段。不允許：超出規格的超頻、換硬體。

## 結報（每組 5 分鐘）

1. 你做了哪些優化、每一項各帶來多少 tok/s；
2. 一個**失敗的嘗試**，以及你從中學到什麼；
3. 你的 Pareto 位置：tok/s vs BPB，你選擇站在哪、為什麼。
