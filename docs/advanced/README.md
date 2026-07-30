# 深度版課程（延伸挑戰）

這裡是本課程的**完整深度版**：原設計為 4 堂課 + 作業 + 雙榜競賽，適合上完
2 小時 workshop 後想深入的人自學，或未來開成完整課程。

| 檔案 | 內容 |
|---|---|
| [lecture1.md](lecture1.md) | 語言模型黑盒子、評估指標（BPB）、window 實驗 |
| [lecture2.md](lecture2.md) | 架構解剖（RMSNorm/RoPE/GQA/SwiGLU）、記憶體預算、設計自己的 config |
| [lecture3.md](lecture3.md) | 量化機制、group size 陷阱、roofline 速度推算 |
| [lecture4.md](lecture4.md) | 部署、優化作業、競賽規則（速度榜 + 品質榜） |

作業骨架在 `student/budget_calc.py`（三條公式挖空，`--check` 自動對答案）。
兩個待挑戰的紀錄：**19.67 tok/s**（naive baseline）、**BPB 0.8135**（stories260K Q8_0）。

助教請看 `docs/course_design.md`——含解答、刻意設下的坑、每堂「給／不給」的節奏設計。
