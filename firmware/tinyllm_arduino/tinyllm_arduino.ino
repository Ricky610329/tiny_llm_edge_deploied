/* TinyLLM on ESP32 — naive baseline sketch.
 *
 * 部署流程：
 *   1. uv run python tools/export_header.py <你的模型_q80.bin> models/tok512.bin
 *      （會產生本資料夾的 model_data.h）
 *   2. Arduino IDE 開這個 sketch，Tools 設定照講義（Partition Scheme 選 Huge APP）
 *   3. Upload，開 Serial Monitor（115200）
 *
 * 這是「能跑就好」的版本：單核心、scalar matmul、fp32 KV cache。
 * 慢是故意的——讓它變快是第 4 堂課的作業。
 */
#include "model_data.h"

#define LLM_PUTS(s)   do { Serial.print(s); } while (0)
#define LLM_LOGF(...) do { Serial.printf(__VA_ARGS__); } while (0)
#define LLM_MILLIS()  ((long)millis())
#include "llm_engine.h"

static unsigned long long seed = 42;

void setup() {
    Serial.begin(115200);
    delay(1500);
    Serial.printf("\n\nTinyLLM  chip: %s  cpu: %d MHz  free heap: %u\n",
                  ESP.getChipModel(), getCpuFrequencyMhz(), ESP.getFreeHeap());
    if (llm_init(MODEL_DATA, MODEL_DATA_LEN, TOKENIZER_DATA) != 0) {
        Serial.println("llm_init failed — check model_data.h");
        while (true) { delay(1000); }
    }
    Serial.printf("model loaded (%u B in flash)  free heap after init: %u\n\n",
                  (unsigned)MODEL_DATA_LEN, ESP.getFreeHeap());
}

void loop() {
    Serial.printf("--- seed %llu ---\n", seed);
    llm_generate("Once upon a time", 100, 0.8f, 0.9f, seed++);
    delay(5000);
}
