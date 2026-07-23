/* TinyLLM on ESP32 — ESP-IDF 側路線（與 Arduino 版共用 llm_engine.h）。
 *
 * 部署流程：
 *   1. uv run python tools/export_header.py <模型_q80.bin> models/tok512.bin
 *   2. idf.py set-target esp32   （或 esp32s3）
 *   3. idf.py flash monitor
 */
#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"

#include "model_data.h"
#define LLM_MILLIS() ((long)(esp_timer_get_time() / 1000))
#include "llm_engine.h"

void app_main(void) {
    printf("\nTinyLLM (ESP-IDF)  free heap: %u\n",
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_DEFAULT));
    if (llm_init(MODEL_DATA, MODEL_DATA_LEN, TOKENIZER_DATA) != 0) {
        printf("llm_init failed — check model_data.h\n");
        return;
    }
    printf("model loaded (%u B in flash)  free heap after init: %u\n\n",
           (unsigned)MODEL_DATA_LEN, (unsigned)heap_caps_get_free_size(MALLOC_CAP_DEFAULT));

    unsigned long long seed = 42;
    while (1) {
        printf("--- seed %llu ---\n", seed);
        llm_generate("Once upon a time", 100, 0.8f, 0.9f, seed++);
        vTaskDelay(pdMS_TO_TICKS(5000));
    }
}
