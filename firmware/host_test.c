/* Host-side harness for the firmware engine.
 *
 * Compiles the exact llm_engine.h + model_data.h that ship to the board and
 * runs generation on the PC, so the port can be verified against runq.c
 * (same model, seed, prompt => identical text) before any hardware exists.
 *
 * Built by tools/build.sh as bin/host_engine.
 * Usage: host_engine [-t temp] [-p topp] [-s seed] [-n steps] [-i prompt]
 */
#include "tinyllm_arduino/model_data.h"
#include "tinyllm_arduino/llm_engine.h"

int main(int argc, char* argv[]) {
    float temperature = 0.8f, topp = 0.9f;
    unsigned long long seed = 42;
    int steps = 100;
    const char* prompt = "Once upon a time";
    for (int i = 1; i < argc - 1; i += 2) {
        if (argv[i][0] != '-') { continue; }
        if (argv[i][1] == 't') { temperature = (float)atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = (float)atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { seed = (unsigned long long)atoll(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
    }
    if (llm_init(MODEL_DATA, MODEL_DATA_LEN, TOKENIZER_DATA) != 0) {
        fprintf(stderr, "llm_init failed\n");
        return 1;
    }
    fprintf(stderr, "model: dim=%d layers=%d heads=%d kv_heads=%d vocab=%d GS=%d seq=%d(clamped %d)\n",
            g_cfg.dim, g_cfg.n_layers, g_cfg.n_heads, g_cfg.n_kv_heads,
            g_cfg.vocab_size, g_gs, g_cfg.seq_len, g_seq_len);
    return llm_generate(prompt, steps, temperature, topp, seed) > 0 ? 0 : 1;
}
