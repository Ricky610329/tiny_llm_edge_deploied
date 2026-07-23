/* llm_engine.h — Q8_0 llama2 inference engine for embedded targets.
 *
 * Adapted from llama2.c runq.c (MIT, Andrej Karpathy). Single header, no file
 * I/O: weights and tokenizer come in as const byte pointers (on ESP32 a const
 * array lives in flash rodata and is memory-mapped through the flash cache).
 *
 * Deliberate changes vs runq.c — all four matter on a microcontroller:
 *   1. No pre-dequantized embedding table: the token's row is dequantized on
 *      the fly (identical math, saves vocab*dim*4 bytes of RAM — 131 KB for
 *      the stories260K baseline, which alone would blow the WROOM heap).
 *   2. seq_len is clamped to LLM_MAX_SEQ_LEN (course deployment spec: 128),
 *      so the KV cache allocation stays inside the RAM budget no matter what
 *      the checkpoint header says.
 *   3. Header/tokenizer fields are read with memcpy: several offsets are not
 *      4-byte aligned, and unaligned word loads from mapped flash crash on
 *      Xtensa (they merely go slow on x86 — you would never catch it on host).
 *   4. printing/time go through LLM_PUTS/LLM_MILLIS/LLM_LOGF macros so the
 *      same file runs under Arduino, ESP-IDF, and host tests.
 *
 * This is the NAIVE baseline: single core, scalar matmul, fp32 KV cache.
 * Making it fast is the point of the course. Start where the time goes.
 */
#pragma once
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <time.h>

#ifndef LLM_PUTS
#define LLM_PUTS(s) do { fputs((s), stdout); fflush(stdout); } while (0)
#endif
#ifndef LLM_LOGF
#define LLM_LOGF(...) do { fprintf(stderr, __VA_ARGS__); } while (0)
#endif
#ifndef LLM_MILLIS
#define LLM_MILLIS() ((long)((double)clock() * 1000.0 / CLOCKS_PER_SEC))
#endif
#ifndef LLM_MAX_SEQ_LEN
#define LLM_MAX_SEQ_LEN 128  /* course deployment spec */
#endif

/* ---------------------------------------------------------------- model */

typedef struct {
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len;
} LlmConfig;

typedef struct {
    int8_t* q;  /* quantized values (points into flash for weights) */
    float* s;   /* scale factors */
} QTensor;

typedef struct {
    QTensor* q_tokens;             /* (vocab, dim) — also the classifier when shared */
    float* rms_att_weight;         /* (layer, dim) */
    float* rms_ffn_weight;         /* (layer, dim) */
    QTensor *wq, *wk, *wv, *wo;    /* attention */
    QTensor *w1, *w2, *w3;         /* ffn */
    float* rms_final_weight;       /* (dim) */
    QTensor* wcls;
} LlmWeights;

typedef struct {
    float *x, *xb, *xb2, *hb, *hb2, *q, *k, *v, *att, *logits;
    QTensor xq, hq;
    float *key_cache, *value_cache;  /* (layer, seq_len, kv_dim) */
} LlmRunState;

static LlmConfig g_cfg;
static LlmWeights g_w;
static LlmRunState g_s;
static int g_gs = 0;       /* quantization group size (from model header) */
static int g_seq_len = 0;  /* runtime seq_len = min(header seq_len, LLM_MAX_SEQ_LEN) */

static void llm_dequantize(const QTensor* qx, float* x, int n) {
    for (int i = 0; i < n; i++) { x[i] = qx->q[i] * qx->s[i / g_gs]; }
}

static void llm_quantize(QTensor* qx, const float* x, int n) {
    for (int group = 0; group < n / g_gs; group++) {
        float wmax = 0.0f;
        for (int i = 0; i < g_gs; i++) {
            float v = fabsf(x[group * g_gs + i]);
            if (v > wmax) { wmax = v; }
        }
        float scale = wmax / 127.0f;
        qx->s[group] = scale;
        for (int i = 0; i < g_gs; i++) {
            qx->q[group * g_gs + i] = (int8_t)roundf(x[group * g_gs + i] / scale);
        }
    }
}

/* map n quantized tensors of size_each elements at *cursor, advancing it */
static QTensor* llm_map_qtensors(const uint8_t** cursor, int n, int size_each) {
    const uint8_t* p = *cursor;
    QTensor* res = (QTensor*)malloc(n * sizeof(QTensor));
    if (!res) { return NULL; }
    for (int i = 0; i < n; i++) {
        res[i].q = (int8_t*)p;                 p += size_each;
        if (((uintptr_t)p & 3u) != 0) {        /* scales must be 4-aligned in flash */
            LLM_LOGF("unaligned scales (tensor dims must be multiples of 4)\n");
            free(res);
            return NULL;
        }
        res[i].s = (float*)p;                  p += (size_each / g_gs) * sizeof(float);
    }
    *cursor = p;
    return res;
}

static int llm_init_model(const uint8_t* data, uint32_t len) {
    uint32_t magic; int version;
    memcpy(&magic, data, 4);
    memcpy(&version, data + 4, 4);
    if (magic != 0x616b3432u || version != 2) {
        LLM_LOGF("bad model: magic=%08x version=%d (need Q8_0 v2 from tools/quantize.py)\n",
                 (unsigned)magic, version);
        return -1;
    }
    memcpy(&g_cfg, data + 8, sizeof(LlmConfig));
    uint8_t shared_classifier = data[36];
    memcpy(&g_gs, data + 37, 4);  /* unaligned offset — memcpy, not a cast */
    g_seq_len = g_cfg.seq_len < LLM_MAX_SEQ_LEN ? g_cfg.seq_len : LLM_MAX_SEQ_LEN;

    const uint8_t* cur = data + 256;
    float* fptr = (float*)cur;
    g_w.rms_att_weight = fptr;   fptr += g_cfg.n_layers * g_cfg.dim;
    g_w.rms_ffn_weight = fptr;   fptr += g_cfg.n_layers * g_cfg.dim;
    g_w.rms_final_weight = fptr; fptr += g_cfg.dim;
    cur = (const uint8_t*)fptr;

    int head_size = g_cfg.dim / g_cfg.n_heads;
    int kv_dim = g_cfg.dim * g_cfg.n_kv_heads / g_cfg.n_heads;
    g_w.q_tokens = llm_map_qtensors(&cur, 1, g_cfg.vocab_size * g_cfg.dim);
    g_w.wq = llm_map_qtensors(&cur, g_cfg.n_layers, g_cfg.dim * g_cfg.dim);
    g_w.wk = llm_map_qtensors(&cur, g_cfg.n_layers, g_cfg.dim * kv_dim);
    g_w.wv = llm_map_qtensors(&cur, g_cfg.n_layers, g_cfg.dim * kv_dim);
    g_w.wo = llm_map_qtensors(&cur, g_cfg.n_layers, g_cfg.dim * g_cfg.dim);
    g_w.w1 = llm_map_qtensors(&cur, g_cfg.n_layers, g_cfg.dim * g_cfg.hidden_dim);
    g_w.w2 = llm_map_qtensors(&cur, g_cfg.n_layers, g_cfg.hidden_dim * g_cfg.dim);
    g_w.w3 = llm_map_qtensors(&cur, g_cfg.n_layers, g_cfg.dim * g_cfg.hidden_dim);
    g_w.wcls = shared_classifier ? g_w.q_tokens
                                 : llm_map_qtensors(&cur, 1, g_cfg.dim * g_cfg.vocab_size);
    if (!g_w.q_tokens || !g_w.wq || !g_w.wk || !g_w.wv || !g_w.wo
        || !g_w.w1 || !g_w.w2 || !g_w.w3 || !g_w.wcls) { return -1; }
    if ((uint32_t)(cur - data) > len) {
        LLM_LOGF("model data truncated: need %u bytes, have %u\n",
                 (unsigned)(cur - data), (unsigned)len);
        return -1;
    }
    (void)head_size;
    return 0;
}

static int llm_init_state(void) {
    LlmConfig* p = &g_cfg;
    int kv_dim = p->dim * p->n_kv_heads / p->n_heads;
    g_s.x   = (float*)calloc(p->dim, sizeof(float));
    g_s.xb  = (float*)calloc(p->dim, sizeof(float));
    g_s.xb2 = (float*)calloc(p->dim, sizeof(float));
    g_s.hb  = (float*)calloc(p->hidden_dim, sizeof(float));
    g_s.hb2 = (float*)calloc(p->hidden_dim, sizeof(float));
    g_s.xq.q = (int8_t*)calloc(p->dim, sizeof(int8_t));
    g_s.xq.s = (float*)calloc(p->dim / g_gs, sizeof(float));
    g_s.hq.q = (int8_t*)calloc(p->hidden_dim, sizeof(int8_t));
    g_s.hq.s = (float*)calloc(p->hidden_dim / g_gs, sizeof(float));
    g_s.q   = (float*)calloc(p->dim, sizeof(float));
    g_s.k   = (float*)calloc(kv_dim, sizeof(float));
    g_s.v   = (float*)calloc(kv_dim, sizeof(float));
    g_s.att = (float*)calloc(p->n_heads * g_seq_len, sizeof(float));
    g_s.logits = (float*)calloc(p->vocab_size, sizeof(float));
    g_s.key_cache   = (float*)calloc((size_t)p->n_layers * g_seq_len * kv_dim, sizeof(float));
    g_s.value_cache = (float*)calloc((size_t)p->n_layers * g_seq_len * kv_dim, sizeof(float));
    if (!g_s.x || !g_s.xb || !g_s.xb2 || !g_s.hb || !g_s.hb2 || !g_s.xq.q || !g_s.xq.s
        || !g_s.hq.q || !g_s.hq.s || !g_s.q || !g_s.k || !g_s.v || !g_s.att
        || !g_s.logits || !g_s.key_cache || !g_s.value_cache) {
        LLM_LOGF("out of memory allocating run state (KV cache = %u bytes)\n",
                 (unsigned)(2u * p->n_layers * g_seq_len * kv_dim * sizeof(float)));
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------- forward */

static void llm_rmsnorm(float* o, const float* x, const float* weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) { ss += x[j] * x[j]; }
    ss = 1.0f / sqrtf(ss / size + 1e-5f);
    for (int j = 0; j < size; j++) { o[j] = weight[j] * (ss * x[j]); }
}

static void llm_softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) { if (x[i] > max_val) { max_val = x[i]; } }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < size; i++) { x[i] /= sum; }
}

/* W (d,n) @ x (n,) -> xout (d,).  >90% of all cycles are spent here.
 * Single core, scalar, streams every weight byte from flash — naive on purpose. */
static void llm_matmul(float* xout, const QTensor* x, const QTensor* w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        int32_t ival = 0;
        int in = i * n;
        for (int j = 0; j <= n - g_gs; j += g_gs) {
            for (int k = 0; k < g_gs; k++) {
                ival += ((int32_t)x->q[j + k]) * ((int32_t)w->q[in + j + k]);
            }
            val += ((float)ival) * w->s[(in + j) / g_gs] * x->s[j / g_gs];
            ival = 0;
        }
        xout[i] = val;
    }
}

static float* llm_forward(int token, int pos) {
    LlmConfig* p = &g_cfg;
    LlmWeights* w = &g_w;
    LlmRunState* s = &g_s;
    float* x = s->x;
    int dim = p->dim;
    int kv_dim = p->dim * p->n_kv_heads / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    /* embedding row dequantized on the fly (see header note #1) */
    {
        const int8_t* qrow = w->q_tokens->q + (size_t)token * dim;
        const float* sall = w->q_tokens->s;
        size_t base = (size_t)token * dim;
        for (int i = 0; i < dim; i++) { x[i] = qrow[i] * sall[(base + i) / g_gs]; }
    }

    for (int l = 0; l < p->n_layers; l++) {
        llm_rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

        llm_quantize(&s->xq, s->xb, dim);
        llm_matmul(s->q, &s->xq, w->wq + l, dim, dim);
        llm_matmul(s->k, &s->xq, w->wk + l, dim, kv_dim);
        llm_matmul(s->v, &s->xq, w->wv + l, dim, kv_dim);

        /* RoPE */
        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1;
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k;
                float v0 = vec[i], v1 = vec[i + 1];
                vec[i]     = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        int loff = l * g_seq_len * kv_dim;
        memcpy(s->key_cache + loff + pos * kv_dim, s->k, kv_dim * sizeof(float));
        memcpy(s->value_cache + loff + pos * kv_dim, s->v, kv_dim * sizeof(float));

        /* multihead attention (heads are independent — hint, hint) */
        for (int h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * head_size;
            float* att = s->att + h * g_seq_len;
            for (int t = 0; t <= pos; t++) {
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) { score += q[i] * k[i]; }
                att[t] = score / sqrtf((float)head_size);
            }
            llm_softmax(att, pos + 1);
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att[t];
                for (int i = 0; i < head_size; i++) { xb[i] += a * v[i]; }
            }
        }

        llm_quantize(&s->xq, s->xb, dim);
        llm_matmul(s->xb2, &s->xq, w->wo + l, dim, dim);
        for (int i = 0; i < dim; i++) { x[i] += s->xb2[i]; }

        llm_rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);
        llm_quantize(&s->xq, s->xb, dim);
        llm_matmul(s->hb, &s->xq, w->w1 + l, dim, hidden_dim);
        llm_matmul(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim);
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= 1.0f / (1.0f + expf(-val));  /* SwiGLU */
            s->hb[i] = val * s->hb2[i];
        }
        llm_quantize(&s->hq, s->hb, hidden_dim);
        llm_matmul(s->xb, &s->hq, w->w2 + l, hidden_dim, dim);
        for (int i = 0; i < dim; i++) { x[i] += s->xb[i]; }
    }

    llm_rmsnorm(x, x, w->rms_final_weight, dim);
    llm_quantize(&s->xq, x, dim);
    llm_matmul(s->logits, &s->xq, w->wcls, dim, p->vocab_size);
    return s->logits;
}

/* ------------------------------------------------------------ tokenizer */

typedef struct { char* str; int id; } LlmTokenIndex;

static struct {
    char** vocab;
    float* vocab_scores;
    LlmTokenIndex* sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
} g_tok;

static int llm_init_tokenizer(const uint8_t* data) {
    g_tok.vocab_size = g_cfg.vocab_size;
    g_tok.vocab = (char**)malloc(g_tok.vocab_size * sizeof(char*));
    g_tok.vocab_scores = (float*)malloc(g_tok.vocab_size * sizeof(float));
    g_tok.sorted_vocab = NULL;
    if (!g_tok.vocab || !g_tok.vocab_scores) { return -1; }
    for (int i = 0; i < 256; i++) {
        g_tok.byte_pieces[i * 2] = (unsigned char)i;
        g_tok.byte_pieces[i * 2 + 1] = '\0';
    }
    size_t off = 0;
    memcpy(&g_tok.max_token_length, data + off, 4); off += 4;
    for (int i = 0; i < g_tok.vocab_size; i++) {  /* offsets are unaligned: memcpy */
        int len;
        memcpy(g_tok.vocab_scores + i, data + off, 4); off += 4;
        memcpy(&len, data + off, 4); off += 4;
        g_tok.vocab[i] = (char*)malloc(len + 1);
        if (!g_tok.vocab[i]) { return -1; }
        memcpy(g_tok.vocab[i], data + off, len); off += len;
        g_tok.vocab[i][len] = '\0';
    }
    return 0;
}

static const char* llm_decode(int prev_token, int token) {
    char* piece = g_tok.vocab[token];
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)g_tok.byte_pieces + byte_val * 2;
    }
    return piece;
}

static void llm_safe_print(const char* piece) {
    if (piece == NULL || piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char b = piece[0];
        if (!(isprint(b) || isspace(b))) { return; }
    }
    LLM_PUTS(piece);
}

static int llm_compare_tokens(const void* a, const void* b) {
    return strcmp(((const LlmTokenIndex*)a)->str, ((const LlmTokenIndex*)b)->str);
}

static int llm_str_lookup(const char* str, const LlmTokenIndex* sorted, int n) {
    LlmTokenIndex key;
    key.str = (char*)str;
    const LlmTokenIndex* res =
        (const LlmTokenIndex*)bsearch(&key, sorted, n, sizeof(LlmTokenIndex), llm_compare_tokens);
    return res != NULL ? res->id : -1;
}

static int llm_encode(const char* text, int8_t bos, int8_t eos, int* tokens) {
    if (g_tok.sorted_vocab == NULL) {
        g_tok.sorted_vocab = (LlmTokenIndex*)malloc(g_tok.vocab_size * sizeof(LlmTokenIndex));
        for (int i = 0; i < g_tok.vocab_size; i++) {
            g_tok.sorted_vocab[i].str = g_tok.vocab[i];
            g_tok.sorted_vocab[i].id = i;
        }
        qsort(g_tok.sorted_vocab, g_tok.vocab_size, sizeof(LlmTokenIndex), llm_compare_tokens);
    }
    char* str_buffer = (char*)malloc(g_tok.max_token_length * 2 + 3);
    size_t str_len = 0;
    int n_tokens = 0;
    if (bos) { tokens[n_tokens++] = 1; }
    if (text[0] != '\0') {
        tokens[n_tokens++] = llm_str_lookup(" ", g_tok.sorted_vocab, g_tok.vocab_size);
    }
    for (const char* c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) { str_len = 0; }
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';
        if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) { continue; }
        int id = llm_str_lookup(str_buffer, g_tok.sorted_vocab, g_tok.vocab_size);
        if (id != -1) {
            tokens[n_tokens++] = id;
        } else {
            for (size_t i = 0; i < str_len; i++) {
                tokens[n_tokens++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0;
    }
    while (1) {  /* greedy BPE merges */
        float best_score = -1e10f;
        int best_id = -1, best_idx = -1;
        for (int i = 0; i < n_tokens - 1; i++) {
            sprintf(str_buffer, "%s%s", g_tok.vocab[tokens[i]], g_tok.vocab[tokens[i + 1]]);
            int id = llm_str_lookup(str_buffer, g_tok.sorted_vocab, g_tok.vocab_size);
            if (id != -1 && g_tok.vocab_scores[id] > best_score) {
                best_score = g_tok.vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }
        if (best_idx == -1) { break; }
        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < n_tokens - 1; i++) { tokens[i] = tokens[i + 1]; }
        n_tokens--;
    }
    if (eos) { tokens[n_tokens++] = 2; }
    free(str_buffer);
    return n_tokens;
}

/* -------------------------------------------------------------- sampler */

static unsigned int llm_random_u32(unsigned long long* state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
static float llm_random_f32(unsigned long long* state) {
    return (llm_random_u32(state) >> 8) / 16777216.0f;
}

typedef struct { float prob; int index; } LlmProbIndex;

static int llm_compare_prob(const void* a, const void* b) {
    const LlmProbIndex* a_ = (const LlmProbIndex*)a;
    const LlmProbIndex* b_ = (const LlmProbIndex*)b;
    if (a_->prob > b_->prob) { return -1; }
    if (a_->prob < b_->prob) { return 1; }
    return 0;
}

static int llm_sample(float* logits, float temperature, float topp,
                      unsigned long long* rng_state, LlmProbIndex* probindex) {
    int n = g_cfg.vocab_size;
    if (temperature == 0.0f) {
        int max_i = 0;
        for (int i = 1; i < n; i++) { if (logits[i] > logits[max_i]) { max_i = i; } }
        return max_i;
    }
    for (int q = 0; q < n; q++) { logits[q] /= temperature; }
    llm_softmax(logits, n);
    float coin = llm_random_f32(rng_state);
    if (topp <= 0 || topp >= 1) {
        float cdf = 0.0f;
        for (int i = 0; i < n; i++) {
            cdf += logits[i];
            if (coin < cdf) { return i; }
        }
        return n - 1;
    }
    /* top-p */
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (logits[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = logits[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(LlmProbIndex), llm_compare_prob);
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) { last_idx = i; break; }
    }
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) { return probindex[i].index; }
    }
    return probindex[last_idx].index;
}

/* ------------------------------------------------------------ public API */

/* Initialize from embedded model + tokenizer data. Returns 0 on success. */
static int llm_init(const uint8_t* model_data, uint32_t model_len, const uint8_t* tok_data) {
    if (llm_init_model(model_data, model_len) != 0) { return -1; }
    if (llm_init_state() != 0) { return -1; }
    if (llm_init_tokenizer(tok_data) != 0) { return -1; }
    return 0;
}

/* Generate up to `steps` positions from `prompt`, printing tokens as they come
 * and a final tok/s line. Returns the number of positions processed, -1 on error. */
static int llm_generate(const char* prompt, int steps, float temperature, float topp,
                        unsigned long long rng_seed) {
    if (prompt == NULL) { prompt = ""; }
    if (temperature < 0.0f) { temperature = 0.0f; }
    if (steps <= 0 || steps > g_seq_len) { steps = g_seq_len; }

    int* prompt_tokens = (int*)malloc((strlen(prompt) + 3) * sizeof(int));
    LlmProbIndex* probindex = (LlmProbIndex*)malloc(g_cfg.vocab_size * sizeof(LlmProbIndex));
    if (!prompt_tokens || !probindex) { free(prompt_tokens); free(probindex); return -1; }
    int num_prompt_tokens = llm_encode(prompt, 1, 0, prompt_tokens);
    if (num_prompt_tokens < 1) { free(prompt_tokens); free(probindex); return -1; }

    long start = 0;
    int next;
    int token = prompt_tokens[0];
    int pos = 0;
    while (pos < steps) {
        float* logits = llm_forward(token, pos);
        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = llm_sample(logits, temperature, topp, &rng_seed, probindex);
        }
        pos++;
        if (next == 1) { break; }  /* BOS delimits stories */
        llm_safe_print(llm_decode(token, next));
        token = next;
        if (start == 0) { start = LLM_MILLIS(); }  /* skip slow first iteration */
    }
    LLM_PUTS("\n");
    if (pos > 1) {
        long end = LLM_MILLIS();
        char buf[64];
        snprintf(buf, sizeof(buf), "achieved tok/s: %.2f\n",
                 (pos - 1) / (double)(end - start) * 1000.0);
        LLM_PUTS(buf);
    }
    free(prompt_tokens);
    free(probindex);
    return pos;
}
