/* eval_bpb: score a model's bits-per-byte on a validation text file.
 *
 * Reuses the inference code from llama2.c verbatim (via #include under
 * TESTING), so the quantized score is bit-exact with what runq.c — and the
 * ESP32 port derived from it — computes at deploy time.
 *
 * Compile (from repo root):
 *   gcc -O3 -DTESTING -Itrain/llama2.c -o bin/eval_bpb_q.exe   eval/eval_bpb.c train/llama2.c/win.c -lm
 *   gcc -O3 -DTESTING -DEVAL_FP32 -Itrain/llama2.c -o bin/eval_bpb_f32.exe eval/eval_bpb.c train/llama2.c/win.c -lm
 *
 * Usage:
 *   eval_bpb_q <model.bin> -z <tokenizer.bin> -f <valset.txt> [-w window]
 *
 * The file is split into documents on "<|endoftext|>". Each document is
 * BPE-encoded with BOS prepended; the model predicts tokens left to right up
 * to `window` positions (default 128 = the course's deployment seq_len).
 * bits  = sum of -log2 p(next token)
 * bytes = UTF-8 byte length of the token pieces actually predicted
 * BPB   = bits / bytes  (tokenizer-neutral; lower is better)
 */
#ifdef EVAL_FP32
#include "run.c"
#else
#include "runq.c"
#endif

#include <math.h>

static const char *DELIM = "<|endoftext|>";

static char *read_entire_file(const char *path, long *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "couldn't open %s\n", path); exit(EXIT_FAILURE); }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc(len + 1);
    if (!buf || fread(buf, 1, len, f) != (size_t)len) {
        fprintf(stderr, "failed to read %s\n", path); exit(EXIT_FAILURE);
    }
    buf[len] = '\0';
    fclose(f);
    *out_len = len;
    return buf;
}

int main(int argc, char *argv[]) {
    char *model_path = NULL, *tokenizer_path = NULL, *text_path = NULL;
    int window = 128;

    if (argc >= 2) { model_path = argv[1]; } else { goto usage; }
    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc || argv[i][0] != '-') { goto usage; }
        if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'f') { text_path = argv[i + 1]; }
        else if (argv[i][1] == 'w') { window = atoi(argv[i + 1]); }
        else { goto usage; }
    }
    if (!tokenizer_path || !text_path) { goto usage; }

    Transformer transformer;
    build_transformer(&transformer, model_path);
    Config *p = &transformer.config;
    if (window <= 1 || window > p->seq_len) { window = p->seq_len; }

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, p->vocab_size);

    long text_len;
    char *text = read_entire_file(text_path, &text_len);

    double total_bits = 0.0;
    long total_tokens = 0, total_bytes = 0;
    int n_docs = 0;
    int *tokens = malloc((text_len + 3) * sizeof(int));

    char *cur = text;
    while (cur && *cur) {
        char *next = strstr(cur, DELIM);
        long doc_len = next ? (next - cur) : (long)strlen(cur);
        char saved = cur[doc_len];
        cur[doc_len] = '\0';
        /* skip leading whitespace so every doc starts like a prompt */
        char *doc = cur;
        while (*doc == '\n' || *doc == '\r' || *doc == ' ') { doc++; }

        if (*doc) {
            int n_tokens = 0;
            encode(&tokenizer, doc, 1, 0, tokens, &n_tokens); /* BOS, no EOS */
            int limit = n_tokens < window ? n_tokens : window;
            for (int pos = 0; pos < limit - 1; pos++) {
                float *logits = forward(&transformer, tokens[pos], pos);
                softmax(logits, p->vocab_size);
                float prob = logits[tokens[pos + 1]];
                if (prob < 1e-10f) { prob = 1e-10f; }
                total_bits += -log2((double)prob);
                char *piece = decode(&tokenizer, tokens[pos], tokens[pos + 1]);
                total_bytes += (long)strlen(piece);
                total_tokens += 1;
            }
            n_docs++;
        }

        cur[doc_len] = saved;
        cur = next ? next + strlen(DELIM) : NULL;
    }

    if (total_bytes == 0) { fprintf(stderr, "no scorable text found\n"); exit(EXIT_FAILURE); }

    double bpb = total_bits / (double)total_bytes;
    double bits_per_token = total_bits / (double)total_tokens;
    printf("docs:            %d\n", n_docs);
    printf("tokens scored:   %ld\n", total_tokens);
    printf("bytes scored:    %ld\n", total_bytes);
    printf("bits per byte:   %.4f\n", bpb);
    printf("token perplexity: %.4f  (2^bits-per-token, window=%d)\n",
           pow(2.0, bits_per_token), window);

    free(tokens);
    free(text);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;

usage:
    fprintf(stderr, "usage: eval_bpb <model.bin> -z <tokenizer.bin> -f <text.txt> [-w window]\n");
    return EXIT_FAILURE;
}
