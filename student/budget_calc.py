"""作業 2a — 部署預算計算機。

補完下面三個 TODO 函式，然後跑：

    uv run python student/budget_calc.py --check

`--check` 用 baseline 模型（stories260K）的實測數字當 ground truth，
三個函式都寫對才會全部 PASS。之後用它檢查你自己的設計：

    uv run python student/budget_calc.py --dim 64 --n-layers 5 --n-heads 8 --n-kv-heads 4

通過預算檢查的 config 才有部署資格——先過這關，再送去訓練。
"""
import argparse
from dataclasses import dataclass

# ---- benchmark 規格（見 lecture2 講義） -------------------------------------
DEPLOY_SEQ_LEN = 128
KV_BUDGET_BYTES = 160 * 1024
FLASH_BUDGET_BYTES = 2 * 1024 * 1024
HEADER_BYTES = 256  # Q8_0 v2 檔案開頭的固定 header


@dataclass
class Config:
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    hidden_dim: int
    shared_classifier: bool = True  # 輸出層與 embedding 共享權重（train.py 預設）


def default_hidden_dim(dim: int, multiple_of: int = 32) -> int:
    """llama2.c model.py 的規則（提供，不用改）：4*dim 的 2/3 向上取整到 multiple_of。"""
    h = int(2 * (4 * dim) / 3)
    return multiple_of * ((h + multiple_of - 1) // multiple_of)


def count_params(cfg: Config) -> int:
    """回傳模型總參數量。

    盤點清單（對照課堂讀過的 model.py）：
      - token embedding：vocab_size × dim
      - 每一層：wq / wk / wv / wo 四個 attention 矩陣（注意：wk、wv 的形狀
        跟 n_kv_heads 有關，不是正方形！）、w1 / w2 / w3 三個 FFN 矩陣
        （與 hidden_dim 有關）、2 個 RMSNorm 向量
      - 最後一個 RMSNorm 向量
      - 分類頭：shared_classifier 時與 embedding 共享，不另計
    """
    raise NotImplementedError("TODO: 作業 2a")


def kv_cache_bytes(cfg: Config, seq_len: int = DEPLOY_SEQ_LEN, bytes_per_elem: int = 4) -> int:
    """回傳 KV cache 需要的 bytes（預設 fp32）。

    課堂推導過：K 和 V 各存什麼、存幾份、每份多大。
    """
    raise NotImplementedError("TODO: 作業 2a")


def q80_file_bytes(cfg: Config, group_size: int = 64) -> int:
    """回傳 Q8_0 v2 量化檔的大小（bytes）。

    構成：
      - 矩陣權重（含 embedding；共享的分類頭不重複存）→ int8，每參數 1 byte，
        另外每 group_size 個參數配一個 fp32 scale
      - RMSNorm 向量維持 fp32
      - 檔案開頭 HEADER_BYTES 的固定 header
    寫完可以拿 models/ 下的實際檔案大小對答案。
    """
    raise NotImplementedError("TODO: 作業 2a")


# ---- 以下不用改 -------------------------------------------------------------

STORIES260K = Config(dim=64, n_layers=5, n_heads=8, n_kv_heads=4,
                     vocab_size=512, hidden_dim=172)
# ground truth：由 models/stories260K.pt 與其 Q8_0(GS=64) 匯出檔實測
EXPECTED = {
    "params": 260_032,
    "kv_bytes@T=128": 163_840,
    "q80_bytes@GS=64": 278_608,
}


def deploy_group_size(cfg: Config, preferred: int = 64) -> int:
    """部署工具（tools/quantize.py）實際會選到的 group size。"""
    gs = preferred
    while gs > 1 and (cfg.dim % gs != 0 or cfg.hidden_dim % gs != 0):
        gs //= 2
    return gs


def run_check() -> int:
    checks = [
        ("params", lambda: count_params(STORIES260K)),
        ("kv_bytes@T=128", lambda: kv_cache_bytes(STORIES260K)),
        ("q80_bytes@GS=64", lambda: q80_file_bytes(STORIES260K, group_size=64)),
    ]
    failed = 0
    for name, fn in checks:
        expected = EXPECTED[name]
        try:
            got = fn()
        except NotImplementedError:
            print(f"[TODO] {name}: 還沒實作")
            failed += 1
            continue
        if got == expected:
            print(f"[PASS] {name} = {got}")
        else:
            print(f"[FAIL] {name}: 你算出 {got}，實測是 {expected}")
            failed += 1
    return failed


def report(cfg: Config) -> None:
    gs = deploy_group_size(cfg)
    params = count_params(cfg)
    kv = kv_cache_bytes(cfg)
    q80 = q80_file_bytes(cfg, group_size=gs)
    print(f"params:        {params:,}")
    print(f"deploy GS:     {gs}")
    print(f"Q8_0 model:    {q80:,} B ({q80 / 1024:.1f} KB)"
          f"  [{'PASS' if q80 <= FLASH_BUDGET_BYTES else 'FAIL'}] budget {FLASH_BUDGET_BYTES:,} B")
    print(f"KV @T={DEPLOY_SEQ_LEN}:     {kv:,} B ({kv / 1024:.1f} KB)"
          f"  [{'PASS' if kv <= KV_BUDGET_BYTES else 'FAIL'}] budget {KV_BUDGET_BYTES:,} B")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="用 stories260K 實測數字驗證你的實作")
    parser.add_argument("--dim", type=int)
    parser.add_argument("--n-layers", type=int)
    parser.add_argument("--n-heads", type=int)
    parser.add_argument("--n-kv-heads", type=int)
    parser.add_argument("--vocab", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, help="不給則按 model.py 預設規則推算")
    parser.add_argument("--multiple-of", type=int, default=32)
    args = parser.parse_args()

    if args.check:
        raise SystemExit(run_check())

    required = [args.dim, args.n_layers, args.n_heads, args.n_kv_heads]
    if any(v is None for v in required):
        parser.error("需要 --dim --n-layers --n-heads --n-kv-heads（或用 --check）")
    hidden = args.hidden_dim or default_hidden_dim(args.dim, args.multiple_of)
    report(Config(dim=args.dim, n_layers=args.n_layers, n_heads=args.n_heads,
                  n_kv_heads=args.n_kv_heads, vocab_size=args.vocab, hidden_dim=hidden))


if __name__ == "__main__":
    main()
