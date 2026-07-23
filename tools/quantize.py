"""Quantize a llama2.c checkpoint (.pt) to Q8_0 v2 format for runq.c / ESP32.

Unlike upstream export.py (which only validates `dim`), this picks the largest
group size that divides BOTH dim and hidden_dim. runq.c's grouped matmul
silently drops the tail of each row when the row length is not a multiple of
the group size (e.g. stories260K's hidden_dim=172 with GS=64 -> garbage
output), so this check is mandatory, not cosmetic.

Usage:
    uv run python tools/quantize.py models/stories260K.pt models/stories260K_q80.bin
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train", "llama2.c"))
from export import load_checkpoint, version2_export  # noqa: E402


def pick_group_size(dim: int, hidden_dim: int, preferred: int = 64) -> int:
    gs = preferred
    while gs > 1 and (dim % gs != 0 or hidden_dim % gs != 0):
        gs //= 2
    return gs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="input .pt checkpoint")
    parser.add_argument("output", help="output .bin (Q8_0 v2)")
    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint)
    p = model.params
    hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
    gs = pick_group_size(p.dim, hidden_dim)
    if gs < 64:
        overhead = 4 / gs  # fp32 scale per group of int8
        print(
            f"WARNING: dim={p.dim}, hidden_dim={hidden_dim} forces group_size={gs} "
            f"(scale overhead {overhead:.0%} of weight bytes). "
            f"For compact models keep both dims multiples of 64 "
            f"(train.py: --multiple_of=64)."
        )
    version2_export(model, args.output, group_size=gs)
    size = os.path.getsize(args.output)
    print(f"group_size={gs}, wrote {size} bytes ({size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
