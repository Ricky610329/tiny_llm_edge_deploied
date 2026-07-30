"""Fetch the first N shards of TinyStories without downloading the full 1.6GB tar.

Streams the tar.gz from HuggingFace and stops as soon as N .json shards are
extracted (~35MB each). Two shards are enough for the workshop: tinystories.py
uses whatever shards exist, and train.py needs at least two (shard 0 becomes
the validation split).

Usage:
    uv run python tools/fetch_shards.py --n 2
"""
import argparse
import os
import tarfile

import requests

URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--out", default=os.path.join("train", "llama2.c", "data", "TinyStories_all_data"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    have = [f for f in os.listdir(args.out) if f.endswith(".json")]
    if len(have) >= args.n:
        print(f"already have {len(have)} shards in {args.out}, skipping")
        return

    resp = requests.get(URL, stream=True, timeout=60)
    resp.raise_for_status()
    got = 0
    with tarfile.open(fileobj=resp.raw, mode="r|gz") as tar:
        for member in tar:
            if not member.name.endswith(".json"):
                continue
            print(f"extracting {member.name} ({member.size / 1e6:.0f} MB)")
            try:
                tar.extract(member, args.out, filter="data")
            except TypeError:  # Python < 3.12 has no filter=
                tar.extract(member, args.out)
            got += 1
            if got >= args.n:
                break
    resp.close()
    print(f"done: {got} new shards in {args.out}")


if __name__ == "__main__":
    main()
