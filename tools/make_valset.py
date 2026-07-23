"""Freeze a validation set for the quality leaderboard.

Takes the first N stories from a TinyStories-format file (documents separated
by <|endoftext|>), normalizes line endings, and writes them back in the same
format. The output file is committed to the repo and must never change during
a course run — it defines the benchmark.

Usage:
    uv run python tools/make_valset.py TinyStoriesV2-GPT4-valid.txt eval/validation_100.txt --n 100
"""
import argparse

DELIM = "<|endoftext|>"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read().replace("\r\n", "\n")

    stories = [s.strip() for s in text.split(DELIM)]
    stories = [s for s in stories if s][: args.n]
    if len(stories) < args.n:
        raise SystemExit(f"only found {len(stories)} stories, wanted {args.n}")

    with open(args.output, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"\n{DELIM}\n".join(stories))
        f.write(f"\n{DELIM}\n")

    total_bytes = sum(len(s.encode("utf-8")) for s in stories)
    print(f"wrote {len(stories)} stories, {total_bytes} story bytes -> {args.output}")


if __name__ == "__main__":
    main()
