"""Episode catalog: registry for dataset builders (used by build-episodes subcommand)."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Build episode datasets")
    p.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    p.add_argument("--out", default="artifacts/episodes")
    args = p.parse_args(argv)
    print(f"build-episodes for {args.dataset} -> {args.out} (stub)")
