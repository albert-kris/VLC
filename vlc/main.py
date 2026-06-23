#!/usr/bin/env python3
"""VLC 包根入口：子命令分发到各模块的 main()。"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable

COMMANDS: dict[str, tuple[str, str, str]] = {
    # ── Core pipeline ──────────────────────────────────────────────────────────
    "train-encoder": (
        "标准→条件编码器 SupCon 训练（LoRA + 投影头）",
        "vlc.model.encoder_trainer",
        "main",
    ),
    "train-unsup": (
        "无监督训练：DEC KL锐化 或 CC对比聚类（无标签）",
        "vlc.model.unsup_trainer",
        "main",
    ),
    "train-rl": (
        "SwAV-REINFORCE DDP 训练（Sinkhorn reward + PPO，双卡）",
        "vlc.train.swav_reinforce_ddp_trainer",
        "main",
    ),
    "cluster": (
        "推理：编码图像集 + KMeans → cluster assignments",
        "vlc.embed.cluster",
        "main",
    ),
    "eval": (
        "迁移评测：对比 VLC 编码器与 CLIP-KMeans",
        "vlc.bench.transfer",
        "main",
    ),
    # ── Utilities ──────────────────────────────────────────────────────────────
    "build-episodes": (
        "（工具）从数据集构建 episode JSON（供分析用）",
        "vlc.episodes.catalog",
        "main",
    ),
    "export-tables": (
        "（工具）导出论文表格到 artifacts/TABLES.md",
        "vlc.bench.export_tables",
        "main",
    ),
}


def _dispatch(command: str, argv: list[str]) -> int:
    if command not in COMMANDS:
        print(f"未知子命令: {command}", file=sys.stderr)
        _print_commands(file=sys.stderr)
        return 2
    _desc, mod_path, fn_name = COMMANDS[command]
    mod = importlib.import_module(mod_path)
    fn: Callable[..., object] = getattr(mod, fn_name)
    fn(argv)
    return 0


def _print_commands(*, file=None) -> None:
    print("子命令:", file=file)
    for name, (desc, _mod, _fn) in COMMANDS.items():
        print(f"  {name:<22} {desc}", file=file)


def build_parser() -> argparse.ArgumentParser:
    epilog = "\n".join(
        ["子命令:"]
        + [f"  {name:<22} {desc}" for name, (desc, _, _) in COMMANDS.items()]
        + ["", "示例: python -m vlc train-encoder --config configs/vlm/encoder_cifar10.yaml"]
    )
    p = argparse.ArgumentParser(
        prog="python -m vlc",
        description="VLC 主实验 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    p.add_argument("command", nargs="?", help="子命令名")
    p.add_argument("args", nargs=argparse.REMAINDER, help="传给子命令的参数")
    return p


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw or raw[0] in ("-h", "--help"):
        build_parser().print_help()
        return 0

    command, rest = raw[0], raw[1:]
    if command.startswith("-"):
        build_parser().print_help()
        return 0

    return _dispatch(command, rest)


if __name__ == "__main__":
    raise SystemExit(main())
