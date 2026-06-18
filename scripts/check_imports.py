#!/usr/bin/env python3
"""Smoke-import vlc subpackages."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "legacy"))

MODULES = [
    "vlc",
    "vlc.core",
    "vlc.data",
    "vlc.baseline",
    "vlc.policy",
    "vlc.bench",
    "vlc.features",  # flat compat
    "vlc.model",
    "vlc.trainer",
]

CLI_MODULES = [
    "vlc.policy.cli",
    "vlc.bench.same_k",
    "vlc.bench.compare_backbones",
    "vlc.bench.dsprites",
    "vlc.policy.ablation_episode_size",
    "vlc.policy.train_llm_vlm",
    "vlc.legacy",
]


def main():
    failed = []
    for name in MODULES:
        try:
            importlib.import_module(name)
            print(f"  {name}: OK")
        except Exception as e:
            print(f"  {name}: FAIL {e}")
            failed.append(name)
    for name in CLI_MODULES:
        try:
            importlib.import_module(name)
            print(f"  {name}: OK")
        except Exception as e:
            print(f"  {name}: FAIL {e}")
            failed.append(name)

    from vlc.data.features import audit_task_data_splits
    from vlc.policy import ClusteringActionPolicy, build_backbone
    from vlc.bench.same_k import run_same_k_suite
    from vlc.main import main as vlc_main

    assert vlc_main([]) == 0
    import numpy as np
    from vlc.pseudo_labels import get_supervision_labels

    y = get_supervision_labels("object", "cifar10", "./data", np.zeros(5, dtype=int), True)
    assert len(y) == 5
    import vlc._legacy as legacy_shim

    assert legacy_shim.load_legacy_lite_module is not None
    print("  symbols: OK")
    if failed:
        sys.exit(1)
    print("\nAll imports OK.")


if __name__ == "__main__":
    main()
