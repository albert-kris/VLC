"""Export transfer evaluation results as LaTeX / Markdown tables.

Reads artifacts/vlm/eval/transfer_{dataset}.json and renders:
  - Table 1: Per-criterion ACC / ARI / NMI for all methods
  - Table 2: Cross-criteria ARI summary

Usage: python -m vlc export-tables --eval-dir artifacts/vlm/eval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def render_method_table(
    results: dict,
    methods: list[str],
    criteria: list[str],
    metric: str = "acc",
) -> str:
    """Render a criterion × method table for one metric."""
    header = "| Criterion | " + " | ".join(m.replace("_", "-") for m in methods) + " |"
    sep = "|---" * (1 + len(methods)) + "|"
    rows = [header, sep]
    for crit in criteria:
        cells = [crit]
        for m in methods:
            r = results.get(crit, {}).get(m, {})
            if "error" in r or metric not in r:
                cells.append("-")
            else:
                cells.append(f"{r[metric]:.3f}")
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def render_full_table_latex(
    results: dict,
    methods: list[str],
    criteria: list[str],
) -> str:
    """Render full ACC/ARI/NMI LaTeX table."""
    n_cols = 1 + len(methods) * 3
    col_spec = "l" + "ccc" * len(methods)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    # Method header spanning 3 columns each
    method_header = "Criterion"
    for m in methods:
        method_header += f" & \multicolumn{{3}}{{c}}{{{m.replace('_', '-')}}}"
    lines.append(method_header + r" \\")
    lines.append(r"\cmidrule(lr){" + "2-4}" * len(methods))  # simplified

    # Metric sub-header
    metric_header = ""
    for _ in methods:
        metric_header += " & ACC & ARI & NMI"
    lines.append(metric_header + r" \\")
    lines.append(r"\midrule")

    # Data rows
    for crit in criteria:
        row = crit.replace("_", r"\_")
        for m in methods:
            r = results.get(crit, {}).get(m, {})
            if "error" in r:
                row += " & - & - & -"
            else:
                row += (
                    f" & {r.get('acc', 0):.3f}"
                    f" & {r.get('ari', 0):.3f}"
                    f" & {r.get('nmi', 0):.3f}"
                )
        lines.append(row + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\caption{VLC Transfer Evaluation}", r"\end{table}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Export VLM eval results as tables")
    p.add_argument("--eval-dir", default="artifacts/vlm/eval", help="Directory with transfer_*.json files")
    p.add_argument("--out", default="artifacts/TABLES_VLM.md", help="Output markdown path")
    p.add_argument("--latex-out", default="artifacts/TABLES_VLM.tex", help="Output LaTeX path")
    args = p.parse_args(argv)

    eval_dir = Path(args.eval_dir)
    md_sections = ["# VLC Transfer Evaluation Tables\n"]
    tex_sections = []

    for json_path in sorted(eval_dir.glob("transfer_*.json")):
        dataset = json_path.stem.replace("transfer_", "")
        with open(json_path, encoding="utf-8") as f:
            results = json.load(f)

        criteria = list(results.keys())
        methods = sorted({m for r in results.values() for m in r if "error" not in r.get(m, {})})

        if not methods:
            continue

        md_sections.append(f"## {dataset.upper()}\n")
        for metric in ["acc", "ari", "nmi"]:
            md_sections.append(f"### {metric.upper()}\n")
            md_sections.append(render_method_table(results, methods, criteria, metric))
            md_sections.append("")

        tex_sections.append(render_full_table_latex(results, methods, criteria))

    md_out = Path(args.out)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text("\n".join(md_sections), encoding="utf-8")
    print(f"Markdown tables → {md_out}")

    tex_out = Path(args.latex_out)
    tex_out.write_text("\n\n".join(tex_sections), encoding="utf-8")
    print(f"LaTeX tables    → {tex_out}")
