"""Generate comparison JSON and PNG/SVG charts from saved predictions; no API calls."""

import argparse
import json
import os
from pathlib import Path
import statistics
import tempfile

from banking77_jev import metrics, write_json


def read_rows(path):
    if not path.exists():
        return {}
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.endswith("\n"):
                break
            row = json.loads(line)
            if row["index"] in rows:
                raise ValueError("Duplicate prediction index")
            rows[row["index"]] = row
    return rows


def percentile(values, q):
    ordered = sorted(values)
    at = (len(ordered) - 1) * q
    lo = int(at)
    return ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (at - lo)


def summarize(rows, labels, target):
    rows = list(rows)
    # The legacy metric helper expects known input-token counts.
    normalized = [{**r, "usage": {"input_tokens": r["usage"].get("input_tokens") or 0}} for r in rows]
    result = metrics(normalized, labels, target)
    latency = [r["api_seconds"] for r in rows]
    costs = [r["estimated_cost_usd"] for r in rows]
    result.update({
        "correct": sum(r["predicted"] == r["expected"] for r in rows),
        "evaluated": len(rows),
        "invalid_rate": sum(not r["valid"] for r in rows) / len(rows) if rows else None,
        "api_mean_seconds": statistics.mean(latency) if rows else None,
        "api_p50_seconds": percentile(latency, 0.5) if rows else None,
        "api_p95_seconds": percentile(latency, 0.95) if rows else None,
        "api_total_seconds": sum(latency),
        "retry_wait_seconds": sum(r["retry_wait_seconds"] for r in rows),
        "pacing_wait_seconds": sum(r["pacing_wait_seconds"] for r in rows),
        "recorded_attempts": sum(r["attempts"] for r in rows),
        "recorded_output_tokens": sum(r["usage"].get("output_tokens") or 0 for r in rows),
        "missing_usage_records": sum(r["usage"].get("input_tokens") is None for r in rows),
        "estimated_cost_usd": sum(costs) if rows and all(c is not None for c in costs) else None,
        "returned_models": sorted({r["model"] for r in rows if r.get("model")}),
    })
    if not rows:
        result["macro_f1_all_classes"] = None
    return result


def create_report(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "comparison.json").read_text())
    labels = manifest["labels"]
    selected = set(manifest["indices"])
    records = {p: read_rows(folder / f"{p}.jsonl") for p in ("jev", "deepseek")}
    for provider, rows in records.items():
        if not set(rows).issubset(selected):
            raise ValueError(f"Unexpected indices in {provider} checkpoint")
    paired = sorted(set(records["jev"]) & set(records["deepseek"]))
    for index in paired:
        a, b = records["jev"][index], records["deepseek"][index]
        if a["text"] != b["text"] or a["expected"] != b["expected"]:
            raise ValueError("Paired records contain different text or ground truth")
    report = {"selected_samples": len(selected), "paired_samples": len(paired),
              "synthetic": manifest.get("synthetic", False),
              "requested_models": manifest.get("models", {}),
              "complete": len(paired) == len(selected),
              "prices_usd_per_million": manifest["prices_usd_per_million"],
              "all_completed": {p: summarize(r.values(), labels, len(selected)) for p, r in records.items()},
              "paired": {p: summarize((r[i] for i in paired), labels, len(selected)) for p, r in records.items()},
              "notes": ["Charts use only paired completed samples; invalid answers count as incorrect.",
                        "API time includes HTTP attempts, excludes retry and pacing sleeps.",
                        "Costs use supplied fixed rates and recorded responses, not invoices.",
                        "Lost responses and exhausted requests may have unrecorded usage/time.",
                        "Tokenizers differ; token counts alone are not a cost comparison."]}
    write_json(folder / "summary.json", report)
    plot_report(folder, report, labels)
    print(f"Report: {folder / 'summary.json'}; paired {len(paired)}/{len(selected)}", flush=True)
    return report


def plot_report(folder, report, labels):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "banking77-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    providers = ["jev", "deepseek"]
    names = ["Jev", "DeepSeek"]
    colors = ["#14877c", "#b74465"]
    title = f"BANKING77 | {report['paired_samples']}/{report['selected_samples']} paired samples"
    if report["synthetic"]:
        title = "SYNTHETIC TEST DATA | " + title
    if not report["complete"]:
        title += " | INCOMPLETE"
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    panels = [("accuracy", "Accuracy", "%", 100),
              ("macro_f1_all_classes", "Macro F1 (all classes)", "%", 100),
              ("invalid_rate", "Invalid answers", "%", 100),
              ("api_p50_seconds", "Median API time (incl. attempts)", "seconds", 1),
              ("api_p95_seconds", "P95 API time (incl. attempts)", "seconds", 1),
              ("estimated_cost_usd", "Estimated cost (recorded responses)", "USD", 1)]
    for ax, (key, name, unit, scale) in zip(axes.flat, panels):
        values = [report["paired"][p][key] for p in providers]
        heights = [v * scale if v is not None else 0 for v in values]
        bars = ax.bar(names, heights, color=colors, width=0.5)
        for provider, bar, value in zip(providers, bars, values):
            if value is None:
                label = "N/A"
            elif key == "accuracy":
                stats = report["paired"][provider]
                label = f"{value * 100:.1f}%\n({stats['correct']}/{stats['evaluated']})"
            elif unit == "%":
                label = f"{value * 100:.1f}%"
            elif unit == "seconds":
                label = f"{value:.2f} s"
            else:
                label = f"${value:.5f}"
            ax.annotate(label, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 5), textcoords="offset points", ha="center")
        ax.set_title(name, fontsize=11)
        ax.set_ylabel(unit)
        ax.set_ylim(0, 110 if unit == "%" else max(max(heights) * 1.3, 0.01))
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle(title, fontsize=15)
    for suffix in ("png", "svg"):
        fig.savefig(folder / f"comparison.{suffix}", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(16, 13), layout="constrained")
    for ax, subset in zip(axes, (labels[:39], labels[39:])):
        y = np.arange(len(subset))
        for offset, provider, color, name in zip((-0.18, 0.18), providers, colors, names):
            values = [report["paired"][provider]["per_class"][label]["f1"] for label in subset]
            bars = ax.barh(y + offset, values, height=0.34, color=color, label=name)
            for bar, value in zip(bars, values):
                ax.text(max(value + 0.012, 0.012), bar.get_y() + bar.get_height() / 2,
                        f"{value * 100:.0f}%", va="center", ha="left", fontsize=6,
                        color="#202020")
        ax.set_yticks(y, [label.replace("_", " ") for label in subset], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.18)
        ax.set_xlabel("F1 score (percentage shown at each bar)")
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(title + " | Per-class F1", fontsize=14)
    overall = []
    for provider, name in zip(providers, names):
        stats = report["paired"][provider]
        accuracy = "N/A" if stats["accuracy"] is None else (
            f"{stats['accuracy'] * 100:.1f}% ({stats['correct']}/{stats['evaluated']})")
        macro_f1 = "N/A" if stats["macro_f1_all_classes"] is None else (
            f"{stats['macro_f1_all_classes'] * 100:.1f}%")
        overall.append(f"{name}: Accuracy {accuracy}, Macro F1 {macro_f1}")
    fig.supxlabel("Overall paired results | " + "   |   ".join(overall), fontsize=10)
    for suffix in ("png", "svg"):
        fig.savefig(folder / f"per_class_f1.{suffix}", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Comparison result folder")
    args = parser.parse_args()
    create_report(args.output)


if __name__ == "__main__":
    main()
