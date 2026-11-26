import argparse
import json
import os
from pathlib import Path


def read_json(path):
    if not path.exists():
        return {}

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metrics",
        default="metrics",
    )

    parser.add_argument(
        "--training",
        default="artifacts/metrics.json",
    )

    parser.add_argument(
        "--out",
        default="metrics/report.md",
    )

    args = parser.parse_args()

    metrics_dir = Path(args.metrics)

    training = read_json(Path(args.training))
    verify = read_json(metrics_dir / "verify.json")
    benchmark = read_json(metrics_dir / "latency.json")

    latency = benchmark.get("latency_ms", {})
    server_latency = benchmark.get(
        "server_latency_ms",
        {},
    )

    accuracy = training.get("accuracy_percent")

    if accuracy is None and "accuracy" in training:
        accuracy = round(training["accuracy"] * 100, 2)

    commit = os.getenv("GITHUB_SHA", "local")[:7]
    event = os.getenv("GITHUB_EVENT_NAME", "local")

    min_accuracy = os.getenv("MIN_ACCURACY")

    if accuracy is not None and min_accuracy is not None:
        quality_gate = (
            "PASS"
            if accuracy >= float(min_accuracy)
            else "FAIL"
        )
    else:
        quality_gate = "N/A"

    lines = [
        "# ML Pipeline Report",
        "",
        "## Run information",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| Commit | {commit} |",
        f"| Event | {event} |",
        f"| Epochs | {training.get('epochs', 'N/A')} |",
        f"| Seed | {training.get('seed', 'N/A')} |",
        "",
        "## Training results",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Accuracy (%) | {accuracy if accuracy is not None else 'N/A'} |",
        f"| Model size (KB) | {training.get('model_size_kb', 'N/A')} |",
        f"| Train samples | {training.get('train_samples', 'N/A')} |",
        f"| Test samples | {training.get('test_samples', 'N/A')} |",
        f"| Minimum accuracy (%) | {min_accuracy or 'N/A'} |",
        f"| Accuracy gate | {quality_gate} |",
        "",
        "## Golden samples",
        "",
        f"Passed: {verify.get('passed', 'N/A')}/{verify.get('total', 'N/A')}",
        "",
        "| File | Expected | Predicted | Result |",
        "|---|---|---|---|",
    ]

    for sample in verify.get("samples", []):
        status = "PASS" if sample["passed"] else "FAIL"

        lines.append(
            f"| {sample['file']} "
            f"| {sample['expected']} "
            f"| {sample['predicted']} "
            f"| {status} |"
        )

    lines.extend([
        "",
        "## Latency benchmark",
        "",
        "| Metric | Client (ms) | Server (ms) |",
        "|---|---|---|",
        f"| Mean | {latency.get('mean', 'N/A')} | {server_latency.get('mean', 'N/A')} |",
        f"| P50 | {latency.get('p50', 'N/A')} | {server_latency.get('p50', 'N/A')} |",
        f"| P95 | {latency.get('p95', 'N/A')} | {server_latency.get('p95', 'N/A')} |",
        "",
        f"Requests: {benchmark.get('requests', 'N/A')}",
        f"Warmup: {benchmark.get('warmup', 'N/A')}",
        "",
    ])

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"Report saved: {output}")


if __name__ == "__main__":
    main()