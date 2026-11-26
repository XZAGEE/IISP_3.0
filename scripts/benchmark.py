import argparse
import json
import statistics
import time
from pathlib import Path

import requests


def percentile(values, percent):
    values = sorted(values)

    position = (len(values) - 1) * percent / 100

    lower = int(position)
    upper = min(lower + 1, len(values) - 1)

    weight = position - lower

    return (
        values[lower] * (1 - weight)
        + values[upper] * weight
    )


def summarize(values):
    return {
        "mean": round(statistics.mean(values), 3),
        "p50": round(percentile(values, 50), 3),
        "p95": round(percentile(values, 95), 3),
    }


def send_request(session, url, audio_path):
    with audio_path.open("rb") as audio:
        start = time.perf_counter()

        response = session.post(
            f"{url.rstrip('/')}/predict",
            files={
                "file": (
                    audio_path.name,
                    audio,
                    "audio/wav",
                )
            },
            timeout=30,
        )

        client_latency = (
            time.perf_counter() - start
        ) * 1000

    response.raise_for_status()

    result = response.json()

    return client_latency, float(result["latency_ms"])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--url",
        default="http://localhost:8000",
    )

    parser.add_argument(
        "--file",
        default="samples/yes.wav",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--out",
        default="metrics/latency.json",
    )

    args = parser.parse_args()

    if args.n < 1:
        parser.error("--n must be at least 1")

    if args.warmup < 0:
        parser.error("--warmup cannot be negative")

    audio_path = Path(args.file)

    client_times = []
    server_times = []

    with requests.Session() as session:
        print(f"Warmup: {args.warmup} requests")

        for _ in range(args.warmup):
            send_request(
                session,
                args.url,
                audio_path,
            )

        print(f"Benchmark: {args.n} requests")

        for index in range(args.n):
            client_time, server_time = send_request(
                session,
                args.url,
                audio_path,
            )

            client_times.append(client_time)
            server_times.append(server_time)

            if (index + 1) % 10 == 0:
                print(f"Completed: {index + 1}/{args.n}")

    report = {
        "requests": args.n,
        "warmup": args.warmup,
        "latency_ms": summarize(client_times),
        "server_latency_ms": summarize(server_times),
    }

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=4))
    print(f"Report saved: {output}")


if __name__ == "__main__":
    main()