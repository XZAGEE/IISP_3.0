import argparse
import json
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--url",
        default="http://localhost:8000",
    )

    parser.add_argument(
        "--expected",
        default="samples/expected.json",
    )

    parser.add_argument(
        "--out",
        default="metrics/verify.json",
    )

    args = parser.parse_args()

    expected_path = Path(args.expected)

    expected = json.loads(
        expected_path.read_text(encoding="utf-8")
    )

    results = []
    passed = 0

    for filename, expected_class in expected.items():
        audio_path = expected_path.parent / filename

        result = {
            "file": filename,
            "expected": expected_class,
            "predicted": None,
            "confidence": None,
            "passed": False,
        }

        try:
            with audio_path.open("rb") as audio:
                response = requests.post(
                    f"{args.url.rstrip('/')}/predict",
                    files={
                        "file": (
                            filename,
                            audio,
                            "audio/wav",
                        )
                    },
                    timeout=30,
                )

            response.raise_for_status()

            prediction = response.json()

            predicted_class = prediction["predicted_class"]

            is_correct = predicted_class == expected_class

            result.update({
                "predicted": predicted_class,
                "confidence": prediction.get("confidence"),
                "passed": is_correct,
            })

            if is_correct:
                passed += 1

        except (requests.RequestException, OSError, ValueError, KeyError) as exc:
            result["error"] = str(exc)

        results.append(result)

        status = "PASS" if result["passed"] else "FAIL"

        print(
            f"[{status}] {filename}: "
            f"expected={expected_class}, "
            f"predicted={result['predicted']}"
        )

    report = {
        "passed": passed,
        "total": len(expected),
        "failed": len(expected) - passed,
        "success": passed == len(expected),
        "samples": results,
    }

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print(f"\nGolden samples: {passed}/{len(expected)}")
    print(f"Report saved: {output}")

    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())