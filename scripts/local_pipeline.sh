#!/usr/bin/env bash

set -e
set -o pipefail

EPOCHS="${1:-3}"
MIN_ACCURACY="${2:-60}"

PORT=8013
CONTAINER=kws-lab3
RESULT_DIR="results/cnn_v1.3_${EPOCHS}epochs"

cleanup() {
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
}

trap cleanup EXIT

cleanup

mkdir -p artifacts metrics "$RESULT_DIR"

rm -f artifacts/model.pt \
      artifacts/labels.json \
      artifacts/metrics.json \
      artifacts/train.log

rm -f metrics/verify.json \
      metrics/latency.json \
      metrics/report.md \
      metrics/service.log \
      metrics/model-info.json

echo "========== TRAIN =========="

MSYS_NO_PATHCONV=1 docker run --rm \
    -v "$(pwd -W)/artifacts:/artifacts" \
    kws-trainer:local \
    --classes yes no up down \
    --epochs "$EPOCHS" \
    --seed 42 \
    --dataset ci_dataset \
    --out /artifacts \
    2>&1 | tee artifacts/train.log

echo "========== BUILD RUNTIME =========="

docker build --target runtime -t kws-api:local .

echo "========== START API =========="

docker run -d \
    --name "$CONTAINER" \
    -p "$PORT:8000" \
    kws-api:local

URL="http://localhost:$PORT"

READY=0

for i in $(seq 1 30); do
    if curl -fsS "$URL/health"; then
        READY=1
        break
    fi

    sleep 2
done

if [ "$READY" -ne 1 ]; then
    docker logs "$CONTAINER"
    exit 1
fi

echo
echo "========== MODEL INFO =========="

curl -fsS "$URL/model-info" \
    | tee metrics/model-info.json

echo
echo "========== GOLDEN SAMPLES =========="

VERIFY_EXIT=0

python scripts/verify_model.py \
    --url "$URL" \
    --expected samples/expected.json \
    --out metrics/verify.json \
    || VERIFY_EXIT=$?

echo "========== ACCURACY GATE =========="

ACCURACY_EXIT=0

python -c "
import json
data = json.load(open('metrics/model-info.json'))
accuracy = data['accuracy']
threshold = float('$MIN_ACCURACY')
print(f'Accuracy: {accuracy}%')
print(f'Threshold: {threshold}%')
raise SystemExit(0 if accuracy >= threshold else 1)
" || ACCURACY_EXIT=$?

echo "========== BENCHMARK =========="

python scripts/benchmark.py \
    --url "$URL" \
    --n 100 \
    --warmup 10 \
    --out metrics/latency.json

echo "========== REPORT =========="

docker logs "$CONTAINER" \
    > metrics/service.log 2>&1

MIN_ACCURACY="$MIN_ACCURACY" \
python scripts/make_report.py \
    --metrics metrics/ \
    --out metrics/report.md

cp artifacts/metrics.json "$RESULT_DIR/"
cp artifacts/train.log "$RESULT_DIR/"
cp metrics/* "$RESULT_DIR/"

cat "$RESULT_DIR/report.md"

echo "========== FINAL RESULT =========="

echo "Golden sample exit code: $VERIFY_EXIT"
echo "Accuracy gate exit code: $ACCURACY_EXIT"

if [ "$VERIFY_EXIT" -ne 0 ] || [ "$ACCURACY_EXIT" -ne 0 ]; then
    echo "LOCAL PIPELINE FAILED"
    exit 1
fi

echo "LOCAL PIPELINE PASSED"