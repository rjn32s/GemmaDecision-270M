# GemmaDecision-270M v0.4.0 — JevBench evaluation package

This package serves the published full-backbone model through JevBench's existing
`typesafe` HTTP adapter. It adds no training, retrieval, benchmark-specific rules,
fallback model or generated explanation. Public measurements are diagnostics;
an official score/rank requires an independent maintainer evaluation.

## Frozen protocol

- Weights: `rajan2k/GemmaDecision-270M`, exact revision
  `785d530221c990671f29976902540101bb9c7647` (v0.4.0).
- JevBench: `d06ee95988da1350eb8ae5511daa0e06bfffe911` (v1.4.2.1).
- Public order: `easy.jsonl` (48), `original.jsonl` (72), `hard.jsonl` (111).
- Every candidate is scored separately with the same complete state and question,
  using the shipped `GemmaJointRanker`. No task IDs, expected answers, family,
  provenance, split or group are supplied to inference.
- Probabilities are **adapter-derived** `softmax(raw_scores / 4.136820402388508)`.
  This temperature was fitted on 300 project calibration examples before this
  benchmark. No JevBench fitting is performed. The public `rank()` interface
  continues to return raw scores; it does not acquire a probability guarantee.
- Candidate wording follows the shipped CLM schema. Choice descriptions and score
  levels are preserved. Noul uses the schema's `false`/`true` candidate descriptions
  and returns `P(yes)`. The upstream adapter calls this transport `native`; that
  field does not mean a separately trained binary probability head.
- Published limits remain 2,048 state/question tokens, 768 candidate tokens and
  2–64 distinct candidates. Refusals return HTTP 422. No silent truncation,
  retried decisions, question-specific thresholds or post-result protocol changes.
- Public reference environment: Modal H100, two CPU cores, 32 GiB RAM; BF16 encoder
  and FP32 head, serial loopback HTTP. CPU uses FP32. No inference warmup;
  first-call latency is retained. Download/load time is reported separately.
- Unmodified upstream `Runner`, scoring and summary code; upstream stop rules
  retained. Unattempted items, attempted failures and valid predictions are
  reported separately. No official composite is calculated locally.

`PROTOCOL.json` records source/data hashes and was published before this v4 run.
See [EXPOSURE.md](EXPOSURE.md) for earlier public-benchmark exposure and provenance.

## Evaluator quickstart

Python 3.11 is the recorded environment. Clone the preregistered integration and
install model dependencies. No token is required to download this public model.

```sh
git clone --branch jevbench-v4-protocol-1 https://github.com/rjn32s/GemmaDecision-270M.git gemmadecision
cd gemmadecision
python -m venv .venv
. .venv/bin/activate
python -m pip install -r benchmark/requirements-server.txt
hf download rajan2k/GemmaDecision-270M --revision 785d530221c990671f29976902540101bb9c7647 --local-dir ../gemmadecision-model
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python benchmark/server.py --model-dir ../gemmadecision-model --device cpu --torch-threads 2 --port 8000
```

Choose `--device cuda` for a CUDA GPU. The server listens on loopback by default,
loads local files only, and suppresses HTTP request logs. After the
download, inference requires no outbound API access. `GET /health` confirms
readiness and model identity. Stop the server with Ctrl+C when finished.

From a second terminal, run the public suite:

```sh
cd gemmadecision
. .venv/bin/activate
git clone https://github.com/fstandhartinger/jevbench.git ../jevbench
git -C ../jevbench checkout d06ee95988da1350eb8ae5511daa0e06bfffe911
python benchmark/run_public.py --harness-dir ../jevbench --output-dir ../gemmadecision-public-run --endpoint http://127.0.0.1:8000
```

The output directory must be new and outside the JevBench checkout. The script
uses the unchanged `TypeSafeAdapter` and `Runner`; it publishes no external data.
It writes per-item records, raw public requests/responses, a manifest, a summary,
and an unattempted list. For private or sealed evaluation, maintainers should use
their own unchanged CLI and keep raw files private:

```sh
PYTHONPATH=../jevbench python -m jevbench.cli run \
  --tasks /absolute/path/to/evaluator-tasks.jsonl \
  --adapter typesafe --endpoint http://127.0.0.1:8000 --key-env '' \
  --model rajan2k/GemmaDecision-270M@785d530221c990671f29976902540101bb9c7647 \
  --results /absolute/private/run/results.jsonl \
  --raw-dir /absolute/private/run/raw --ledger /absolute/private/run/ledger.jsonl \
  --cap-usd 20 --reserve-usd 0.02
```

The harness's $0.02 reservation is an accounting allowance for an unknown tariff,
**not an actual API bill**. This model has no advertised hosted inference tariff;
per-decision tariff remains unknown/null. Modal experiment compute is reported
separately. A maintainer-selected reference tariff must be labelled an estimate.
The token count includes repeated state/question tokens for every separately
encoded candidate; generated output tokens are zero.

## Reproduction and terms

Run lightweight contract checks; no model is loaded by these tests:

```sh
python benchmark/test_server.py --model-dir ../gemmadecision-model --harness-dir ../jevbench
JEVBENCH_TEST_HARNESS=../jevbench python -m unittest discover -s benchmark -p 'test_run_public.py'
```

`modal_benchmark.py` is the
author's budget-checked cloud launcher and references the existing project volume;
other evaluators should use the portable commands above or adapt their own cloud
deployment. Model weights retain Gemma terms; source code retains Apache-2.0.
Neither upstream authors nor benchmark maintainers endorse this submission.
