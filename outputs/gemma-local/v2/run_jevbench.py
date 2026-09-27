"""Run the published adapter against a caller-provided JevBench task file."""
import argparse
from dataclasses import asdict
import json
from jevbench.tasks import load_jsonl
from jevbench_adapter import GemmaDecisionAdapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=".")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    adapter = GemmaDecisionAdapter(model=args.model, revision=args.revision, device=args.device)
    tasks = load_jsonl(args.tasks)
    # Exclusive output creation avoids overwriting an earlier benchmark run.
    with open(args.output, "x") as output:
        for index, task in enumerate(tasks):
            response = adapter.run(task)
            output.write(json.dumps({"task_id": task.id, **asdict(response)}) + "\n")
            output.flush()
            if (index + 1) % 25 == 0:
                print(f"Completed {index + 1}/{len(tasks)}", flush=True)


if __name__ == "__main__":
    main()
