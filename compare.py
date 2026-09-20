"""Paced Jev versus DeepSeek evaluation: --pilot selects two examples per intent."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

from banking77_jev import (
    ENDPOINT, INSTRUCTIONS, load_banking_data, load_records,
    jev_credentials, payload, request_json, run_lock, write_json,
)

DEEPSEEK_ENDPOINT = "https://api.deepseek.com/chat/completions"


def select_indices(dataset, pilot=False, seed=42):
    if not pilot:
        return list(range(len(dataset)))
    groups = {i: [] for i in range(len(dataset.features["label"].names))}
    for index, row in enumerate(dataset):
        groups[row["label"]].append(index)
    rng = random.Random(seed)
    selected = []
    for indices in groups.values():
        if len(indices) < 2:
            raise ValueError("Pilot requires at least two examples per class.")
        selected.extend(rng.sample(indices, 2))
    rng.shuffle(selected)
    return selected


def deepseek_payload(text, labels, model):
    criteria = payload("", labels, model)["questions"]["intent"]["criteria"]
    return {
        "model": model, "thinking": {"type": "disabled"}, "temperature": 0,
        "max_tokens": 128, "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": INSTRUCTIONS +
             '\nReturn only JSON: {"label": "one exact category key"}.\nCategories:\n' +
             json.dumps(criteria, ensure_ascii=False)},
            {"role": "user", "content": text},
        ],
    }


def parse_prediction(provider, data, labels):
    raw = None
    try:
        if provider == "jev":
            answer = data["answers"]["intent"]
            choice = answer["choice"]
            raw = answer
        else:
            completion = data["choices"][0]
            raw = completion["message"]["content"]
            if completion.get("finish_reason") != "stop":
                raise ValueError("Incomplete completion")
            choice = json.loads(raw)["label"]
        if not isinstance(choice, str) or choice not in labels:
            raise ValueError("Unknown label")
        return {"predicted": choice, "valid": True, "raw_answer": raw}
    except (KeyError, IndexError, TypeError, ValueError):
        # Invalid model answers count as errors, without extra chances at the task.
        return {"predicted": None, "valid": False, "raw_answer": raw}


def normalized_usage(provider, data):
    usage = data.get("usage", {})
    if provider == "jev":
        return {"input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"), "cached_tokens": 0}
    return {"input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "cached_tokens": usage.get("prompt_cache_hit_tokens")}


def estimate_cost(provider, usage, prices):
    rate = prices[provider]
    incoming, outgoing, cached = (usage[k] for k in ("input_tokens", "output_tokens", "cached_tokens"))
    if rate["input"] is None or incoming is None:
        return None
    if provider == "jev":
        return incoming * rate["input"] / 1_000_000
    if any(v is None for v in (rate["output"], rate["cached"], outgoing, cached)):
        return None
    return ((incoming - cached) * rate["input"] + cached * rate["cached"] +
            outgoing * rate["output"]) / 1_000_000


def execute(args):
    import httpx
    from plot_comparison import create_report

    jev_key, jev_endpoint = jev_credentials()
    keys = {"jev": jev_key,
            "deepseek": os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_KEY")}
    if not args.dry_run and not all(keys.values()):
        raise RuntimeError("Set TYPESAFE_API_KEY (or JEV_API_KEY) and DEEPSEEK_API_KEY.")
    dataset = load_banking_data(args.split, args.revision)
    labels = dataset.features["label"].names
    indices = select_indices(dataset, args.pilot, args.seed)
    models = {"jev": args.jev_model, "deepseek": args.deepseek_model}
    prices = {"jev": {"input": args.jev_input_price},
              "deepseek": {"input": args.deepseek_input_price,
                           "cached": args.deepseek_cached_price, "output": args.deepseek_output_price}}
    config = {"format_version": 1, "dataset": "PolyAI/banking77", "split": args.split,
              "revision": args.revision, "fingerprint": dataset._fingerprint,
              "pilot": args.pilot, "seed": args.seed, "indices": indices, "labels": labels,
              "models": models, "prices_usd_per_million": prices,
              "templates": {"jev": payload("", labels, models["jev"]),
                            "deepseek": deepseek_payload("", labels, models["deepseek"])}}
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if args.dry_run:
        count = len(indices)
        print(json.dumps({"samples_per_model": count, "total_requests": 2 * count,
                          "pilot": args.pilot, "models": models,
                          "minimum_wait_seconds": 2 * (max(0, count - 1) * args.delay +
                              max(0, (count - 1) // args.batch_size) * args.batch_delay),
                          "indices": indices if args.pilot else indices[:10]}, indent=2))
        return
    output = args.output or Path("results") / ("pilot" if args.pilot else f"full-{args.split}")
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output), ExitStack() as stack:
        manifest_path = output / "comparison.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text())["signature"] != signature:
                raise RuntimeError("Run settings differ. Use a new --output folder.")
        else:
            if any(output.glob("*.jsonl")):
                raise RuntimeError("Predictions exist without a comparison manifest.")
            write_json(manifest_path, {"signature": signature, **config})
        records = {name: load_records(output / f"{name}.jsonl") for name in models}
        clients = {name: stack.enter_context(httpx.Client(
            headers={"Authorization": f"Bearer {keys[name]}"}, timeout=args.timeout,
            follow_redirects=False)) for name in models}
        files = {name: stack.enter_context((output / f"{name}.jsonl").open("a", encoding="utf-8"))
                 for name in models}
        completed_this_run = {name: 0 for name in models}
        next_batch_pause = {name: args.batch_size for name in models}
        selected = set(indices)
        for name in models:
            if not set(records[name]).issubset(selected):
                raise RuntimeError("Checkpoint contains samples outside this selection.")
        def call_provider(provider, index):
            row = dataset[index]
            body = (payload if provider == "jev" else deepseek_payload)(
                row["text"], labels, models[provider])
            data = request_json(clients[provider], body, args.retries, args.delay,
                                jev_endpoint if provider == "jev" else DEEPSEEK_ENDPOINT)
            usage = normalized_usage(provider, data)
            return {"index": index, "text": row["text"], "expected": labels[row["label"]],
                    **parse_prediction(provider, data, labels),
                    "provider": provider, "model": data.get("model"), "usage": usage,
                    "estimated_cost_usd": estimate_cost(provider, usage, prices),
                    **data["_timing"]}

        pending_indices = [i for i in indices if any(i not in records[p] for p in models)]
        try:
            with ThreadPoolExecutor(max_workers=args.concurrency * len(models)) as executor:
                for start in range(0, len(pending_indices), args.concurrency):
                    chunk = pending_indices[start:start + args.concurrency]
                    tasks = [(provider, index) for index in chunk for provider in models
                             if index not in records[provider]]
                    if not tasks:
                        continue
                    wait = args.delay if any(completed_this_run.values()) else 0.0
                    for provider in {provider for provider, _ in tasks}:
                        if completed_this_run[provider] >= next_batch_pause[provider]:
                            wait = max(wait, args.delay + args.batch_delay)
                            while completed_this_run[provider] >= next_batch_pause[provider]:
                                next_batch_pause[provider] += args.batch_size
                    if wait:
                        time.sleep(wait)
                    futures = {executor.submit(call_provider, provider, index): provider
                               for provider, index in tasks}
                    first_error = None
                    for future in as_completed(futures):
                        provider = futures[future]
                        try:
                            record = future.result()
                        except Exception as exc:
                            first_error = first_error or exc
                            continue
                        record["pacing_wait_seconds"] = wait / len(tasks)
                        files[provider].write(json.dumps(record, ensure_ascii=False) + "\n")
                        files[provider].flush()
                        os.fsync(files[provider].fileno())
                        records[provider][record["index"]] = record
                        completed_this_run[provider] += 1
                        print(f"{provider}: {len(records[provider])}/{len(indices)} "
                              f"correct={record['predicted'] == record['expected']} "
                              f"api={record['api_seconds']:.2f}s", flush=True)
                    if first_error:
                        raise first_error
        finally:
            create_report(output)


def main():
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).with_name(".env"), override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="2 samples per class = 154 per model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--revision", default="master")
    parser.add_argument("--jev-model", default="jev-latest")
    parser.add_argument("--deepseek-model", default="deepseek-v4-pro")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=3,
                        help="Examples per provider processed concurrently")
    parser.add_argument("--delay", type=float, default=0.3)
    parser.add_argument("--batch-delay", type=float, default=2)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--jev-input-price", type=float, help="USD per 1M input tokens")
    parser.add_argument("--deepseek-input-price", type=float, help="USD per 1M cache-miss tokens")
    parser.add_argument("--deepseek-cached-price", type=float, help="USD per 1M cache-hit tokens")
    parser.add_argument("--deepseek-output-price", type=float, help="USD per 1M output tokens")
    args = parser.parse_args()
    numbers = [args.delay, args.batch_delay, args.timeout, args.jev_input_price,
               args.deepseek_input_price, args.deepseek_cached_price, args.deepseek_output_price]
    if any(v is not None and (not math.isfinite(v) or v < 0) for v in numbers):
        parser.error("Durations and prices must be finite and nonnegative.")
    if args.timeout == 0 or args.batch_size < 1 or args.concurrency < 1 or args.retries < 0:
        parser.error("timeout/batch-size/concurrency must be positive; retries must be nonnegative.")
    try:
        execute(args)
    except KeyboardInterrupt:
        print("Stopped. Rerun the same command to resume.", file=sys.stderr)
        return 130
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
