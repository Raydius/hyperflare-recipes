"""Passkey retrieval at chosen depths of prompts below and beyond a package's native context.

One serving container per run (`flare run PACKAGE`, history drafts off), or with `--base-url` an
instance served elsewhere (the console's), which the probe neither starts nor stops. Each cell of a
lengths × depths grid sends the frozen workload's service catalog, sized to the target prompt
tokens, with one passkey sentence spliced between catalog lines at the depth fraction, and asks
for the pass key under `top_k=1` (the runtime refuses `temperature=0`). A cell passes when the
reply contains the passkey. Cells run longest length first, depths ascending, so no prompt is a
prefix of an earlier one. `summary.json` holds every cell, a pass/fail grid, and `all_passed`.

`--task tracking` replaces the single needle with RULER's multi-hop variable tracking
(https://github.com/NVIDIA/RULER): a chain of assignment sentences, the first giving a variable a
five-digit value and each later one assigning the previous variable to a new one, spread through
the prompt in chain order from the cell's depth. The question names the last variable, and a cell
passes when the reply contains the first value, which no sentence states for that variable.
`--compare` pairs summaries of a native and a scaled package, cell by cell.

The defaults are the Qwen3.8-Flash-Next arm (`enable_thinking` off, 32 reply tokens, the Qwen-measured
`TOKENS_PER_RECORD`). The Muse Glimmer arm passes `--template-kwargs '{"reasoning_strength": "low"}'`
and `--max-tokens 256` because its template reasons in the `to=self` channel before content, and
`--tokens-per-record 56.54`, the pinned Glimmer tokenizer's rate on the catalog.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
from string import ascii_uppercase
import subprocess

from qualify.dflash2_long_context import TOKENS_PER_RECORD, draft_counts, measure, records_for
from qualify.dflash2_served import PORT, Server, container_command, healthy
from qualify.provenance import write_json
from qualify.speculative_workload import SAMPLING, _catalog, _message, value_sha

LENGTHS = (200_000, 320_000, 520_000, 1_000_000)
DEPTHS = (0.1, 0.5, 0.9)
QUESTION = "What is the pass key? The pass key is"
TASKS = ("passkey", "tracking")
HOPS = 4


def needle(passkey):
    return f"The pass key is {passkey}. Remember it. {passkey} is the pass key."


def passkey_for(seed, length, depth, records):
    """A five-digit passkey for one cell, redrawn while it occurs in that cell's catalog text."""
    generator = random.Random(f"{seed}:{length}:{depth}")
    catalog = _catalog(records)
    while True:
        candidate = f"{generator.randrange(10_000, 100_000)}"
        if candidate not in catalog:
            return candidate


def chain(seed, length, depth, records, passkey, hops=HOPS):
    """Variable names and assignment sentences of one tracking cell: the value, then `hops - 1` copies."""
    generator = random.Random(f"tracking:{seed}:{length}:{depth}")
    catalog, names = _catalog(records), []
    while len(names) < hops:
        name = "".join(generator.choices(ascii_uppercase, k=5))
        if name not in names and name not in catalog:
            names.append(name)
    sentences = [f"VAR {names[0]} = {passkey}."]
    sentences += [f"VAR {name} = VAR {previous}." for previous, name in zip(names, names[1:])]
    return names, sentences


def tracking_question(name):
    return f"What is the numeric value of VAR {name}? The value of VAR {name} is"


def prompt(records, depth, passkey, *, model, seed=0, template_kwargs=None, max_tokens=32, task="passkey",
           length=None):
    """The catalog with the needle at depth, then the pass-key question, as a benchmark case.

    `template_kwargs` replaces the default `{"enable_thinking": False}` (`add_generation_prompt`
    is always on) and `max_tokens` the 32-token reply budget. With `task="tracking"` the chain's
    sentences sit at even steps from `depth` to the end of the catalog and the question names the
    last variable; `length` then seeds the variable names.
    """
    lines = _catalog(records).split("\n")
    if task == "tracking":
        names, sentences = chain(seed, length, depth, records, passkey)
        haystack, start = [], 0
        for hop, sentence in enumerate(sentences):
            at = round((depth + hop * (1 - depth) / len(sentences)) * records)
            haystack += lines[start:at] + [sentence]
            start = at
        haystack, question = "\n".join(haystack + lines[start:]), tracking_question(names[-1])
    else:
        at = round(depth * records)
        haystack, question = "\n".join(lines[:at] + [needle(passkey)] + lines[at:]), QUESTION
    request = dict(model=model, messages=[
        _message("system", "Use only the supplied service records."),
        _message("user", haystack),
        _message("user", question)],
        tools=[], chat_template_kwargs={**(template_kwargs or {"enable_thinking": False}), "add_generation_prompt": True},
        **{**SAMPLING, "top_k": 1}, seed=seed, max_tokens=max_tokens, stream=True, stream_options={"include_usage": True},
        # No token-id echo: a prompt past the native context would render the first event beyond
        # the benchmark reader's 1 MiB line bound.
        return_token_ids=False, return_token_timestamps=False)
    return dict(id=f"{task}-{records}-{depth}", request=request, request_sha256=value_sha(request),
                prompt_tokens=[], expected_output_semantics={"kind": "text"})


def check(text, passkey):
    return passkey in text


def cell(observation, delta, *, length, depth, passkey, task="passkey"):
    analysis = observation["analysis"]
    usage = analysis.get("usage") or {}
    elapsed = observation.get("elapsed_ns")
    content = analysis.get("content") or ""
    return {"case": observation["case"], "task": task, "length": length, "depth": depth, "passkey": passkey,
            "prompt_tokens": usage.get("prompt_tokens"), "generated_tokens": usage.get("completion_tokens") or 0,
            "reply": content, "passed": check(content, passkey), "finish_reason": analysis.get("finish_reason"),
            "completed": "client_error" not in observation and analysis["checks"]["no_stream_error"]
                         and analysis["checks"]["one_valid_terminal"],
            "client_error": observation.get("client_error"),
            "request_seconds": elapsed / 1e9 if elapsed else None,
            "accepted_per_round": draft_counts(delta)["accepted_per_round"]}


def summary(cells):
    grid = {}
    for row in cells:
        grid.setdefault(str(row["length"]), {})[str(row["depth"])] = row["passed"]
    return {"cells": cells, "grid": grid, "all_passed": bool(cells) and all(row["passed"] for row in cells)}


def compare(pairs):
    """Pair each cell of a native summary with the scaled summary's cell of the same task, length and depth."""
    rows = []
    for native, scaled in pairs:
        other = {(row["task"], row["length"], row["depth"]): row for row in scaled["cells"]}
        for row in native["cells"]:
            match = other.get((row["task"], row["length"], row["depth"]))
            if match is not None:
                rows.append({"task": row["task"], "length": row["length"], "depth": row["depth"],
                             "native_passed": row["passed"], "scaled_passed": match["passed"],
                             "native_reply": row["reply"], "scaled_reply": match["reply"],
                             "replies_identical": row["reply"] == match["reply"]})
    return {"pairs": rows, "native_passed": sum(row["native_passed"] for row in rows),
            "scaled_passed": sum(row["scaled_passed"] for row in rows),
            "replies_identical": sum(row["replies_identical"] for row in rows)}


def _cells(args, base_url, *, measure):
    cells = []
    for length in sorted(args.lengths, reverse=True):
        records = records_for(length, args.tokens_per_record)
        for depth in sorted(args.depths):
            passkey = passkey_for(args.seed, length, depth, records)
            case = prompt(records, depth, passkey, model=args.model_id, seed=args.seed,
                          template_kwargs=args.template_kwargs, max_tokens=args.max_tokens,
                          task=args.task, length=length)
            folder = args.output / f"{length}-{depth}"
            observation, delta = measure(base_url, case, folder, timeout=args.timeout, model_id=args.model_id)
            row = cell(observation, delta, length=length, depth=depth, passkey=passkey, task=args.task)
            cells.append(row)
            print(json.dumps({k: row[k] for k in ("length", "depth", "prompt_tokens", "passed", "completed",
                                                  "request_seconds", "accepted_per_round")}), flush=True)
    return cells


def run(args, *, popen=subprocess.Popen, stop=subprocess.run, ready=healthy, measure=measure):
    args.output.mkdir(parents=True, exist_ok=True)
    if args.base_url:
        # An instance somebody else serves (the console's): no container of our own, nothing stopped at the end.
        if not ready(args.base_url):
            raise RuntimeError(f"{args.base_url} is not healthy")
        cells = _cells(args, args.base_url, measure=measure)
    else:
        base_url = f"http://127.0.0.1:{args.port}"
        command = container_command(args.image, args.package, name="long-context-passkey", uid=os.getuid(), gid=os.getgid(),
                                    home=Path.home(), publisher_keys=args.publisher_keys, trust_store=args.trust_store,
                                    port=args.port, history_drafts=False, max_context_tokens=args.max_context_tokens)
        with Server(command, "long-context-passkey", base_url, args.output / "server.log",
                    popen=popen, stop=stop, ready=ready, ready_timeout=args.ready_timeout):
            cells = _cells(args, base_url, measure=measure)
    result = summary(cells)
    write_json(args.output / "summary.json", result)
    print(json.dumps({"grid": result["grid"], "all_passed": result["all_passed"],
                      "output": str(args.output / "summary.json")}), flush=True)
    return result


def _numbers(kind):
    return lambda text: tuple(kind(item) for item in text.split(",") if item)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=None,
                        help="an already-served instance (the console's, such as http://127.0.0.1:8100); no container is started")
    parser.add_argument("--image", help="serving image tag, such as local/flare-ignition:2ca523b2 (without --base-url)")
    parser.add_argument("--package", type=Path, help="package model.safetensors (without --base-url)")
    parser.add_argument("--publisher-keys", type=Path, help="publisher profile JSON that signed the package (without --base-url)")
    parser.add_argument("--trust-store", type=Path, help="trust store approving that signer (without --base-url)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="ckpt", help="model id the package's responses carry")
    parser.add_argument("--max-context-tokens", type=int, default=None,
                        help="served context to request from the package (omit to serve the package's prepared ceiling; the flag is the fork's, upstream flare has none)")
    parser.add_argument("--lengths", type=_numbers(int), default=LENGTHS, help="target prompt tokens, comma-separated")
    parser.add_argument("--tokens-per-record", type=float, default=TOKENS_PER_RECORD,
                        help="catalog tokens per record for the served tokenizer (default: the Qwen3.8 measurement)")
    parser.add_argument("--template-kwargs", type=json.loads, default=None,
                        help='chat_template_kwargs JSON replacing {"enable_thinking": false}')
    parser.add_argument("--max-tokens", type=int, default=32, help="reply token budget per cell")
    parser.add_argument("--depths", type=_numbers(float), default=DEPTHS, help="needle depth fractions, comma-separated")
    parser.add_argument("--seed", type=int, default=48_300, help="passkey and sampling seed")
    parser.add_argument("--task", choices=TASKS, default="passkey",
                        help="passkey retrieval, or multi-hop variable tracking")
    parser.add_argument("--compare", type=Path, nargs="+", metavar="SUMMARY",
                        help="native and scaled summary.json files in pairs; writes --output/comparison.json and runs nothing")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--timeout", type=float, default=900, help="per-request deadline in seconds")
    parser.add_argument("--ready-timeout", type=float, default=900, help="seconds to wait for /health")
    args = parser.parse_args()
    if args.compare:
        if len(args.compare) % 2:
            parser.error("--compare takes native and scaled summaries in pairs")
        summaries = [json.loads(path.read_text()) for path in args.compare]
        args.output.mkdir(parents=True, exist_ok=True)
        result = compare(list(zip(summaries[::2], summaries[1::2])))
        write_json(args.output / "comparison.json", result)
        print(json.dumps({key: value for key, value in result.items() if key != "pairs"} | {"pairs": len(result["pairs"])}))
        return
    if not args.base_url and not (args.image and args.package and args.publisher_keys and args.trust_store):
        parser.error("--base-url, or --image, --package, --publisher-keys and --trust-store")
    result = run(args)
    raise SystemExit(0 if result["all_passed"] else 1)


if __name__ == "__main__":
    main()
