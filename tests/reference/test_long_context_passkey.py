"""Needle placement, pass criterion, grid, and serving order of the passkey probe; no container runs."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from qualify import dflash2_served as ds
from qualify import long_context_passkey as pk
from qualify.speculative_workload import _catalog


def test_prompt_places_the_needle_at_depth_over_the_catalog():
    case = pk.prompt(4800, 0.5, "48213", model="ckpt")
    messages = case["request"]["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "user"]
    lines = messages[1]["content"].split("\n")
    assert lines[2400] == pk.needle("48213")
    assert "\n".join(lines[:2400] + lines[2401:]) == _catalog(4800)
    assert messages[2]["content"] == pk.QUESTION
    request = case["request"]
    assert request["top_k"] == 1 and request["max_tokens"] == 32 and request["model"] == "ckpt"
    assert request["return_token_ids"] is False and request["return_token_timestamps"] is False
    assert case["id"] == "passkey-4800-0.5" and case["prompt_tokens"] == []
    assert case["request_sha256"] == pk.value_sha(request)


def test_needle_at_the_edges_stays_inside_the_catalog():
    first = pk.prompt(100, 0.0, "11111", model="ckpt")["request"]["messages"][1]["content"].split("\n")
    last = pk.prompt(100, 1.0, "11111", model="ckpt")["request"]["messages"][1]["content"].split("\n")
    assert first[0] == pk.needle("11111") and last[-1] == pk.needle("11111") and len(first) == len(last) == 101


def test_passkey_is_five_digits_absent_from_the_catalog_and_stable():
    records = pk.records_for(200_000)
    key = pk.passkey_for(48_300, 200_000, 0.1, records)
    assert len(key) == 5 and key.isdigit() and key not in _catalog(records)
    assert key == pk.passkey_for(48_300, 200_000, 0.1, records)
    assert key != pk.passkey_for(48_300, 200_000, 0.5, records)


def test_check_requires_the_exact_passkey():
    assert pk.check("48213.", "48213")
    assert pk.check("The pass key is 48213", "48213")
    assert not pk.check("I do not know", "48213")
    assert not pk.check("48212", "48213")


def observation(content, prompt_tokens, *, generated=4, elapsed_ns=2_000_000_000):
    return {"case": "x", "elapsed_ns": elapsed_ns,
            "analysis": {"usage": {"prompt_tokens": prompt_tokens, "completion_tokens": generated},
                         "finish_reason": "stop", "content": content,
                         "checks": {"no_stream_error": True, "one_valid_terminal": True}}}


def delta(rounds, offered, accepted, width=4):
    return {f'flare_speculative_rounds_total{{batch="1",width="{width}"}}': rounds,
            f'flare_speculative_offered_drafts_total{{batch="1",width="{width}"}}': offered,
            f'flare_speculative_accepted_drafts_total{{batch="1",width="{width}"}}': accepted,
            'flare_speculative_drafts_total{depth="0",source="mtp",outcome="accepted"}': accepted,
            'flare_speculative_rounds_total{batch="1",width="1"}': 10}


def test_cells_and_summary_build_the_grid():
    good = pk.cell(observation("48213.", 320_500), delta(100, 300, 180), length=320_000, depth=0.5, passkey="48213")
    assert good["passed"] and good["completed"] and good["request_seconds"] == 2.
    assert good["accepted_per_round"] == 180 / 110
    bad = pk.cell(observation("no idea", 200_100), delta(100, 300, 180), length=200_000, depth=0.1, passkey="77777")
    result = pk.summary([good, bad])
    assert result["grid"] == {"320000": {"0.5": True}, "200000": {"0.1": False}}
    assert result["all_passed"] is False
    assert pk.summary([good])["all_passed"] is True and pk.summary([])["all_passed"] is False


class Process:
    def __init__(self, command, *, stdout, stderr):
        self.command = command
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0


def run_with_fakes(tmp_path, lengths, max_context_tokens, reply):
    started, sent, stopped = [], [], []

    def popen(command, *, stdout, stderr):
        process = Process(command, stdout=stdout, stderr=stderr)
        started.append(command)
        return process

    def measure(base_url, case, folder, *, timeout, model_id):
        sent.append(case["id"])
        records = int(case["id"].split("-")[1])
        passkey = case["request"]["messages"][1]["content"].split("The pass key is ")[1].split(".")[0]
        return observation(reply(passkey), round(records * 66.6)), delta(100, 300, 180)

    args = SimpleNamespace(task="passkey", image="img", package=tmp_path / "pk/models/d/e/model.safetensors",
                           publisher_keys=tmp_path / "keys.json", trust_store=tmp_path / "trust.toml",
                           output=tmp_path / "out", model_id="ckpt", max_context_tokens=max_context_tokens,
                           lengths=lengths, depths=(0.5, 0.1, 0.9), seed=48_300, port=8111, timeout=1, ready_timeout=1,
                           base_url=None, tokens_per_record=pk.TOKENS_PER_RECORD, template_kwargs=None, max_tokens=32)
    result = pk.run(args, popen=popen, stop=lambda command, **_: stopped.append(command), ready=lambda url: True, measure=measure)
    return result, started, sent, stopped, args


def test_run_serves_once_and_sends_the_longest_length_first(tmp_path):
    result, started, sent, stopped, args = run_with_fakes(tmp_path, (200_000, 1_000_000, 320_000, 520_000), 1_048_576,
                                                          lambda passkey: f"{passkey}.")
    r = pk.records_for
    assert sent[0] == f"passkey-{r(1_000_000)}-0.1" and sent[-1] == f"passkey-{r(200_000)}-0.9"
    assert sent == [f"passkey-{r(length)}-{depth}" for length in (1_000_000, 520_000, 320_000, 200_000)
                    for depth in (0.1, 0.5, 0.9)]
    assert len(started) == 1
    command = started[0]
    assert "--history-drafts" not in command and command[command.index("--max-context-tokens") + 1] == "1048576"
    assert command == ds.container_command("img", args.package, name="long-context-passkey", uid=os.getuid(), gid=os.getgid(),
                                           home=Path.home(), publisher_keys=args.publisher_keys, trust_store=args.trust_store,
                                           port=8111, history_drafts=False, max_context_tokens=1_048_576)
    assert stopped == [["docker", "stop", "-t", "120", "long-context-passkey"]]
    assert result["all_passed"] is True and (tmp_path / "out/summary.json").is_file()
    assert sorted(result["grid"]) == sorted(["1000000", "520000", "320000", "200000"])


def test_run_without_a_context_ceiling_omits_the_flag(tmp_path):
    result, started, sent, stopped, args = run_with_fakes(tmp_path, (200_000,), None, lambda passkey: passkey)
    assert "--max-context-tokens" not in started[0] and result["all_passed"] is True


def test_native_baseline_run_takes_one_length_and_fails_on_a_miss(tmp_path):
    result, started, sent, stopped, args = run_with_fakes(tmp_path, (200_000,), 262_144, lambda passkey: "no idea")
    assert len(sent) == 3 and list(result["grid"]) == ["200000"] and result["grid"]["200000"] == {"0.1": False, "0.5": False, "0.9": False}
    assert result["all_passed"] is False and started[0][started[0].index("--max-context-tokens") + 1] == "262144"


def test_prompt_defaults_keep_the_qwen_request_and_accept_the_glimmer_overrides():
    default = pk.prompt(100, 0.5, "48213", model="ckpt")["request"]
    assert default["chat_template_kwargs"] == {"enable_thinking": False, "add_generation_prompt": True}
    assert default["max_tokens"] == 32
    glimmer = pk.prompt(100, 0.5, "48213", model="Muse-Glimmer-30B",
                        template_kwargs={"reasoning_strength": "low"}, max_tokens=256)["request"]
    assert glimmer["chat_template_kwargs"] == {"reasoning_strength": "low", "add_generation_prompt": True}
    assert glimmer["max_tokens"] == 256 and glimmer["top_k"] == 1


def test_records_for_takes_the_served_tokenizer_rate():
    assert pk.records_for(1_000_000) == round(1_000_000 / pk.TOKENS_PER_RECORD)
    assert pk.records_for(1_000_000, 56.54) == round(1_000_000 / 56.54) == 17687


def test_run_threads_the_glimmer_arm_into_every_cell(tmp_path):
    cases = []

    def measure(base_url, case, folder, *, timeout, model_id):
        cases.append(case)
        passkey = case["request"]["messages"][1]["content"].split("The pass key is ")[1].split(".")[0]
        return observation(f"{passkey}.", 1), delta(100, 300, 180)

    args = SimpleNamespace(task="passkey", image="img", package=tmp_path / "pk/models/d/e/model.safetensors",
                           publisher_keys=tmp_path / "keys.json", trust_store=tmp_path / "trust.toml",
                           output=tmp_path / "out", model_id="Muse-Glimmer-30B", max_context_tokens=None,
                           lengths=(1_000_000,), depths=(0.5,), seed=48_300, port=8111, timeout=1, ready_timeout=1,
                           base_url=None, tokens_per_record=56.54, template_kwargs={"reasoning_strength": "low"}, max_tokens=256)
    result = pk.run(args, popen=lambda command, **_: Process(command, stdout=None, stderr=None),
                    stop=lambda command, **_: None, ready=lambda url: True, measure=measure)
    assert result["all_passed"] and cases[0]["id"] == "passkey-17687-0.5"
    request = cases[0]["request"]
    assert request["chat_template_kwargs"] == {"reasoning_strength": "low", "add_generation_prompt": True}
    assert request["max_tokens"] == 256 and request["model"] == "Muse-Glimmer-30B"


def test_run_against_a_served_url_starts_and_stops_nothing(tmp_path):
    started, urls = [], []

    def measure(base_url, case, folder, *, timeout, model_id):
        urls.append(base_url)
        passkey = case["request"]["messages"][1]["content"].split("The pass key is ")[1].split(".")[0]
        return observation(f"{passkey}.", 1), delta(100, 300, 180)

    args = SimpleNamespace(task="passkey", base_url="http://127.0.0.1:8100", image=None, package=None, publisher_keys=None, trust_store=None,
                           output=tmp_path / "out", model_id="Muse-Glimmer-30B", max_context_tokens=None,
                           lengths=(200_000,), depths=(0.1, 0.9), seed=48_300, port=8111, timeout=1, ready_timeout=1,
                           tokens_per_record=pk.TOKENS_PER_RECORD, template_kwargs=None, max_tokens=32)
    result = pk.run(args, popen=lambda command, **_: started.append(command), stop=lambda command, **_: started.append(command),
                    ready=lambda url: url == "http://127.0.0.1:8100", measure=measure)
    assert started == [] and urls == ["http://127.0.0.1:8100"] * 2 and result["all_passed"]
    args.base_url = "http://127.0.0.1:9"
    with pytest.raises(RuntimeError):
        pk.run(args, popen=lambda *a, **k: None, stop=lambda *a, **k: None, ready=lambda url: False, measure=measure)


def test_tracking_prompt_spreads_the_chain_in_order_from_the_depth():
    case = pk.prompt(4800, 0.1, "48213", model="ckpt", task="tracking", length=320_000)
    messages = case["request"]["messages"]
    lines = messages[1]["content"].split("\n")
    names, sentences = pk.chain(0, 320_000, 0.1, 4800, "48213")
    assert len(names) == len(set(names)) == pk.HOPS and all(len(name) == 5 and name.isupper() for name in names)
    assert sentences[0] == f"VAR {names[0]} = 48213."
    assert sentences[1:] == [f"VAR {b} = VAR {a}." for a, b in zip(names, names[1:])]
    # Chain order, at even steps from the depth to the end; each earlier sentence shifts the next by one line.
    assert [lines.index(sentence) for sentence in sentences] == [480 + 1080 * hop + hop for hop in range(pk.HOPS)]
    assert "\n".join(line for line in lines if line not in sentences) == _catalog(4800)
    assert sum("48213" in line for line in lines) == 1
    assert messages[2]["content"] == pk.tracking_question(names[-1]) and names[-1] in messages[2]["content"]
    assert case["id"] == "tracking-4800-0.1"
    assert names == pk.chain(0, 320_000, 0.1, 4800, "48213")[0] != pk.chain(0, 320_000, 0.5, 4800, "48213")[0]


def test_passkey_request_is_unchanged_by_the_task_argument():
    default = pk.prompt(4800, 0.5, "48213", model="ckpt")
    assert default == pk.prompt(4800, 0.5, "48213", model="ckpt", task="passkey", length=320_000)
    assert default["request_sha256"] == "c5a36f45d77c34b12e564fadfd48a67aeb3fc48e26141cc95bb90aa7ee85b352"


def test_tracking_passes_only_on_the_chain_value():
    names, _ = pk.chain(0, 320_000, 0.1, 4800, "48213")
    passed = pk.cell(observation("48213", 320_500), delta(100, 300, 180), length=320_000, depth=0.1,
                     passkey="48213", task="tracking")
    assert passed["passed"] and passed["task"] == "tracking"
    for reply in (f"VAR {names[0]}", "48212", ""):
        assert not pk.cell(observation(reply, 320_500), delta(100, 300, 180), length=320_000, depth=0.1,
                           passkey="48213", task="tracking")["passed"]


def test_compare_pairs_native_and_scaled_cells():
    def summaries(task, replies):
        cells = [pk.cell(observation(replies(length, depth), length), delta(1, 1, 1), length=length, depth=depth,
                         passkey="48213", task=task)
                 for length in (200_000, 64_000, 8_000) for depth in (0.1, 0.5, 0.9)]
        return pk.summary(cells)

    native = [summaries(task, lambda length, depth: "48213") for task in pk.TASKS]
    scaled = [summaries(task, lambda length, depth: "48213" if length != 8_000 else "no") for task in pk.TASKS]
    # A scaled summary also holds lengths the native package cannot serve; they pair with nothing.
    scaled[0]["cells"].append(pk.cell(observation("48213", 1_000_000), delta(1, 1, 1), length=1_000_000, depth=0.1,
                                      passkey="48213"))
    result = pk.compare(list(zip(native, scaled)))
    assert len(result["pairs"]) == 18
    assert result["native_passed"] == 18 and result["scaled_passed"] == 12 and result["replies_identical"] == 12
    assert {pair["task"] for pair in result["pairs"]} == set(pk.TASKS)
    assert all(pair["replies_identical"] == (pair["length"] != 8_000) for pair in result["pairs"])
