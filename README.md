> **Coding agents:** skip this README and read [AGENTS.md](AGENTS.md) directly.

# Hyperflare Recipes

Official model recipes, reusable PyTorch and b12x model components, and
qualification drivers for preparing runtime cores served by
[Hyperflare](https://github.com/NVIDIA/hyperflare-inference) (Flare / Helios).

## Fork notes

This is `Raydius/hyperflare-recipes`, a fork of
[local-inference-lab/hyperflare-recipes](https://github.com/local-inference-lab/hyperflare-recipes).
`main` mirrors upstream unchanged; `deploy` carries the fork's recipe patches on upstream
`07d7bb39` and installs `ignition` from the core fork
[Raydius/flare-local-inference](https://github.com/Raydius/flare-local-inference) at
`0efa58ce` (its `deploy`), whose `FORK.md` describes the two-repository workflow. The
FlashInfer source is the fork pin `Raydius/flashinfer@d73da5c4`: upstream's locked
`0aa01681` plus two commits that reject unresolved planning stubs in compiler workers and
route the MLA compress memo through the program cache; two of the fork's three eviction
cases fail on unmodified `0aa01681` and all three pass with them. Commits carried over from
the core fork's stack keep a `Split-From:` trailer naming their combined-stack source.

"Still needed" is judged against upstream `07d7bb39`: none of the recipes, flags or
component changes below is present there.

### Active patches

- **Muse Glimmer recipe `muse_glimmer_nvfp4` (`9834def8`, `85463f36`, `37f3613d`, `c4f4fa7c`, `8fe522a3`, `d77a2055`, `906ee060`)** — added 2026-11-06 · Prepared (`hyperflare_recipes/recipes`)
  - What it does: Serves `original-el8/Muse-Glimmer-30B-NVFP4` (alias `muse-glimmer-30b-nvfp4`) on one GPU at TP1: W4A4 NVFP4 decoder projections with a BF16 head, up to four images per request, eight concurrent requests by default and sixteen at `--max-concurrency 16` with 128 packed verification rows (served at 16 on one RTX PRO 6000: 57.3 GiB of KV, 9,076,736 tokens), and a declared 1,048,576-token context over the native 131,072 (no rope scaling; the sliding layers stay bounded at 2,048). Drafting uses the `z-lab/Muse-Glimmer-30B-DFlash2` block drafter (pinned `b54ffdd1`) at 15 draft tokens; `ignition export --draft-model` supplies the drafter offline. Passkey retrieval at 1M: 14 of 15 cells within a 256-token cap, the 15th at 1,024
  - On `1a532755`: the recipe's image profile is `ImagePreparationKind` 11 and `ImageSettings` field 11, because upstream's Inkling padded grid holds 10. A Glimmer core exported before 2026-11-15 records 10; a hand-run `flare run` of it on a binary built from this base is not rejected, reads the profile as the padded grid, and gives invalid image results, so such a core stays on the image that exported it. The text and draft forwards take their metadata per call: Glimmer's attention is its own module (`muse_glimmer_nvfp4/attention.py`), `muse_spark/attention.py` is upstream's text, and the draft path uses upstream's shared `DFlash2Preparation`. The recipe's tests in `tests/recipes/muse_glimmer_nvfp4` pass on SM120 on this base: attention parity under graph replay with synthetic weights, and, against the checkpoint and its DFlash2 drafter on a 96 GB GB202, vision parity, the text reference comparison and the width-16 draft round
  - **Still needed:** Yes — until [PR #229](https://github.com/NVIDIA/spark-local-inference/pull/229) merges (it carries the recipe rebased on `972e28e4`, without this fork's compilation lowering, the renumbering, or the per-call forwards)
- **Elephant-1 recipe `elephant1` (`2e8f5d58`, `c2f2d052`, `e65c6c30`, `8797f6db`)** — added 2026-11-16 · Prepared (`hyperflare_recipes/recipes`, `hyperflare_recipes/components`)
  - What it does: Serves `baselabs/elephant-1` (alias `elephant-1-9b`, served name `Elephant-1`), a text-only Qwen3.5-9B decision model, inside 12.5 GB: preparation packs the BF16 body projections to NVFP4 W4A16 (`pack_body_nvfp4` in `ignition/components/weights.py`) and keeps the embedding, head, norms and Gated DeltaNet scalars in BF16; the 333 vision tensors are skipped and image inputs refused. Context 16,384 tokens, concurrency up to 4. A decision is read with Completions `max_tokens: 1` and `logprob_token_ids` naming the option letters. The shared Gated DeltaNet buffers and the dense Qwen decoder take their widths and query format from each model's head counts and cache. Preparation needs the FlashInfer GDN head-ratio patch below. Core `gb202-tp1-202611200624` (exported on GB202 at concurrency 4) serves on an RTX 5060 Ti in 11,596 MiB (12.16 GB), and the concurrency-2 core `gb202-tp1-202611171634` in 10,428 MiB; that card is in no Flare hardware platform list
  - **Still needed:** Yes — upstream has no Elephant-1 recipe; its Gated DeltaNet buffers assumed 48 value heads
- **DSpark drafting and DaisyRose-27B in the dense Qwen recipe (`4d91893d`, `c0cd13b4`)** — added 2026-11-19 · Prepared (`hyperflare_recipes/recipes/qwen38_dense`, `hyperflare_recipes/components`)
  - What it does: `--speculative-method dspark` (or `speculative_method = "dspark"` in the configuration) prepares the block draft `RadixArk/Qwen3.8-27B-DSpark` (pinned `b9a5dbdf`) for `qwen38_dense`, beside the recipe's default `dflash2`: five noncausal draft layers reading target layers 5, 19, 33, 47 and 61, a rank-256 Markov bias, FP8 paged draft history beside the target's pages, and 0 to 7 draft tokens (DFlash2 keeps 15). The draft loads and stores through the shared DFlash2 components (`projector="dspark"` in `ignition/components/dflash2_weights.py`). `--draft-head-quantization`, `--draft-body-quantization` and `ignition bench --speculation-study` are refused with DSpark. Declares `RadixArk/DaisyRose-27B-NVFP4` (pinned `c4d85c5d`), a fine-tune stored in the tensor layout of `nvidia/Qwen3.8-27B-NVFP4`, as `daisyrose-27b`. Qualified on an RTX PRO 6000 with one language-only core: 7 drafts at fixed width 8, 16 sequences, 1,048,576-token context, 3.29 accepted drafts per round over 619 rounds of chat traffic; the draft graphs match the eager reference `ignition/reference/qwen38_dense_dspark.py` on an RTX 5060 Ti. Throughput at fixed concurrency is not measured ([docs/model-preparation.md](docs/model-preparation.md), [docs/TODO.md](docs/TODO.md))
  - **Still needed:** Yes — upstream declares DSpark only for DeepSeek V4/V4.1, Kimi K3/K3.1 and MiniMax M3, not for the dense Qwen recipe; no PR yet
- **Rope-scaled context for the dense Qwen recipe (`3e48c406`)** — added 2026-11-19 · Prepared (`hyperflare_recipes/recipes/qwen38_dense`)
  - What it does: The dense decoder applies a declared `RopeScaling` (frequency-only YaRN) to its rotary tables, and `--rope-parameters` works for dense checkpoints. `daisyrose-27b-yarn4` declares factor 4 over 262,144 tokens, a 1,048,576-token context; the draft's own rotary is not scaled. A passkey probe (`qualify/long_context_passkey.py`) on the served core retrieved the key at depths 0.25 and 0.75 of prompts up to 1,009,836 tokens; quality past 262,144 tokens beyond passkey retrieval is not measured
  - **Still needed:** Yes — upstream's dense recipe declares no rope scaling
- **History drafts on the parallel-draft round (`95752d66`)** — added 2026-10-28 · Prepared (`hyperflare_recipes/components`)
  - What it does: `flare run --history-drafts` copies decoded-history tokens into parallel-draft (DFlash2) rounds through `substitute_copies` and the copy mask, as it already did for MTP. A/B at fixed-7 on a Qwen3.8 DFlash2 package of the retired `qwen38_dflash` recipe (2026-11-06, `qualify/results/dflash2-history-drafts/`, `2e91f176`): off 497.7 / 346.4 tok/s, on 380.7 / 341.6 tok/s (c1 / c8 `code_edit` medians; 52 % of copies accepted)
  - **Still needed:** Yes — upstream has history drafts only for MTP rounds. Upstream's `qwen38` DFlash2 path declares no copy inputs, so no Qwen 3.8 DFlash2 package takes the flag since `qwen38_dflash` was retired
- **`--draft-model` for `qwen38` DFlash2 proposals** — added 2026-11-10 · Prepared (`ignition/`)
  - What it does: `ignition export --recipe qwen38` with speculative method `dflash2` loads the DFlash2 draft from the `--draft-model` folder (`config.json`, `model.safetensors`) and downloads the pinned `incoai/Qwen3.8-Flash-Next-DFlash2` snapshot only without the flag. With MTP proposals the flag is still refused ("loads no external draft")
  - **Still needed:** Yes — upstream's `qwen38` DFlash2 path reads the draft only from the hub cache, so an offline export or a draft other than the pin needs the folder
- **FlashInfer pin `Raydius/flashinfer@cf1ac676a`** — added 2026-10-29, re-cut 2026-11-15, 2026-11-18 and 2026-11-19 · Prepared (`pyproject.toml`)
  - What it does: Upstream's pin `local-inference-lab/flashinfer@598c8547` plus one commit (branch `evict-planned-jits-598c8547`): a compiler worker rejects a job whose result is an unresolved planning stub, and `compile_compress` memoizes through the program cache so its kernels are evicted with the rest. Upstream's own planned-JIT registry (`_PLANNED_TRITON_JITS`), which since `4fd836dc` also covers application (non-b12x) JITs, replaces the fork's registry, which is no longer carried
  - **Still needed:** Yes — on unmodified `598c8547` the eviction case for application JITs passes, and the stub-rejection and `compile_compress` cases fail; both pass with the commit ([FlashInfer PR #67](https://github.com/local-inference-lab/flashinfer/pull/67), open, head `7245b1e0` since 2026-11-15; the `598c8547` re-cut is pushed to the fork remote as branch `evict-planned-jits-598c8547`, not to the PR)
- **Elephant-1 qualification tools** — added 2026-11-16 · Tooling (`qualify/`)
  - What it does: `elephant1_decision_parity.py` renders 160 decision probes by the vendor rules and compares a served core with the vendor runtime (OneJev `81ce62f`) in two stages, the mask and the storage precision; `--masks` scores mask ablations, and `core --reverse` with `report --recached` compares answers from an empty prefix cache with cached ones. `elephant1_block_attention_probe.py` checks the dense noncausal attention operator at Elephant-1's geometry and the prefill storage arithmetic. Results in `qualify/results/elephant1-decision-parity/` and `qualify/results/elephant1-mask-feasibility/`
  - **Still needed:** Yes — fork measurement tooling, not upstreamed
- **DFlash2 qualification tools (`bee64b00`, `25640c8d`, `a75e845d`, `9f764425`, `d3919444`, `d77a2055`, `4f434bdb`)** — added 2026-10-28 · Tooling (`qualify/`)
  - What it does: `dflash2_acceptance.py`, `dflash2_served.py`, `dflash2_draft_sweep.py`, `dflash2_long_context.py` and their results: acceptance, served DFlash2-vs-MTP comparison, draft-count sweep, past-262k probe. `dflash2_served.py` and `dflash2_draft_sweep.py` accept only packages with eight request slots and mixed graphs, which are packages of the retired `qwen38_dflash` recipe; `dflash2_console_probe.py` and `dflash2_console_rate.py` run the same workload against a console-served package of any prepared concurrency (results: `qualify/results/dflash2-paths-c16/`). In those results, A16 decode (`force_moe_decode_a16`) added to upstream's `qwen38` DFlash2 path with the MXFP8 draft lowers `code_edit` from 555.9 to 518.7 tok/s at concurrency 1 and the 16-client aggregate from 1,231.2 to 1,152.0 tok/s, with non-overlapping ranges over three trials each, so the fork serves that path with A16 decode off. Earlier results of the retired recipe: served comparison `qualify/results/dflash2-served/`, draft sweep `dflash2-draft-sweep/`, history-drafts A/B `dflash2-history-drafts/`, requalification runs `dflash2-requalify/`, past-262k probe `dflash2-long-context/`. `long_context_passkey.py`: five-digit passkey retrieval over prompt length × depth, against a launched package or a running server (`--base-url`), reporting width-16 speculative rounds and accepted drafts per round
  - **Still needed:** Yes — fork measurement tooling, not upstreamed

## Design principles

- **Embrace PyTorch and FlashInfer/b12x at build time; serve from Rust.**
  Recipes load checkpoints, autotune b12x operators, compile complete forwards,
  and capture CUDA graphs into a runtime core exported by `ignition` and
  `flare_heliopause`. Helios in `NVIDIA/hyperflare-inference` serves that core
  without Python.
- **Build on the model ecosystem.** Reuse canonical Transformers forwards where
  their dataflow fits, and replace only the smallest coherent modules needed for
  layout, state, or execution control.
- **Share reusable execution mechanisms without a rigid framework.** Keep
  model-specific composition in `hyperflare_recipes/recipes/` and reusable
  projections, MoE, attention, MLA, GDN, Mamba, head exchange, and speculation
  blocks in `hyperflare_recipes/components/`.
- **Keep core Flare small and recipe-free.** Core `ignition` in
  `NVIDIA/hyperflare-inference` owns preparation orchestration, planning,
  export, and the 13 Helios-ABI component interfaces; Helios owns scheduling,
  cache storage, and request handling.

## Repository layout

- `hyperflare_recipes/recipes/` — official model recipes (`deepseek4`,
  `deepseek41`, `gemma4`, `glm53`, `inkling2` including `inkling2_mini`,
  `kimi3`, `kimi31`, `mimo_v2`, `minimax_m3`, `muse_spark`, `nemotron_h`,
  `qwen38`, `qwen38_dense`, `step5`).
- `hyperflare_recipes/components/` — reusable b12x and Triton model building
  blocks.
- `hyperflare_recipes/fixtures/` — operator-tuning benchmark fixtures.
- `hyperflare_recipes/reference/` — numerical reference implementations.
- `hyperflare_recipes/cli.py` — `ignition` CLI entrypoint wrapping
  `ignition.cli.main(...)` from `NVIDIA/hyperflare-inference`.
- `qualify/`, `tests/`, `scripts/`, `tools/` — model qualification drivers,
  benchmarks, and recipe test suites.
- `docs/` — [model preparation and serving](docs/model-preparation.md),
  [model development](docs/model-development.md),
  [preparation development](docs/preparation-development.md),
  [GPU performance work](docs/performance-work.md),
  [direct execution](docs/direct-execution.md),
  [compressed scale layout](docs/tp-checkpoint-layout.md), and
  [commit design review](docs/design-review.md).

## Usage

Install dependencies (including `ignition[prepare]` and `flare-heliopause` from
`NVIDIA/hyperflare-inference`) and export a model core:

```sh
uv sync --extra test
.venv/bin/ignition export MODEL --gpus 0
```

Serve the exported core with the native `flare` executable from
`NVIDIA/hyperflare-inference`:

```sh
flare run CORE
```

All core preparation and publishing subcommands (`export`, `debug-serve`,
`bench`, `join`, `keygen`, `key-password`, `sign`, `edit`, `combine`,
`convert`) are provided by `ignition.cli` and inherited automatically. See
[model preparation and serving](docs/model-preparation.md) for hardware setup
and per-model commands.

## Building an independent recipe repository

External recipe repositories do not need to merge into Flare or
`hyperflare-recipes`. Depend on `ignition` from `NVIDIA/hyperflare-inference`
(and optionally `hyperflare-recipes` to reuse b12x model components) and expose
a CLI entrypoint calling `ignition.cli.main`:

```python
from pathlib import Path
from ignition.cli import main as ignition_main
from .recipes import RECIPES

def main(argv=None) -> int:
    return ignition_main(
        argv,
        recipes=RECIPES,
        prog="my-recipes",
        source_roots=(("my_recipes", Path(__file__).resolve().parent),),
    )
```

## Licensing and contributions

Project code is licensed under [Apache 2.0](LICENSE). Contributions require the
Developer Certificate of Origin sign-off described in
[CONTRIBUTING.md](CONTRIBUTING.md) and commit design review in
[docs/design-review.md](docs/design-review.md). Third-party attributions are
listed in [NOTICE](NOTICE); upstream terms remain applicable to third-party code
and assets.

The DeepSeek recipe assets retain their
[MIT license](hyperflare_recipes/recipes/deepseek41/assets/LICENSE). Upstream
chat templates are fetched with the model checkpoints and annotated locally
using bundled insertion patches. Gemma uses pinned Google sources, and Kimi K3 /
K3.1 segmented renderers are included as project code. See
[sampling and frontend metadata](docs/model-development.md#sampling-and-frontend-metadata).
