# Raydius fork: recipe patch notes

This fork carries local recipes and recipe changes on top of
[local-inference-lab/hyperflare-recipes](https://github.com/local-inference-lab/hyperflare-recipes).
It is the recipes half of the Raydius Flare fork; the core half is
[Raydius/flare-local-inference](https://github.com/Raydius/flare-local-inference).

- **Base:** upstream recipes `7d70e230`, installed with upstream core
  `c915b5f0`, which is 27 commits past core's `v0.3.2` tag; upstream has no
  later tag. `main` mirrors upstream unchanged and `deploy` carries the fork's
  commits.
- **Core commit:** `pyproject.toml` names `ignition` by the core fork's URL and
  one exact commit, and `uv.lock` records it for `ignition` and
  `flare-heliopause`. A release carries the same tag, `vMAJOR.MINOR.PATCH-rN`,
  in both forks, and this fork's tagged commit locks the core fork's tagged commit.
- **FlashInfer:** `Raydius/flashinfer` `675984cd`, which is
  `local-inference-lab/flashinfer` `0d48af53`, the revision upstream recipes
  `7d70e230` locks, plus one commit.
- **Workflow:** the core fork's
  [FORK.md](https://github.com/Raydius/flare-local-inference/blob/deploy/FORK.md)
  specifies the branch model, the mirror sync, and the move to a newer upstream
  for both forks. Its
  [README](https://github.com/Raydius/flare-local-inference/blob/deploy/README.md)
  holds the full patch notes, including the core-side patches.
- **Open checks:** [docs/TODO.txt](docs/TODO.txt).

## Patches in this repository

Each verdict is judged against upstream recipes `7d70e230` installed with core
`c915b5f0`. "Kept" and "adapted" patches are absent from upstream;
an adapted patch was also fitted to an upstream mechanism on this base.

| Patch | Paths | Verdict |
|---|---|---|
| Muse Glimmer recipe `muse_glimmer_nvfp4` (alias `muse-glimmer-30b-nvfp4`) | `hyperflare_recipes/recipes/muse_glimmer_nvfp4/`, `hyperflare_recipes/reference/muse_glimmer.py` | Adapted: the bootstrap no longer passes the collective settings upstream removed (`be1a9a11`) |
| Elephant-1 recipe `elephant1` (alias `elephant-1-9b`) | `hyperflare_recipes/recipes/elephant1/`, `hyperflare_recipes/components/weights.py`, `gated_delta.py` | Adapted to upstream's dense Qwen decoder with image input and its fused query and key normalization, rotation and cache write (`a2746f49`): the recipe passes its attention class and the cache element type through upstream's constructor, and no longer passes the collective settings upstream removed |
| Elephant-1 trained block mask, prepared side | `hyperflare_recipes/recipes/elephant1/attention.py`, `contract.py` | Kept |
| DSpark drafting and DaisyRose-27B in the dense Qwen recipe (`--speculative-method dspark`, aliases `daisyrose-27b`, `daisyrose-27b-yarn4`) | `hyperflare_recipes/recipes/qwen38_dense/`, `hyperflare_recipes/components/dflash2_weights.py` | Adapted: the draft's Markov tensors are declared weight fields and the vision encoder is charged once, through `vision_bytes`, and the DSpark context is built inside upstream's non-sampling and terminal prefill graph variants (`a2746f49`) |
| Rope-scaled context for the dense Qwen recipe | `hyperflare_recipes/recipes/qwen38_dense/` | Adapted: the tuning identity takes the rope scaling, and the scaled frequencies feed upstream's fused normalization and rotation (`a2746f49`) |
| `qwen3.8-flash-yarn4` and rope-scaled Flash Next rotary tables | `hyperflare_recipes/recipes/qwen38/` | Adapted: the scaled rotary tables feed upstream's fused query and key normalization and rotation (`35059e30`), and the native-context bound of the SM103 score workspace sits in upstream's expression for sharded selection (`eb2b087b`) |
| `--draft-model` for `qwen38` DFlash2 proposals | `hyperflare_recipes/recipes/qwen38/recipe.py` | Kept |
| History drafts on the parallel-draft round | `hyperflare_recipes/components/parallel_draft.py`, `packed_speculation.py` | Kept |
| Elephant-1 and DFlash2 qualification tools | `qualify/elephant1_*.py`, `qualify/dflash2_*.py`, `qualify/long_context_passkey.py` | Kept |
| Image input for the dense Qwen recipe | | Retired: upstream `32009ad1` prepares the dense vision encoder |
| FP8 paged prefill kernel call passes `kv_tile_ranges` | | Retired: upstream `ce095402` passes the argument |
| Qwen preparation accepts the locked Transformers release | | Retired: upstream `069f7189` requires the locked release |

Qualification on this base: the CPU suite (`pytest tests -q`) fails no test
that it passes on upstream `7d70e230`, apart from three cases that need an
input a recipes-only checkout lacks and pass when it is supplied. Three cores
were exported and served on this base on one RTX PRO 6000: Flash Next with
factor-4 rope scaling and the DFlash2 draft at 8 sequences, Muse Glimmer with
its DFlash2 drafter at 16 sequences, and Elephant-1 at concurrency 4; each
answered its served checks. The GPU tests, the DSpark export and the
measurements that have not run on this base are listed in
[docs/TODO.txt](docs/TODO.txt).

---

> **Coding agents:** skip this README and read [AGENTS.md](AGENTS.md) directly.

# Hyperflare Recipes

Official model recipes, reusable PyTorch and b12x model components, and
qualification drivers for preparing runtime cores served by
[Hyperflare](https://github.com/NVIDIA/hyperflare-inference) (Flare / Helios).

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
