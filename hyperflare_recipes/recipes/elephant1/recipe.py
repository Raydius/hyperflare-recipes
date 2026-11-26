# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-GPU text-only Elephant-1 preparation for calibrated decision reads."""

from types import MappingProxyType

from ignition.declaration import Checkpoint, RankCosts, RankUsage, Recipe, SamplingDefaults

from .config import MODEL_ID, PAGE_TOKENS, REVISION, TRAINED_CONTEXT


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            core_storage=None, ranks=None):
    from .bootstrap import PreparationConfig, prepare as prepare_model

    if plan.world_size != 1 or ranks is not None:
        raise ValueError("Elephant-1 preparation supports one local GPU")
    if plan.speculation is not None:
        raise ValueError("Elephant-1 serves single-position decisions without learned speculation")
    if plan.benchmark_selection is not None:
        raise ValueError("Elephant-1 has no benchmark selection; it serves single-position decisions")
    from ignition.preparation.bootstrap import common_preparation_kwargs

    config = PreparationConfig(prompt_logprobs=plan.prompt_logprobs,
        **common_preparation_kwargs(plan, compile_forwards=compile_forwards, kernel_cache=kernel_cache,
                                    retain_graphs=retain_graphs))
    return prepare_model(snapshot, config, progress=progress, core_storage=core_storage)


def usage(prepared):
    return (RankUsage(prepared.memory_report["pages"], prepared.memory_report["free_device_bytes"]),)


RECIPE = Recipe(
    name="elephant1", served_model_name="Elephant-1", prepare=prepare, usage=usage,
    # A decision read is one prefill and one scored position, so the served defaults are deterministic.
    sampling_defaults=SamplingDefaults(temperature=1., top_p=1., top_k=1, seed=0),
    checkpoints=(Checkpoint(alias="elephant-1-9b", repository=MODEL_ID, revision=REVISION,
        architecture="Qwen3_5ForConditionalGeneration", stored_bytes=18_819_722_392,
        # Text tensors only: 6.92B body parameters at NVFP4 W4A16, the BF16 embedding and vocabulary head.
        device_bytes=7_963_437_024),
    ),
    context=TRAINED_CONTEXT, page_tokens=PAGE_TOKENS, concurrency=2, max_concurrency=4,
    # A block of the trained mask prefills whole, and a prefix block can fill the context.
    max_draft_tokens=0, prefill_chunk=TRAINED_CONTEXT, language_model_only=True,
    # Continuation and KV geometry, and the graph allowance. At the full context with the KV cache capped at
    # the concurrency, the serving process held 10,428 MiB at concurrency 2 and 11,596 MiB at concurrency 4 on
    # an RTX 5060 Ti; each request adds one state row and 128 KV pages.
    ranks=MappingProxyType({1: RankCosts(55_714_048, 4_194_304, 2 << 30)}),
)
