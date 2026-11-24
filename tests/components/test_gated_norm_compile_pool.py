# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""A b12x compiler worker plans, compiles and publishes a recipe component's Triton program."""


def test_worker_publishes_the_ignition_triton_program_it_planned(monkeypatch, tmp_path):
    """One b12x worker plans, then compiles, a flare.gated_rms_norm job and publishes it."""
    for variable in ("TRITON_CACHE_DIR", "B12X_COMPILE_CACHE_DIR", "FLASHINFER_WORKSPACE_BASE",
                     "TORCHINDUCTOR_CACHE_DIR"):
        monkeypatch.setenv(variable, str(tmp_path / variable.lower()))
    from b12x._lib.compile_plan import compiled_program_available
    from b12x._lib.compile_pool import CompileJob, CompilePool
    from hyperflare_recipes.components.gated_norm_preparation import CONTRACT, Query

    rows, heads, dim = 8, 64, 64
    activation = ((rows, heads, dim), (heads * dim, dim, 1), "bfloat16")
    query = Query(
        operands=(activation, activation, ((dim,), (1,), "bfloat16"), activation, ((1,), (1,), "int32")),
        options=(("EPS", 1e-5), ("HEADS", heads), ("DIM", dim), ("HEAD_BLOCK", 8), ("ROUND_NORMALIZED", False),
                 ("BF16_WEIGHT", True), ("SILU_GATE", False), ("NORM_BEFORE_GATE", True),
                 ("WEIGHT_PER_HEAD", False)))
    job = CompileJob.create("hyperflare_recipes.components.gated_norm_preparation:compile_norm",
                            CONTRACT.encode_query(query), 4, CompileJob.DEVICE_ORDINAL)
    pool = CompilePool(device_ordinal=0, compute_capability=(12, 0),
                       device_uuid="GPU-00000000-0000-0000-0000-000000000000", product_name="NVIDIA test target",
                       sm_count=188, max_shared_memory_per_block=232448,
                       max_shared_memory_per_multiprocessor=233472, workers=1)
    try:
        programs = tuple(pool.submit([job]))
        pool.wait(programs, deadline=None)
        assert programs and pool.ready(programs)
        available = [compiled_program_available(program) for program in programs]
        assert available == [True] * len(programs)
        assert list(tmp_path.rglob("__grp___gated_rms_norm.json"))
    finally:
        pool.close(terminate=True)
