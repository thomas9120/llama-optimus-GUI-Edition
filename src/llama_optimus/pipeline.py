# llama_optimus/pipeline.py
# Shared run pipeline used by both the CLI and the GUI.
#
# Extracted from cli.main(): takes a settings dict, resolves the llama-bench
# binary, validates paths, estimates max -ngl, runs warmup and the 3-stage
# Optuna optimization. Contains no argparse and no interactive input() calls,
# so it can be driven by any front-end (CLI, GUI, tests).

import os
import sys
import platform
from pathlib import Path

from .core import run_optimization, estimate_max_ngl, warmup_until_stable
from .search_space import SEARCH_SPACE, max_threads

# standard drop-in location for llama.cpp builds: ~/.llama-optimus/llama/bin
LLAMA_BIN_DIR = Path.home() / ".llama-optimus" / "llama" / "bin"


class PipelineError(Exception):
    """User-facing pipeline error (bad paths, missing binaries, ...)."""


def resolve_llama_bench_path(llama_bin_path):
    """
    Given a llama.cpp bin directory, return the full path to the llama-bench
    binary (handles the flat prebuilt layout and the CMake Release/ subfolder).
    Raises PipelineError when not found.
    """
    if platform.system() == "Windows":
        # Prebuilt binaries from GitHub releases use a flat directory layout,
        # while CMake source builds place binaries under a Release/ subdirectory.
        for cand in (f"{llama_bin_path}/llama-bench.exe",
                     f"{llama_bin_path}/Release/llama-bench.exe"):
            if Path(cand).is_file():
                return cand
        raise PipelineError(
            f"ERROR: llama-bench.exe not found.\n"
            f"  Searched:\n"
            f"    {llama_bin_path}/llama-bench.exe\n"
            f"    {llama_bin_path}/Release/llama-bench.exe"
        )
    else:
        llama_bench_path = f"{llama_bin_path}/llama-bench"
        if not Path(llama_bench_path).is_file():
            raise PipelineError(f"ERROR: llama-bench not found at {llama_bench_path}")
        return llama_bench_path


def resolve_defaults(settings):
    """Fill in preset-dependent defaults (only where the user left values unset)."""
    s = dict(settings)  # do not mutate the caller's dict
    quick = s.get("preset") == "quick"

    if s.get("trials") is None:   s["trials"] = 5 if quick else 45
    if s.get("repeat") is None:   s["repeat"] = 2 if quick else 3
    if s.get("n_tokens") is None: s["n_tokens"] = 20 if quick else 192
    if quick:
        s["no_warmup"] = True
        print("Quick preset: 5 trials, 2 repeats, no warmup, 20 tokens per test.")
    return s


def run_pipeline(settings, stop_event=None):
    """
    Run the full llama-optimus pipeline.

    settings keys (all optional except llama_bin and model):
        llama_bin       path to llama.cpp build/bin folder
        model           path to model .gguf file
        preset          'default' | 'quick'
        trials, repeat, n_tokens, metric ('tg'|'pp'|'mean'),
        ngl_max         (None = auto-estimate)
        no_warmup, n_warmup_runs (default 35), n_warmup_tokens (default 128),
        override_mode   ('scan'|'none'|'custom'; default 'scan'),
        verbose

    stop_event: optional threading.Event; when set, the pipeline aborts
    cleanly between llama-bench runs (used by the GUI Stop button).

    Raises PipelineError for user-fixable problems (missing llama-bench/model),
    or core.OptimizationStopped when stop_event is set mid-run.
    """
    s = resolve_defaults(settings)

    llama_bin_path = s.get("llama_bin")
    model_path = s.get("model")
    metric = s.get("metric") or "tg"

    # Quick check if paths are set. ERROR msg if None or empty.
    if not llama_bin_path or not model_path:
        raise PipelineError(
            "ERROR: llama bin or model path not set. "
            "Provide both paths (GUI fields or --llama-bin/--model CLI flags)."
        )

    llama_bench_path = resolve_llama_bench_path(llama_bin_path)

    if not os.path.isfile(model_path):
        raise PipelineError(
            f"ERROR: model file not found at: {model_path}\n"
            f"  The path must point to an existing .gguf file."
        )

    print("")
    print("#################")
    print("# LLAMA-OPTIMUS #")
    print("#################")

    print("")
    print(f"Number of CPUs: {max_threads}.")
    print(f"Path to 'llama-bench':{llama_bench_path}")  # in llama.cpp/tools/
    print(f"Path to 'model.gguf' file:{model_path}")
    print("")

    ngl_max = s.get("ngl_max")
    if ngl_max is not None:
        SEARCH_SPACE['gpu_layers']['high'] = ngl_max
        print("")
        print(f"User-specified maximum -ngl set to {ngl_max}")
        print("")
    else:
        print("")
        print("########################################################################")
        print("# Find maximum number of model layers that can be written to your VRAM #")
        print("########################################################################")
        print("")

        SEARCH_SPACE['gpu_layers']['high'] = estimate_max_ngl(
            llama_bench_path=llama_bench_path, model_path=model_path,
            min_ngl=0, max_ngl=SEARCH_SPACE['gpu_layers']['high'],
            stop_event=stop_event)
        print("")
        print(f"Setting maximum -ngl to {SEARCH_SPACE['gpu_layers']['high']}")
        print("")

    # system warm-up before optimization
    max_ngl_wup = SEARCH_SPACE['gpu_layers']['high']

    if s.get("no_warmup"):
        print("")
        print("#####################################################")
        print("# !!!Optimization running without system warmup!!!  #")
        print("#####################################################")
        print("")
    else:
        print("")
        print("#######################")
        print("# Starting warmup...  #")
        print("#######################")
        print("")

        n_warmup_runs = s.get("n_warmup_runs")
        if n_warmup_runs is None:
            n_warmup_runs = 35

        # in case n_warmup_runs is set to < 4, warn about the minimum number of warmup runs
        if n_warmup_runs < 4:
            print("")
            print("#########################################################################")
            print("# Setting a minimum of 4 warmup runs.                                   #")
            print('# For no warmup, pass the --no-warmup flag during llama-optimus launch  #')
            print("#########################################################################")
            print("")

        # launch warmup
        warmup_until_stable(llama_bench_path=llama_bench_path, model_path=model_path, metric=metric,
                            ngl=max_ngl_wup, min_runs=4, n_warmup_runs=n_warmup_runs,
                            n_warmup_tokens=s.get("n_warmup_tokens") or 128,
                            max_threads=max_threads, stop_event=stop_event)

    print("")
    print("##################################")
    print("# Starting Optimization Loop...  #")
    print("##################################")
    print("")

    run_optimization(n_trials=s["trials"], n_tokens=s["n_tokens"], metric=metric,
                     repeat=s["repeat"], llama_bench_path=llama_bench_path,
                     model_path=model_path, llama_bin_path=llama_bin_path,
                     override_mode=s.get("override_mode") or "scan",
                     verbose=bool(s.get("verbose")), stop_event=stop_event)
