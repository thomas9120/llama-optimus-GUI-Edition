# llama_optimus/cli.py
# handle parsing, validation, and env setup

import argparse, configparser, os, shutil, sys
import platform
from pathlib import Path
from .override_patterns import OVERRIDE_PATTERNS
from .search_space import max_threads
from .pipeline import run_pipeline, resolve_llama_bench_path, PipelineError, LLAMA_BIN_DIR

from llama_optimus import __version__

CONFIG_PATH = Path.home() / ".llama-optimus.cfg"


def _find_llama_bins(home=None):
    """Return llama.cpp bin dirs found in common locations (deduped, best guess first)."""
    home = home or Path.home()
    exe = "llama-bench.exe" if os.name == "nt" else "llama-bench"
    cands = []
    # anywhere on PATH (covers llama.cpp prebuilt releases added to PATH)
    which = shutil.which("llama-bench")
    if which:
        cands.append(os.path.dirname(which))
    # standard drop-in location managed by llama-optimus (and its GUI downloader)
    dropin = home / ".llama-optimus" / "llama" / "bin"
    if (dropin / exe).is_file():
        cands.append(str(dropin))
    # common source-build / prebuilt locations
    for rel in ("llama.cpp/build/bin", "llama.cpp/build/bin/Release", "llama.cpp/bin",
                ".cache/llama.cpp/bin", ".llama.cpp/bin"):
        d = home / rel
        if (d / exe).is_file():
            cands.append(str(d))
    return list(dict.fromkeys(cands))  # dedupe, keep order


def _find_models(home=None, max_results=25):
    """Return up to max_results .gguf files found in common model directories."""
    home = home or Path.home()
    roots = (home / ".lmstudio/models", home / ".cache/lm-studio/models",
             home / ".cache/huggingface", home / "models")
    out = []
    for root in roots:
        if root.is_dir():
            for p in root.rglob("*.gguf"):
                out.append(str(p))
                if len(out) >= max_results:
                    return out
    return out


def _pick(kind, options):
    """Numbered picker in the terminal; 0 = enter path manually."""
    print(f"Found {len(options)} {kind} candidate(s):")
    for i, o in enumerate(options, 1):
        print(f"  [{i}] {o}")
    print("  [0] enter path manually")
    while True:
        choice = input(f"Choose 1-{len(options)} (or 0 for manual): ").strip()
        if choice.isdigit() and 0 <= int(choice) <= len(options):
            if int(choice) == 0:
                return input(f"Enter path to {kind}: ").strip()
            return options[int(choice) - 1]
        print("Invalid choice, try again.")


def _saved_path(key):
    """Return a path remembered from a previous interactive session, or None."""
    cp = configparser.ConfigParser()
    if CONFIG_PATH.is_file():
        cp.read(CONFIG_PATH)
    if cp.has_option("paths", key):
        return cp.get("paths", key)
    return None


def _save_paths(llama_bin, model):
    cp = configparser.ConfigParser()
    if CONFIG_PATH.is_file():
        cp.read(CONFIG_PATH)
    cp["paths"] = {"llama_bin": llama_bin, "model": model}
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        cp.write(f)
    print(f"Paths saved to {CONFIG_PATH} (reused on next launch; delete the file to reset).")


def main():
    parser = argparse.ArgumentParser(
        description="llama-optimus: Benchmark & tune llama.cpp.",
        epilog="""
        Example usage:

            llama-optimus --llama-bin my_path_to/llama.cpp/build/bin --model my_path_to/models/my-model.gguf --trials 35 --metric tg

        quick sanity run (~minutes, low accuracy):

            llama-optimus --preset quick
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
        )
    parser.add_argument("--trials", type=int, help="Number of Optuna/optimization trials (default: 45; 'quick' preset: 5)")
    parser.add_argument("--preset", choices=["default", "quick"], default="default",
        help="'quick': fast sanity run (5 trials, 2 repeats, no warmup, 20 tokens); "
             "'default': thorough optimization")
    parser.add_argument("--model", type=str, help="Path to model .gguf file (overrides env var; auto-detected/prompted if omitted)")
    parser.add_argument("--llama-bin", type=str, help="Path to llama.cpp build/bin folder (overrides env var; auto-detected/prompted if omitted)")

    parser.add_argument("--doctor", action="store_true", help="Run environment sanity checks (llama-bench, model, system) and exit")

    parser.add_argument("--metric", type=str, default="tg", choices=["tg", "pp", "mean"], help="Which throughput " \
        "metric to optimize: 'tg' (token generation, default), 'pp' (prompt processing), or 'mean' (average of both)")

    parser.add_argument("--ngl-max",type=int, help="Maximum number of model layers for -ngl "
        "(skip estimation if provided; estimation runs by default).")

    parser.add_argument("--repeat", "-r", type=int, help="Number of llama-bench runs per configuration "
        "(higher = more robust, lower = faster; default: 3; 'quick' preset: 2; for quick assessment: 1)")

    parser.add_argument("--n-tokens", type=int, help="Number of tokens used in llama-bench to test " \
        "velocity of prompt processing and text generation. Keep in mind there is large variability in tok/s outputs. " \
        "If n_tokens is too low, uncertainty takes over, optimization may suffer. Still, if you need to lower it, " \
        "try to operate with n_tokens > 70 and --repeat 3. " \
        "For fast exploration/testing/debug: --n-tokens 10 --repeat 2 is fine")

    parser.add_argument("--n-warmup-tokens", "-nwt", type=int, default=128, help="Number of tokens passed to " \
        "llama-bench during each warmup loop. In case of large models (and you getting small tg tokens/s), "
        "if n_warmup_tokens is too large, it can happen that you warmup in the first warmup cycle, and you end "
        "up not detecting the warmup. ")

    parser.add_argument("--n-warmup-runs", type=int, default=35, help="Maximum warm-up iterations before trials " \
    "begin. To skip warm-up completely, use the --no-warmup flag; Otherwise, there will be a minimum " \
    "number of warmup runs, which is set with `min_runs=3` in core function definition")

    parser.add_argument("--no-warmup", action="store_true", help="Skip the initial system warmup phase before "
        "optimization (for debugging/testing).")

    parser.add_argument("--verbose", action="store_true", help="Show full llama-bench commands and Optuna logs "
        "for every trial (debug).")

    #parser.add_argument('--version', "-v", action='version', version='llama-optimus v0.1.0')
    parser.add_argument("--version", "-v", action='version', version=f'llama-optimus v{__version__}')

    parser.add_argument("--override-mode", type=str, default="scan", choices=["none", "scan", "custom"],
    help=f"'none': do not scan this parameter; scan: 'scan' over preset override-tensor patterns; " \
    f"'custom': (future) user provides their own pattern(s). Available override patterns: {OVERRIDE_PATTERNS.keys()}" )

    args = parser.parse_args()

    if args.doctor:
        _doctor(args)
        return

    # Set paths based on CLI flags, env vars, or prompt user to provide it
    # Resolve llama_bin_path  (priority: CLI flag > env var > saved config > auto-detect > interactive prompt)
    prompted = False
    llama_bin_path = (args.llama_bin or os.environ.get("LLAMA_BIN") or _saved_path("llama_bin"))
    if not llama_bin_path:
        bins = _find_llama_bins()
        if len(bins) == 1:
            print(f"Auto-detected llama.cpp binaries at: {bins[0]}")
            llama_bin_path = bins[0]
        elif bins:
            llama_bin_path = _pick("llama.cpp bin folder", bins)
        else:
            llama_bin_path = input("Please, provide the path to your 'llama.cpp/build/bin' ").strip()
        prompted = True

    # Resolve model_path  (priority: CLI flag > env var > saved config > auto-detect > interactive prompt)
    model_path = (args.model or os.environ.get("MODEL_PATH") or _saved_path("model"))
    if not model_path:
        models = _find_models()
        if models:
            model_path = _pick("model (.gguf)", models)
        else:
            model_path = input("Please, provide the path to your 'ai_model.gguf' ").strip()
        prompted = True

    # Quick check if paths are set. ERROR msg if None or empty.
    if not llama_bin_path or not model_path:
        print("ERROR: LLAMA_BIN or MODEL_PATH not set. Set via environment variable, " \
        "pass via CLI flags, or provide paths just after launching llama-optimus. " \
        "Go to your terminal, navigate to your_path_to/llama.cpp/buil/bin and type 'pwd' to resolve the entire path. " \
        "Go to your terminal, navigate to your_path_to_AI_models/ and type 'pwd' to resolve the path. " \
        "Note: you must pass /path_to_model/model_name.gguf; e.g. your_path_model/gemma3_12B.gguf .", file=sys.stderr)
        sys.exit(1)

    # friendly check of the llama-bench binary (before it crashes deep inside a trial)
    try:
        resolve_llama_bench_path(llama_bin_path)
    except PipelineError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    # friendly check of the model path (before it crashes deep inside a trial)
    if not os.path.isfile(model_path):
        sys.exit(
            f"ERROR: model file not found at: {model_path}\n"
            f"  The path must point to an existing .gguf file.\n"
            f"  Fix it with: llama-optimus --model /full/path/to/model.gguf"
        )

    # remember paths given interactively, so the user is never asked twice
    if prompted and llama_bin_path and model_path:
        _save_paths(llama_bin_path, model_path)

    try:
        run_pipeline({
            "llama_bin": llama_bin_path,
            "model": model_path,
            "preset": args.preset,
            "trials": args.trials,
            "repeat": args.repeat,
            "n_tokens": args.n_tokens,
            "metric": args.metric,
            "ngl_max": args.ngl_max,
            "no_warmup": args.no_warmup,
            "n_warmup_runs": args.n_warmup_runs,
            "n_warmup_tokens": args.n_warmup_tokens,
            "override_mode": args.override_mode,
            "verbose": args.verbose,
        })
    except PipelineError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


def _doctor(args):
    """Sanity-check the environment: llama-bench runs, model loads, system info."""
    import subprocess

    print("########################################")
    print("# llama-optimus doctor                 #")
    print("########################################")
    print("")
    print(f"llama-optimus v{__version__}")
    print(f"Python {platform.python_version()} | {platform.system()} {platform.release()} | CPUs: {max_threads}")
    print("")

    # 1. llama-bench: explicit paths first, then auto-detect
    bench_path = None
    bin_dir = args.llama_bin or os.environ.get("LLAMA_BIN") or _saved_path("llama_bin")
    if bin_dir:
        exe = "llama-bench.exe" if platform.system() == "Windows" else "llama-bench"
        for cand in (f"{bin_dir}/{exe}", f"{bin_dir}/Release/{exe}"):
            if Path(cand).is_file():
                bench_path = cand
                break
        if not bench_path:
            print(f"[FAIL] llama-bench not found in --llama-bin/LLAMA_BIN path: {bin_dir}")
    else:
        bins = _find_llama_bins()
        if bins:
            bench_path = f"{bins[0]}/{'llama-bench.exe' if platform.system() == 'Windows' else 'llama-bench'}"
            print(f"[OK]   llama-bench auto-detected at: {bins[0]}")
        else:
            print("[FAIL] llama-bench not found (no --llama-bin, no env var, nothing in common locations)")
            print(f"       Build llama.cpp or download prebuilt binaries (drop them into {LLAMA_BIN_DIR}), "
                  "then pass --llama-bin /path/to/bin")

    if bench_path:
        try:
            r = subprocess.run([bench_path, "--version"], capture_output=True, text=True, timeout=60)
            first_line = (r.stdout or r.stderr).strip().splitlines()
            if r.returncode == 0:
                print(f"[OK]   llama-bench runs: {first_line[0] if first_line else bench_path}")
            else:
                print(f"[WARN] llama-bench exited with code {r.returncode}: {first_line[:2]}")
        except Exception as e:
            print(f"[FAIL] could not execute llama-bench: {e}")

    # 2. model
    model = args.model or os.environ.get("MODEL_PATH") or _saved_path("model")
    if model:
        if Path(model).is_file():
            print(f"[OK]   model file exists: {model}")
        else:
            print(f"[FAIL] model file not found: {model}")
    else:
        models = _find_models()
        if models:
            print(f"[OK]   {len(models)} model(s) auto-detected, e.g.: {models[0]}")
        else:
            print("[FAIL] no model set (no --model, no env var) and none found in common locations")
            print("       Pass --model /full/path/to/model.gguf")

    print("")
    print("Doctor done. All [OK] = ready to run: llama-optimus --preset quick")


if __name__ == "__main__":

    main()
