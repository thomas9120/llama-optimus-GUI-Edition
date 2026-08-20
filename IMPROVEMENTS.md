# Improvement Log — Install & Beginner-Friendly UX

Progress log for making install and user flow easier. Check off items as they are done.

Legend: [x] done · [ ] not started · [~] in progress / partially done

---

## Install friction

- [ ] **1. Auto-detect llama.cpp and models instead of asking for paths**
      Search common locations before prompting: `~/llama.cpp/build/bin`, official
      prebuilt-release folder, `~/.cache/llama.cpp`, LM Studio / Ollama model dirs,
      HF cache (`~/.cache/huggingface`). When prompting, offer a picker
      ("Found 3 models: [1] gemma3-12b.gguf … [2] enter path manually").

- [ ] **2. Fix README/code mismatches that confuse debugging**
      - README says Python 3.10+, `pyproject.toml` says `>=3.8`
      - README documents defaults that don't match code: trials "35" vs 45,
        `-r` "2" vs 3, `--n-tokens` "60" vs 192
      - Placeholder author email `YOUR_EMAIL@yourdomain.com` in pyproject
      - Typos in headings ("Intallation II", "lauch", "propted")

- [ ] **3. Add a zero-thought install path**
      - Recommend `pipx install llama-optimus` first (no venv lecture)
      - Add Windows section: `set LLAMA_BIN=...` / PowerShell `$env:LLAMA_BIN=...`;
        error message currently tells users to run `pwd`, meaningless on Windows

## User-flow friction

- [x] **4. Remember paths between runs**
      First run prompts for two paths; every later run prompts again unless the
      user knows about env vars. Save a small config after the interactive prompt
      and reuse it. Priority: CLI flag > env var > saved config > prompt.
      Kills the whole "Option A/B/C" section of the README.
      DONE: `~/.llama-optimus.cfg` (configparser). Saved only when entered
      interactively; CLI flags and env vars still win. Covered by
      `test_saved_paths_roundtrip`.

- [ ] **5. Validate the model path with a friendly error**
      `--model` typos currently surface as a raw llama-bench crash deep inside a
      trial. One `Path(model_path).is_file()` check with a clear message.

- [x] **6. Add `--preset quick` + progress/ETA instead of teaching flag incantations**
      One flag that sets sane quick values (trials=5, repeat=2, no-warmup,
      n-tokens=20). Default run prints estimated total work and Ctrl+C note.
      Per-trial progress line: "trial 12/45 | best: 71.2 tok/s | ~22m remaining".
      DONE: `--preset quick|default`; explicit user flags still win. Per-stage
      `_TrialProgress` callback prints trial count, best-so-far, elapsed and
      estimated remaining time; total-work estimate printed up front.

- [x] **7. Print copy-paste commands with real paths, not `$LLAMA_BIN/$MODEL`**
      A beginner copies the command, runs it, gets "model not found". Both
      absolute paths are already in scope — just print them.
      DONE: server command now uses the real `llama-server` path (derived from
      the validated llama-bench location, `.exe` suffix on Windows) and the
      real model path.

- [x] **8. Save results to a file + show improvement %**
      Everything is stdout-only today; the best config scrolls away after a
      30-minute run. Write `optimus_results.txt` (best config, llama-server and
      llama-bench commands, optimized-vs-default comparison) and print its path.
      Also compute the [TBD] % improvement — it's the payoff number.
      DONE: final benches now run with `-o csv` and are parsed
      (`_parse_bench_csv`, covered by tests); tg/pp improvement % printed and
      written to `optimus_results.txt` in the cwd together with best config and
      all copy-paste commands.

- [ ] **9. Quieter, friendlier progress**
      Compact one-line-per-trial summary instead of raw Optuna logs; final
      summary table.

- [ ] **10. Optional: `llama-optimus doctor`**
      One-command sanity check: llama-bench found + `--version` runs, model
      loads, Python/OS/GPU summary.

---

## Session notes

- 2025-08-20: Initial audit done. Implemented batch 1: **#4, #6, #7, #8**.
  Tests added (`test_parse_bench_csv`, `test_pct`, `test_saved_paths_roundtrip`);
  full suite passes (4/4). CLI smoke-tested: prompt → save → second run needs
  no prompt; `--preset quick` applies 5 trials / 2 reps / 20 tokens / no warmup.
- Remaining, in rough value order: #5 (model-path validation, ~4 lines),
  #2/#3 (README + pyproject fixes), #9 (quieter Optuna logs), #1 (auto-detect),
  #10 (doctor).
- Note: final benches switched from `--progress` table to `-o csv` output so
  results can be parsed; the comparison summary is printed instead.
