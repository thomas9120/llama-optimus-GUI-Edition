# test/test_core.py
import os
from pathlib import Path

def test_search_space_shape():
    """Minimal smoke test to ensure SEARCH_SPACE is defined and has expected keys."""
    from llama_optimus.core import SEARCH_SPACE
    assert isinstance(SEARCH_SPACE, dict)
    assert 'batch_size' in SEARCH_SPACE


SAMPLE_CSV = (
    "build_info,model,n_batch,n_ubatch,n_gpu_layers,threads,n_gen,n_prompt,avg_ts,stddev_ts\n"
    ",m,512,128,99,8,128,0,71.20,1.10\n"
    ",m,512,128,99,8,0,256,402.50,3.30\n"
)

def test_parse_bench_csv():
    from llama_optimus.core import _parse_bench_csv
    res = _parse_bench_csv(SAMPLE_CSV)
    assert res == {"tg": 71.20, "pp": 402.50}
    # missing rows -> None
    empty = _parse_bench_csv(SAMPLE_CSV.splitlines()[0] + "\n")
    assert empty == {"tg": None, "pp": None}

def test_pct():
    from llama_optimus.core import _pct
    assert _pct(110.0, 100.0) == "+10.0%"
    assert _pct(90.0, 100.0) == "-10.0%"
    assert _pct(None, 100.0) == "n/a"
    assert _pct(110.0, 0.0) == "n/a"

def test_saved_paths_roundtrip(monkeypatch):
    """Paths saved via _save_paths are returned by _saved_path on next launch."""
    import tempfile
    from llama_optimus import cli
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setattr(cli, "CONFIG_PATH", Path(d) / "cfg.ini")
        assert cli._saved_path("llama_bin") is None
        cli._save_paths("/some/llama/bin", "/some/model.gguf")
        assert cli._saved_path("llama_bin") == "/some/llama/bin"
        assert cli._saved_path("model") == "/some/model.gguf"


def test_find_llama_bins_and_models():
    """Auto-detection finds a source-build llama-bench dir and .gguf models under a fake home."""
    import tempfile
    from llama_optimus import cli
    with tempfile.TemporaryDirectory() as d:
        home = Path(d)
        exe = "llama-bench.exe" if os.name == "nt" else "llama-bench"
        (home / "llama.cpp/build/bin").mkdir(parents=True)
        (home / "llama.cpp/build/bin" / exe).write_text("")
        bins = cli._find_llama_bins(home=home)
        assert len(bins) == 1 and bins[0] == str(home / "llama.cpp/build/bin")

        (home / ".lmstudio/models/vendor").mkdir(parents=True)
        (home / ".lmstudio/models/vendor/m.gguf").write_text("")
        (home / "models").mkdir()
        (home / "models/n.gguf").write_text("")
        (home / "models/not_a_model.txt").write_text("")
        found = cli._find_models(home=home)
        assert sorted(found) == sorted([str(home / ".lmstudio/models/vendor/m.gguf"), str(home / "models/n.gguf")])

        # empty home -> nothing found, no crash
        with tempfile.TemporaryDirectory() as d2:
            assert cli._find_llama_bins(home=Path(d2)) == []
            assert cli._find_models(home=Path(d2)) == []

