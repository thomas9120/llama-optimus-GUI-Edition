# llama_optimus/gui.py
# Basic tkinter GUI for llama-optimus.
#
# Same functionality as the CLI: pick llama.cpp bin + model, set optimization
# options, run the pipeline and watch the live log. Includes a downloader that
# fetches prebuilt llama.cpp binaries into ~/.llama-optimus/llama/bin.
#
# Standard library only (tkinter); the heavy lifting lives in pipeline.py /
# core.py, whose print() output is streamed into the log pane.

import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

from llama_optimus import __version__
from .pipeline import run_pipeline, PipelineError, LLAMA_BIN_DIR
from .core import OptimizationStopped
from . import llamacpp_dl
from .cli import _find_llama_bins, _find_models, _saved_path, _save_paths

_DONE = "__LLAMA_OPTIMUS_DONE__"  # sentinel pushed onto the log queue by the worker


class _QueueWriter:
    """File-like object forwarding write() calls to a queue (stdout bridge)."""

    def __init__(self, q, is_stderr=False):
        self.q, self.is_stderr = q, is_stderr

    def write(self, s):
        if s:
            self.q.put((s, self.is_stderr))

    def flush(self):
        pass


class OptimusGui(tk.Tk):
    POLL_MS = 100

    def __init__(self):
        super().__init__()
        self.title(f"llama-optimus v{__version__}")
        self.minsize(780, 560)

        self._log_q = queue.Queue()
        self._worker = None
        self._stop_event = threading.Event()
        self._inputs = []  # all form widgets, for enable/disable

        self._build_form()
        self._build_log()
        self._build_buttons()
        self.after(self.POLL_MS, self._poll_log)

        self.log(f"llama-optimus v{__version__}")
        self.log(f"Tip: drop a llama.cpp build into {LLAMA_BIN_DIR} and it is found automatically.")
        self.log("")

    # ------------------------------------------------------------------ form

    def _build_form(self):
        frm = ttk.LabelFrame(self, text="Settings", padding=10)
        frm.pack(fill="x", padx=10, pady=(10, 5))
        frm.columnconfigure(1, weight=1)

        # row 0: llama.cpp bin
        ttk.Label(frm, text="llama.cpp bin:").grid(row=0, column=0, sticky="w", pady=2)
        self.var_bin = tk.StringVar()
        self.cmb_bin = ttk.Combobox(frm, textvariable=self.var_bin)
        self.cmb_bin.grid(row=0, column=1, sticky="we", padx=(6, 6), pady=2)
        self._inputs.append(self.cmb_bin)
        self._track_grid(ttk.Button(frm, text="Browse...", command=self._browse_bin),
                         row=0, column=2, sticky="e")

        # row 1: model
        ttk.Label(frm, text="Model (.gguf):").grid(row=1, column=0, sticky="w", pady=2)
        self.var_model = tk.StringVar()
        self.cmb_model = ttk.Combobox(frm, textvariable=self.var_model)
        self.cmb_model.grid(row=1, column=1, sticky="we", padx=(6, 6), pady=2)
        self._inputs.append(self.cmb_model)
        self._track_grid(ttk.Button(frm, text="Browse...", command=self._browse_model),
                         row=1, column=2, sticky="e")

        # row 2: preset / metric / override mode
        frm2 = ttk.Frame(frm)
        frm2.grid(row=2, column=0, columnspan=3, sticky="we", pady=(8, 0))
        self.var_preset = self._option(frm2, "Preset:", ["default", "quick"], "default")
        self.var_metric = self._option(frm2, "Metric:", ["tg", "pp", "mean"], "tg")
        self.var_override = self._option(frm2, "Override mode:", ["scan", "none"], "scan")

        # row 3: numeric options
        frm3 = ttk.Frame(frm)
        frm3.grid(row=3, column=0, columnspan=3, sticky="we", pady=(6, 0))
        self.var_trials = self._spin(frm3, "Trials:", 1, 2000, None, "preset default")
        self.var_repeat = self._spin(frm3, "Repeat:", 1, 100, None)
        self.var_tokens = self._spin(frm3, "N tokens:", 1, 100000, None)
        self.var_ngl = self._spin(frm3, "NGL max:", 0, 999, None, "auto-estimate")

        # row 4: warmup + verbose
        frm4 = ttk.Frame(frm)
        frm4.grid(row=4, column=0, columnspan=3, sticky="we", pady=(6, 0))
        self.var_warmup = tk.BooleanVar(value=True)
        self._check(frm4, "Warmup:", self.var_warmup)
        self.var_warmup_runs = self._spin(frm4, "runs", 1, 999, 35)
        self.var_warmup_tokens = self._spin(frm4, "tokens", 1, 100000, 128)
        self.var_verbose = tk.BooleanVar(value=False)
        self._check(frm4, "Verbose", self.var_verbose)

        self._populate_defaults()

    def _track_grid(self, widget, **grid_kw):
        """grid() a widget and remember it for later disabling."""
        widget.grid(**grid_kw)
        self._inputs.append(widget)
        return widget

    def _option(self, parent, label, values, default):
        box = ttk.Frame(parent)
        box.pack(side="left", padx=(0, 18))
        ttk.Label(box, text=label).pack(side="left")
        var = tk.StringVar(value=default)
        cmb = ttk.Combobox(box, textvariable=var, values=values, width=10, state="readonly")
        cmb.pack(side="left", padx=(4, 0))
        self._inputs.append(cmb)
        return var

    def _spin(self, parent, label, lo, hi, default, hint=""):
        box = ttk.Frame(parent)
        box.pack(side="left", padx=(0, 14))
        ttk.Label(box, text=label).pack(side="left")
        var = tk.StringVar(value="" if default is None else str(default))
        spb = ttk.Spinbox(box, from_=lo, to=hi, textvariable=var, width=7)
        spb.pack(side="left", padx=(4, 0))
        self._inputs.append(spb)
        if hint:
            ttk.Label(box, text=f"({hint})", foreground="gray").pack(side="left", padx=(3, 0))
        return var

    def _check(self, parent, label, var):
        cb = ttk.Checkbutton(parent, text=label, variable=var)
        cb.pack(side="left")
        self._inputs.append(cb)
        return cb

    def _populate_defaults(self):
        # same priority order as the CLI: env var > saved config > auto-detect
        bins = _find_llama_bins()
        if str(LLAMA_BIN_DIR) not in bins:
            bins.insert(0, str(LLAMA_BIN_DIR))  # always offer the drop-in dir
        self.cmb_bin["values"] = bins
        saved_bin = os.environ.get("LLAMA_BIN") or _saved_path("llama_bin")
        if saved_bin:
            self.var_bin.set(saved_bin)
        elif len(bins) > 2:
            self.var_bin.set(bins[1])  # skip the (possibly empty) drop-in entry
        elif len(bins) == 2:
            self.var_bin.set(bins[1])
        elif bins:
            self.var_bin.set(bins[0])

        models = _find_models()
        self.cmb_model["values"] = models
        saved_model = os.environ.get("MODEL_PATH") or _saved_path("model")
        if saved_model:
            self.var_model.set(saved_model)
        elif models:
            self.var_model.set(models[0])

    # ------------------------------------------------------------------- log

    def _build_log(self):
        frm = ttk.LabelFrame(self, text="Log", padding=4)
        frm.pack(fill="both", expand=True, padx=10, pady=5)
        frm.rowconfigure(0, weight=1)
        frm.columnconfigure(0, weight=1)
        self.txt_log = tk.Text(frm, wrap="none", height=18, state="disabled",
                               font=("Consolas", 9) if os.name == "nt" else ("Courier New", 9))
        ysb = ttk.Scrollbar(frm, orient="vertical", command=self.txt_log.yview)
        xsb = ttk.Scrollbar(frm, orient="horizontal", command=self.txt_log.xview)
        self.txt_log.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        self.txt_log.grid(row=0, column=0, sticky="nsew")
        ysb.grid(row=0, column=1, sticky="ns")
        xsb.grid(row=1, column=0, sticky="we")
        self.txt_log.tag_config("err", foreground="#b00020")
        self.txt_log.tag_config("ok", foreground="#0a7d32")

    def log(self, msg, tag=None):
        self.txt_log.configure(state="normal")
        self.txt_log.insert("end", msg + "\n", tag or ())
        self.txt_log.see("end")
        self.txt_log.configure(state="disabled")

    def _poll_log(self):
        drain = True
        while drain:
            drain = False
            try:
                s, is_err = self._log_q.get_nowait()
                drain = True
                if s == _DONE:
                    self._worker = None
                    self.btn_run.configure(state="normal")
                    self.btn_stop.configure(state="disabled")
                    self._set_form_state("normal")
                else:
                    self.txt_log.configure(state="normal")
                    self.txt_log.insert("end", s, ("err",) if is_err else ())
                    self.txt_log.see("end")
                    self.txt_log.configure(state="disabled")
            except queue.Empty:
                pass
        self.after(self.POLL_MS, self._poll_log)

    # --------------------------------------------------------------- buttons

    def _build_buttons(self):
        frm = ttk.Frame(self, padding=(10, 0, 10, 10))
        frm.pack(fill="x")
        self.btn_run = ttk.Button(frm, text="Run", command=self._on_run)
        self.btn_run.pack(side="left")
        self.btn_stop = ttk.Button(frm, text="Stop", command=self._on_stop, state="disabled")
        self.btn_stop.pack(side="left", padx=(8, 0))
        self.btn_dl = ttk.Button(frm, text="Download llama.cpp...", command=self._on_download)
        self.btn_dl.pack(side="right")

    def _set_form_state(self, state):
        for w in self._inputs:
            try:
                w.configure(state=state)
            except tk.TclError:
                pass

    # --------------------------------------------------------------- actions

    def _browse_bin(self):
        d = filedialog.askdirectory(title="Select llama.cpp build/bin folder",
                                    initialdir=self.var_bin.get() or str(Path.home()))
        if d:
            self.var_bin.set(d)

    def _browse_model(self):
        init = Path(self.var_model.get()).parent if self.var_model.get() else Path.home()
        f = filedialog.askopenfilename(title="Select model .gguf file",
                                       filetypes=[("GGUF model files", "*.gguf"), ("All files", "*.*")],
                                       initialdir=str(init))
        if f:
            self.var_model.set(f)

    @staticmethod
    def _spin_value(var):
        v = var.get().strip()
        return int(v) if v.isdigit() else None

    def _collect_settings(self):
        return {
            "llama_bin": self.var_bin.get().strip(),
            "model": self.var_model.get().strip(),
            "preset": self.var_preset.get(),
            "metric": self.var_metric.get(),
            "trials": self._spin_value(self.var_trials),
            "repeat": self._spin_value(self.var_repeat),
            "n_tokens": self._spin_value(self.var_tokens),
            "ngl_max": self._spin_value(self.var_ngl),
            "no_warmup": not self.var_warmup.get(),
            "n_warmup_runs": self._spin_value(self.var_warmup_runs) or 35,
            "n_warmup_tokens": self._spin_value(self.var_warmup_tokens) or 128,
            "override_mode": self.var_override.get(),
            "verbose": self.var_verbose.get(),
        }

    # ------------------------------------------------------------------ run

    def _on_run(self):
        if self._worker is not None:
            return  # already running

        settings = self._collect_settings()
        if not settings["llama_bin"] or not settings["model"]:
            messagebox.showerror(
                "llama-optimus",
                "Please set both the llama.cpp bin folder and the model (.gguf) path.")
            return

        # remember paths (same ~/.llama-optimus.cfg the CLI uses)
        _save_paths(settings["llama_bin"], settings["model"])

        self._stop_event = threading.Event()
        self.btn_run.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self._set_form_state("disabled")

        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout = _QueueWriter(self._log_q)
        sys.stderr = _QueueWriter(self._log_q, is_stderr=True)
        self._worker = threading.Thread(
            target=self._run_pipeline, args=(settings, stdout, stderr), daemon=True)
        self._worker.start()

    def _run_pipeline(self, settings, stdout, stderr):
        try:
            run_pipeline(settings, stop_event=self._stop_event)
            self._log_q.put((f"\nRun finished. Results saved to "
                             f"{Path.cwd() / 'optimus_results.txt'}\n", False))
        except OptimizationStopped:
            self._log_q.put(("\nOptimization stopped by user.\n", False))
        except PipelineError as e:
            self._log_q.put((f"\n{e}\n", True))
        except Exception as e:
            self._log_q.put((f"\nUnexpected error: {type(e).__name__}: {e}\n", True))
        finally:
            sys.stdout, sys.stderr = stdout, stderr
            self._log_q.put((_DONE, False))

    def _on_stop(self):
        self._stop_event.set()
        self.btn_stop.configure(state="disabled")
        self.log("Stopping... (will abort after the current llama-bench run)")

    # ------------------------------------------------------------ downloader

    def _on_download(self):
        if self._worker is not None:
            messagebox.showinfo("llama-optimus", "Wait for the current run to finish first.")
            return
        _DownloadDialog(self)


class _DownloadDialog(tk.Toplevel):
    """Pick and download a prebuilt llama.cpp release into ~/.llama-optimus/llama/bin."""

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.cancel_event = threading.Event()
        self.assets = []
        self.title("Download llama.cpp")
        self.transient(app)
        self.grab_set()

        ttk.Label(self, text=f"Prebuilt binaries will be extracted to:\n{LLAMA_BIN_DIR}",
                  justify="left").pack(padx=12, pady=(12, 6), anchor="w")

        self.lst = tk.Listbox(self, width=74, height=10, selectmode="single")
        self.lst.pack(padx=12, pady=4, fill="both", expand=True)

        self.var_status = tk.StringVar(value="Fetching llama.cpp releases from GitHub...")
        ttk.Label(self, textvariable=self.var_status, foreground="gray",
                  wraplength=520, justify="left").pack(padx=12, pady=2, anchor="w")

        btns = ttk.Frame(self)
        btns.pack(padx=12, pady=(4, 12), fill="x")
        self.btn_dl = ttk.Button(btns, text="Download", command=self._on_download, state="disabled")
        self.btn_dl.pack(side="left")
        self.btn_cancel = ttk.Button(btns, text="Cancel", command=self._on_cancel)
        self.btn_cancel.pack(side="right")

        self.after(50, self._fetch_assets)

    def _fetch_assets(self):
        def work():
            try:
                assets = llamacpp_dl.list_prebuilt_assets()
                self.after(0, lambda: self._fill(assets))
            except llamacpp_dl.NotAvailableError as e:
                self.after(0, lambda msg=str(e): self._fail(msg))
        threading.Thread(target=work, daemon=True).start()

    def _fill(self, assets):
        self.assets = assets
        for a in assets:
            self.lst.insert("end", f"{a['tag']:>10}  {llamacpp_dl._pretty_variant(a['name']):<16} "
                                   f"{a['size_mb']:8.1f} MB")
        self.lst.selection_set(0)
        self.var_status.set(f"{len(assets)} prebuilt asset(s) found. Select one and click Download.")
        self.btn_dl.configure(state="normal")

    def _fail(self, msg):
        self.var_status.set(msg)
        self.btn_dl.configure(state="disabled")
        self.btn_cancel.configure(text="Close")

    def _on_download(self):
        sel = self.lst.curselection()
        if not sel:
            return
        asset = self.assets[sel[0]]
        self.btn_dl.configure(state="disabled")
        self.lst.configure(state="disabled")
        self.btn_cancel.configure(text="Cancel download")
        self.var_status.set(f"Downloading {asset['name']} ...")

        def progress(done, total):
            if total:
                pct = done / total * 100
                self.after(0, lambda p=pct, d=done: self.var_status.set(
                    f"Downloading {asset['name']} ... {p:5.1f}%  ({d / 1e6:7.1f} MB)"))

        def work():
            try:
                llamacpp_dl.download_asset(asset, progress=progress, cancel_event=self.cancel_event)
                self.after(0, lambda: self._done(asset))
            except llamacpp_dl.NotAvailableError as e:
                self.after(0, lambda msg=str(e): self._fail(msg))
            except Exception as e:
                self.after(0, lambda msg=str(e): self._fail(f"Download failed: {msg}"))

        threading.Thread(target=work, daemon=True).start()

    def _done(self, asset):
        self.var_status.set(f"Done. Extracted to {LLAMA_BIN_DIR} — already selected as your llama.cpp bin.")
        self.btn_cancel.configure(text="Close")
        self.app.log(f"llama.cpp {asset['tag']} ({asset['name']}) installed to {LLAMA_BIN_DIR}", "ok")
        # refresh + select the drop-in dir in the main window
        bins = _find_llama_bins()
        if str(LLAMA_BIN_DIR) not in bins:
            bins.insert(0, str(LLAMA_BIN_DIR))
        self.app.cmb_bin["values"] = bins
        self.app.var_bin.set(str(LLAMA_BIN_DIR))

    def _on_cancel(self):
        self.cancel_event.set()
        self.destroy()


def main():
    try:
        import tkinter  # noqa: F401
    except ImportError:
        print("tkinter is not available in this Python build. "
              "Use the CLI instead: llama-optimus --help")
        sys.exit(1)

    app = OptimusGui()
    app.mainloop()


if __name__ == "__main__":
    main()
