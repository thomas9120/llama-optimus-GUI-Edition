# llama_optimus/llamacpp_dl.py
# Download prebuilt llama.cpp binaries from GitHub releases into the
# standard llama/bin drop-in directory (~/.llama-optimus/llama/bin).
#
# Standard library only: urllib, zipfile, json.

import json
import os
import tempfile
import urllib.request
import urllib.error
import zipfile
from pathlib import Path

from .pipeline import LLAMA_BIN_DIR

RELEASES_URL = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page={n}"
_TIMEOUT = 60  # seconds, per HTTP request


def _fetch_json(url, timeout=_TIMEOUT):
    req = urllib.request.Request(url, headers={"User-Agent": "llama-optimus",
                                               "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def list_prebuilt_assets(max_releases=10, platform_name=None):
    """
    Return prebuilt llama.cpp release assets for this platform.

    Only assets from the newest release that has prebuilt binaries are returned
    (asset names starting with 'llama-'; companion 'cudart-*' runtime zips are
    skipped). Returns a list of dicts: {"tag", "name", "url", "size_mb"}.
    Raises NotAvailableError on network/API failure or when no prebuilt assets
    are published for this platform (e.g. Linux/macOS).
    """
    platform_name = platform_name or os.name  # 'nt' on Windows
    if platform_name != "nt":
        raise NotAvailableError(
            "llama.cpp does not publish prebuilt binaries for this platform.\n"
            "Build llama.cpp from source (or copy your own build) into:\n"
            f"  {LLAMA_BIN_DIR}"
        )

    try:
        releases = _fetch_json(RELEASES_URL.format(n=max_releases))
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise NotAvailableError(f"Could not reach GitHub to list llama.cpp releases: {e}")

    for rel in releases:
        assets = []
        for asset in rel.get("assets", []):
            name = asset.get("name", "")
            # e.g. llama-b10517-bin-win-cuda-12.4-x64.zip / -bin-win-vulkan-x64.zip / -bin-win-cpu-x64.zip
            # (skip companion cudart-llama-* runtime zips)
            if name.startswith("llama-") and "bin-win-" in name and name.endswith(".zip"):
                assets.append({
                    "tag": rel.get("tag_name", "?"),
                    "name": name,
                    "url": asset.get("browser_download_url", ""),
                    "size_mb": asset.get("size", 0) / 1e6,
                })
        if assets:
            return assets  # newest release that has prebuilt binaries

    raise NotAvailableError(
            "No prebuilt Windows binaries found in the latest llama.cpp releases.\n"
            "Build llama.cpp from source (or copy your own build) into:\n"
            f"  {LLAMA_BIN_DIR}"
        )


def _pretty_variant(asset_name):
    """Extract a short backend variant label from the asset filename."""
    # llama-b10517-bin-win-cuda-12.4-x64.zip -> cuda-12.4 x64
    parts = asset_name.replace(".zip", "").split("-bin-win-")
    if len(parts) < 2:
        return asset_name
    variant = parts[1]
    for arch in ("-x64", "-arm64"):
        if variant.endswith(arch):
            return variant[:-len(arch)] + " " + arch[1:]
    return variant


def download_asset(asset, dest_dir=None, progress=None, cancel_event=None):
    """
    Download a prebuilt llama.cpp release asset (zip) and extract it into dest_dir
    (default: ~/.llama-optimus/llama/bin).

    progress: optional callable(done_bytes, total_bytes) called as data streams.
    cancel_event: optional threading.Event; download aborts when set.

    Returns the destination directory Path.
    """
    dest_dir = Path(dest_dir) if dest_dir else LLAMA_BIN_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not asset.get("url"):
        raise NotAvailableError("Asset has no download URL.")

    req = urllib.request.Request(asset["url"], headers={"User-Agent": "llama-optimus"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp, \
         tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            if cancel_event is not None and cancel_event.is_set():
                tmp.close()
                os.unlink(tmp.name)
                raise NotAvailableError("Download cancelled.")
            chunk = resp.read(1 << 16)  # 64 KiB
            if not chunk:
                break
            tmp.write(chunk)
            done += len(chunk)
            if progress:
                progress(done, total)
        tmp_path = tmp.name

    # extract only after the full download succeeded (no partial state)
    try:
        with zipfile.ZipFile(tmp_path) as zf:
            zf.extractall(dest_dir)
    finally:
        os.unlink(tmp_path)

    return dest_dir


class NotAvailableError(Exception):
    """Download not possible (platform, network, cancelled). User-facing message."""
