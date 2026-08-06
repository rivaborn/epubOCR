"""Fan one book's OCR across every free LLM unit in the fleet.

``ocr_book`` already batches cache-misses (``run_batch``), but every engine pointed at
exactly one server, so a 6-unit homelab OCR'd a book on one GPU. ``PoolEngine`` wraps N
sub-engines — one per unit — and spreads a batch over them.

**Dynamic, not round-robin.** Units are wildly unequal (measured 2026-08-06: paddleocr
on the 3070 Ti at ~5 s/page vs surya2 on a GB10 at ~25 s/page), so pages are pulled from
a shared queue as each worker frees up. A static split would leave the fast unit idle
waiting for the slow one.

**The pool is homogeneous by construction.** ``storage.cache_key`` keys every page on
``engine.identity()``, so a pool whose members ran different models could not express
"this page came from chandra, that one from surya2" — a re-run would silently re-OCR, and
worse, one book would mix two transcription styles. :func:`build_pool` therefore refuses
members whose ``identity()`` disagrees. Comparing *different* models is what the eval
harness is for; this is for making one model go faster.

Discovery (``pool = "auto"``) asks LLMConfig which units are free and already serving the
model, so the pool never triggers a cold load or evicts someone else's work — an idle
unit holding your model is free capacity; a busy one is not.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Config
from .base import OCREngine, OcrResult

_DEFAULT_GATEWAY = "http://192.168.1.40:11430"


@dataclass
class WorkerStat:
    unit: str
    pages: int = 0
    seconds: float = 0.0
    errors: int = 0

    @property
    def secs_per_page(self) -> float | None:
        return self.seconds / self.pages if self.pages else None


@dataclass
class PoolEngine(OCREngine):
    """An ``OCREngine`` that distributes ``run_batch`` across per-unit sub-engines."""

    members: list[tuple[str, OCREngine]]          # (unit label, engine)
    # Pages handed to a member per call. NOT 1: a member's own run_batch is what engages
    # its concurrency (Surya's SURYA_INFERENCE_PARALLEL fan-out, or the threaded batch for
    # the OpenAI-vision engines), so single-page hand-offs would serialize every unit and
    # make the pool slower than one well-batched engine. Small enough to still load-balance.
    chunk: int = 16
    name: str = "pool"
    wants_preprocess: bool = False
    stats: dict[str, WorkerStat] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if not self.members:
            raise RuntimeError("OCR pool has no members — no free unit is serving this model")
        first = self.members[0][1]
        self.name = first.name
        self.wants_preprocess = first.wants_preprocess
        self.stats = {unit: WorkerStat(unit) for unit, _ in self.members}

    def identity(self) -> str:
        """The members' shared identity — enforced by :func:`build_pool`, so the page
        cache is keyed exactly as it would be for a single-unit run of the same model."""
        return self.members[0][1].identity()

    @property
    def preferred_batch(self) -> int:
        """Pages ``ocr_book`` should hand over per call: enough for EVERY unit to get at
        least one chunk.

        Without this the pool silently degrades to one unit. ``ocr_book``'s own
        ``_OCR_BATCH`` (32) is a checkpoint size chosen for a single engine; if it is
        <= ``chunk`` then every call contains one chunk, the first worker takes it and
        every other worker returns immediately. Measured the hard way: a 359-page book
        with chunk=32 sent all 359 pages to spark1 and none to spark2, with no error.
        """
        return self.chunk * len(self.members)

    @property
    def units(self) -> list[str]:
        return [u for u, _ in self.members]

    def run(self, image_path: Path) -> OcrResult:
        return self.run_batch([image_path])[0]

    def run_batch(self, image_paths: list[Path]) -> list[OcrResult]:
        results: list[OcrResult | None] = [None] * len(image_paths)
        chunk = max(1, self.chunk)
        starts = iter(range(0, len(image_paths), chunk))
        cursor_lock = threading.Lock()

        def worker(unit: str, engine: OCREngine) -> None:
            while True:
                with cursor_lock:
                    try:
                        start = next(starts)
                    except StopIteration:
                        return
                idx = list(range(start, min(start + chunk, len(image_paths))))
                paths = [image_paths[i] for i in idx]
                t0 = time.perf_counter()
                try:
                    got = engine.run_batch(paths)
                    if len(got) != len(paths):
                        raise RuntimeError(
                            f"unit returned {len(got)} results for {len(paths)} pages")
                except Exception as exc:  # noqa: BLE001
                    # One dead unit must not lose the book: record empty pages (the builder
                    # routes empty text to facsimile) and keep the other units going.
                    got = [OcrResult(text="", words=[], mean_conf=None, engine=self.name,
                                     meta={"error": f"{type(exc).__name__}: {exc}"})
                           for _ in paths]
                    with self._lock:
                        self.stats[unit].errors += len(paths)
                dt = time.perf_counter() - t0
                with self._lock:
                    self.stats[unit].pages += len(paths)
                    self.stats[unit].seconds += dt
                for i, res in zip(idx, got):
                    res.meta.setdefault("unit", unit)
                    results[i] = res

        with ThreadPoolExecutor(max_workers=len(self.members),
                                thread_name_prefix="ocrpool") as ex:
            list(ex.map(lambda m: worker(*m), self.members))

        # Never return a short list: ocr_book zips these back onto the page list BY
        # POSITION, so a dropped element would silently shift every later page's text
        # onto the wrong page — the worst failure this pipeline could have.
        missing = [i for i, r in enumerate(results) if r is None]
        if missing:
            raise RuntimeError(f"OCR pool lost {len(missing)} page(s) at indices {missing[:5]}")
        return [r for r in results if r is not None]

    def summary(self) -> str:
        rows = []
        for unit, st in self.stats.items():
            spp = f"{st.secs_per_page:.1f}s/pg" if st.secs_per_page else "-"
            err = f" {st.errors} err" if st.errors else ""
            rows.append(f"{unit}: {st.pages}pg {spp}{err}")
        return " | ".join(rows)


# --------------------------------------------------------------------------- discovery

def free_units(gateway: str, model: str, *, timeout: float = 15.0,
               require_idle: bool = True) -> list[dict]:
    """Units that are reachable, holding ``model``, and usable — capacity we can take
    without a cold load or an eviction.

    ``require_idle`` skips units whose ``usage`` is ``active``. That is right for
    discovery (``pool = "auto"``): a busy unit is someone else's work. It is *wrong* for
    an explicitly named unit — ``active`` only means traffic within the gateway's 60 s
    window, so a unit you just finished using still reads active and naming it would
    fail for a minute after your own previous run.

    A non-preemptible lease is always disqualifying: that is a real claim by another
    caller, whichever way the unit was chosen.
    """
    import json
    import urllib.request

    with urllib.request.urlopen(f"{gateway.rstrip('/')}/api/status", timeout=timeout) as r:
        status = json.loads(r.read().decode("utf-8"))

    out: list[dict] = []
    for lane in status.get("lanes", []):
        if not lane.get("reachable", True) or not lane.get("enabled", True):
            continue
        if lane.get("swap_in_progress"):
            continue
        if require_idle and lane.get("usage") == "active":
            continue
        lease = lane.get("lease") or {}
        if lease and not lease.get("preemptible", True):
            continue
        for m in (lane.get("loaded_models") or []):
            if m.get("model") == model:
                out.append({"unit": lane.get("id"), "host": lane.get("host") or "",
                            "port": m.get("port") or 0})
                break
    return out


def _lane_url(gateway: str, unit: str) -> str:
    """The lane-scoped gateway path — the supported remote route to a specific unit."""
    return f"{gateway.rstrip('/')}/lane/{unit}/v1"


def unit_url(gateway: str, unit: dict, *, direct: bool = True) -> str:
    """Best endpoint for a discovered unit.

    Prefer the node's **own** ``host:port``, because a Spark serves ONE model per slot
    port and Surya's attach path insists the endpoint's first (and only) ``/v1/models``
    entry equal its checkpoint — a lane-scoped gateway catalog lists the unit's whole
    catalog, so surya2 cannot attach through it (it fails "Model mismatch ... got
    'gemma-4-26b-fp8'"). Direct also skips a proxy hop on every page.

    GPU lanes report no reachable port (their vLLM relay is not LAN-exposed — the WSL
    mirrored-networking dead end), so those fall back to the lane path, which is fine for
    every engine except surya2.
    """
    if direct and unit.get("host") and unit.get("port"):
        return f"http://{unit['host']}:{unit['port']}/v1"
    return _lane_url(gateway, unit["unit"])


def build_pool(config: Config, engine_name: str, *, make_engine) -> PoolEngine:
    """Build a pool from ``[ocr]`` config.

    ``make_engine(url, model)`` builds one sub-engine for a unit — supplied by
    ``ocr.get_engine`` so this module stays free of per-engine imports.

    Config (all under ``[ocr]``)::

        pool = "auto"                    # or a list of unit ids: ["spark1", "spark2"]
        pool_gateway = "http://192.168.1.40:11430"
        pool_model = "surya-ocr-2"       # served name to look for / request
        pool_urls = ["http://192.168.1.50:8000/v1", ...]   # explicit, bypasses discovery
    """
    ocr = config.raw.get("ocr", {}) or {}
    gateway = ocr.get("pool_gateway") or _DEFAULT_GATEWAY
    model = ocr.get("pool_model") or ocr.get(f"{engine_name}_model")
    spec = ocr.get("pool")

    direct = bool(ocr.get("pool_direct", True))
    members: list[tuple[str, OCREngine]] = []
    if ocr.get("pool_urls"):                       # explicit endpoints, no discovery
        for i, url in enumerate(ocr["pool_urls"]):
            members.append((f"u{i}", make_engine(url, model)))
    else:
        if spec == "auto" or spec is True:
            found = free_units(gateway, model)
        elif isinstance(spec, list):
            # Explicitly named: only require that the unit actually serves the model.
            avail = {u["unit"]: u for u in free_units(gateway, model, require_idle=False)}
            missing = [u for u in spec if u not in avail]
            if missing:
                raise RuntimeError(
                    f"unit(s) {missing} are not serving '{model}' — "
                    f"serving it: {sorted(avail) or 'none'}")
            found = [avail[u] for u in spec]
        else:
            raise RuntimeError(
                f"[ocr] pool must be \"auto\", a list of unit ids, or use pool_urls; got {spec!r}")
        if not found:
            raise RuntimeError(
                f"no free unit is serving '{model}' — load it on a unit, or set "
                f"[ocr] pool_urls to address servers directly")
        for u in found:
            members.append((u["unit"], make_engine(unit_url(gateway, u, direct=direct), model)))

    ids = {eng.identity() for _, eng in members}
    if len(ids) > 1:
        raise RuntimeError(
            "OCR pool members disagree on identity(): " + ", ".join(sorted(ids)) +
            ". A pool must run ONE model (the page cache is keyed on identity) — use the "
            "eval harness to compare different models.")
    return PoolEngine(members=members, chunk=int(ocr.get("pool_chunk", 16)))
