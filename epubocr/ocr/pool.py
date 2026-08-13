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

Discovery (``pool = "auto"``) asks LLMConfig which units are usable: ones already serving
the model first, and failing that it **loads the model onto an idle unit**. That fits this
fleet specifically — the units are idle >95% of the time, and LLMConfig's leases already
arbitrate contention (priority displaces, and a displaced long-running job restarts when a
unit frees up), so refusing to cold-load would be a self-imposed limit rather than good
manners. Never taken: a unit mid-request, one mid-swap, or one under a non-preemptible
lease.
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
#
# Fleet philosophy this implements: the LLM units are idle >95% of the time, and LLMConfig
# already arbitrates the rest — leases carry priority, a higher-priority job displaces a
# lower one, and the displaced job is restarted when a unit frees up. So a batch OCR run
# is a legitimate tenant of an IDLE unit, not a guest that must find one already warmed.
# An earlier version of this module refused to cold-load on principle; that was the right
# rule for a contended fleet and the wrong one for this fleet, where it just meant epubocr
# could not run at all unless somebody had hand-loaded the model first.
#
# What is still refused, always: a unit that is ACTIVE (someone is mid-request), one
# mid-swap, and one holding a NON-PREEMPTIBLE lease. Those are real claims; idleness is not.

def _get(gateway: str, path: str, timeout: float = 20.0):
    import json
    import urllib.request

    with urllib.request.urlopen(f"{gateway.rstrip('/')}{path}", timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def resolve_role(gateway: str, model: str, timeout: float = 10.0) -> str:
    """Turn a ROLE name into the model id it resolves to; pass anything else back.

    Needed because this module matches `model` against RESIDENCY — `/api/status`
    reports served names and knows nothing about roles — so a bare role would
    never match a loaded unit and would re-trigger autoload on every run, even
    with the model already up.

    An unreadable gateway returns the input unchanged, which degrades to exactly
    the pre-role behaviour rather than to an error.
    """
    try:
        data = _get(gateway, "/api/roles", timeout)
    except Exception:      # noqa: BLE001 — old gateway or one that is down
        return model
    for row in data.get("roles", []):
        if row.get("role") == model and row.get("resolves_to"):
            return row["resolves_to"]
    return model


def _post(gateway: str, path: str, body: dict, timeout: float = 30.0):
    import json
    import urllib.request

    req = urllib.request.Request(f"{gateway.rstrip('/')}{path}",
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _usable(lane: dict, *, require_idle: bool) -> bool:
    if not lane.get("reachable", True) or not lane.get("enabled", True):
        return False
    if lane.get("swap_in_progress"):
        return False
    if require_idle and lane.get("usage") == "active":
        return False
    lease = lane.get("lease") or {}
    return not (lease and not lease.get("preemptible", True))


def free_units(gateway: str, model: str, *, timeout: float = 15.0,
               require_idle: bool = True, status: dict | None = None) -> list[dict]:
    """Usable units **already serving** ``model``.

    ``require_idle`` skips units whose ``usage`` is ``active``. That is right for
    discovery (``pool = "auto"``): a busy unit is someone else's work. It is *wrong* for
    an explicitly named unit — ``active`` only means traffic within the gateway's 60 s
    window, so a unit you just finished using still reads active and naming it would
    fail for a minute after your own previous run.
    """
    status = status if status is not None else _get(gateway, "/api/status", timeout)
    out: list[dict] = []
    for lane in status.get("lanes", []):
        if not _usable(lane, require_idle=require_idle):
            continue
        for m in (lane.get("loaded_models") or []):
            if m.get("model") == model:
                out.append({"unit": lane.get("id"), "host": lane.get("host") or "",
                            "port": m.get("port") or 0})
                break
    return out


def loadable_units(gateway: str, model: str, *, status: dict | None = None,
                   exclude: set[str] | None = None) -> list[dict]:
    """Idle units whose catalog can serve ``model`` right now, best first.

    Fit is **not** re-derived here: each unit's catalog entry carries LLMConfig's own
    server-side ``addable`` verdict and its reason (VRAM headroom, ``needs_empty_node``,
    a multi-node recipe on a single node). Recomputing that in the client is how a
    gray-out and a real load refusal end up disagreeing.

    ``free`` units are preferred over merely ``idle`` ones: an idle unit holds someone's
    model, and loading ours evicts it. Both are permitted — that is what the lease system
    is for — but we spend the empty ones first.
    """
    status = status if status is not None else _get(gateway, "/api/status")
    exclude = exclude or set()
    out: list[dict] = []
    for lane in status.get("lanes", []):
        unit = lane.get("id")
        if unit in exclude or not _usable(lane, require_idle=True):
            continue
        try:
            cat = _get(gateway, f"/api/models?lane={unit}")
        except Exception:  # noqa: BLE001 - a unit we cannot ask about is simply not a candidate
            continue
        entries = [e for group in ("spark", "vllm", "ollama") for e in (cat.get(group) or [])]
        for e in entries:
            if e.get("served_name") != model and e.get("alias") != model:
                continue
            if not e.get("addable", True):
                continue
            out.append({"unit": unit, "alias": e.get("alias") or model,
                        "server": e.get("server") or ("spark" if lane.get("kind") == "spark"
                                                      else "vllm"),
                        "empty": not (lane.get("loaded_models") or [])})
            break
    out.sort(key=lambda u: not u["empty"])          # empty units first
    return out


def autoload(gateway: str, model: str, want: int, *, timeout_s: float = 900.0,
             poll_s: float = 10.0, log=print) -> list[dict]:
    """Ask LLMConfig to load ``model`` onto up to ``want`` idle units; return the ones
    that came up, as ``free_units`` entries.

    Loads are requested together and then awaited together — a Spark cold load runs into
    minutes (>=250 s measured), so serializing them would multiply the wait by the number
    of units for no reason.
    """
    import time

    cands = loadable_units(gateway, model)[:max(1, want)]
    if not cands:
        return []
    for c in cands:
        log(f"[pool] loading {model} on {c['unit']} (idle)...")
        try:
            _post(gateway, "/api/load",
                  {"server": c["server"], "model": c["alias"], "lane": c["unit"]})
        except Exception as exc:  # noqa: BLE001
            log(f"[pool] load request failed on {c['unit']}: {exc}")

    wanted = {c["unit"] for c in cands}
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        # Residency is the only proof: a load returns a job id immediately, and
        # `sparkrun stop`-style rc=0 lies mean the job's own status is not evidence.
        got = [u for u in free_units(gateway, model, require_idle=False) if u["unit"] in wanted]
        if len(got) >= len(wanted):
            return got
        time.sleep(poll_s)
    got = [u for u in free_units(gateway, model, require_idle=False) if u["unit"] in wanted]
    if got:
        log(f"[pool] {len(got)}/{len(wanted)} unit(s) came up within {timeout_s:.0f}s; "
            f"proceeding with those")
    return got


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
        pool_autoload = true             # load onto an idle unit when nothing serves it
        pool_autoload_units = 1          # how many idle units to load onto
        pool_max_units = 4               # cap on members when several already serve it
        pool_load_timeout_s = 900        # a Spark cold load runs into minutes
        pool_urls = ["http://192.168.1.50:8000/v1", ...]   # explicit, bypasses discovery
    """
    ocr = config.raw.get("ocr", {}) or {}
    gateway = ocr.get("pool_gateway") or _DEFAULT_GATEWAY
    model = ocr.get("pool_model") or ocr.get(f"{engine_name}_model")
    # `pool_model` may name a ROLE ("ocr"). Resolve it once, here, so every use
    # below — residency matching, autoload, and the per-unit sub-engines — sees
    # the same concrete served name.
    model = resolve_role(gateway, model)
    spec = ocr.get("pool")

    direct = bool(ocr.get("pool_direct", True))
    members: list[tuple[str, OCREngine]] = []
    if ocr.get("pool_urls"):                       # explicit endpoints, no discovery
        for i, url in enumerate(ocr["pool_urls"]):
            members.append((f"u{i}", make_engine(url, model)))
    else:
        if spec == "auto" or spec is True:
            found = free_units(gateway, model)[:int(ocr.get("pool_max_units", 4))]
            if not found and ocr.get("pool_autoload", True):
                # Nothing resident. On this fleet that is the NORMAL state (units idle
                # >95% of the time), not an error — ask LLMConfig to load onto an idle
                # unit. Displacement, priority and restart are the lease system's job.
                found = autoload(gateway, model, int(ocr.get("pool_autoload_units", 1)),
                                 timeout_s=float(ocr.get("pool_load_timeout_s", 900)))
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
                f"no unit is serving '{model}' and none could be loaded — every unit is "
                f"busy, mid-swap, non-preemptibly leased, or cannot fit it. Load it by "
                f"hand (`llmconfig load ...`), set [ocr] pool_urls to address a server "
                f"directly, or raise [ocr] pool_autoload_units.")
        for u in found:
            members.append((u["unit"], make_engine(unit_url(gateway, u, direct=direct), model)))

    ids = {eng.identity() for _, eng in members}
    if len(ids) > 1:
        raise RuntimeError(
            "OCR pool members disagree on identity(): " + ", ".join(sorted(ids)) +
            ". A pool must run ONE model (the page cache is keyed on identity) — use the "
            "eval harness to compare different models.")
    return PoolEngine(members=members, chunk=int(ocr.get("pool_chunk", 16)))
