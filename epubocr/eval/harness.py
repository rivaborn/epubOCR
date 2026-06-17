"""Run OCR engines over a gold set and report fidelity metrics (epubOCR.md §5).

The output table is what picks the per-book default engine — engine choice is
empirical, not a priori. Engines that fail to initialize (e.g. Tesseract not
installed) are reported as unavailable rather than aborting the comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Config
from ..ocr import get_engine
from ..pipeline import ocr_image
from ..storage import BookProject
from .metrics import PageScore, score_engine


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


@dataclass
class EngineReport:
    engine: str
    n: int = 0
    mean_cer: float | None = None
    mean_wer: float | None = None
    mean_insertion: float | None = None
    error: str | None = None
    per_page: list[PageScore] = field(default_factory=list)


def run_eval(project: BookProject, gold: dict[int, str], engine_names: list[str],
             cfg: Config) -> list[EngineReport]:
    manifest = project.read_json(project.manifest_path)
    page_by_index = {p["index"]: p for p in manifest["pages"]}

    reports: list[EngineReport] = []
    for name in engine_names:
        try:
            engine = get_engine(name, cfg)
        except Exception as exc:  # noqa: BLE001
            reports.append(EngineReport(engine=name, error=f"init: {exc}"))
            continue

        scores: list[PageScore] = []
        error: str | None = None
        for idx, gold_text in gold.items():
            page = page_by_index.get(idx)
            if not page:
                continue
            imgs = page.get("extracted_images") or []
            if not imgs:
                continue
            src = project.pages / imgs[0]
            if not src.exists():
                continue
            try:
                result, _ = ocr_image(engine, src, project, cfg)
            except Exception as exc:  # noqa: BLE001
                error = f"run: {exc}"
                break
            scores.append(score_engine(name, gold_text, result.text))

        reports.append(EngineReport(
            engine=name, n=len(scores),
            mean_cer=_mean([s.cer for s in scores]),
            mean_wer=_mean([s.wer for s in scores]),
            mean_insertion=_mean([s.insertion_rate for s in scores]),
            error=error if not scores else None,
            per_page=scores,
        ))
    return reports


def format_table(reports: list[EngineReport]) -> str:
    def fmt(x):
        return f"{x:.3f}" if isinstance(x, float) else " n/a "

    rows = ["engine      |  n | CER   | WER   | insert | note",
            "------------+----+-------+-------+--------+-----"]
    for r in reports:
        note = r.error or ("best" if _is_best(r, reports) else "")
        rows.append(f"{r.engine:11s} | {r.n:>2} | {fmt(r.mean_cer)} | {fmt(r.mean_wer)} "
                    f"| {fmt(r.mean_insertion)}  | {note}")
    return "\n".join(rows)


def _is_best(r: EngineReport, reports: list[EngineReport]) -> bool:
    scored = [x for x in reports if x.mean_cer is not None]
    if not scored or r.mean_cer is None:
        return False
    return r.mean_cer == min(x.mean_cer for x in scored)
