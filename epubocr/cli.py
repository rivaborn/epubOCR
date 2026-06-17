"""epubocr command-line interface.

Per-stage, resumable, cache-aware entry point (epubOCR.md §11). Working commands:
``show-config``, ``endpoints``, ``ingest``, ``ocr-page``. Later stages print a clear
"not wired yet" message until a real test EPUB drives them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import typer

from . import __version__
from .config import load_config

# Books carry em-dashes/ligatures and our output uses arrows; force UTF-8 so printing
# doesn't crash on the Windows console's cp1252 default.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

app = typer.Typer(add_completion=False, help="Fidelity-first OCR pipeline for image-only EPUBs.")


def _resolve_project(book: str, cfg):
    """Accept either a book slug under projects_root or a path to an EPUB."""
    from .storage import BookProject

    p = Path(book)
    if p.exists() and p.is_file():
        return BookProject.for_epub(p, cfg.projects_root)
    return BookProject(root=cfg.projects_root / book)


@app.command()
def version():
    """Print the epubocr version."""
    typer.echo(f"epubocr {__version__}")


@app.command("show-config")
def show_config():
    """Show the resolved configuration (endpoints, roles, models, thresholds)."""
    cfg = load_config()
    typer.echo(f"projects_root : {cfg.projects_root}")
    typer.echo(f"ocr engine    : {cfg.ocr_default_engine}")
    typer.echo("endpoints     :")
    for name, ep in cfg.endpoints.items():
        typer.echo(f"  - {name:7s} {ep.base_url}")
    typer.echo("roles         :")
    for role, ep in cfg.roles.items():
        typer.echo(f"  - {role:13s} -> {ep}  ({cfg.model(role, endpoint=ep)})")
    f = cfg.fidelity
    typer.echo(f"fidelity      : cer<={f.max_cer_vs_ocr} insertion<={f.max_insertion_ratio} "
               f"len_delta<={f.max_net_length_delta}")


@app.command()
def endpoints():
    """List models available at each configured endpoint (live reachability check)."""
    from .llm.client import LLMClient

    cfg = load_config()
    for name, ep in cfg.endpoints.items():
        typer.echo(f"\n[{name}] {ep.base_url}")
        try:
            models = LLMClient(ep).list_models()
            for m in models:
                typer.echo(f"  - {m}")
        except Exception as exc:  # noqa: BLE001 - surface any reachability error
            typer.secho(f"  unreachable: {exc}", fg=typer.colors.RED)


@app.command()
def ingest(epub: Path = typer.Argument(..., exists=True, dir_okay=False)):
    """Unpack an EPUB, classify spine pages, extract page images, write manifest.json."""
    from .ingest import ingest as run_ingest
    from .storage import BookProject

    cfg = load_config()
    project = BookProject.for_epub(epub, cfg.projects_root)
    pages = run_ingest(epub, project)

    counts: dict[str, int] = {}
    for p in pages:
        counts[p.page_type.value] = counts.get(p.page_type.value, 0) + 1
    typer.echo(f"ingested {len(pages)} pages -> {project.manifest_path}")
    for t, n in sorted(counts.items()):
        typer.echo(f"  {t:6s} {n}")


@app.command()
def ocr(
    book: str = typer.Argument(..., help="Book slug under projects_root, or path to an ingested EPUB"),
    engine: str = typer.Option(None, help="tesseract|surya|paddle|vlm (default from config)"),
    force: bool = typer.Option(False, help="ignore cache and re-OCR every page"),
    limit: int = typer.Option(None, help="OCR only the first N image pages (sampling big books)"),
):
    """Run OCR over an ingested book's image pages → ocr/page_XXXX.{raw.json,text.txt}."""
    from .pipeline import ocr_book
    from .storage import BookProject

    cfg = load_config()
    project = _resolve_project(book, cfg)
    if not project.manifest_path.exists():
        typer.secho(f"no manifest at {project.manifest_path} — run `epubocr ingest` first.",
                    fg=typer.colors.RED)
        raise typer.Exit(code=2)

    results = ocr_book(project, engine or cfg.ocr_default_engine, cfg, force=force, limit=limit)
    n_cached = sum(1 for r in results if r.cached)
    n_degenerate = sum(1 for r in results if r.degenerate)
    typer.echo(f"OCR'd {len(results)} image pages ({n_cached} from cache, "
               f"{n_degenerate} degenerate->facsimile) -> {project.ocr}")
    for r in results:
        conf = f"{r.mean_conf:.2f}" if r.mean_conf is not None else "n/a"
        flag = "  DEGENERATE->facsimile" if r.degenerate else ""
        typer.echo(f"  page {r.index:>4}  [{r.engine}]  conf={conf}  chars={r.text_chars}{flag}")


@app.command("ocr-page")
def ocr_page(
    image: Path = typer.Argument(..., exists=True, dir_okay=False),
    engine: str = typer.Option(None, help="tesseract|surya|paddle|vlm (default from config)"),
):
    """Run one image through an OCR engine and print the text (quick smoke test)."""
    from .ocr import get_engine

    cfg = load_config()
    eng = get_engine(engine or cfg.ocr_default_engine, cfg)
    result = eng.run(image)
    conf = f"{result.mean_conf:.2f}" if result.mean_conf is not None else "n/a"
    typer.echo(f"[{result.engine}] mean_conf={conf}\n{'-' * 40}\n{result.text}")


@app.command("eval")
def eval_cmd(
    book: str = typer.Argument(..., help="Book slug under projects_root, or path to an ingested EPUB"),
    gold: Path = typer.Option(..., exists=True, dir_okay=False, help="JSON: {page_index: ground_truth_text}"),
    engines: str = typer.Option("vlm", help="comma-separated: vlm,tesseract,surya,paddle"),
):
    """Score OCR engines against a gold set — the numbers pick the default engine."""
    import json

    from .eval.harness import format_table, run_eval

    cfg = load_config()
    project = _resolve_project(book, cfg)
    gold_map = {int(k): v for k, v in json.loads(gold.read_text(encoding="utf-8")).items()}
    names = [e.strip() for e in engines.split(",") if e.strip()]
    reports = run_eval(project, gold_map, names, cfg)
    typer.echo(format_table(reports))


@app.command()
def build(
    book: str = typer.Argument(..., help="Book slug under projects_root, or path to an ingested EPUB"),
    use_llm: bool = typer.Option(False, "--llm/--no-llm", help="run the LLM structure pass (behind the fidelity verifier)"),
    title: str = typer.Option(None, help="title for the output EPUB"),
    cleanup_endpoint: str = typer.Option(None, help="override cleanup endpoint (ollama|vllm)"),
    cleanup_model: str = typer.Option(None, help="override cleanup model (e.g. reuse a loaded model)"),
):
    """Assemble an improved EPUB from the manifest + OCR (run `ingest` then `ocr` first)."""
    from .pipeline import build_book
    from .validate import structural_checks

    cfg = load_config()
    project = _resolve_project(book, cfg)
    if not project.manifest_path.exists():
        typer.secho(f"no manifest at {project.manifest_path} — run `epubocr ingest` first.",
                    fg=typer.colors.RED)
        raise typer.Exit(code=2)

    out_path, summary = build_book(project, cfg, use_llm=use_llm, title=title,
                                   cleanup_endpoint=cleanup_endpoint, cleanup_model=cleanup_model)
    typer.echo(f"built {out_path}")
    typer.echo(f"  pages={summary['docs']} reflowable={summary['reflowable']} "
               f"facsimile={summary['facsimile']} preserved={summary['preserved']} "
               f"held={summary['held']}")

    checks = structural_checks(project.read_json(project.manifest_path))
    for w in checks.warnings:
        typer.secho(f"  warn: {w}", fg=typer.colors.YELLOW)
    for e in checks.errors:
        typer.secho(f"  error: {e}", fg=typer.colors.RED)
    typer.secho("  structural checks: OK" if checks.ok else "  structural checks: FAILED",
                fg=typer.colors.GREEN if checks.ok else typer.colors.RED)


if __name__ == "__main__":
    app()
