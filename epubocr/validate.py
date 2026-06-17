"""Validation: EPUBCheck + cheap structural checks (epubOCR.md §9)."""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CheckResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    note: str = ""


def epubcheck(epub_path: Path, jar_path: Path | None = None) -> CheckResult:
    """Run EPUBCheck 5.x if Java + the jar are available; else report what's missing.

    Prefer the current 5.x jar over the PyPI wrapper's bundled 4.2.6. Set the jar via
    ``jar_path`` or the ``EPUBCHECK_JAR`` env var.
    """
    import os
    java = shutil.which("java")
    jar = jar_path or (Path(os.environ["EPUBCHECK_JAR"]) if os.environ.get("EPUBCHECK_JAR") else None)
    if not java:
        return CheckResult(ok=False, note="java not found on PATH — install a JRE for EPUBCheck")
    if not jar or not Path(jar).exists():
        return CheckResult(ok=False, note="epubcheck.jar not found — set EPUBCHECK_JAR to the 5.x jar")

    proc = subprocess.run(
        [java, "-jar", str(jar), "--json", "-", str(epub_path)],
        capture_output=True, text=True,
    )
    try:
        report = json.loads(proc.stdout)
        msgs = report.get("messages", [])
        errors = [m["message"] for m in msgs if m.get("severity") in ("ERROR", "FATAL")]
        warnings = [m["message"] for m in msgs if m.get("severity") == "WARNING"]
        return CheckResult(ok=not errors, errors=errors, warnings=warnings)
    except json.JSONDecodeError:
        return CheckResult(ok=proc.returncode == 0, note=proc.stderr.strip()[:500])


def structural_checks(manifest: dict) -> CheckResult:
    """Cheap, dependency-free sanity checks over the ingest manifest."""
    errors: list[str] = []
    warnings: list[str] = []
    pages = manifest.get("pages", [])
    if not pages:
        errors.append("no pages in manifest")
    empties = [p["index"] for p in pages if p.get("type") == "empty"]
    if empties:
        warnings.append(f"{len(empties)} empty page(s): {empties[:10]}")
    missing = [p["index"] for p in pages if p.get("type") in ("image", "cover") and not p.get("extracted_images")]
    if missing:
        errors.append(f"{len(missing)} image page(s) with no extracted image: {missing[:10]}")
    return CheckResult(ok=not errors, errors=errors, warnings=warnings)
