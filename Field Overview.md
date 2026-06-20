# Field Overview — scanned-document → e-book / text OCR projects

A survey of existing open-source (and a few commercial) projects that tackle the same problem space
as `epubocr` — turning scanned/image-only documents into clean, searchable, or reflowable output —
and where this project sits relative to them. Compiled June 2026.

## The problem space

"Improve an image-only book" splits into a few distinct jobs that different tools optimize for:

- **Document parsing** — image/PDF pages → structured **markdown/JSON** (for reading or RAG).
- **Searchable facsimile** — keep the page image, add an invisible/hidden **OCR text layer** under it.
- **E-book conversion** — produce an actual **EPUB** (reflowable or fixed-layout).

`epubocr` is specifically **image-only EPUB → improved EPUB**, choosing *per page* between reflowable
text and searchable facsimile based on OCR confidence — a niche most tools don't target.

## Landscape at a glance

| Project              | Input → Output                 | OCR / engine            | Focus                                  | Confidence→facsimile fallback |
| -------------------- | ------------------------------ | ----------------------- | -------------------------------------- | ----------------------------- |
| **pdf-craft**        | scanned PDF → EPUB / Markdown  | DeepSeek-OCR (local)    | **scanned books** → EPUB (auto-TOC)    | No — commits to clean output  |
| **OCRmyPDF**         | scanned PDF → searchable PDF   | Tesseract               | **searchable facsimile** (lossless)    | Facsimile-only (no reflow)    |
| **Marker**           | PDF/image/EPUB → Markdown/JSON | **Surya**               | fast, general doc parsing              | No                            |
| **docling** (IBM)    | PDF/DOCX/HTML → DoclingDocument| layout + VLM            | modular parsing, RAG                   | No                            |
| **MinerU**           | PDF/Office → Markdown/JSON     | PP-OCR + VLM            | broad hardware, CJK, agentic           | No                            |
| **olmOCR** (AllenAI) | PDF → linearized text          | Qwen2.5-VL fine-tune    | document linearization                 | No                            |
| **Calibre**          | many → EPUB (+OCR via plugins) | external (Tesseract)    | general e-book conversion              | No                            |
| **epubocr** (this)   | **image-only EPUB → EPUB**     | Surya + VLM challenger  | **fidelity-first per-page routing**    | **Yes — the core idea**       |

## Closest in spirit

- **[pdf-craft](https://github.com/oomol-lab/pdf-craft)** (`oomol-lab`) — the most directly comparable:
  purpose-built for scanned **books** → EPUB/Markdown, runs locally (DeepSeek-OCR), auto-generates a
  TOC, and handles headers/footers, footnotes, formulas, and tables. Differences from `epubocr`: it is
  **PDF-in** (not image-only EPUB), and it commits to clean output — no confidence-based facsimile
  fallback, so on a badly faded scan it would likely emit garbled/confabulated text rather than
  preserving the page image. For most scanned-book → EPUB jobs this is the tool to reach for.
- **[OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF)** — the canonical **searchable-facsimile** tool:
  adds an invisible OCR text layer *under* the page image, losslessly, without reflowing. This is
  exactly `epubocr`'s **facsimile fallback** philosophy ("preserve the page, make it searchable"), for
  PDF. It implicitly endorses our conclusion: when a scan is too poor to reflow, keep the image + a
  text layer instead of fabricating text.

## OCR + layout pipelines (markdown-oriented)

Mature pipelines doing the "OCR → structured text" core. `epubocr` uses **Surya** directly; **Marker**
is essentially the production version of that same approach.

- **[Marker](https://github.com/datalab-to/marker)** (`datalab-to`) — built on Surya (our engine),
  fast, outputs markdown, accepts PDF/image/**EPUB** input. The most mature take on our OCR+layout core.
- **[docling](https://github.com/docling-project/docling)** (IBM) — modular doc → structured
  `DoclingDocument`; RAG-oriented.
- **[MinerU](https://github.com/opendatalab/mineru)** (`opendatalab`) — PDF/Office → markdown/JSON;
  broadest hardware support (NVIDIA/AMD/CN accelerators); strong CJK.
- **[olmOCR](https://github.com/allenai/olmocr)** (AllenAI) — PDF page linearization (Qwen2.5-VL
  fine-tune); document-anchoring to curb hallucination.
- **PyMuPDF4LLM** — native (born-digital) PDF → markdown; not for scans, but cheap when text is present.

## Simpler / general converters

- **[Calibre](https://calibre-ebook.com/)** `ebook-convert` + OCR workflows; **[pdf2epubEX](https://github.com/dodeeric/pdf2epubEX)**
  (fixed-layout EPUB from PDF); **[fabriziosalmi/pdf-ocr](https://github.com/fabriziosalmi/pdf-ocr)**
  (Flask OCR app); **[phuc-nt/scan-to-ebook](https://github.com/phuc-nt/scan-to-ebook)** (Manga OCR +
  Pandoc). Commercial: **ABBYY FineReader**, **iLoveOCR**.

## Where `epubocr` fits

**Distinctive:**
- **Image-only EPUB → improved EPUB** specifically — most tools are PDF-in. Image-EPUBs are a real
  niche (Internet Archive / Calibre exports, comics, photo books).
- **Fidelity-first per-page routing** — OCR confidence + cross-engine **consensus** decide
  reflowable-vs-facsimile, with a facsimile fallback when the scan is too faded. The markdown pipelines
  assume a recoverable source and don't fall back; on a degraded scan they hallucinate (as we observed
  with a VLM on *This Town* — see [comparison.md](comparison.md)).
- An **eval harness** (CER/WER/insertion) and **degeneracy/fidelity guards** baked in.

**Behind:** Marker, pdf-craft, docling, and MinerU are far more mature and capable. A pragmatic path is
to use **Marker** (it already shares our Surya core) or **pdf-craft** (for books) for the heavy
lifting, and keep `epubocr`'s confidence-routing + facsimile-fallback layer on top for low-quality
scans where committing to clean text would fabricate it.

## "Would converting the EPUB to a PDF and running Marker do better?"

A natural question, since Marker is the mature tool. For a **faded scan like *This Town*, no** — and
the PDF step is pointless. The reasoning is counterintuitive:

- **Marker uses the same OCR engine we do (Surya).** On the faded pages its raw recognition would be
  ~identical to ours (mean confidence ≈ 0.49, mostly garbled) — it can't read faded ink any better,
  because it's the same model on the same pixels. The bottleneck is **scan quality, not the tool**.
- **EPUB → PDF changes nothing.** The EPUB is image-only (page-scan JPEGs); converting to PDF just
  re-wraps the same JPEGs. And Marker **accepts EPUB directly** (`--force_ocr`), so the PDF detour adds
  nothing (a re-encode can only lose quality).

Where each actually wins:

| Scenario                          | Better tool | Why                                                                        |
| --------------------------------- | ----------- | -------------------------------------------------------------------------- |
| Clean, readable scanned book      | **Marker**  | far better layout, tables, equations, reading order; optional LLM refine   |
| Faded scan (e.g. *This Town*)     | **epubocr** | same OCR, but we keep low-confidence pages as **facsimile** instead of emitting garbage |
| Raw OCR accuracy on the faded ink | tie         | identical Surya engine                                                      |

Two reasons ours is the better result *specifically* on a degraded scan: Marker commits to **markdown**
and has no "too low-confidence → keep the page image" path, so it would output the garbled/hallucinated
text for the ~88% of unreadable pages; and getting an EPUB back out (markdown → EPUB via pandoc/Calibre)
**discards the page images**, losing the facsimile option that is the only sane output here.

**Takeaway:** it isn't "ours vs Marker" — it's **Marker (or pdf-craft) for OCR + structure on readable
books, with epubocr's confidence-routing + facsimile-fallback layer on top** for pages too degraded to
trust. For *This Town*, no tool rescues it; a cleaner source scan is the only real fix.

## Sources

- [Best Open-Source PDF-to-Markdown Tools in 2026 (Marker vs Docling vs MinerU vs pdf-craft)](https://themenonlab.blog/blog/best-open-source-pdf-to-markdown-tools-2026)
- [PDF Craft: Convert scanned PDF books to EPUB (OOMOL)](https://oomol.com/blog/2025/07/14/Convert-scanned-PDF-books-to-EPUB/)
- [OCRmyPDF documentation](https://ocrmypdf.org/)
- [pdf2epubEX](https://github.com/dodeeric/pdf2epubEX) · [scan-to-ebook](https://github.com/phuc-nt/scan-to-ebook) · [fabriziosalmi/pdf-ocr](https://github.com/fabriziosalmi/pdf-ocr)
