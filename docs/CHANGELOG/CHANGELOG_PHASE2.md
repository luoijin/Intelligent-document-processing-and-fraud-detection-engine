# Changelog — Phase 2: Baseline Structured Extraction

**Roadmap reference:** `docs/06-ROADMAP-MILESTONES.md`, Phase 2
**Exit criteria (roadmap):** "Implement the rule-based extractor
(`03-MODEL-DEVELOPMENT.md §2.1`). Baseline field-level accuracy
measured and recorded against SROIE ground truth (this number becomes
your benchmark)."
**Status:** ✅ Met — see [Validation](#validation) below.

---

## Summary

Phase 2 implements the rule-based baseline entity extractor
(`app/extraction.py`) on top of the Phase 1 OCR layer: given OCR
tokens, it heuristically extracts `merchant_name`, `date`, and
`total_amount`. It is exposed via a new API endpoint and covered by an
automated exit-criteria test that measures and records field-level
accuracy, following the same pattern established by `test_phase1.py`.

**Deviation from `docs/06-ROADMAP-MILESTONES.md` Phase 2 / SROIE ground
truth, consistent with the Phase 1 deviation already documented in
`CHANGELOG_PHASE1.md`:** the real SROIE dataset (and its ground-truth
annotations) still requires a manual, authenticated download this
environment cannot perform. `scripts/generate_sample_receipts.py` was
extended to also emit `data/sample_receipts/ground_truth.json`
(merchant_name/date/total_amount per sample) so the exit criteria are
measurable today. **The Phase 2 baseline accuracy below has been
measured against synthetic samples, not real SROIE ground truth.**
Re-run `tests/test_phase2.py` against real SROIE images + labels
(drop them into `data/raw/`, point `SAMPLE_DIR`/`GROUND_TRUTH_PATH` at
that folder) before treating this exit criterion as fully satisfied
against the original dataset. Expect materially lower accuracy on real
SROIE data — the synthetic samples are clean, unskewed-text renders
with no handwriting, folds, glare, or OCR misreads beyond what the
deskew step itself introduces; the 100% baseline recorded here is a
ceiling on synthetic data, not a realistic estimate for real receipts.

**Bug fix (unrelated to Phase 2 scope but blocking it):** `tests/test_phase0.py`
and `tests/test_phase1.py` both computed `REPO = Path(__file__).parent`,
which resolved to `tests/` once those files were moved into that
directory (see git history — they were originally at the repo root
when `CHANGELOG_PHASE1.md`'s validation transcript was captured). This
silently broke every path derived from `REPO` (sample directory,
required-files list, etc.) without raising an import error, so the
break would not have been obvious until someone actually ran
`python tests/test_phase1.py` and got a false "no samples found"
failure. Both now compute `REPO = Path(__file__).parent.parent`.
`test_phase0.py`'s `check_fastapi_skeleton()` had a second, related
bug: it added `REPO / "app"` to `sys.path` and did `from main import
app`, which only worked when this file lived at the repo root (Python's
implicit script-directory `sys.path` entry happened to also expose the
repo root, making `app.ocr`'s absolute import resolvable by accident).
It now adds `REPO` itself and imports `from app.main import app`,
matching how the package is actually imported everywhere else. Fixed
now because Phase 2 needs a working `tests/test_phase2.py` in the same
directory, and shipping a fourth test file next to two silently-broken
ones was worse than fixing them.

---

## Files Added

| File | Purpose |
|---|---|
| `app/extraction.py` (~215 lines) | Rule-based baseline extractor: `_group_lines`, `_extract_merchant_name`, `_extract_date`, `_extract_total_amount`, `extract_fields()`, `to_dict()` |
| `tests/test_phase2.py` (~150 lines) | Phase 2 exit-criteria validation: runs OCR + extraction over samples, computes per-field and overall accuracy against `ground_truth.json`, appends the result to `experiments.csv` |
| `experiments.csv` | Model/metric log per `docs/03-MODEL-DEVELOPMENT.md` §4 ("simple experiments.csv log"); one row per `test_phase2.py` run so far |
| `docs/CHANGELOG/CHANGELOG_PHASE2.md` | This file |

## Files Modified

| File | Change | Reason |
|---|---|---|
| `app/main.py` | Added `POST /v1/extraction/baseline` endpoint; bumped app `version` from `0.2.0` → `0.3.0` | Expose the Phase 2 extractor over HTTP |
| `scripts/generate_sample_receipts.py` | Restructured `SAMPLE_RECEIPTS` to pair each receipt's render lines with a `ground_truth` dict; `main()` now also writes `data/sample_receipts/ground_truth.json` | Phase 2's accuracy measurement needs known-correct field values per sample, not just images |
| `tests/test_phase0.py` | Fixed `REPO` path resolution; fixed `check_fastapi_skeleton()`'s import approach | See "Bug fix" above |
| `tests/test_phase1.py` | Fixed `REPO` path resolution | See "Bug fix" above |
| `README.md` | Phase 2 status, API section, project structure, testing section, roadmap table | Reflect current implementation state |
| `docs/CODEBASE-DOCUMENTATION.md` | Added §2.6 (`app/extraction.py`), §3.4 (`POST /v1/extraction/baseline`), updated executive summary / data flow / tech stack sections to remove Phase 2's `[PLANNED]` markers now that it exists | Keep this doc accurate to `main`, per its own stated scope |

## Files Not Modified (confirmed via diff)

`app/preprocessing.py`, `app/ocr.py`, `Dockerfile`, `docker-compose.yml`,
`.env`, `.env.example`, `.gitignore`, `requirements.txt` — Phase 2 is
pure-Python (stdlib `re`/`datetime`/`dataclasses` + the existing
`OcrToken`/`OcrResult` types), so no new dependencies, container
changes, or config changes were needed.

## Directories Created (not tracked by git — see `.gitignore`)

None new — `data/sample_receipts/` already existed from Phase 1;
`ground_truth.json` is written into it by the updated generator script.

---

## New API Surface

### `POST /v1/extraction/baseline`

New endpoint. Accepts the same input as `POST /v1/ocr/extract`
(`multipart/form-data`, `file`, `image/jpeg`/`image/png`), runs OCR
then the baseline extractor, and returns structured fields. Not the
full `POST /v1/documents` contract in `docs/04-API-SPEC.md` — no
`document_id`, no `anomaly_score`, no persistence; those require the
Phase 4 anomaly model and Phase 5 service wrapper. See
`docs/CODEBASE-DOCUMENTATION.md` §3.4 for the full request/response
contract.

**Example response:**
```json
{
  "merchant_name": "GROCERY MART",
  "date": "2026-09-01",
  "total_amount": 299.5,
  "currency": null,
  "extraction_confidence": 1.0,
  "ocr_confidence_avg": 94.33
}
```

---

## Extraction Heuristics (implementation notes)

Per `docs/03-MODEL-DEVELOPMENT.md` §2.1:

- **`merchant_name`**: the plan's wording is "largest/topmost text
  block (often the header)". Tesseract's `image_to_data` output (the
  Phase 1 OCR engine) does not expose font size, so there is no
  "largest" signal available — only position. The implementation uses
  the topmost non-empty line as a proxy. This is a documented
  narrowing of the heuristic to the signal actually available, not a
  silent substitution — flagged here the same way Phase 1 flagged its
  OCR-engine substitution.
- **`date`**: first token matching one of several common date regex
  patterns (`YYYY-MM-DD`, `YYYY/MM/DD`, `MM/DD/YYYY`, `MM-DD-YYYY`,
  `DD.MM.YYYY`, `DD-Mon-YYYY`, `DD/Mon/YYYY`), normalized to
  `YYYY-MM-DD` in the output. Only patterns represented in the
  synthetic sample data (`YYYY-MM-DD`) have been exercised in testing;
  the others are included per the plan's general "common date regex
  patterns" wording but are unvalidated until real/varied receipt data
  is available — noted as a gap, not hidden.
- **`total_amount`**: largest currency-formatted number on a line
  containing a total/amount-due keyword; falls back to the largest
  currency-formatted number anywhere on the document if no such line
  is found (e.g., OCR misreads "TOTAL" itself). No currency-symbol
  detection is implemented — `currency` is always `null` in the
  response. None of the current sample data includes a currency
  symbol to detect against, so this was left unimplemented rather than
  guessed at; see Known Gaps below.

## Known Gaps Carried Forward (not blockers for Phase 2, tracked for later phases)

- `currency` field is always `null` — no currency-symbol/code
  detection exists. Revisit if/when sample or real data includes
  currency markers to detect.
- Merchant-name extraction uses line position only, not visual size —
  see "Extraction Heuristics" above. A trained model (Phase 3) is
  expected to do meaningfully better here since it can use true
  layout/visual features.
- Date-pattern coverage beyond `YYYY-MM-DD` is unvalidated against
  real data (see above).
- The 100% baseline accuracy recorded below is against clean synthetic
  renders and should not be read as a realistic accuracy estimate for
  real, noisy receipt photos — see "Deviation" above.
- `line_items` extraction (listed as a stretch field in
  `docs/02-DATA-STRATEGY.md` §2) is not implemented — out of scope for
  the Phase 2 baseline per `docs/03-MODEL-DEVELOPMENT.md` §2.1, which
  only names merchant/date/total.

## Next Phase

Per `docs/06-ROADMAP-MILESTONES.md`, Phase 3 (Trained Extraction
Model) fine-tunes LayoutLMv3 and must document its field-level F1
against the Phase 2 baseline recorded in `experiments.csv`
(`overall_field_accuracy=1.000` on synthetic data — expect this number
to look very different, and be far more meaningful, once measured
against real SROIE data instead).

---

## Validation

```
$ python tests/test_phase0.py
✅ All required files present
✅ FastAPI app imports successfully
✅ requirements.txt contains all Phase 0 dependencies
🎉 Phase 0 skeleton is ready!

$ python tests/test_phase1.py
✅ Sample images found in data/sample_receipts
  ✅ sample_receipt_01.png: 21 tokens, avg_confidence=94.3
  ✅ sample_receipt_02.png: 17 tokens, avg_confidence=88.7
  ✅ sample_receipt_03.png: 17 tokens, avg_confidence=91.1
3/3 sample images processed successfully with text + bounding boxes.
✅ Deskew path handled 2 rotated sample(s) without error
🎉 Phase 1 exit criteria met.

$ python tests/test_phase2.py
✅ Ground truth found at data/sample_receipts/ground_truth.json
  sample_receipt_01.png: merchant_name=✅, date=✅, total_amount=✅
  sample_receipt_02.png: merchant_name=✅, date=✅, total_amount=✅
  sample_receipt_03.png: merchant_name=✅, date=✅, total_amount=✅

Field-level accuracy (n=3):
  merchant_name: 100.0%
  date: 100.0%
  total_amount: 100.0%
  overall: 100.0%
✅ Recorded metrics to experiments.csv
🎉 Phase 2 exit criteria met: baseline field-level accuracy measured
   and recorded (this is the benchmark Phase 3 must beat).
```

End-to-end API validation (via `TestClient`, not just the pipeline
functions directly):

```
GET /health -> 200 {'status': 'ok', 'models_loaded': True}
POST /v1/extraction/baseline -> 200 {
  'merchant_name': 'GROCERY MART', 'date': '2026-09-01',
  'total_amount': 299.5, 'currency': None,
  'extraction_confidence': 1.0, 'ocr_confidence_avg': 94.33
}
POST /v1/extraction/baseline (bad content-type) -> 422 (rejected as expected)
```
