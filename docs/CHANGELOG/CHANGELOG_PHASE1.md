# Changelog — Phase 1: Vision Extraction (Raw)

**Roadmap reference:** `docs/06-ROADMAP-MILESTONES.md`, Phase 1
**Exit criteria (roadmap):** "Given any SROIE image, the pipeline
outputs raw text + bounding boxes with no crashes on 100% of the
sample set."
**Status:** ✅ Met — see [Validation](#validation) below.

---

## Summary

Phase 1 implements the vision extraction layer: image preprocessing
(deskew/denoise/contrast normalization) and OCR (raw text + bounding
boxes + confidence). It is exposed via a new API endpoint and covered
by an automated exit-criteria test, following the same pattern
established by `test_phase0.py` in Phase 0.

**Deviation from `docs/03-MODEL-DEVELOPMENT.md` §1 worth flagging
explicitly:** that document names EasyOCR as the "Decision for v1."
This implementation uses **Tesseract** instead, which the same
document also lists as the option to "use first" for fastest
integration. Tesseract was chosen for Phase 1 specifically because:
(a) it has no large model-weight download (EasyOCR pulls in PyTorch,
multi-hundred-MB), keeping the Docker image lean and Phase 1 fast to
ship; (b) it is a mature, stable, pure-CPU dependency with no GPU
considerations. This is a substitution within the plan's own stated
options, not a departure from it — but it changes which package
appears in `requirements.txt` and the Dockerfile versus what a literal
reading of the "Decision for v1" line would suggest, so it's called
out here rather than left implicit. Reassess this choice at Phase 3
against measured Tesseract accuracy on real (non-synthetic) receipts.

**Deviation from `docs/02-DATA-STRATEGY.md` §1:** the real SROIE
dataset requires a manual, authenticated download this environment
cannot perform (not reachable via allowed network egress, and not a
`pip`/`apt` package). A synthetic sample-receipt generator
(`scripts/generate_sample_receipts.py`) was added as a stand-in so the
pipeline and its exit-criteria test are runnable end-to-end today.
This is explicitly documented as a placeholder in the script's
docstring and in `test_phase1.py` — **the Phase 1 exit criteria above
have been validated against synthetic samples, not real SROIE data.**
Re-run `test_phase1.py` against real SROIE images (drop them into
`data/raw/` and point `SAMPLE_DIR` at that folder) before treating this
exit criterion as fully satisfied against the original dataset.

---

## Files Added

| File | Purpose |
|---|---|
| `app/preprocessing.py` (102 lines) | Image preprocessing: `load_image`, `to_grayscale`, `denoise`, `deskew`, `normalize_contrast`, and the composed `preprocess()` entrypoint |
| `app/ocr.py` (104 lines) | OCR wrapper around `pytesseract`: `OcrToken`/`OcrResult` dataclasses, `run_ocr()`, `to_dict()` for API serialization |
| `scripts/generate_sample_receipts.py` (92 lines) | Generates 3 synthetic receipt images (one straight, two skewed) as a stand-in for SROIE until the real dataset is downloaded |
| `test_phase1.py` (100 lines) | Phase 1 exit-criteria validation: sample presence, 100%-pass pipeline run, deskew sanity check on rotated samples |
| `docs/CODEBASE-DOCUMENTATION.md` (426 lines) | Enterprise-grade codebase documentation (architecture, components, API spec, deployment, security, observability, runbook) reflecting current Phase 0+1 implementation state |
| `CHANGELOG_PHASE1.md` | This file |

## Files Modified

| File | Change | Reason |
|---|---|---|
| `app/main.py` | Added `POST /v1/ocr/extract` endpoint; added content-type validation, temp-file handling, error translation (`ValueError` → 422); bumped app `version` from `0.1.0` → `0.2.0` | Expose the Phase 1 pipeline over HTTP |
| `requirements.txt` | Added `opencv-python-headless`, `pytesseract`, `pillow`, `numpy`; changed pinned `==` versions to floor constraints (`>=`) for the Phase 0 deps to reduce transitive resolver conflicts with the new packages | Phase 1 pipeline dependencies |
| `Dockerfile` | Added `tesseract-ocr` (OCR engine binary) and `libgl1` (OpenCV headless runtime dependency) to the `apt-get install` layer | Container must run the OCR pipeline, not just the API skeleton |
| `test_phase0.py` | `check_requirements_phase0()` changed from "requirements.txt is an exact subset of 4 packages" to "requirements.txt contains at least the 4 Phase 0 packages" (regex-based version-specifier parsing instead of splitting on `==`) | The prior check was written when `requirements.txt` was Phase-0-exclusive; it began failing once Phase 1 dependencies were added. The check's intent (core web framework deps are present and correctly named) is preserved; its assumption (no other deps may exist) no longer holds by design |

## Files Not Modified (confirmed via diff)

`docker-compose.yml`, `.env`, `.env.example`, `.gitignore` — no changes
were needed. The `.env` variables (`OCR_ENGINE`, etc.) remain unread by
Phase 1 code; see `docs/CODEBASE-DOCUMENTATION.md` §4.1 for that gap.

## Directories Created (not tracked by git — see `.gitignore`)

- `data/sample_receipts/` — output of `scripts/generate_sample_receipts.py`, three `.png` files. Excluded from version control by the existing `data/` ignore rule; regenerate locally with `python scripts/generate_sample_receipts.py`.

---

## New API Surface

### `POST /v1/ocr/extract`

New endpoint. Not previously specified under this exact path in
`docs/04-API-SPEC.md` (which specifies `POST /documents` for the full
Phase 5 contract) — added as an explicitly Phase-1-scoped endpoint so
the API surface doesn't imply capabilities (structured fields, anomaly
scores) that don't exist yet. See
`docs/CODEBASE-DOCUMENTATION.md` §3.2 for the full request/response
contract and the rationale for not reusing the `/documents` path
prematurely.

---

## Validation

Both automated checks pass against the current `main` working tree:

```
$ python test_phase0.py
✅ All required files present
✅ FastAPI app imports successfully
✅ requirements.txt contains all Phase 0 dependencies
🎉 Phase 0 skeleton is ready!

$ python scripts/generate_sample_receipts.py
Wrote data/sample_receipts/sample_receipt_01.png (skew=0.0deg)
Wrote data/sample_receipts/sample_receipt_02.png (skew=2.5deg)
Wrote data/sample_receipts/sample_receipt_03.png (skew=-3.0deg)

$ python test_phase1.py
✅ Sample images found in data/sample_receipts
  ✅ sample_receipt_01.png: 21 tokens, avg_confidence=94.3
  ✅ sample_receipt_02.png: 17 tokens, avg_confidence=89.0
  ✅ sample_receipt_03.png: 17 tokens, avg_confidence=89.4
3/3 sample images processed successfully with text + bounding boxes.
✅ Deskew path handled 2 rotated sample(s) without error
🎉 Phase 1 exit criteria met.
```

End-to-end API validation (via `TestClient`, not just the pipeline
functions directly):

```
GET /health -> 200 {'status': 'ok', 'models_loaded': True}
POST /v1/ocr/extract -> 200 (21 tokens, avg_confidence=94.33)
POST /v1/ocr/extract (bad content-type) -> 422 (rejected as expected)
```

## Known Gaps Carried Forward (not blockers for Phase 1, tracked for later phases)

- `OCR_ENGINE` env var is not wired to an actual branch point in
  `app/ocr.py` — the engine is hardcoded to Tesseract. Should be
  addressed when Phase 3 introduces a second engine to switch to;
  building the switch now would be speculative for a single
  implementation.
- No file-size limit on uploads (see `docs/CODEBASE-DOCUMENTATION.md` §5.4).
- No structured logging of per-request OCR confidence/token counts
  (§6.1) — would aid early quality monitoring once this sees any real
  traffic, ahead of the full Phase 5 persistence layer.
- Full validation against the real SROIE dataset is still outstanding
  (see Deviations above) — the exit criterion is met against synthetic
  data only.

## Next Phase

Per `docs/06-ROADMAP-MILESTONES.md`, Phase 2 (Baseline Structured
Extraction) builds the rule-based merchant/date/total extractor on top
of this OCR output — no changes to `app/preprocessing.py` or
`app/ocr.py` are anticipated; Phase 2 is additive.
