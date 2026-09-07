# Roadmap & Milestones

Each phase has a hard **exit criterion** — do not move to the next phase
until it's met. This prevents the common failure mode of this kind of
project: partially building every layer and never having a working system.

## Phase 0 — Setup
- Repo scaffolding, Docker/docker-compose skeleton, empty FastAPI app
  with a working `/health` endpoint.
- **Exit criteria:** `docker compose up` returns a 200 from `/health`.

## Phase 1 — Vision Extraction (Raw)
- Ingest SROIE dataset. Run EasyOCR over sample receipts.
- **Exit criteria:** given any SROIE image, the pipeline outputs raw text
  + bounding boxes with no crashes on 100% of the sample set.

## Phase 2 — Baseline Structured Extraction
- Implement the rule-based extractor (`03-MODEL-DEVELOPMENT.md §2.1`).
- **Exit criteria:** baseline field-level accuracy measured and recorded
  against SROIE ground truth (this number becomes your benchmark).

## Phase 3 — Trained Extraction Model
- Label/prepare training data, fine-tune LayoutLMv3.
- **Exit criteria:** fine-tuned model's F1 documented and compared
  against the Phase 2 baseline, with the comparison written down (even
  if the trained model doesn't win — that's a valid, documentable result).

## Phase 4 — Anomaly / Fraud Layer
- Build feature pipeline, inject synthetic fraud, train Isolation Forest,
  then XGBoost.
- **Exit criteria:** precision/recall on synthetic fraud validation set
  documented for both models.

## Phase 5 — Service Wrapper
- Wire Phases 1-4 behind the FastAPI endpoints in `04-API-SPEC.md`.
- **Exit criteria:** a single `POST /documents` call with a real receipt
  image returns a complete response (fields + anomaly score) end-to-end.

## Phase 6 — Deployment
- Containerize, optionally deploy to a free-tier host.
- **Exit criteria:** the service is reachable via `docker compose up`
  from a clean checkout on another machine (proves reproducibility).

## Phase 7 (Stretch) — Active Learning Loop
- Review queue, correction ingestion, periodic retraining script.
- **Exit criteria:** a documented before/after metric improvement after
  one retraining cycle using corrected data.

## Suggested Pacing

| Phase | Focus | Note |
|---|---|---|
| 0-1 | Plumbing | Should be quick; don't over-engineer |
| 2 | Baseline | Resist the urge to skip — you need this number |
| 3 | The hardest phase | Budget the most time here |
| 4 | Fastest to feel "done" | Good motivation boost after Phase 3 |
| 5-6 | Engineering polish | This is what differentiates you in interviews |
| 7 | Optional | Only if time allows |
