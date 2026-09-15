# Changelog — Phase 5: Service Wrapper

**Roadmap reference:** `docs/06-ROADMAP-MILESTONES.md`, Phase 5
**Exit criteria (roadmap):** "A single `POST /documents` call with a
real receipt image returns a complete response (fields + anomaly
score) end-to-end."
**Status:** ✅ Met — see [Validation](#validation) below.

---

## Summary

Wires Phase 1 (OCR) → Phase 2 (baseline extraction) → Phase 4 (feature
building + anomaly scoring) → a new SQLite Result Store → a new
`POST /v1/documents` endpoint matching `docs/04-API-SPEC.md`'s
`POST /documents` contract, plus `GET /v1/documents/{document_id}` for
retrieval.

## Documented Deviation: Phase 3 Is Not Wired In

`docs/06-ROADMAP-MILESTONES.md` describes Phase 5 as wiring
"Phases 1-4" — at the time this was built, Phase 3 (trained LayoutLMv3
extractor) is scaffolded but not executed (see `CHANGELOG_PHASE3.md`):
no GPU and no network access to `huggingface.co` in the build
environment, both required to actually fine-tune the model. Rather
than block Phase 5 on an infrastructure gap outside this project's
control, `POST /v1/documents` wires the Phase 2 rule-based extractor
instead.

This is **not** a silent substitution: `extract_fields()` returns the
same `ExtractedFields` shape regardless of whether it's the rule-based
implementation or a future Phase 3 model, so swapping one in is a
one-line import change in `app/main.py`'s `create_document()`, not a
re-architecture. Once Phase 3 actually produces a fine-tuned model
(per its own Next Steps), re-point that import and re-run
`tests/test_phase5.py` — no other change needed.

## New Components

### `app/store.py` — Result Store
SQLite (`data/app.db`, gitignored) per `docs/01-ARCHITECTURE.md` §2.7 /
`docs/05-DEPLOYMENT-FREE-TIER.md`. One `documents` table. Exposes
`insert_document`, `get_document`, `get_prior_records_for_merchant`,
`get_all_record_hashes` — persistence only, no field extraction or
scoring logic lives here.

### `POST /v1/documents`
Full pipeline. Feature building at request time uses **live** merchant
history from the Result Store (not the frozen Phase 4 training stats)
— a fresh deployment starts with zero history per merchant and
exercises `app/features.py`'s cold-start path (`amount_zscore = 0.0`)
by design; stats sharpen as more documents are persisted. This mirrors
how a real system accumulates history, rather than baking in Phase 4's
synthetic training distribution.

**Anomaly model choice:** Isolation Forest, not XGBoost. Per
`CHANGELOG_PHASE4.md`, the two tied on best-F1 (0.667) but at different
operating points — IF: precision=1.000, recall=0.500; XGBoost:
precision=0.778, recall=0.583. `docs/03-MODEL-DEVELOPMENT.md` §3.2
states false-positive rate matters more than raw accuracy in a
fraud-review context, so the perfectly-precise model was chosen for
production scoring. Documented choice, revisit if XGBoost's precision
improves on more/real data.

**Review threshold:** `0.6`, matching Isolation Forest's own best-F1
threshold from the Phase 4 validation sweep — a documented v1 default,
not derived from real-world fraud outcomes (none exist yet; see
`docs/07-TESTING-EVALUATION.md` §4).

### `GET /v1/documents/{document_id}`
Retrieval by id, 404 if unknown. Not in the roadmap's narrow Phase 5
exit criterion but a natural, low-cost pairing with the Result Store —
`GET /documents` (list+filter) and `POST /documents/{id}/review`
(active-learning correction ingestion) from `docs/04-API-SPEC.md` are
deliberately deferred to Phase 7 (Review Queue) scope, not built here,
to avoid scope creep beyond this phase's exit criterion.

## Files Added / Modified

| File | Change |
|---|---|
| `app/store.py` | New — SQLite Result Store |
| `app/main.py` | Added startup hook (DB init + anomaly model load), `POST /v1/documents`, `GET /v1/documents/{document_id}`; bumped version 0.3.0 → 0.5.0 |
| `requirements.txt` | Added `scikit-learn`, `xgboost`, `joblib` — these are now genuine inference-time deps (Isolation Forest scores every request), unlike Phase 3's training-only `requirements-phase3.txt` |
| `tests/test_phase5.py` | Exit-criteria test (TestClient, in-process real request cycle) |
| `docs/CHANGELOG/CHANGELOG_PHASE5.md` | This file |

Not committed (gitignored, regenerable): `data/app.db`.

## Known Gaps

- No auth (documented v1 gap per project instructions §2.7 — not
  silently fixed here).
- `GET /v1/documents` (list/filter) and
  `POST /v1/documents/{id}/review` from `docs/04-API-SPEC.md` are not
  implemented — deferred to Phase 7 scope.
- Cold-start merchant stats mean the very first few documents for any
  merchant get a neutral `amount_zscore` regardless of how unusual
  their amount actually is — expected and documented, not a bug.
- No file-size limit on uploads.
- `record_hash`-based duplicate detection is exact-match only (same
  limitation noted in `CHANGELOG_PHASE4.md`).

## Validation

```
$ python tests/test_phase5.py
GET /health -> 200 {'status': 'ok', 'models_loaded': True}

POST /v1/documents (sample_receipt_01.png) -> 200 in 0.48s
  {'document_id': '829704c9-...', 'status': 'processed',
   'fields': {'merchant_name': 'GROCERY MART', 'date': '2026-09-01',
              'total_amount': 299.5, 'currency': None},
   'extraction_confidence': 1.0, 'anomaly_score': 0.0,
   'review_required': False}
✅ Response has the complete docs/04-API-SPEC.md shape

GET /v1/documents/829704c9-... -> 200
✅ GET /v1/documents/{id} round-trips the persisted record

GET /v1/documents/does-not-exist -> 404
✅ Unknown document_id returns 404

POST /v1/documents (bad content-type) -> 422
✅ Bad content-type rejected with 422, not a stack trace

🎉 Phase 5 exit criteria met.
```

0.48s well under the < 5s p95 target (`docs/04-API-SPEC.md` §3), on
local CPU, single warm request — not a statistically meaningful p95
sample, just confirms no gross latency problem.

## Next Phase

Phase 6 (Deployment) — containerize and confirm `docker compose up`
brings up this full service from a clean checkout on another machine.
Note: the `Dockerfile`/`docker-compose.yml` predate Phase 4/5's new
`scikit-learn`/`xgboost` dependencies and models directory; re-verify
the container build picks these up before calling Phase 6 done.
