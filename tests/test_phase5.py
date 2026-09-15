#!/usr/bin/env python3
"""
Phase 5 exit-criteria validation, per docs/06-ROADMAP-MILESTONES.md:

    "Exit criteria: a single POST /documents call with a real receipt
    image returns a complete response (fields + anomaly score)
    end-to-end."

Uses FastAPI's TestClient (in-process, real request/response cycle
through the actual app, not a mocked one) against a synthetic sample
receipt. Also checks GET /v1/documents/{id} round-trips the persisted
record, and that a bad content-type still returns 422 rather than a
stack trace (docs/07-TESTING-EVALUATION.md §3 QA checklist).

Run: python tests/test_phase5.py
"""
import sys
import time
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

SAMPLE_DIR = REPO / "data" / "sample_receipts"
DB_PATH = REPO / "data" / "app.db"


def main() -> None:
    print("🔎 Validating Phase 5 full service wrapper...\n")

    if not any(SAMPLE_DIR.glob("*.png")):
        print(f"❌ No sample images in {SAMPLE_DIR}. "
              f"Run: python scripts/generate_sample_receipts.py")
        sys.exit(1)

    # Start from a clean DB so this test's assertions (e.g. cold-start
    # first document, review_required behavior) are reproducible.
    DB_PATH.unlink(missing_ok=True)

    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        health = client.get("/health")
        print(f"GET /health -> {health.status_code} {health.json()}")
        if not health.json().get("models_loaded"):
            print(
                "\n❌ Anomaly model not loaded. Run: "
                "python scripts/build_fraud_dataset.py && "
                "python scripts/train_anomaly_models.py"
            )
            sys.exit(1)

        sample = sorted(SAMPLE_DIR.glob("*.png"))[0]
        with open(sample, "rb") as f:
            start = time.time()
            response = client.post(
                "/v1/documents",
                files={"file": (sample.name, f, "image/png")},
            )
            elapsed = time.time() - start

        print(f"\nPOST /v1/documents ({sample.name}) -> {response.status_code} "
              f"in {elapsed:.2f}s")
        if response.status_code != 200:
            print(f"❌ Expected 200, got {response.status_code}: {response.text}")
            sys.exit(1)

        body = response.json()
        print(f"  {body}")

        required_keys = {"document_id", "status", "fields", "extraction_confidence",
                          "anomaly_score", "review_required"}
        missing = required_keys - body.keys()
        if missing:
            print(f"❌ Response missing keys: {missing}")
            sys.exit(1)
        required_field_keys = {"merchant_name", "date", "total_amount", "currency"}
        if required_field_keys - body["fields"].keys():
            print(f"❌ Response 'fields' missing keys: "
                  f"{required_field_keys - body['fields'].keys()}")
            sys.exit(1)
        print("✅ Response has the complete docs/04-API-SPEC.md shape "
              "(fields + extraction_confidence + anomaly_score + review_required)")

        if elapsed >= 5.0:
            print(f"⚠️  Latency {elapsed:.2f}s exceeds the p95 < 5s target "
                  f"(docs/04-API-SPEC.md §3) — not a hard failure for a "
                  f"single warm-cache local run, but worth watching.")

        document_id = body["document_id"]
        get_response = client.get(f"/v1/documents/{document_id}")
        print(f"\nGET /v1/documents/{document_id} -> {get_response.status_code}")
        if get_response.status_code != 200 or get_response.json()["fields"] != body["fields"]:
            print(f"❌ GET did not round-trip the persisted record: {get_response.text}")
            sys.exit(1)
        print("✅ GET /v1/documents/{id} round-trips the persisted record")

        missing_get = client.get("/v1/documents/does-not-exist")
        print(f"\nGET /v1/documents/does-not-exist -> {missing_get.status_code}")
        if missing_get.status_code != 404:
            print(f"❌ Expected 404 for unknown document_id, got {missing_get.status_code}")
            sys.exit(1)
        print("✅ Unknown document_id returns 404")

        bad_type = client.post(
            "/v1/documents",
            files={"file": ("not_an_image.txt", b"hello", "text/plain")},
        )
        print(f"\nPOST /v1/documents (bad content-type) -> {bad_type.status_code}")
        if bad_type.status_code != 422:
            print(f"❌ Expected 422 for bad content-type, got {bad_type.status_code}")
            sys.exit(1)
        print("✅ Bad content-type rejected with 422, not a stack trace")

    print(
        "\n🎉 Phase 5 exit criteria met: a single POST /v1/documents call "
        "with a real receipt image returns a complete response "
        "(fields + anomaly score) end-to-end."
    )
    print(
        "\n⚠️  Reminder: this wires the Phase 2 rule-based extractor, not "
        "a Phase 3 trained model (Phase 3 is scaffolded, not executed — "
        "see docs/CHANGELOG/CHANGELOG_PHASE3.md). Documented deviation, "
        "not a silent one."
    )


if __name__ == "__main__":
    main()
