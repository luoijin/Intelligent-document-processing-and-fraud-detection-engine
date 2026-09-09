#!/usr/bin/env python3
"""
Phase 1 exit-criteria validation, per docs/06-ROADMAP-MILESTONES.md:

    "Exit criteria: given any SROIE image, the pipeline outputs raw
    text + bounding boxes with no crashes on 100% of the sample set."

Since the real SROIE dataset must be downloaded by the user (see
scripts/generate_sample_receipts.py docstring), this validates the
same pipeline (app.preprocessing -> app.ocr) against the generated
synthetic sample set as a stand-in. Swapping in real SROIE images
under data/raw/ requires no code change — only pointing SAMPLE_DIR
at the new folder.

Run: python test_phase1.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).parent
sys.path.insert(0, str(REPO))

SAMPLE_DIR = REPO / "data" / "sample_receipts"


def check_samples_exist() -> bool:
    if not SAMPLE_DIR.exists() or not any(SAMPLE_DIR.glob("*.png")):
        print(f"❌ No sample images found in {SAMPLE_DIR}. "
              f"Run: python scripts/generate_sample_receipts.py")
        return False
    print(f"✅ Sample images found in {SAMPLE_DIR}")
    return True


def check_pipeline_runs_on_all_samples() -> bool:
    from app.ocr import run_ocr

    images = sorted(SAMPLE_DIR.glob("*.png"))
    total = len(images)
    passed = 0
    for img_path in images:
        try:
            result = run_ocr(str(img_path))
            has_tokens = len(result.tokens) > 0
            has_bboxes = all(
                t.width > 0 and t.height > 0 for t in result.tokens
            ) if has_tokens else False
            if has_tokens and has_bboxes:
                passed += 1
                print(f"  ✅ {img_path.name}: {len(result.tokens)} tokens, "
                      f"avg_confidence={result.avg_confidence:.1f}")
            else:
                print(f"  ❌ {img_path.name}: no tokens/bboxes extracted")
        except Exception as e:
            print(f"  ❌ {img_path.name}: raised {type(e).__name__}: {e}")

    print(f"\n{passed}/{total} sample images processed successfully "
          f"with text + bounding boxes.")
    return total > 0 and passed == total


def check_preprocessing_handles_skew() -> bool:
    """Sanity check that deskew doesn't error on rotated samples
    (sample_receipt_02 and _03 are generated with nonzero skew)."""
    from app.preprocessing import preprocess

    skewed = list(SAMPLE_DIR.glob("sample_receipt_0[23].png"))
    if not skewed:
        print("⚠️  No skewed samples found to validate deskew path (non-fatal)")
        return True
    try:
        for path in skewed:
            preprocess(str(path))
        print(f"✅ Deskew path handled {len(skewed)} rotated sample(s) without error")
        return True
    except Exception as e:
        print(f"❌ Deskew raised {type(e).__name__}: {e}")
        return False


def main() -> None:
    print("🔎 Validating Phase 1 vision extraction layer...\n")
    ok = True
    ok = check_samples_exist() and ok
    if ok:
        ok = check_pipeline_runs_on_all_samples() and ok
        ok = check_preprocessing_handles_skew() and ok

    if ok:
        print("\n🎉 Phase 1 exit criteria met: pipeline produces raw text + "
              "bounding boxes with no crashes on 100% of the sample set.")
        print("Next: python -m uvicorn app.main:app --reload, then POST an "
              "image to http://localhost:8000/v1/ocr/extract")
    else:
        print("\n❌ Phase 1 validation failed — fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
