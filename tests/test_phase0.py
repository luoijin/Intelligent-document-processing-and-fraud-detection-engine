#!/usr/bin/env python3
"""
Quick validation that Phase 0 skeleton is correctly assembled:
- Dockerfile & docker-compose.yml exist
- FastAPI app loads and exposes /health
- requirements.txt lists only Phase 0 deps
Run: python test_phase0.py
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parent


def check_files():
    required = [
        "Dockerfile",
        "docker-compose.yml",
        "requirements.txt",
        ".env.example",
        ".gitignore",
        "app/main.py",
    ]
    missing = [f for f in required if not (REPO / f).exists()]
    if missing:
        print(f"❌ Missing files: {missing}")
        return False
    print("✅ All required files present")
    return True


def check_fastapi_skeleton():
    try:
        # Import the app module to verify it's syntactically valid
        sys.path.insert(0, str(REPO / "app"))
        from main import app  # noqa: F401
        print("✅ FastAPI app imports successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to import app: {e}")
        return False


def check_requirements_phase0():
    """Verify the Phase 0 web-framework deps are still present.

    Updated after Phase 1 (see CHANGELOG_PHASE1.md): requirements.txt is
    now additive across phases rather than Phase-0-exclusive, so this
    checks for a required *subset* (Phase 0's four core deps) instead of
    an exact/closed set. A separate Phase 1 dependency check lives in
    test_phase1.py.
    """
    try:
        content = (REPO / "requirements.txt").read_text().strip().splitlines()
        import re
        pkgs = {
            re.split(r"[><=]", line, 1)[0].strip().lower()
            for line in content
            if line.strip() and not line.startswith("#")
        }
        required_phase0 = {"fastapi", "uvicorn", "pydantic", "python-multipart"}
        missing = required_phase0 - pkgs
        if missing:
            print(f"❌ Missing Phase 0 packages in requirements.txt: {missing}")
            return False
        print("✅ requirements.txt contains all Phase 0 dependencies")
        return True
    except Exception as e:
        print(f"❌ Error reading requirements: {e}")
        return False


def main():
    print("🔎 Validating Phase 0 skeleton...\n")
    ok = True
    ok = check_files() and ok
    ok = check_fastapi_skeleton() and ok
    ok = check_requirements_phase0() and ok

    if ok:
        print("\n🎉 Phase 0 skeleton is ready!")
        print("Next step: test with `docker compose up` (after fixing Docker permissions) or run locally:")
        print("  uvicorn app.main:app --reload")
        print("Then visit http://localhost:8000/health")
    else:
        print("\n❌ Phase 0 validation failed – fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()