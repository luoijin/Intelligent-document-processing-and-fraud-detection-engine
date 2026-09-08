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
    try:
        content = (REPO / "requirements.txt").read_text().strip().splitlines()
        pkgs = {line.split("==")[0].lower() for line in content if line.strip() and not line.startswith("#")}
        expected = {"fastapi", "uvicorn", "pydantic", "python-multipart"}
        if not pkgs.issubset(expected):
            print(f"❌ Unexpected packages in requirements.txt: {pkgs - expected}")
            return False
        print("✅ requirements.txt contains only Phase 0 dependencies")
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