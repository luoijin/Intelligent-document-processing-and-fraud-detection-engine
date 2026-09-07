# Deployment — Free-Tier Stack

This project is designed to run entirely on free tiers. This document
lists the specific choices and their limits so you know when you'd need
to upgrade (out of scope for this project, but good to know).

## 1. Local Development (primary target)

- **Everything runs locally first.** `docker compose up` should bring up
  the API + SQLite with zero paid dependencies. This is the required,
  always-free path — treat any hosted deployment below as optional polish.

## 2. Containerization

- **Docker** (free) — one `Dockerfile` for the API service.
- **docker-compose** — orchestrates the API container + volume-mounted
  SQLite file. No paid orchestration needed at this scale.

## 3. Optional Hosted Demo (if you want a shareable link)

| Layer | Free-tier option | Limit to know about |
|---|---|---|
| API hosting | Render.com free web service, or Railway free tier, or Hugging Face Spaces (Docker SDK) | Free instances sleep after inactivity; cold starts ~30-60s |
| Database | Stay on SQLite (file-based) for a demo, or Supabase/Neon free Postgres tier | Free Postgres tiers cap storage (~500MB-1GB) — plenty for a portfolio demo |
| Model hosting | Bundle model weights in the container image (small models) or load from Hugging Face Hub free hosting at runtime | Large model images may exceed free-tier container size limits — keep LayoutLM in fp16/quantized if needed |
| Frontend (optional) | Static demo page on GitHub Pages or Vercel free tier | N/A for this project's scope |

## 4. CI (optional, still free)

- GitHub Actions free tier (2,000 minutes/month for public repos) for:
  - Linting
  - Running the test suite from `07-TESTING-EVALUATION.md`
  - Building the Docker image

## 5. Explicit Cost Guardrails

- Do **not** call paid OCR/vision APIs (Google Vision, AWS Textract,
  Azure Form Recognizer) even for comparison — they have free trial
  credits but are not free long-term and are outside this project's
  constraint.
- Do **not** use GPT-4V/Claude/other hosted LLM APIs for extraction in
  this project's core pipeline — use open, locally-runnable models so
  the whole system remains reproducible at zero cost. (Fine — and
  encouraged — to use an LLM API as a *design/coding assistant* while
  building; that's separate from what the shipped system depends on.)
- Colab/Kaggle free GPU sessions are time-limited and can disconnect —
  checkpoint model training regularly so you don't lose progress.

## 6. Environment Configuration

Use a `.env.example` file (never commit a real `.env`) documenting:
```
DATABASE_URL=sqlite:///./data.db
MODEL_DIR=./models
OCR_ENGINE=easyocr
LOG_LEVEL=info
```
