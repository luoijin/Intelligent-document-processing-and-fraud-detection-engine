# Project Charter — Intelligent Document Processing & Fraud Detection Engine

## 1. Purpose

A portfolio-grade, production-shaped system that ingests unstructured
document images (receipts, invoices), extracts structured data via
CV + NLP, and flags statistically anomalous or fraudulent records via a
tabular ML model. The project exists to demonstrate **AI/ML Engineering**
competency — pipeline design, model integration, evaluation discipline, and
deployment — not novel research.

## 2. Problem Statement

Organizations across fintech, business operations, and government spend
significant manual effort converting paper/image-based documents into
verified structured data, and manually auditing that data for errors or
fraud. This project builds a minimal but real version of that pipeline.

## 3. Goals (in priority order)

| # | Goal | Why it matters for an AI/ML Engineer portfolio |
|---|------|-------------------------------------------------|
| 1 | Working end-to-end pipeline on one document type (receipts) | Proves you can ship a full system, not just a notebook |
| 2 | Fine-tuned entity extraction model with measured improvement over a baseline | Proves you understand evaluation, not just training |
| 3 | Tabular anomaly/fraud detection model with engineered features | Proves classic ML skill, not just deep learning |
| 4 | Served behind an API with logging | Proves you understand deployment, not just modeling |
| 5 | (Stretch) Active learning / retraining loop | Proves you understand the ML lifecycle |

## 4. Non-Goals (explicitly out of scope for v1)

- Multi-document-type generalization (invoices, tax forms, IDs) — v2+
- Handwriting recognition — v2+
- Real-time streaming ingestion
- Multi-user auth, billing, or a production-grade frontend
- Any paid API, paid compute, or paid dataset (see §6 Cost Constraint)

## 5. Success Criteria

A v1 is "done" when:
1. A receipt image can be POSTed to a local API and a structured JSON
   response with extracted fields + a fraud/anomaly score is returned.
2. The extraction model has a documented accuracy/F1 measured against a
   held-out labeled set, compared against a rule-based baseline.
3. The anomaly detector has been validated against injected synthetic
   fraud cases with a documented precision/recall.
4. The whole pipeline runs from a single command (`docker compose up` or
   equivalent) with no manual steps.

## 6. Cost Constraint

**This project must be buildable and runnable at $0.** Every tool,
dataset, model, and hosting choice in the companion documents is selected
to have a genuinely free tier or be fully open-source/self-hostable. See
`05-DEPLOYMENT-FREE-TIER.md` for the specific free-tier stack and its
limits, and `02-DATA-STRATEGY.md` for free dataset sources.

## 7. Roles

Solo project — you act as Data Engineer, ML Engineer, Backend Engineer,
and QA simultaneously. Documents in this set are written so each phase
can be picked up independently once the previous one's exit criteria are met.

## 8. Companion Documents

| Document | Covers |
|---|---|
| `01-ARCHITECTURE.md` | System design, components, data flow |
| `02-DATA-STRATEGY.md` | Datasets, schema, labeling process |
| `03-MODEL-DEVELOPMENT.md` | OCR, NER/LayoutLM, anomaly model plans |
| `04-API-SPEC.md` | Service contract / endpoints |
| `05-DEPLOYMENT-FREE-TIER.md` | Free hosting & infra choices |
| `06-ROADMAP-MILESTONES.md` | Phased plan with exit criteria |
| `07-TESTING-EVALUATION.md` | Metrics, test plan, QA checklist |
