# Email Router: AI-powered inbound email classification

> **Status: Under development.** This project is being built step by step as a
> learning project. Features, results and structure will change.

A zero-cost, local-first reimplementation of the Misumi USA / Atos agentic
email routing case study (originally built on Amazon Bedrock). The system
classifies inbound customer emails, extracts data from attachments, matches
customers, and routes low-confidence cases to human review.

All data in this repository is **synthetic**. No real customer data is used.

## Goal

Replace manual email triage with an AI-assisted workflow:

1. Classify each email as `order`, `quote`, `return`, `support` or `other`
2. Return a confidence score and a short reason for each decision
3. Extract fields (part number, quantity, PO number) from PDF attachments
4. Match the sender to a customer record
5. Route confident results automatically and send uncertain ones to a human

## Tech stack

- **LLM:** [Ollama](https://ollama.com) running `gemma3:4b` locally (no API cost)
- **Language:** Python 3.13
- **Validation:** Pydantic (structured, validated model output)
- **Planned:** pdfplumber (PDF extraction), pandas (evaluation), Streamlit (review dashboard)

The model layer is designed to be swappable, so Ollama can later be replaced
with Amazon Bedrock without changing the rest of the pipeline.

## Project structure

```
.
├── data/
│   └── emails.jsonl      # 50 synthetic labeled emails (5 classes x 10)
├── src/
│   ├── load_data.py      # loads and summarizes the dataset
│   └── classify.py       # LLM-based email classifier
├── eval/                 # evaluation scripts (planned)
├── docs/                 # architecture notes (planned)
└── requirements.txt
```

## Getting started

Requirements: Python 3.10+, [Ollama](https://ollama.com) installed and running.

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd <repo-folder>

# 2. Create a virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Download the model
ollama pull gemma3:4b

# 4. Inspect the dataset
python src/load_data.py

# 5. Try the classifier on a few emails
python src/classify.py
```

## Dataset

`data/emails.jsonl` contains one email per line with these fields:

| Field | Description |
|---|---|
| `id` | Unique identifier |
| `subject`, `body` | Email content (the only fields sent to the model) |
| `label` | Ground-truth class, used only for evaluation |
| `has_attachment` | Whether the email has an attachment |
| `hard` | Marks deliberately ambiguous examples (e.g. a quote approval that is really an order) |

## Roadmap

- [x] Step 0: Environment setup
- [x] Step 1: Define classes, generate synthetic data, data loader
- [ ] Step 2: Classifier prototype (code written, testing in progress)
- [ ] Step 3: Evaluation harness (accuracy, confusion matrix, prompt iteration)
- [ ] Step 4: Attachment data extraction
- [ ] Step 5: Customer matching
- [ ] Step 6: End-to-end pipeline
- [ ] Step 7: Human review dashboard
- [ ] Step 8: Architecture docs (target design on AWS)

## Results

Evaluation results will be added after Step 3.

## Acknowledgements

Inspired by the public case study
*"Misumi USA: Automating industrial component configuration and order routing
with Amazon Bedrock"* (Atos / AWS). This is an independent learning project and
is not affiliated with Misumi, Atos or AWS.
