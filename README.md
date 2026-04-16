# AI Orchestration Utility

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A lightweight **multi-agent AI orchestration platform** with metrics tracking, Docker, and CI/CD integration.  
Designed for **production-ready experimentation with LLMs**, evaluation of outputs, and orchestration of complex AI tasks.

---

## **🚀 Features**

- **Multi-Agent Orchestration**  
  Run multiple instances of AI agents concurrently with flexible task assignment.

- **Metrics Tracking**  
  Evaluate AI outputs using:
  - BLEU  
  - METEOR  
  - ROUGE  
  - Cosine Similarity  
  - Diversity Score  
  - Coverage Score  
  - Hallucination Rate  
  - F1, Precision, Recall  

- **Dockerized Environment**  
  Fully reproducible builds, including NLTK resources.

- **CI/CD Ready**  
  - Unit tests run automatically on GitHub Actions  
  - Integration tests run locally (excluded from CI/CD for speed)  

- **Extensible**  
  Add new agents, metrics, or connectors with minimal effort.

---

## **📂 Repository Structure**

```text
ai-orchestration-utility/
├─ backend/
│  ├─ agents/
│  ├─ api/
│  ├─ core/
│  ├─ evaluators/
│  ├─ experiments/
│  ├─ metrics/
│  ├─ models/
│  ├─ orchestrator/
│  ├─ rag/
│  ├─ scoring/
│  └─ tests/
│     ├─ unit/
│     └─ integration/
├─ frontend/
├─ utils/
│  └─ setup_nltk.py
├─ requirements.txt
├─ Dockerfile
└─ README.md
```

- `backend/tests/unit/` → Unit tests used in CI/CD.
- `backend/tests/integration/` → Integration tests for local validation.
- `utils/setup_nltk.py` → Ensures NLTK data (e.g., `punkt`) is available locally or in Docker.

---

## **⚡ Quick Start**

### **1️⃣ Clone the repo**

```bash
git clone https://github.com/NadiaR96/ai-orchestration-utility.git
cd ai-orchestration-utility
```

### **2️⃣ Install Python dependencies**

```bash
pip install -r requirements.txt
```

### **3️⃣ Setup NLTK data**

```bash
python utils/setup_nltk.py
```

### **4️⃣ Run unit tests**

```bash
python -m unittest discover -s backend/tests/unit -p "test_*.py"
```

### **5️⃣ Optional: Run full test suite**

```bash
python -m unittest discover -s backend/tests -p "test_*.py"
```

### **6️⃣ Optional: Run in Docker**

```bash
docker build -t ai-orchestration-utility:latest .
docker run --rm ai-orchestration-utility:latest
```

## **🛠️ CI/CD Workflow**

Runs on push or pull request to main branch.

Steps:
- Checkout code
- Setup Python
- Install dependencies
- Setup NLTK resources
- Run unit tests
- Build Docker image
- Run Docker container for verification

Integration tests run on `workflow_dispatch` only (they load real HuggingFace models and require secrets).

## **📈 Extending the Platform**

- Add new agents: place implementations in `backend/agents/` and route them in `backend/orchestrator/orchestrator.py`.
- Add new metrics: implement metric logic in `backend/metrics/metrics_tracker.py`.
- Add new scoring strategies: add scorer classes in `backend/scoring/` and register them in `backend/scoring/registry.py`.
- Extend experiment workflows: use `backend/experiments/` and expose routes through `backend/api/`.

## **🎯 Why This Project Matters**

- Demonstrates multi-agent orchestration and RAG architecture.
- Provides a practical evaluation stack for quality, hallucination, and efficiency metrics.
- Shows production fundamentals: test coverage, API boundaries, and containerized execution.

## **📄 License**

MIT License. See LICENSE
 for details.

---

## **🔐 Environment Variables**

The backend automatically loads a .env file from the repository root when the backend package is imported.

For Hugging Face access, the preferred key is HF_TOKEN.

Supported token keys:
- HF_TOKEN
- HUGGINGFACEHUB_API_TOKEN
- HUGGINGFACE_HUB_TOKEN

If HF_TOKEN is missing but one of the alias keys is present, the backend maps it to HF_TOKEN automatically.