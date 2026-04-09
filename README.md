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


ai-orchestration-utility/
├─ orchestrator.py
├─ metrics/
│ └─ metrics_tracker.py
├─ agents/
│ └─ ...
├─ tests/
│ ├─ test_agents.py
│ ├─ test_metrics.py
│ ├─ test_orchestrator.py
│ └─ test_agents_integration.py
├─ utils/
│ └─ setup_nltk.py
├─ requirements.txt
├─ Dockerfile
└─ README.md


- `utils/setup_nltk.py` → Ensures NLTK data (e.g., `punkt`) is available locally or in Docker.  
- `tests/` → Unit tests run in CI/CD; integration tests are optional.  

---

## **⚡ Quick Start**

### **1️⃣ Clone the repo**

```bash
git clone https://github.com/NadiaR96/ai-orchestration-utility.git
cd ai-orchestration-utility
2️⃣ Install Python dependencies
pip install -r requirements.txt
3️⃣ Setup NLTK data
python utils/setup_nltk.py
4️⃣ Run unit tests
python -m unittest discover -s tests -p "test_*.py"
5️⃣ Optional: Run in Docker
docker build -t ai-orchestration-utility:latest .
docker run --rm ai-orchestration-utility:latest
6️⃣ Run integration tests locally (optional)
python -m unittest discover -s tests -p "test_agents_integration.py"
🛠️ CI/CD Workflow
Runs on push or pull request to main branch
Steps:
Checkout code
Setup Python
Install dependencies
Setup NLTK resources
Run unit tests only
Build Docker image
Run Docker container for verification

Integration tests are excluded from CI/CD to keep pipelines fast.

📈 Extending the Platform
Add new agents → Place in agents/ and update orchestrator
Add new metrics → Add to metrics/metrics_tracker.py
Hugging Face integration → Replace placeholder agent logic with HF models
Monitoring & Logging → Extend orchestrator to track latency, cost, hallucination live
🎯 Why This Project Matters
Demonstrates multi-agent orchestration and RAG architecture
Provides production-ready evaluation metrics
Shows modern engineering skills: Docker, CI/CD, testing
Perfect for portfolio, blog posts, and interview demos
📄 License

MIT License. See LICENSE
 for details.