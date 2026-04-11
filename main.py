from backend.orchestrator import Orchestrator

if __name__ == "__main__":
    orchestrator = Orchestrator()

    task = "Explain multi-agent AI systems in simple terms"

    result = orchestrator.process_task(
        task=task,
        metrics=["meteor"],  # optional
        retrieval="rag"
    )

    print("\n=== RESULT ===")
    print(result["output"])

    print("\n=== METRICS ===")
    print(result["metrics"])