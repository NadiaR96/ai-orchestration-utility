from orchestrator import Orchestrator

if __name__ == "__main__":
    orchestrator = Orchestrator()
    task = {"input": "Explain multi-agent AI systems in simple terms"}
    result = orchestrator.process_task(task)

    print("\n=== RESULT ===")
    print(result["output"])
    print("\n=== EVALUATION ===")
    print(result["evaluation"])