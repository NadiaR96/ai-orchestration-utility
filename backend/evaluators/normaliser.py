class Normaliser:
    def normalise(self, metrics: dict) -> dict:
        flat = {}

        for k, v in metrics.items():

            # dict metrics (ROUGE)
            if isinstance(v, dict):
                flat[k] = sum(v.values()) / len(v)

            # tensors
            elif hasattr(v, "item"):
                flat[k] = float(v.item())

            # floats / ints
            elif isinstance(v, (int, float)):
                flat[k] = float(v)

            else:
                flat[k] = 0.0
                
        cost = flat.get("cost", 0.0)
        latency = flat.get("latency", 0.0)

        flat["cost_norm"] = min(cost, 1.0)  # ok for now
        flat["latency_norm"] = min(latency / 5.0, 1.0)
        flat["quality_norm"] = (
             0.5 * flat.get("bert_score", 0)
             + 0.3 * flat.get("faithfulness", 0)
        )

        return flat