class EvaluatorAgent:
    """
    Evaluates outputs from other agents and returns a score or feedback.
    """
    def __init__(self):
        pass

    def evaluate(self, input_text: str, agent_output: str) -> dict:
        """
        Simple scoring example:
        - Checks length
        - Simple keyword presence
        """
        score = 0
        feedback = []

        if len(agent_output.split()) > 5:
            score += 1
        else:
            feedback.append("Output too short.")

        if "AI" in agent_output:
            score += 1
        else:
            feedback.append("Missing key term 'AI'.")

        return {"score": score, "feedback": feedback}