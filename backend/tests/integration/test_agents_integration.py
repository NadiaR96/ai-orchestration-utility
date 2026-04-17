import unittest
from backend.agents.hf_agent import HuggingFaceAgent

class TestHuggingFaceAgentIntegration(unittest.TestCase):
    def setUp(self):
        # lightweight test model
        self.agent = HuggingFaceAgent(model_name="distilgpt2")

    def test_run_real_model(self):
        output = self.agent.run("Hello world")
        self.assertIsInstance(output, str)
        self.assertGreater(len(output), 0)