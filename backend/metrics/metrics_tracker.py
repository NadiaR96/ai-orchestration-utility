import nltk
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn


class MetricsTracker:
    def __init__(self):
        self._rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    def bert_score(self, candidate, reference, model_type='bert-base-uncased'):
        # Normalise inputs
        if isinstance(candidate, str):
            candidate = [candidate]
            if isinstance(reference, str):
                reference = [reference]

                # Safety: ensure equal lengths
                if len(candidate) != len(reference):
                    return 0.0

        try:
            _, _, F1 = bert_score_fn(
                candidate,
                reference,
                lang='en',
                model_type=model_type
            )

            # F1 is a tensor → safely convert
            return float(F1.mean().item())
        except Exception:
            return 0.0

    def rouge(self, output, reference):
        scores = self._rouge.score(reference, output)
        return {k: v.fmeasure for k, v in scores.items()}

    def bleu(self, output, reference):
        return len(set(output.split()) & set(reference.split())) / max(1, len(output.split()))

    def hallucination_rate(self, output, reference_tokens):
        out = set(output.split())
        ref = set(reference_tokens)
        return len(out - ref) / max(1, len(out))

    def faithfulness(self, output, chunks):
        context = " ".join(chunks)
        out = set(output.lower().split())
        ctx = set(context.lower().split())
        return len(out & ctx) / max(1, len(out))

    def diversity_score(self, text: str) -> float:
        tokens = nltk.word_tokenize(text)
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)

    def perplexity(self, candidate: str, reference: str) -> float:
        tokens = nltk.word_tokenize(candidate)
        if not tokens or not reference:
            return 0.0

        ref_tokens = set(nltk.word_tokenize(reference))
        overlap = sum(1 for t in tokens if t in ref_tokens)

        prob = overlap / len(tokens)

        # Clamp probability
        prob = max(prob, 1e-3)  # higher floor = more stable

        perplexity = 1 / prob

        # Hard cap to prevent explosion
        return min(perplexity, 50.0)

    def compute_all(self, output, reference, chunks):
        if isinstance(reference, list):
            reference = " ".join(reference)

        results = {}

        # Only compute reference-based metrics if reference exists
        if reference:
            results["bert_score"] = self.bert_score([output], [reference])
            results["bleu"] = self.bleu(output, reference)
            results["rouge"] = self.rouge(output, reference)
            results["perplexity"] = self.perplexity(output, reference)
            results["hallucination"] = self.hallucination_rate(output, reference.split())
        else:
            # Default safe values
            results["bert_score"] = 0.0
            results["bleu"] = 0.0
            results["rouge"] = 0.0
            results["perplexity"] = 0.0
            results["hallucination"] = 0.0

        # Always compute these
        results["faithfulness"] = self.faithfulness(output, chunks)
        results["diversity"] = self.diversity_score(output)

        return results
