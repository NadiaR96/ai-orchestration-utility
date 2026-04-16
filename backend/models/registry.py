MODEL_REGISTRY = {
    "small": "distilgpt2",
    "default": "google/flan-t5-base",
    "quality": "facebook/bart-large-cnn"
}


def get_model(name: str):
    return MODEL_REGISTRY.get(name, MODEL_REGISTRY["small"])