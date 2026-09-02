class OpenAIEmbeddingProvider:
    def __init__(self, client, model_name):
        self.client = client
        self.model_name = model_name
        self.last_usage = {"prompt_tokens": 0, "total_tokens": 0}

    def embed(self, text: str):
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]):
        result = self.client.embeddings.create(
            input=texts,
            model=self.model_name,
        )
        usage = getattr(result, "usage", None)
        self.last_usage = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }
        return [item.embedding for item in sorted(result.data, key=lambda item: item.index)]