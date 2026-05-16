from __future__ import annotations

class FastEmbedder:
    _instance: "FastEmbedder | None" = None

    def __new__(cls) -> "FastEmbedder":
        if cls._instance is None:
            from sentence_transformers import SentenceTransformer

            instance = super().__new__(cls)
            instance.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            cls._instance = instance
        return cls._instance

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self.model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

    def embed_one(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]
