import json
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


MODEL_NAME = "BAAI/bge-m3"
METADATA_PATH = "output/metadata.jsonl"
SAVE_PATH = "output/product_embeddings.npy"


class Embedder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(self.device)
        self.model.eval()

    def encode(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.model(**inputs)

        emb = output.last_hidden_state.mean(dim=1)
        emb = F.normalize(emb, p=2, dim=1)

        return emb.cpu().numpy()


def main():
    embedder = Embedder()

    names = []

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            raw = record.get("raw_data", {})
            name = str(raw.get("품목명", ""))
            names.append(name)

    print("상품명 개수:", len(names))

    vectors = embedder.encode(names)

    np.save(SAVE_PATH, vectors)

    print("저장 완료:", SAVE_PATH)


if __name__ == "__main__":
    main()