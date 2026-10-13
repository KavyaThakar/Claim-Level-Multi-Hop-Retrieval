import re
from rank_bm25 import BM25Okapi

from data_loader import load_corpus


def tokenize(text):
    """
    Simple tokenizer for BM25 retrieval.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    return text.split()


class BM25Retriever:

    def __init__(self):

        print("🔄 Loading scientific corpus...")

        self.corpus = load_corpus()

        print("🔄 Preparing BM25 index...")

        self.documents = []

        for paper in self.corpus:

            title = paper.get("title", "")

            abstract = paper.get("abstract", [])

            if isinstance(abstract, list):
                abstract = " ".join(abstract)

            full_text = title + " " + abstract

            self.documents.append({
                "doc_id": paper.get("doc_id"),
                "title": title,
                "text": full_text
            })

        tokenized_corpus = [
            tokenize(doc["text"])
            for doc in self.documents
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)

        print(f"✅ BM25 index created for {len(self.documents)} papers")


    def retrieve(self, query, top_k=5):

        tokenized_query = tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []

        for index in ranked_indices:

            document = self.documents[index]

            results.append({
                "doc_id": document["doc_id"],
                "title": document["title"],
                "score": float(scores[index])
            })

        return results


if __name__ == "__main__":

    retriever = BM25Retriever()

    print("\n" + "=" * 70)
    print("🔎 SCIENTIFIC LITERATURE RETRIEVAL TEST")
    print("=" * 70)

    query = "0-dimensional biomaterials lack inductive properties."

    print("\n🧠 Query:")
    print(query)

    results = retriever.retrieve(query, top_k=5)

    print("\n📚 TOP RETRIEVED PAPERS:\n")

    for rank, result in enumerate(results, start=1):

        print(f"{rank}. Document ID: {result['doc_id']}")
        print(f"   Title: {result['title']}")
        print(f"   BM25 Score: {result['score']:.4f}")
        print()