import re
import numpy as np

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

from data_loader import load_corpus


def tokenize(text):
    """
    Tokenize text for BM25.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    return text.split()


class HybridRetriever:

    def __init__(self):

        print("🔄 Loading scientific corpus...")

        self.corpus = load_corpus()

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

        # -----------------------------
        # BM25 INDEX
        # -----------------------------

        print("⚡ Building BM25 index...")

        tokenized_corpus = [
            tokenize(doc["text"])
            for doc in self.documents
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)

        # -----------------------------
        # DENSE SEMANTIC MODEL
        # -----------------------------

        print("🧠 Loading semantic model...")

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        print("🧠 Creating document embeddings...")

        document_texts = [
            doc["text"]
            for doc in self.documents
        ]

        self.document_embeddings = self.model.encode(
            document_texts,
            convert_to_tensor=True,
            show_progress_bar=True
        )

        print(
            f"✅ Hybrid retrieval system ready "
            f"for {len(self.documents)} papers!"
        )


    def bm25_search(self, query, top_k=20):

        tokenized_query = tokenize(query)

        scores = self.bm25.get_scores(
            tokenized_query
        )

        ranked_indices = np.argsort(scores)[::-1][:top_k]

        return ranked_indices


    def semantic_search(self, query, top_k=20):

        query_embedding = self.model.encode(
            query,
            convert_to_tensor=True
        )

        cosine_scores = util.cos_sim(
            query_embedding,
            self.document_embeddings
        )[0]

        top_results = np.argsort(
            cosine_scores.cpu().numpy()
        )[::-1][:top_k]

        return top_results


    def retrieve(self, query, top_k=5, candidate_k=20):

        # BM25 candidates
        bm25_indices = self.bm25_search(
            query,
            candidate_k
        )

        # Semantic candidates
        semantic_indices = self.semantic_search(
            query,
            candidate_k
        )

        # ---------------------------------
        # Reciprocal Rank Fusion
        # ---------------------------------

        rrf_scores = {}

        k_rrf = 60

        for rank, index in enumerate(
            bm25_indices,
            start=1
        ):

            rrf_scores[index] = (
                rrf_scores.get(index, 0)
                + 1 / (k_rrf + rank)
            )

        for rank, index in enumerate(
            semantic_indices,
            start=1
        ):

            rrf_scores[index] = (
                rrf_scores.get(index, 0)
                + 1 / (k_rrf + rank)
            )

        # Sort fused results
        ranked_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        results = []

        for index, score in ranked_results[:top_k]:

            document = self.documents[index]

            results.append({
                "doc_id": document["doc_id"],
                "title": document["title"],
                "score": float(score)
            })

        return results


if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("🔬 HYBRID SCIENTIFIC RETRIEVAL")
    print("=" * 70)

    retriever = HybridRetriever()

    query = (
        "0-dimensional biomaterials lack "
        "inductive properties."
    )

    print("\n🧠 QUERY:")
    print(query)

    print("\n📚 TOP RETRIEVED PAPERS:\n")

    results = retriever.retrieve(
        query,
        top_k=5
    )

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"{rank}. Document ID: "
            f"{result['doc_id']}"
        )

        print(
            f"   Title: {result['title']}"
        )

        print(
            f"   Hybrid Score: "
            f"{result['score']:.6f}"
        )

        print()