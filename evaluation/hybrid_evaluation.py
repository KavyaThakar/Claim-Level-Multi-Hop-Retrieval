import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from data_loader import load_claims
from prepare_dataset import extract_claim_evidence
from hybrid_retrieval import HybridRetriever


def group_claims_by_relevant_docs(prepared_data):
    """
    Group all relevant document IDs belonging to the same claim.
    """

    grouped = {}

    for item in prepared_data:

        claim_id = item["claim_id"]

        if claim_id not in grouped:
            grouped[claim_id] = {
                "claim": item["claim"],
                "relevant_docs": set()
            }

        grouped[claim_id]["relevant_docs"].add(
            item["doc_id"]
        )

    return grouped


def precision_at_k(retrieved_docs, relevant_docs, k):

    retrieved_k = retrieved_docs[:k]

    relevant_found = sum(
        1 for doc_id in retrieved_k
        if doc_id in relevant_docs
    )

    return relevant_found / k


def reciprocal_rank(retrieved_docs, relevant_docs):

    for rank, doc_id in enumerate(
        retrieved_docs,
        start=1
    ):

        if doc_id in relevant_docs:
            return 1 / rank

    return 0


def evaluate(grouped_claims, retriever, k=5):

    precision_scores = []
    reciprocal_ranks = []

    total = len(grouped_claims)

    for index, (_, item) in enumerate(
        grouped_claims.items(),
        start=1
    ):

        claim = item["claim"]
        relevant_docs = item["relevant_docs"]

        results = retriever.retrieve(
            claim,
            top_k=k
        )

        retrieved_docs = [
            result["doc_id"]
            for result in results
        ]

        precision = precision_at_k(
            retrieved_docs,
            relevant_docs,
            k
        )

        rr = reciprocal_rank(
            retrieved_docs,
            relevant_docs
        )

        precision_scores.append(precision)
        reciprocal_ranks.append(rr)

        if index % 20 == 0:
            print(f"Processed {index}/{total} claims")

    avg_precision = (
        sum(precision_scores)
        / len(precision_scores)
    )

    mrr = (
        sum(reciprocal_ranks)
        / len(reciprocal_ranks)
    )

    return avg_precision, mrr


if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("📊 HYBRID RETRIEVAL EVALUATION")
    print("=" * 70)

    print("\n🔄 Loading SciFact development claims...")

    dev_claims = load_claims("dev")

    prepared_data = extract_claim_evidence(
        dev_claims
    )

    grouped_claims = group_claims_by_relevant_docs(
        prepared_data
    )

    print(
        f"🧠 Unique claims to evaluate: "
        f"{len(grouped_claims)}"
    )

    print("\n🔄 Initializing Hybrid Retriever...")

    retriever = HybridRetriever()

    print("\n🚀 Starting evaluation...\n")

    precision, mrr = evaluate(
        grouped_claims,
        retriever,
        k=5
    )

    print("\n" + "=" * 70)
    print("📈 FINAL RESULTS")
    print("=" * 70)

    print(f"\nPrecision@5: {precision:.4f}")
    print(f"MRR@5:       {mrr:.4f}")

    print("\n✅ Hybrid evaluation completed!")
    