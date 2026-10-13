import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from data_loader import load_claims
from prepare_dataset import extract_claim_evidence
from retrieval import BM25Retriever


def group_claims_by_relevant_docs(prepared_data):

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

    for index, item in enumerate(
        grouped_claims.values(),
        start=1
    ):

        results = retriever.retrieve(
            item["claim"],
            top_k=k
        )

        retrieved_docs = [
            result["doc_id"]
            for result in results
        ]

        precision_scores.append(
            precision_at_k(
                retrieved_docs,
                item["relevant_docs"],
                k
            )
        )

        reciprocal_ranks.append(
            reciprocal_rank(
                retrieved_docs,
                item["relevant_docs"]
            )
        )

        if index % 20 == 0:
            print(f"Processed {index}/{total} claims")

    return (
        sum(precision_scores) / len(precision_scores),
        sum(reciprocal_ranks) / len(reciprocal_ranks)
    )


if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("BM25 GROUPED RETRIEVAL EVALUATION")
    print("=" * 60)

    dev_claims = load_claims("dev")

    prepared_data = extract_claim_evidence(dev_claims)

    grouped_claims = group_claims_by_relevant_docs(
        prepared_data
    )

    print(
        f"\nUnique claims: {len(grouped_claims)}"
    )

    retriever = BM25Retriever()

    precision, mrr = evaluate(
        grouped_claims,
        retriever,
        k=5
    )

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(f"\nPrecision@5: {precision:.4f}")
    print(f"MRR@5:       {mrr:.4f}")

    print("\nEvaluation completed!")