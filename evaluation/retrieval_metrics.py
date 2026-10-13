import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from data_loader import load_claims
from prepare_dataset import extract_claim_evidence
from retrieval import BM25Retriever


def precision_at_k(retrieved_doc_ids, relevant_doc_ids, k):
    """
    Calculate Precision@K.
    """

    retrieved_k = retrieved_doc_ids[:k]

    relevant_retrieved = sum(
        1 for doc_id in retrieved_k
        if doc_id in relevant_doc_ids
    )

    return relevant_retrieved / k


def reciprocal_rank(retrieved_doc_ids, relevant_doc_ids):
    """
    Calculate Reciprocal Rank.
    """

    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):

        if doc_id in relevant_doc_ids:
            return 1 / rank

    return 0


def evaluate_retrieval(claims, retriever, k=5):

    precision_scores = []
    reciprocal_ranks = []

    for index, claim_item in enumerate(claims):

        claim_text = claim_item["claim"]

        relevant_doc_ids = {claim_item["doc_id"]}

        results = retriever.retrieve(
            claim_text,
            top_k=k
        )

        retrieved_doc_ids = [
            result["doc_id"]
            for result in results
        ]

        precision = precision_at_k(
            retrieved_doc_ids,
            relevant_doc_ids,
            k
        )

        rr = reciprocal_rank(
            retrieved_doc_ids,
            relevant_doc_ids
        )

        precision_scores.append(precision)
        reciprocal_ranks.append(rr)

        # Show progress
        if (index + 1) % 20 == 0:
            print(f"Processed {index + 1}/{len(claims)} claims")

    avg_precision = sum(precision_scores) / len(precision_scores)

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    return avg_precision, mrr


if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("📊 BM25 RETRIEVAL EVALUATION")
    print("=" * 70)

    print("\n🔄 Loading SciFact development data...")

    dev_claims = load_claims("dev")

    prepared_claims = extract_claim_evidence(dev_claims)

    print(
        f"\n🧠 Evaluating {len(prepared_claims)} "
        "claim-evidence pairs..."
    )

    retriever = BM25Retriever()

    precision, mrr = evaluate_retrieval(
        prepared_claims,
        retriever,
        k=5
    )

    print("\n" + "=" * 70)
    print("📈 RESULTS")
    print("=" * 70)

    print(f"\nPrecision@5: {precision:.4f}")
    print(f"MRR@5:        {mrr:.4f}")

    print("\n✅ Evaluation completed!")