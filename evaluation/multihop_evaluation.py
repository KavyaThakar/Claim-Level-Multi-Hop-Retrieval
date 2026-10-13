import sys
import json
from pathlib import Path


# =========================================
# PROJECT PATH SETUP
# =========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT / "src"))

from multihop_retrieval import GraphMultiHopRetriever


# =========================================
# DATA PATH
# =========================================

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "scifact"
    / "data"
    / "claims_dev.jsonl"
)


# =========================================
# LOAD CLAIMS
# =========================================

def load_claims():

    claims = []

    print("📂 Loading SciFact development claims...")

    with open(DATA_PATH, "r", encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            if line:

                claims.append(json.loads(line))

    print(f"✅ Loaded {len(claims)} claims")

    return claims


# =========================================
# GET RELEVANT DOCUMENTS
# =========================================

def get_relevant_documents(claim):

    evidence = claim.get("evidence", {})

    return {

        str(doc_id)

        for doc_id in evidence.keys()
    }


# =========================================
# PRECISION @ K
# =========================================

def precision_at_k(
    retrieved_docs,
    relevant_docs,
    k=5
):

    top_k = retrieved_docs[:k]

    relevant_found = sum(

        1

        for doc_id in top_k

        if str(doc_id) in relevant_docs
    )

    return relevant_found / k


# =========================================
# RECIPROCAL RANK @ K
# =========================================

def reciprocal_rank_at_k(
    retrieved_docs,
    relevant_docs,
    k=5
):

    top_k = retrieved_docs[:k]

    for rank, doc_id in enumerate(
        top_k,
        start=1
    ):

        if str(doc_id) in relevant_docs:

            return 1 / rank

    return 0.0


# =========================================
# MAIN EVALUATION
# =========================================

def evaluate():

    print("\n" + "=" * 70)
    print("🕸️ IMPROVED GRAPH MULTI-HOP RETRIEVAL EVALUATION")
    print("=" * 70)

    # -------------------------------------
    # LOAD CLAIMS
    # -------------------------------------

    claims = load_claims()

    # -------------------------------------
    # INITIALIZE RETRIEVER
    # -------------------------------------

    print("\n🧠 Initializing Improved Multi-Hop Retriever...")

    retriever = GraphMultiHopRetriever()

    precision_scores = []
    reciprocal_ranks = []

    evaluated_claims = 0

    print("\n🚀 Starting evaluation...\n")

    # -------------------------------------
    # PROCESS CLAIMS
    # -------------------------------------

    for index, claim_data in enumerate(
        claims,
        start=1
    ):

        claim_text = claim_data.get(
            "claim",
            ""
        )

        relevant_docs = get_relevant_documents(
            claim_data
        )

        # Skip claims without evidence

        if not claim_text or not relevant_docs:

            continue

        try:

            results = retriever.retrieve(

                query=claim_text,

                initial_k=10,

                max_concepts=12,

                hybrid_candidate_k=100,

                top_k=5
            )

            retrieved_docs = [

                str(result["doc_id"])

                for result in results
            ]

            # ---------------------------------
            # PRECISION @ 5
            # ---------------------------------

            precision = precision_at_k(

                retrieved_docs,

                relevant_docs,

                k=5
            )

            precision_scores.append(precision)

            # ---------------------------------
            # MRR @ 5
            # ---------------------------------

            rr = reciprocal_rank_at_k(

                retrieved_docs,

                relevant_docs,

                k=5
            )

            reciprocal_ranks.append(rr)

            evaluated_claims += 1

        except Exception as error:

            print(
                f"\n⚠️ Error processing "
                f"claim {index}: {error}"
            )

        # Progress

        if index % 10 == 0:

            print(
                f"Processed "
                f"{index}/{len(claims)} claims"
            )

    # =====================================
    # FINAL METRICS
    # =====================================

    average_precision = (

        sum(precision_scores)
        / len(precision_scores)

        if precision_scores
        else 0.0
    )

    mrr_at_5 = (

        sum(reciprocal_ranks)
        / len(reciprocal_ranks)

        if reciprocal_ranks
        else 0.0
    )

    # =====================================
    # FINAL RESULTS
    # =====================================

    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS")
    print("=" * 70)

    print(
        f"\nClaims Evaluated: "
        f"{evaluated_claims}"
    )

    print(
        f"\nPrecision@5: "
        f"{average_precision:.4f}"
    )

    print(
        f"MRR@5:       "
        f"{mrr_at_5:.4f}"
    )

    # =====================================
    # SYSTEM COMPARISON
    # =====================================

    print("\n" + "=" * 70)
    print("📈 SYSTEM COMPARISON")
    print("=" * 70)

    print("\nBM25")

    print("Precision@5: 0.1862")
    print("MRR@5:       0.7772")

    print("\nHybrid Retrieval")

    print("Precision@5: 0.1915")
    print("MRR@5:       0.8009")

    print("\nImproved Graph Multi-Hop Retrieval")

    print(
        f"Precision@5: "
        f"{average_precision:.4f}"
    )

    print(
        f"MRR@5:       "
        f"{mrr_at_5:.4f}"
    )

    print("\n" + "=" * 70)
    print("✅ IMPROVED MULTI-HOP EVALUATION COMPLETED!")
    print("=" * 70)


# =========================================
# RUN
# =========================================

if __name__ == "__main__":

    evaluate()