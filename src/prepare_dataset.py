from data_loader import load_claims


def extract_claim_evidence(claims):
    """
    Extract claim, document IDs, evidence sentence IDs,
    and verification labels from SciFact.
    """

    prepared_data = []

    for item in claims:

        claim_id = item.get("id")
        claim_text = item.get("claim")
        evidence = item.get("evidence", {})

        # Skip claims without evidence
        if not evidence:
            continue

        for doc_id, evidence_sets in evidence.items():

            for evidence_set in evidence_sets:

                label = evidence_set.get("label")

                sentence_ids = evidence_set.get(
                    "sentences", []
                )

                prepared_data.append({
                    "claim_id": claim_id,
                    "claim": claim_text,
                    "doc_id": int(doc_id),
                    "label": label,
                    "sentence_ids": sentence_ids
                })

    return prepared_data


if __name__ == "__main__":

    print("\n🔬 Preparing SciFact Claim-Evidence Data...\n")

    train_claims = load_claims("train")
    dev_claims = load_claims("dev")

    train_data = extract_claim_evidence(train_claims)
    dev_data = extract_claim_evidence(dev_claims)

    print(f"\n✅ Training claim-evidence pairs: {len(train_data)}")
    print(f"✅ Development claim-evidence pairs: {len(dev_data)}")

    print("\n📄 SAMPLE PREPARED DATA:\n")

    for item in train_data[:5]:

        print("Claim ID:", item["claim_id"])
        print("Claim:", item["claim"])
        print("Document ID:", item["doc_id"])
        print("Label:", item["label"])
        print("Evidence Sentences:", item["sentence_ids"])

        print("-" * 60)