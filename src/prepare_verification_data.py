import json
import random
from pathlib import Path


# =========================================
# PROJECT PATH
# =========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SCIFACT_DATA = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "scifact"
    / "data"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "verification"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================
# LABEL MAPPING
# =========================================

LABEL_MAP = {
    "SUPPORT": 0,
    "CONTRADICT": 1,
    "NEI": 2
}


# =========================================
# LOAD JSONL FILE
# =========================================

def load_jsonl(file_path):

    data = []

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if line:

                data.append(
                    json.loads(line)
                )

    return data


# =========================================
# LOAD SCIENTIFIC CORPUS
# =========================================

def load_corpus():

    corpus_path = (
        SCIFACT_DATA
        / "corpus.jsonl"
    )

    print("📚 Loading SciFact corpus...")

    corpus_data = load_jsonl(
        corpus_path
    )

    corpus = {}

    for paper in corpus_data:

        doc_id = str(
            paper.get("doc_id")
        )

        title = paper.get(
            "title",
            ""
        )

        abstract = paper.get(
            "abstract",
            []
        )

        # Convert abstract sentences
        # into one scientific evidence text

        if isinstance(abstract, list):

            abstract_text = " ".join(
                abstract
            )

        else:

            abstract_text = str(
                abstract
            )

        full_text = (
            title
            + " "
            + abstract_text
        )

        corpus[doc_id] = full_text

    print(
        f"✅ Loaded {len(corpus)} papers"
    )

    return corpus


# =========================================
# GET EVIDENCE TEXT
# =========================================

def get_evidence_text(
    paper,
    evidence_sentence_ids
):

    # SciFact abstracts are lists
    # of sentences

    abstract = paper.get(
        "abstract",
        []
    )

    if not isinstance(
        abstract,
        list
    ):

        return str(abstract)

    evidence_sentences = []

    for sentence_id in evidence_sentence_ids:

        try:

            sentence_id = int(
                sentence_id
            )

            if (
                0 <= sentence_id
                < len(abstract)
            ):

                evidence_sentences.append(
                    abstract[sentence_id]
                )

        except Exception:

            continue

    # If evidence sentence IDs exist

    if evidence_sentences:

        return " ".join(
            evidence_sentences
        )

    # Otherwise use the complete abstract

    return " ".join(
        abstract
    )


# =========================================
# CREATE PAPER LOOKUP
# =========================================

def load_papers():

    corpus_path = (
        SCIFACT_DATA
        / "corpus.jsonl"
    )

    corpus_data = load_jsonl(
        corpus_path
    )

    papers = {}

    for paper in corpus_data:

        papers[
            str(paper["doc_id"])
        ] = paper

    return papers


# =========================================
# PROCESS CLAIM FILE
# =========================================

def prepare_split(
    filename,
    papers,
    corpus
):

    claims_path = (
        SCIFACT_DATA
        / filename
    )

    print(
        f"\n📂 Processing {filename}..."
    )

    claims = load_jsonl(
        claims_path
    )

    dataset = []

    paper_ids = list(
        corpus.keys()
    )

    for index, claim_data in enumerate(
        claims,
        start=1
    ):

        claim = claim_data.get(
            "claim",
            ""
        )

        evidence = claim_data.get(
            "evidence",
            {}
        )

        claim_id = claim_data.get(
            "id",
            index
        )

        # ---------------------------------
        # CASE 1: SUPPORT / CONTRADICT
        # ---------------------------------

        if evidence:

            for doc_id, evidence_sets in (
                evidence.items()
            ):

                doc_id = str(doc_id)

                paper = papers.get(
                    doc_id
                )

                if not paper:

                    continue

                # SciFact can contain
                # multiple evidence sets

                for evidence_set in evidence_sets:

                    label = evidence_set.get(
                        "label"
                    )

                    if label not in [
                        "SUPPORT",
                        "CONTRADICT"
                    ]:

                        continue

                    evidence_sentence_ids = (
                        evidence_set.get(
                            "evidence",
                            []
                        )
                    )

                    evidence_text = (
                        get_evidence_text(

                            paper,

                            evidence_sentence_ids
                        )
                    )

                    dataset.append({

                        "claim_id": claim_id,

                        "claim": claim,

                        "evidence": evidence_text,

                        "label": LABEL_MAP[label],

                        "label_name": label,

                        "document_id": doc_id
                    })

        # ---------------------------------
        # CASE 2: NEI
        # ---------------------------------

        else:

            # Create a negative scientific
            # document for NEI

            random_doc_id = random.choice(
                paper_ids
            )

            evidence_text = corpus[
                random_doc_id
            ]

            dataset.append({

                "claim_id": claim_id,

                "claim": claim,

                "evidence": evidence_text,

                "label": LABEL_MAP["NEI"],

                "label_name": "NEI",

                "document_id": random_doc_id
            })

        # Progress

        if index % 50 == 0:

            print(
                f"Processed "
                f"{index}/{len(claims)} claims"
            )

    print(
        f"✅ Created "
        f"{len(dataset)} verification examples"
    )

    return dataset


# =========================================
# SAVE DATASET
# =========================================

def save_dataset(
    dataset,
    filename
):

    output_path = (
        OUTPUT_DIR
        / filename
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        for item in dataset:

            file.write(
                json.dumps(item)
                + "\n"
            )

    print(
        f"💾 Saved: {output_path}"
    )


# =========================================
# DISPLAY STATISTICS
# =========================================

def display_statistics(
    dataset,
    split_name
):

    label_counts = {

        "SUPPORT": 0,

        "CONTRADICT": 0,

        "NEI": 0
    }

    for item in dataset:

        label_name = item[
            "label_name"
        ]

        label_counts[
            label_name
        ] += 1

    print("\n" + "=" * 60)

    print(
        f"📊 {split_name.upper()} DATASET STATISTICS"
    )

    print("=" * 60)

    print(
        f"\nTotal Examples: "
        f"{len(dataset)}"
    )

    print(
        f"SUPPORT:     "
        f"{label_counts['SUPPORT']}"
    )

    print(
        f"CONTRADICT:  "
        f"{label_counts['CONTRADICT']}"
    )

    print(
        f"NEI:         "
        f"{label_counts['NEI']}"
    )


# =========================================
# MAIN
# =========================================

def main():

    print("\n" + "=" * 70)

    print(
        "🧠 PREPARING SCIFACT "
        "CLAIM VERIFICATION DATA"
    )

    print("=" * 70)

    # Reproducibility

    random.seed(42)

    # -------------------------------------
    # LOAD DATA
    # -------------------------------------

    papers = load_papers()

    corpus = load_corpus()

    # -------------------------------------
    # PREPARE TRAIN
    # -------------------------------------

    train_data = prepare_split(

        "claims_train.jsonl",

        papers,

        corpus
    )

    save_dataset(

        train_data,

        "train.jsonl"
    )

    display_statistics(

        train_data,

        "TRAIN"
    )

    # -------------------------------------
    # PREPARE DEV
    # -------------------------------------

    dev_data = prepare_split(

        "claims_dev.jsonl",

        papers,

        corpus
    )

    save_dataset(

        dev_data,

        "dev.jsonl"
    )

    display_statistics(

        dev_data,

        "DEVELOPMENT"
    )

    # -------------------------------------
    # COMPLETED
    # -------------------------------------

    print("\n" + "=" * 70)

    print(
        "✅ VERIFICATION DATA PREPARATION COMPLETED!"
    )

    print("=" * 70)

    print(
        "\n📁 Output Location:"
    )

    print(OUTPUT_DIR)


if __name__ == "__main__":

    main()