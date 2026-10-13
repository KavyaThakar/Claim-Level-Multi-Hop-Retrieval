import json
from pathlib import Path


# Get the main project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# SciFact dataset location
DATA_DIR = BASE_DIR / "data" / "raw" / "scifact" / "data"


def load_jsonl(file_path):
    """
    Load a JSONL file and return a list of dictionaries.
    """
    data = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                data.append(json.loads(line))

    return data


def load_corpus():
    """
    Load the scientific paper corpus.
    """

    corpus_path = DATA_DIR / "corpus.jsonl"

    corpus = load_jsonl(corpus_path)

    print(f"📚 Loaded {len(corpus)} scientific papers")

    return corpus


def load_claims(split="train"):
    """
    Load claims from train, dev, or test dataset.
    """

    file_name = f"claims_{split}.jsonl"
    claims_path = DATA_DIR / file_name

    claims = load_jsonl(claims_path)

    print(f"📝 Loaded {len(claims)} {split} claims")

    return claims


if __name__ == "__main__":

    print("\n🔬 Loading SciFact Dataset...\n")

    corpus = load_corpus()

    train_claims = load_claims("train")
    dev_claims = load_claims("dev")
    test_claims = load_claims("test")

    print("\n📄 SAMPLE SCIENTIFIC PAPER:\n")

    sample_paper = corpus[0]

    print("Paper ID:", sample_paper.get("doc_id"))
    print("Title:", sample_paper.get("title"))
    print("Abstract:")

    abstract = sample_paper.get("abstract", [])

    if isinstance(abstract, list):
        print(" ".join(abstract))
    else:
        print(abstract)

    print("\n🧠 SAMPLE CLAIM:\n")

    sample_claim = train_claims[0]

    print("Claim ID:", sample_claim.get("id"))
    print("Claim:", sample_claim.get("claim"))
    print("Evidence:", sample_claim.get("evidence"))