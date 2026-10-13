
import re
from pathlib import Path

import networkx as nx
import spacy

from data_loader import load_corpus


class ScientificEvidenceGraph:

    def __init__(self):

        print("🧠 Loading spaCy NLP model...")

        self.nlp = spacy.load("en_core_web_sm")

        print("📚 Loading scientific corpus...")

        self.corpus = load_corpus()

        # Create an undirected graph
        self.graph = nx.Graph()


    def extract_concepts(self, text):
        """
        Extract important concepts from scientific text using:
        - Named Entities
        - Noun Chunks
        """

        doc = self.nlp(text)

        concepts = set()

        # ---------------------------------
        # NAMED ENTITIES
        # ---------------------------------

        for ent in doc.ents:

            concept = ent.text.strip().lower()

            concept = re.sub(
                r"[^a-z0-9\s-]",
                "",
                concept
            )

            if len(concept) > 2:
                concepts.add(concept)

        # ---------------------------------
        # NOUN CHUNKS
        # ---------------------------------

        for chunk in doc.noun_chunks:

            concept = chunk.text.strip().lower()

            # Remove unnecessary characters
            concept = re.sub(
                r"[^a-z0-9\s-]",
                "",
                concept
            )

            # Remove extra spaces
            concept = re.sub(
                r"\s+",
                " ",
                concept
            ).strip()

            # Ignore very short concepts
            if len(concept) > 2:
                concepts.add(concept)

        return concepts


    def build_graph(self):

        print("\n" + "=" * 60)
        print("🕸️ BUILDING SCIENTIFIC EVIDENCE GRAPH")
        print("=" * 60 + "\n")

        for index, paper in enumerate(self.corpus):

            doc_id = paper.get("doc_id")

            title = paper.get("title", "")

            abstract = paper.get("abstract", [])

            # Convert abstract list into text
            if isinstance(abstract, list):
                abstract = " ".join(abstract)

            # Combine title and abstract
            text = f"{title} {abstract}"

            # ---------------------------------
            # CREATE DOCUMENT NODE
            # ---------------------------------

            document_node = f"DOC_{doc_id}"

            self.graph.add_node(
                document_node,
                node_type="document",
                doc_id=doc_id,
                title=title
            )

            # ---------------------------------
            # EXTRACT SCIENTIFIC CONCEPTS
            # ---------------------------------

            concepts = self.extract_concepts(text)

            # ---------------------------------
            # CREATE CONCEPT NODES
            # ---------------------------------

            for concept in concepts:

                concept_node = f"CONCEPT_{concept}"

                self.graph.add_node(
                    concept_node,
                    node_type="concept",
                    name=concept
                )

                # Connect paper to concept
                self.graph.add_edge(
                    document_node,
                    concept_node,
                    relation="contains"
                )

            # ---------------------------------
            # SHOW PROGRESS
            # ---------------------------------

            if (index + 1) % 100 == 0:

                print(
                    f"Processed "
                    f"{index + 1}/{len(self.corpus)} papers"
                )

        print("\n" + "=" * 60)
        print("✅ GRAPH CONSTRUCTION COMPLETED!")
        print("=" * 60)

        print(
            f"\n📄 Document nodes: "
            f"{len(self.corpus)}"
        )

        print(
            f"🕸️ Total graph nodes: "
            f"{self.graph.number_of_nodes()}"
        )

        print(
            f"🔗 Total graph edges: "
            f"{self.graph.number_of_edges()}"
        )

        return self.graph


    def save_graph(self):
        """
        Save the completed graph to disk so we don't
        need to rebuild it every time.
        """

        # Project root directory
        BASE_DIR = Path(__file__).resolve().parent.parent

        # Processed data directory
        processed_dir = BASE_DIR / "data" / "processed"

        # Create directory if it doesn't exist
        processed_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # Graph file path
        graph_path = (
            processed_dir
            / "scientific_evidence_graph.gpickle"
        )

        print("\n💾 Saving graph...")

        # Save graph using pickle
        import pickle

        with open(graph_path, "wb") as file:
            pickle.dump(
                self.graph,
                file
            )

        print(
            f"✅ Graph saved successfully!"
        )

        print(
            f"📁 Location: {graph_path}"
        )


if __name__ == "__main__":

    # ---------------------------------
    # CREATE GRAPH
    # ---------------------------------

    evidence_graph = ScientificEvidenceGraph()

    graph = evidence_graph.build_graph()

    # ---------------------------------
    # SAVE GRAPH
    # ---------------------------------

    evidence_graph.save_graph()