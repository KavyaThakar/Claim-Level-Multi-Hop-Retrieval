import sys
import pickle
import math
import re

from pathlib import Path
from collections import defaultdict


# =========================================
# PROJECT PATH
# =========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT / "src"))

from hybrid_retrieval import HybridRetriever


# =========================================
# GRAPH MULTI-HOP RETRIEVER
# =========================================

class GraphMultiHopRetriever:

    def __init__(self):

        print("\n" + "=" * 70)
        print("🕸️ INITIALIZING IMPROVED GRAPH MULTI-HOP RETRIEVER")
        print("=" * 70)

        # ---------------------------------
        # LOAD HYBRID RETRIEVER
        # ---------------------------------

        print("\n🧠 Loading Hybrid Retriever...")

        self.hybrid_retriever = HybridRetriever()

        # ---------------------------------
        # LOAD SCIENTIFIC EVIDENCE GRAPH
        # ---------------------------------

        graph_path = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "scientific_evidence_graph.gpickle"
        )

        print("\n🕸️ Loading Scientific Evidence Graph...")

        if not graph_path.exists():

            raise FileNotFoundError(
                f"\n❌ Graph file not found:\n{graph_path}"
            )

        with open(graph_path, "rb") as file:

            self.graph = pickle.load(file)

        print("✅ Graph loaded successfully!")

        print(
            f"🕸️ Total Nodes: "
            f"{self.graph.number_of_nodes()}"
        )

        print(
            f"🔗 Total Edges: "
            f"{self.graph.number_of_edges()}"
        )

        # ---------------------------------
        # CREATE DOCUMENT LOOKUP
        # ---------------------------------

        self.document_map = {

            str(doc["doc_id"]): doc

            for doc in self.hybrid_retriever.documents
        }

        print(
            f"📚 Document lookup ready: "
            f"{len(self.document_map)} papers"
        )

        # ---------------------------------
        # TOTAL DOCUMENT COUNT
        # ---------------------------------

        self.total_documents = len(
            self.document_map
        )


    # =========================================
    # TEXT TOKENIZATION
    # =========================================

    def tokenize(self, text):

        text = text.lower()

        tokens = re.findall(
            r"[a-zA-Z0-9]+",
            text
        )

        return set(tokens)


    # =========================================
    # GET CONNECTED CONCEPTS
    # =========================================

    def get_connected_concepts(self, doc_id):

        document_node = f"DOC_{doc_id}"

        if document_node not in self.graph:

            return []

        concepts = []

        for neighbor in self.graph.neighbors(
            document_node
        ):

            node_data = self.graph.nodes[neighbor]

            if node_data.get("node_type") == "concept":

                concepts.append(neighbor)

        return concepts


    # =========================================
    # GET DOCUMENTS FROM CONCEPT
    # =========================================

    def get_documents_from_concept(
        self,
        concept_node
    ):

        if concept_node not in self.graph:

            return []

        documents = []

        for neighbor in self.graph.neighbors(
            concept_node
        ):

            node_data = self.graph.nodes[neighbor]

            if node_data.get("node_type") == "document":

                doc_id = node_data.get("doc_id")

                if doc_id is not None:

                    documents.append(str(doc_id))

        return documents


    # =========================================
    # GET CONCEPT NAME
    # =========================================

    def get_concept_name(
        self,
        concept_node
    ):

        node_data = self.graph.nodes[concept_node]

        name = node_data.get(
            "name",
            concept_node
        )

        return str(name)


    # =========================================
    # CONCEPT SPECIFICITY
    # =========================================

    def calculate_specificity(
        self,
        concept_node
    ):

        # Number of connected documents

        connected_docs = (
            self.get_documents_from_concept(
                concept_node
            )
        )

        document_frequency = len(
            connected_docs
        )

        # IDF-style specificity
        #
        # Rare concepts → higher score
        # Common concepts → lower score

        specificity = math.log(

            (self.total_documents + 1)
            /
            (document_frequency + 1)

        )

        return specificity


    # =========================================
    # QUERY-CONCEPT RELEVANCE
    # =========================================

    def calculate_query_relevance(
        self,
        query,
        concept_name
    ):

        query_tokens = self.tokenize(query)

        concept_tokens = self.tokenize(
            concept_name
        )

        if not concept_tokens:

            return 0.0

        overlap = len(
            query_tokens.intersection(
                concept_tokens
            )
        )

        # Fraction of concept terms
        # appearing in query

        relevance = overlap / len(
            concept_tokens
        )

        return relevance


    # =========================================
    # NORMALIZE SCORES
    # =========================================

    def normalize_scores(self, scores):

        if not scores:

            return {}

        maximum = max(scores.values())

        if maximum <= 0:

            return {
                key: 0.0
                for key in scores
            }

        return {

            key: value / maximum

            for key, value
            in scores.items()
        }


    # =========================================
    # GET HYBRID SCORE
    # =========================================

    def get_result_score(self, result):

        return float(

            result.get(
                "hybrid_score",

                result.get(
                    "score",

                    result.get(
                        "final_score",
                        0.0
                    )
                )
            )
        )


    # =========================================
    # IMPROVED MULTI-HOP RETRIEVAL
    # =========================================

    def retrieve(

        self,
        query,

        initial_k=10,

        max_concepts=12,

        hybrid_candidate_k=100,

        top_k=10,

        hybrid_weight=0.75,

        graph_weight=0.25
    ):

        # =====================================
        # HOP 1 — HYBRID RETRIEVAL
        # =====================================

        initial_results = (

            self.hybrid_retriever.retrieve(

                query,

                top_k=initial_k
            )
        )

        initial_doc_ids = [

            str(result["doc_id"])

            for result in initial_results
        ]

        # -------------------------------------
        # GET EXTENDED HYBRID CANDIDATES
        # -------------------------------------

        extended_results = (

            self.hybrid_retriever.retrieve(

                query,

                top_k=hybrid_candidate_k
            )
        )

        hybrid_scores = {}

        for result in extended_results:

            doc_id = str(
                result["doc_id"]
            )

            hybrid_scores[doc_id] = (
                self.get_result_score(
                    result
                )
            )

        # =====================================
        # HOP 2 — CONCEPT EXTRACTION
        # =====================================

        concept_seed_frequency = defaultdict(int)

        for doc_id in initial_doc_ids:

            concepts = (

                self.get_connected_concepts(
                    doc_id
                )
            )

            for concept in concepts:

                concept_seed_frequency[
                    concept
                ] += 1

        # =====================================
        # CONCEPT IMPORTANCE SCORING
        # =====================================

        concept_scores = {}

        for concept, frequency in (
            concept_seed_frequency.items()
        ):

            concept_name = (
                self.get_concept_name(
                    concept
                )
            )

            # Query relevance

            query_relevance = (
                self.calculate_query_relevance(

                    query,

                    concept_name
                )
            )

            # Specificity

            specificity = (
                self.calculate_specificity(
                    concept
                )
            )

            # Seed document frequency
            # normalized by initial results

            seed_strength = (

                frequency / initial_k
            )

            # ---------------------------------
            # FINAL CONCEPT SCORE
            # ---------------------------------

            # Query relevance is boosted,
            # but concepts with no direct
            # lexical overlap still receive
            # a small graph-based contribution.

            concept_score = (

                (
                    0.60 * query_relevance
                    +
                    0.40 * seed_strength
                )

                * specificity
            )

            concept_scores[
                concept
            ] = concept_score

        # -------------------------------------
        # SORT CONCEPTS
        # -------------------------------------

        ranked_concepts = sorted(

            concept_scores.items(),

            key=lambda item: item[1],

            reverse=True
        )

        selected_concepts = [

            concept

            for concept, score
            in ranked_concepts[
                :max_concepts
            ]
        ]

        # =====================================
        # HOP 3 — WEIGHTED GRAPH EXPANSION
        # =====================================

        graph_scores = defaultdict(float)

        for concept in selected_concepts:

            concept_importance = (
                concept_scores[concept]
            )

            connected_docs = (

                self.get_documents_from_concept(
                    concept
                )
            )

            for doc_id in connected_docs:

                # Add weighted concept
                # importance to each document

                graph_scores[doc_id] += (
                    concept_importance
                )

        # =====================================
        # BUILD CANDIDATE SET
        # =====================================

        candidate_doc_ids = set(
            hybrid_scores.keys()
        )

        candidate_doc_ids.update(
            graph_scores.keys()
        )

        # =====================================
        # NORMALIZE SCORES
        # =====================================

        normalized_hybrid = (

            self.normalize_scores(
                hybrid_scores
            )
        )

        normalized_graph = (

            self.normalize_scores(
                graph_scores
            )
        )

        # =====================================
        # FINAL RERANKING
        # =====================================

        final_results = []

        for doc_id in candidate_doc_ids:

            hybrid_score = (

                normalized_hybrid.get(
                    doc_id,
                    0.0
                )
            )

            graph_score = (

                normalized_graph.get(
                    doc_id,
                    0.0
                )
            )

            # ---------------------------------
            # FINAL SCORE
            # ---------------------------------

            final_score = (

                hybrid_weight
                *
                hybrid_score

                +

                graph_weight
                *
                graph_score
            )

            document = self.document_map.get(
                str(doc_id)
            )

            if document:

                final_results.append({

                    "doc_id": str(doc_id),

                    "title": document.get(
                        "title",
                        "Unknown Title"
                    ),

                    "hybrid_score": hybrid_score,

                    "graph_score": graph_score,

                    "final_score": final_score
                })

        # =====================================
        # SORT RESULTS
        # =====================================

        final_results = sorted(

            final_results,

            key=lambda result:
            result["final_score"],

            reverse=True
        )

        # Add ranks

        for rank, result in enumerate(

            final_results,

            start=1
        ):

            result["rank"] = rank

        return final_results[:top_k]


# =========================================
# TEST
# =========================================

if __name__ == "__main__":

    retriever = GraphMultiHopRetriever()

    test_query = (

        "0-dimensional biomaterials "
        "lack inductive properties."
    )

    print("\n" + "=" * 70)
    print("🧠 IMPROVED GRAPH MULTI-HOP RETRIEVAL")
    print("=" * 70)

    print("\nClaim:")
    print(test_query)

    results = retriever.retrieve(

        query=test_query,

        initial_k=10,

        max_concepts=12,

        hybrid_candidate_k=100,

        top_k=10
    )

    print("\n" + "=" * 70)
    print("📚 FINAL RERANKED RESULTS")
    print("=" * 70)

    for result in results:

        print()

        print(
            f"{result['rank']}. "
            f"Document ID: "
            f"{result['doc_id']}"
        )

        print(
            f"   Title: "
            f"{result['title']}"
        )

        print(
            f"   Hybrid Score: "
            f"{result['hybrid_score']:.4f}"
        )

        print(
            f"   Graph Score: "
            f"{result['graph_score']:.4f}"
        )

        print(
            f"   ⭐ Final Score: "
            f"{result['final_score']:.4f}"
        )

    print("\n" + "=" * 70)
    print("✅ IMPROVED MULTI-HOP RETRIEVAL COMPLETED!")
    print("=" * 70)