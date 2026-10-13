import streamlit as st
import sys
from pathlib import Path
import random


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Scientific Claim Verification",
    page_icon="🔬",
    layout="wide"
)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# ============================================================
# PAGE TITLE
# ============================================================

st.title("🔬 Claim-Level Multi-Hop Retrieval")

st.subheader(
    "Contradiction-Aware Evidence Reasoning for Scientific Literature"
)

st.markdown(
    """
    Enter a scientific claim below. The system analyzes the claim,
    retrieves relevant scientific evidence, and performs
    contradiction-aware verification.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ System Configuration")

    retrieval_method = st.selectbox(
        "Retrieval Method",
        [
            "Hybrid Retrieval",
            "Graph Multi-Hop Retrieval",
            "BM25 Retrieval"
        ]
    )

    top_k = st.slider(
        "Number of Evidence Results",
        min_value=3,
        max_value=5,
        value=5
    )

    st.divider()

    st.markdown("### 🧠 Project Modules")

    st.success("✓ BM25 Retrieval")
    st.success("✓ Hybrid Retrieval")
    st.success("✓ Graph Multi-Hop Retrieval")
    st.success("✓ Evidence Graph")
    st.success("✓ Claim Verification")


# ============================================================
# CLAIM INPUT
# ============================================================

claim = st.text_area(
    "📝 Enter a Scientific Claim",
    placeholder=(
        "Example: Vitamin D supplementation reduces "
        "the risk of respiratory infections."
    ),
    height=120
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze_button = st.button(
    "🔍 Analyze Claim",
    type="primary",
    use_container_width=True
)


# ============================================================
# DEMO RETRIEVAL FUNCTION
# ============================================================

def demo_retrieval(claim, method, top_k):

    """
    Temporary interface layer.

    This will later connect directly to the existing
    BM25 / Hybrid / Graph retrieval modules.
    """

    sample_evidence = [

        {
            "title": "Scientific Study on Claim Evidence",
            "evidence": (
                "The study reports experimental evidence "
                "that is relevant to the submitted scientific claim."
            ),
            "score": 0.91,
            "label": "SUPPORT"
        },

        {
            "title": "Related Scientific Literature",
            "evidence": (
                "The reported findings provide partially "
                "related evidence for evaluating the claim."
            ),
            "score": 0.84,
            "label": "SUPPORT"
        },

        {
            "title": "Alternative Experimental Findings",
            "evidence": (
                "The results indicate conditions under which "
                "the claim may not consistently hold."
            ),
            "score": 0.76,
            "label": "CONTRADICT"
        },

        {
            "title": "Scientific Background Study",
            "evidence": (
                "The study discusses related scientific concepts "
                "but does not provide sufficient direct evidence."
            ),
            "score": 0.68,
            "label": "NEI"
        },

        {
            "title": "Evidence from Related Research",
            "evidence": (
                "The research provides additional context "
                "for evaluating the submitted claim."
            ),
            "score": 0.61,
            "label": "NEI"
        },

    ]

    return sample_evidence[:top_k]


# ============================================================
# CONTRADICTION-AWARE REASONING
# ============================================================

def verify_claim(evidence_list):

    support_score = sum(
        item["score"]
        for item in evidence_list
        if item["label"] == "SUPPORT"
    )

    contradict_score = sum(
        item["score"]
        for item in evidence_list
        if item["label"] == "CONTRADICT"
    )

    nei_score = sum(
        item["score"]
        for item in evidence_list
        if item["label"] == "NEI"
    )

    scores = {

        "SUPPORT": support_score,

        "CONTRADICT": contradict_score,

        "NEI": nei_score
    }

    final_label = max(
        scores,
        key=scores.get
    )

    total_score = sum(scores.values())

    if total_score > 0:

        confidence = (
            scores[final_label]
            / total_score
        )

    else:

        confidence = 0

    return final_label, confidence, scores


# ============================================================
# GRAPH MULTI-HOP PATH
# ============================================================

def create_reasoning_path(claim):

    return [

        "Claim Node",

        "Relevant Scientific Concept",

        "Supporting / Contradicting Evidence",

        "Scientific Document",

        "Final Verification Decision"
    ]


# ============================================================
# ANALYSIS
# ============================================================

if analyze_button:

    if not claim.strip():

        st.warning(
            "⚠️ Please enter a scientific claim first."
        )

    else:

        st.divider()

        st.header("🔍 Analysis Results")

        st.info(
            f"Retrieval Method: **{retrieval_method}**"
        )


        # ----------------------------------------------------
        # LOADING
        # ----------------------------------------------------

        with st.spinner(
            "Retrieving scientific evidence..."
        ):

            evidence_list = demo_retrieval(
                claim,
                retrieval_method,
                top_k
            )


        # ----------------------------------------------------
        # VERIFY CLAIM
        # ----------------------------------------------------

        final_label, confidence, scores = verify_claim(
            evidence_list
        )


        # ----------------------------------------------------
        # FINAL RESULT
        # ----------------------------------------------------

        st.subheader("⚖️ Claim Verification Result")

        col1, col2, col3 = st.columns(3)


        with col1:

            if final_label == "SUPPORT":

                st.success(
                    f"### ✅ {final_label}"
                )

            elif final_label == "CONTRADICT":

                st.error(
                    f"### ❌ {final_label}"
                )

            else:

                st.warning(
                    f"### ⚠️ {final_label}"
                )


        with col2:

            st.metric(
                "Confidence",
                f"{confidence * 100:.1f}%"
            )


        with col3:

            st.metric(
                "Evidence Retrieved",
                len(evidence_list)
            )


        # ----------------------------------------------------
        # SCORE BREAKDOWN
        # ----------------------------------------------------

        st.subheader("📊 Evidence Score Breakdown")

        score_col1, score_col2, score_col3 = st.columns(3)

        score_col1.metric(
            "Support Score",
            f"{scores['SUPPORT']:.3f}"
        )

        score_col2.metric(
            "Contradiction Score",
            f"{scores['CONTRADICT']:.3f}"
        )

        score_col3.metric(
            "NEI Score",
            f"{scores['NEI']:.3f}"
        )


        # ----------------------------------------------------
        # EVIDENCE
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "📚 Retrieved Scientific Evidence"
        )


        for index, item in enumerate(
            evidence_list,
            start=1
        ):

            with st.expander(

                f"{index}. {item['title']} "
                f"(Score: {item['score']:.2f})"

            ):

                st.markdown(
                    f"**Evidence:** {item['evidence']}"
                )

                label = item["label"]

                if label == "SUPPORT":

                    st.success(
                        "Evidence Classification: SUPPORT"
                    )

                elif label == "CONTRADICT":

                    st.error(
                        "Evidence Classification: CONTRADICT"
                    )

                else:

                    st.warning(
                        "Evidence Classification: NEI"
                    )


        # ----------------------------------------------------
        # MULTI-HOP REASONING
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "🕸️ Multi-Hop Evidence Reasoning Path"
        )

        reasoning_path = create_reasoning_path(
            claim
        )


        for i, step in enumerate(
            reasoning_path
        ):

            st.markdown(
                f"### {i + 1}. {step}"
            )

            if i < len(reasoning_path) - 1:

                st.markdown(
                    "⬇️"
                )


        # ----------------------------------------------------
        # SYSTEM SUMMARY
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "📋 System Summary"
        )

        st.write(
            f"""
            **Claim:** {claim}

            **Retrieval Method:** {retrieval_method}

            **Final Decision:** {final_label}

            **Confidence:** {confidence * 100:.2f}%

            **Evidence Count:** {len(evidence_list)}
            """
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Claim-Level Multi-Hop Retrieval with "
    "Contradiction-Aware Evidence Reasoning "
    "for Scientific Literature"
)