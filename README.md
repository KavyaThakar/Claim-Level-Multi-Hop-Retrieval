# Claim-Level Multi-Hop Retrieval with Contradiction-Aware Evidence Reasoning for Scientific Literature

## 🚀 Live Demo

Try the deployed application here:

🔗 **[Scientific Claim Verification – Streamlit App](https://happylimbasiya22-claim-level-multi-hop-retrieval-app-mrjcjj.streamlit.app/)**

## 📌 Project Overview

This project implements a **claim-level scientific literature retrieval and evidence reasoning system**.

The system retrieves relevant scientific papers for a given claim and uses a **graph-based multi-hop retrieval approach** to discover additional related evidence through shared scientific concepts.

The project is being developed as part of an **Information Retrieval and Natural Language Processing (IRNLP)** subject project.

---

## 🎯 Project Objectives

The main objectives of this project are:

- Retrieve scientific literature relevant to a given claim.
- Implement efficient document retrieval using **BM25**.
- Improve retrieval using **Hybrid Retrieval**.
- Build a **Scientific Evidence Graph** connecting papers and scientific concepts.
- Perform **Graph-Based Multi-Hop Retrieval** to discover additional evidence.
- Handle contradictory scientific evidence using claim verification.
- Fine-tune **DeBERTa** for scientific claim verification using the **SciFact dataset**.
- Evaluate retrieval and classification performance using standard metrics.

---

# 🏗️ System Architecture

```text
                        Scientific Claim
                               │
                               ▼
                     ┌──────────────────┐
                     │  Hybrid Retrieval │
                     │  BM25 + Semantic  │
                     └─────────┬────────┘
                               │
                               ▼
                         Initial Papers
                               │
                               ▼
                     ┌──────────────────┐
                     │ Scientific Evidence│
                     │      Graph         │
                     └─────────┬────────┘
                               │
                               ▼
                       Concept Expansion
                               │
                               ▼
                        Multi-Hop Search
                               │
                               ▼
                     Related Scientific Papers
                               │
                               ▼
                     Graph-Aware Reranking
                               │
                               ▼
                       Evidence Candidates
                               │
                               ▼
                  ┌────────────────────────┐
                  │ Claim Verification      │
                  │ Fine-tuned DeBERTa      │
                  └───────────┬────────────┘
                              │
                              ▼
                 SUPPORT / CONTRADICT / NEI
