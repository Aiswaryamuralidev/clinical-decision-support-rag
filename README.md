# 🩺 Clinical Decision Support with Medical RAG

An evidence-grounded clinical research assistant that answers biomedical and COVID-19 questions **only from retrieved research passages**, shows the supporting evidence, and has every answer scored by an **independent LLM judge**.

> Capstone project — AI & Emerging Technologies, IIT Hyderabad (TalentSprint), Cohort 9 · Group 1
> ⚠️ Research support only. This tool does not diagnose, prescribe or replace clinical judgement.

---

## 🎯 Problem
General-purpose LLMs can **hallucinate** — give confident answers with no evidence behind them. In medicine that is dangerous. This project grounds every answer in published research and makes it checkable.

## 🧠 Approach
| Step | What happens |
|---|---|
| **1. Data** | PubMedQA + COVID-QA from [RAGBench](https://huggingface.co/datasets/galileo-ai/ragbench) (train split) |
| **2. Clean & de-duplicate** | 20,852 rows → **11,018** unique document sets; repeated passages removed |
| **3. Chunk** | 1,200-character chunks, 200 overlap → **53,938 chunks** |
| **4. Embed & index** | `all-MiniLM-L6-v2` vectors in **ChromaDB** + a **BM25** keyword index |
| **5. Hybrid retrieval** | Top 5 from each, fused with **Reciprocal Rank Fusion** (weights 0.5 / 0.5) |
| **6. Generate** | **Gemma 3 4B Instruct** (local GPU) answers only from the retrieved context |
| **7. Judge** | **GPT-OSS 120B** (via Groq) scores each answer 1–5 against the reference answer, with reasoning |
| **8. UI** | **Gradio** web app with Answer, Evidence and Score tabs |

**Why hybrid search?** Vector search catches meaning ("heart attack" ≈ "myocardial infarction"); BM25 catches exact terms (drug names, gene codes). RRF combines them by rank, rewarding chunks both searches agree on.

**Why an independent judge?** A model grading its own answers is lenient. Our results confirmed it (see below).

---

## 📊 Results
50 questions per dataset, scored 1–5 (1 = completely incorrect, 5 = completely accurate).

| Answering model | Judge | PubMedQA | COVID-QA |
|---|---|---|---|
| Gemma 3 4B | Gemma 3 4B (self-judge) | 3.78 | – |
| Gemma 3 4B | **GPT-OSS 120B** | **3.37** | **2.49** |
| Llama 3.2 3B | GPT-OSS 120B | 2.54 | 2.41 |

**Key findings**
- **Self-judging inflates scores:** Gemma rates its own answers 3.78; the independent judge gives 3.37.
- **COVID-QA is harder:** the answer is one specific fact inside a long article, so retrieval often misses it.
- **Model size matters:** Gemma 4B beats Llama 3B on PubMedQA with the same judge.
- With 50 questions per dataset, differences below ~0.2–0.3 are not significant.

---

## 🖥️ User interface
- **Answer** — the answer with question-type, risk-level and source-count labels, plus a disclaimer
- **Evidence** — the retrieved chunks, with question words highlighted
- **Score** — one-click LLM-as-a-judge score with reasoning
- **Safety** — high-risk topics (dosage, pregnancy, interactions) get a warning; personal diagnosis/prescription requests are refused
- **Feedback** — "Was this correct?" saved to CSV for human review; session history downloadable

<!-- Add screenshots: ![Answer tab](images/ui_answer.png) -->

---

## 🚀 How to run (Kaggle)
1. Create a **Hugging Face Read token** and accept the [Gemma 3 4B](https://huggingface.co/google/gemma-3-4b-it) licence.
2. Create a free **Groq API key** at [console.groq.com](https://console.groq.com).
3. On Kaggle: import the notebook → Settings: **GPU T4 x2**, **Internet On**.
4. Add-ons → Secrets: add `HF_TOKEN` and `GROQ_API_KEY`, and tick both.
5. **Run All.** The last cell prints a `gradio.live` link to open the app.

> 🔐 Keys are read from Secrets only — never paste API keys into the notebook.

## 🧰 Tech stack
Python · LangChain · Hugging Face Transformers · ChromaDB · BM25 (rank_bm25) · sentence-transformers · Groq API · Gradio · Kaggle T4 GPU

---

## ⚠️ Limitations
- Only 50 questions per dataset could be evaluated (free-GPU limits)
- Reference answers in RAGBench are LLM-written, not clinician-written
- Retrieval metrics (Recall@5, MRR) not measured in the final run
- Safety triage uses keyword rules, so unusual wording can slip through

## 🔮 Future work
- Biomedical embeddings (PubMedBERT, MedCPT) and a re-ranker to improve retrieval
- MedGemma 4B as the answering model
- Larger evaluation set with human-checked samples
- Permanent hosting (Hugging Face Spaces) and live sources (PubMed, FDA alerts)

---


Aiiswarya M K
