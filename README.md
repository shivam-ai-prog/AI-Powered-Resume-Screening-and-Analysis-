# CareerForge AI

**Dual-sided AI Career Agent** for candidates and recruiters.

CareerForge AI helps candidates improve resumes and prepare for interviews, and helps recruiters shortlist the best-fit candidates from a resume database using hybrid matching, vector search (RAG), and LLM analysis.

---

## Features

### Phase 1 — Candidate Mode
- Upload a resume (PDF)
- Optional target role (e.g. Backend Engineer, AI Engineer)
- Resume parsing (experience, skills, projects, education, achievements, certifications)
- Resume analysis with realistic scoring
- Strengths, weaknesses, and gap analysis
- Career advisor recommendations:
  - Missing skills to learn
  - Recommended certifications
  - Recommended projects
- Improved professional summary
- Resume bullet suggestions
- Full improved resume + career growth section
- Tailored interview questions

### Phase 2 — Recruiter Mode
- Paste a job description (JD)
- JD parsing (required/preferred skills, experience level, keywords)
- Candidate database (SQLite)
- Vector search over candidates (Chroma + sentence embeddings)
- RAG retrieval of top relevant candidates
- LLM-based ranking, strengths, and gaps
- Top-N shortlist (default Top 5)
- Tailored interview questions per shortlisted candidate
- Recruiter report

### UI
- Gradio interface with two tabs:
  - **Candidate Mode**
  - **Recruiter Mode** (RAG + LangGraph)

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Orchestration | LangGraph |
| LLM | Groq (`openai/gpt-oss-120b`) via LangChain |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector DB | ChromaDB |
| Structured DB | SQLite |
| PDF parsing | pdfplumber |
| UI | Gradio |
| Schema validation | Pydantic |
| Environment | Python, Kaggle / local notebook |

---

## Architecture

### Candidate Pipeline (Phase 1)
```text
Resume PDF
   → Extract text
   → Parse resume (LLM + Pydantic)
   → Analyze resume (score, gaps, career advice)
   → Generate content (summary, suggestions, improved resume, questions)
   → Candidate report
```

### Recruiter Pipeline (Phase 2 — RAG)
```text
Job Description
   → Analyze JD
   → Embed JD
   → Vector search (Chroma) → Top K candidates
   → LLM match / rank retrieved candidates
   → Generate interview questions
   → Recruiter shortlist report
```

---

## Project Structure (suggested)

```text
CareerForge-AI/
├── README.md
├── requirements.txt
├── careerforge_candidates.db          # SQLite candidate store
├── careerforge_chroma/                # Chroma vector index (generated)
├── notebooks/
│   └── careerforgeai.ipynb            # Main notebook
├── data/
│   └── resumes/                       # Sample / test PDFs
└── src/                               # (optional) modularized code
    ├── parser.py
    ├── analyzer.py
    ├── generator.py
    ├── recruiter.py
    ├── vector_store.py
    and ui.py
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/CareerForge-AI.git
cd CareerForge-AI
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

Suggested `requirements.txt`:
```text
transformers
accelerate
bitsandbytes
sentencepiece
protobuf
langgraph
langchain
langchain-core
langchain-community
langchain-groq
pdfplumber
pydantic
reportlab
sentence-transformers
chromadb
gradio
torch
```

### 3. Set API keys
Create a Groq API key from [console.groq.com](https://console.groq.com) and set:

```python
GROQ_API_KEY = "gsk_your_key_here"
```

Or use environment variables:
```bash
export GROQ_API_KEY="gsk_your_key_here"
```

### 4. Prepare data
- Place resume PDFs in your data folder
- Load/parse candidates into SQLite
- Build the Chroma vector index once

---

## How to Run

### In Kaggle / Jupyter
1. Open `careerforgeai.ipynb`
2. Run setup cells (installs, model, helpers)
3. Run Phase 1 cells (parser, analyzer, generator, LangGraph)
4. Run Phase 2 cells (SQLite, JD analyzer, matching, Chroma, RAG LangGraph)
5. Launch Gradio UI:
```python
demo.launch(share=True)
```

### Candidate Mode
1. Upload a resume PDF
2. Enter optional target role
3. Click **Analyze Resume**

### Recruiter Mode
1. Paste a job description
2. Set Top N (default 5)
3. Click **Find Matching Candidates**

---

## Example Usage

### Candidate analysis
```python
result = run_careerforge(
    resume_pdf_path="data/resumes/Resume_01_Aarav_Mehta.pdf",
    target_role="Backend Engineer"
)
```

### Recruiter shortlist (RAG + LangGraph)
```python
jd_text = """
Job Title: Backend Engineer
Required: Python, Go, microservices, PostgreSQL, Redis, Kafka, Docker, Kubernetes
Preferred: AWS, fintech
Experience: 2+ years
"""

result = run_recruiter_pipeline_rag(jd_text, top_n=5, retrieve_k=8)
```

---

## Key Design Decisions

- **No hallucinated experience** in improved resumes
- **Career growth section** for missing skills, certs, and projects
- **SQLite** for persistent candidate storage
- **Vector DB + RAG** so the LLM only scores retrieved candidates (faster, cheaper, scalable)
- **LangGraph** for both candidate and recruiter workflows
- **Dual UI** for real-world demo value

---

## Current Limitations

- PDF download of improved resume still needs polish
- Education extraction quality depends on resume formatting
- Free Groq tiers have daily token limits
- Vector index must be rebuilt when new candidates are added

---

## Roadmap

- [ ] Reliable improved-resume PDF export
- [ ] Auto-update vector index when new resumes are inserted
- [ ] Better education parsing across resume formats
- [ ] Simple auth + multi-user demo
- [ ] Deploy Gradio / Streamlit app publicly
- [ ] Evaluation metrics for ranking quality

---

## Demo Tips for Recruiters / Judges

1. **Candidate tab:** upload a strong resume and a weak/fresher resume  
2. **Role mismatch:** same resume with different target roles (e.g. Backend vs AI)  
3. **Recruiter tab:** paste Backend, AI, Frontend, and DevOps JDs  
4. Show that vector retrieval + LLM re-ranking produces sensible Top 5 lists  

---

## License

MIT License (or your preferred license)

---

## Acknowledgements

- Groq for fast LLM inference
- LangGraph / LangChain
- ChromaDB
- sentence-transformers
- Gradio

---

Built as a practical AI agent project for real-world job application and recruiting workflows.
