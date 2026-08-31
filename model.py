"""
CareerForge AI - core model and pipeline logic.

This module contains the code extracted/refactored from the original Kaggle notebook:
- Resume PDF extraction
- LLM resume parsing and analysis
- Resume content generation
- Job-description parsing
- Candidate matching
- Chroma/SentenceTransformer retrieval
- LangGraph candidate and recruiter workflows

Environment variables:
    GROQ_API_KEY       Required for Groq.
    GROQ_MODEL         Optional; defaults to openai/gpt-oss-120b.
    CANDIDATE_DB_PATH  Optional SQLite candidate database path.
    CHROMA_PATH        Optional persistent Chroma directory.
    EMBEDDING_MODEL    Optional SentenceTransformer model.
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Dict, List, Optional, TypedDict

import chromadb
import pdfplumber
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
CANDIDATE_DB_PATH = os.getenv("CANDIDATE_DB_PATH", "data/careerforge_candidates.db")
CHROMA_PATH = os.getenv("CHROMA_PATH", "data/careerforge_chroma")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Create a .env file or export GROQ_API_KEY before running."
    )

llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=0.2,
    api_key=GROQ_API_KEY,
)

embedder = SentenceTransformer(EMBEDDING_MODEL)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name="candidates",
    metadata={"hnsw:space": "cosine"},
)


# ---------------------------------------------------------------------------
# Pydantic data models
# ---------------------------------------------------------------------------

class Education(BaseModel):
    degree: str = ""
    institution: str = ""
    years: str = ""
    details: Optional[str] = None


class Experience(BaseModel):
    title: str = ""
    company: str = ""
    duration: str = ""
    bullets: List[str] = Field(default_factory=list)


class Project(BaseModel):
    name: str = ""
    technologies: Optional[str] = None
    bullets: List[str] = Field(default_factory=list)


class Skills(BaseModel):
    technical: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    soft: List[str] = Field(default_factory=list)
    raw_text: Optional[str] = None


class StructuredResume(BaseModel):
    name: str = ""
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None
    headline: Optional[str] = None
    summary: Optional[str] = None
    education: List[Education] = Field(default_factory=list)
    experience: List[Experience] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    skills: Skills = Field(default_factory=Skills)
    achievements: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    raw_text: Optional[str] = None


class ResumeAnalysis(BaseModel):
    overall_score: float = 0.0
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    missing_skills_for_role: List[str] = Field(default_factory=list)
    recommended_certifications: List[str] = Field(default_factory=list)
    recommended_projects: List[str] = Field(default_factory=list)
    improvement_areas: List[str] = Field(default_factory=list)
    feedback_summary: str = ""
    approval_status: str = "Needs Improvement"


class ResumeSuggestion(BaseModel):
    original_bullet: str
    improved_bullet: str
    reason: str


class GeneratedContent(BaseModel):
    improved_summary: Optional[str] = None
    resume_suggestions: List[ResumeSuggestion] = Field(default_factory=list)
    interview_questions: List[str] = Field(default_factory=list)
    full_tailored_resume_text: Optional[str] = None


class JobRequirements(BaseModel):
    job_title: str = ""
    company: Optional[str] = None
    location: Optional[str] = None
    required_skills: List[str] = Field(default_factory=list)
    preferred_skills: List[str] = Field(default_factory=list)
    experience_level: Optional[str] = None
    responsibilities: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    education_requirements: Optional[str] = None
    raw_text: Optional[str] = None


class CandidateMatch(BaseModel):
    candidate_name: str
    overall_score: float = 0.0
    llm_score: float = 0.0
    strengths: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)
    summary: str = ""
    interview_questions: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def call_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 1500,
) -> str:
    messages = []
    if system_prompt:
        messages.append(("system", system_prompt))
    messages.append(("human", prompt))
    response = llm.invoke(messages)
    return response.content.strip()


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract selectable text from a PDF resume."""
    text_parts: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts).strip()


def clean_llm_json(response: str) -> dict:
    """Parse JSON from an LLM response, tolerating markdown code fences."""
    response = response.strip()

    if "```json" in response:
        response = response.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in response:
        response = response.split("```", 1)[1].split("```", 1)[0]

    response = response.strip()

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(response[start:end])
        raise ValueError("Could not parse JSON from LLM response")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def normalize_resume_data(data: dict) -> dict:
    if isinstance(data.get("projects"), list):
        for project in data["projects"]:
            if not isinstance(project, dict):
                continue
            technologies = project.get("technologies")
            if isinstance(technologies, list):
                project["technologies"] = " • ".join(map(str, technologies))
            elif technologies is None:
                project["technologies"] = ""

    for field in ("achievements", "certifications"):
        if isinstance(data.get(field), list):
            cleaned = []
            for item in data[field]:
                if isinstance(item, dict):
                    cleaned.append(
                        item.get("text")
                        or item.get("title")
                        or item.get("name")
                        or str(item)
                    )
                else:
                    cleaned.append(str(item))
            data[field] = cleaned

    skills = data.get("skills")
    if isinstance(skills, dict):
        for key in ("technical", "tools", "soft"):
            value = skills.get(key)
            if isinstance(value, str):
                skills[key] = [s.strip() for s in value.split(",") if s.strip()]

    return data


# ---------------------------------------------------------------------------
# Candidate-side pipeline
# ---------------------------------------------------------------------------

def parse_resume_with_llm(raw_text: str) -> StructuredResume:
    system_prompt = """You are an expert resume parser.
Extract information and return ONLY a valid JSON object.

Rules:
- Extract all degrees, institutions, years, and CGPA/details when present.
- Extract all certifications as plain strings.
- Extract awards, hackathons, publications, and honors as achievements.
- Do not invent information.
- If a section exists, do not return an empty list.
- Return pure JSON with no markdown or explanation."""

    user_prompt = f"""
Extract information from this resume using exactly this JSON structure:

{{
  "name": "string",
  "email": "string or null",
  "phone": "string or null",
  "location": "string or null",
  "linkedin": "string or null",
  "headline": "string or null",
  "summary": "string or null",
  "education": [
    {{"degree": "string", "institution": "string", "years": "string", "details": "string or null"}}
  ],
  "experience": [
    {{"title": "string", "company": "string", "duration": "string", "bullets": ["string"]}}
  ],
  "projects": [
    {{"name": "string", "technologies": "string", "bullets": ["string"]}}
  ],
  "skills": {{
    "technical": ["string"],
    "tools": ["string"],
    "soft": ["string"],
    "raw_text": "string"
  }},
  "achievements": ["string"],
  "certifications": ["string"]
}}

Resume:
\"\"\"
{raw_text[:12000]}
\"\"\"
"""

    response = call_llm(user_prompt, system_prompt=system_prompt, max_tokens=2000)

    try:
        data = normalize_resume_data(clean_llm_json(response))
        data["raw_text"] = raw_text
        return StructuredResume(**data)
    except Exception as exc:
        raise ValueError(f"Resume parsing failed: {exc}") from exc


def analyze_resume(
    structured_resume: StructuredResume,
    target_role: Optional[str] = None,
) -> ResumeAnalysis:
    system_prompt = """You are a strict and honest technical career coach and resume reviewer.
Rules:
1. Never invent experience, companies, projects, or skills.
2. Only use information present in the resume.
3. Score realistically.
4. If score > 90, approval_status must be Approved.
5. Return only valid JSON."""

    exp_text = ""
    for exp in structured_resume.experience:
        exp_text += f"\n{exp.title} at {exp.company} ({exp.duration})\n"
        exp_text += "\n".join(f"  • {b}" for b in exp.bullets)

    projects_text = "\n".join(
        f"- {p.name} ({p.technologies or ''}): {', '.join(p.bullets)}"
        for p in structured_resume.projects
    ) or "No projects listed"

    education_text = "\n".join(
        f"- {e.degree} | {e.institution} | {e.years} | {e.details or ''}"
        for e in structured_resume.education
    ) or "None"

    skills_text = ", ".join(
        structured_resume.skills.technical
        + structured_resume.skills.tools
        + structured_resume.skills.soft
    ) or "Not provided"

    user_prompt = f"""
Analyze this resume honestly for the target role.

Target Role: {target_role or "Not specified"}
Name: {structured_resume.name}
Headline: {structured_resume.headline or "Not provided"}
Summary: {structured_resume.summary or "Not provided"}
Skills: {skills_text}

Experience:
{exp_text or "No experience listed"}

Projects:
{projects_text}

Education:
{education_text}

Achievements:
{", ".join(structured_resume.achievements) or "None"}

Certifications:
{", ".join(structured_resume.certifications) or "None"}

Return exactly:
{{
  "overall_score": 75.0,
  "strengths": [],
  "weaknesses": [],
  "missing_skills_for_role": [],
  "recommended_certifications": [],
  "recommended_projects": [],
  "improvement_areas": [],
  "feedback_summary": "",
  "approval_status": "Approved" or "Needs Improvement"
}}

Scoring:
90-100 Excellent
75-89 Strong
60-74 Average
Below 60 Weak/Fresher
"""

    response = call_llm(user_prompt, system_prompt=system_prompt, max_tokens=1800)
    data = clean_llm_json(response)

    score = safe_float(data.get("overall_score"))
    status = "Approved" if score > 90 else data.get("approval_status", "Needs Improvement")

    return ResumeAnalysis(
        overall_score=score,
        strengths=[str(x) for x in data.get("strengths", [])],
        weaknesses=[str(x) for x in data.get("weaknesses", [])],
        missing_skills_for_role=[str(x) for x in data.get("missing_skills_for_role", [])],
        recommended_certifications=[str(x) for x in data.get("recommended_certifications", [])],
        recommended_projects=[str(x) for x in data.get("recommended_projects", [])],
        improvement_areas=[str(x) for x in data.get("improvement_areas", [])],
        feedback_summary=str(data.get("feedback_summary", "")),
        approval_status=status,
    )


def generate_content_from_resume(
    structured_resume: StructuredResume,
    analysis: ResumeAnalysis,
    target_role: Optional[str] = None,
) -> GeneratedContent:
    system_prompt = """You are an expert career coach and resume writer.
Return only valid JSON.
Never invent work experience, projects, skills, or achievements."""

    exp_text = "\n".join(
        f"{e.title} at {e.company}: {' | '.join(e.bullets[:2])}"
        for e in structured_resume.experience[:3]
    )
    projects_text = "\n".join(
        f"- {p.name}: {', '.join(p.bullets[:2])}"
        for p in structured_resume.projects[:3]
    )

    prompt1 = f"""
Generate improved content for this candidate.

Name: {structured_resume.name}
Target Role: {target_role or "Software Engineer"}
Current Summary: {structured_resume.summary or "Not provided"}

Strengths:
{chr(10).join("- " + x for x in analysis.strengths[:3])}

Weaknesses:
{chr(10).join("- " + x for x in analysis.weaknesses[:3])}

Experience:
{exp_text or "None"}

Projects:
{projects_text or "None"}

Return:
{{
  "improved_summary": "3-line professional summary",
  "resume_suggestions": [
    {{"original_bullet": "", "improved_bullet": "", "reason": ""}}
  ],
  "interview_questions": ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"]
}}
"""

    data1 = clean_llm_json(call_llm(prompt1, system_prompt, max_tokens=1800))

    suggestions = [
        ResumeSuggestion(
            original_bullet=str(s.get("original_bullet", "")),
            improved_bullet=str(s.get("improved_bullet", "")),
            reason=str(s.get("reason", "")),
        )
        for s in data1.get("resume_suggestions", [])
    ]

    improved_summary = data1.get("improved_summary")
    questions = [str(q) for q in data1.get("interview_questions", [])]

    prompt2 = f"""
Create a complete improved resume in plain text followed by Career Growth Recommendations.

Candidate: {structured_resume.name}
Headline: {structured_resume.headline or target_role or "Software Engineer"}
Improved Summary: {improved_summary or structured_resume.summary or ""}
Skills: {", ".join(structured_resume.skills.technical[:15])}
Achievements: {", ".join(structured_resume.achievements) or "None"}
Certifications: {", ".join(structured_resume.certifications) or "None"}

Experience:
{exp_text or "None"}

Projects:
{projects_text or "None"}

Missing Skills:
{", ".join(analysis.missing_skills_for_role) or "None"}

Recommended Certifications:
{", ".join(analysis.recommended_certifications) or "None"}

Recommended Projects:
{chr(10).join("- " + x for x in analysis.recommended_projects) or "None"}

Preserve facts from the candidate. Do not invent experience.

Return JSON:
{{"full_improved_resume": "complete plain text resume and career growth section"}}
"""

    data2 = clean_llm_json(call_llm(prompt2, system_prompt, max_tokens=2200))

    return GeneratedContent(
        improved_summary=improved_summary,
        resume_suggestions=suggestions,
        interview_questions=questions,
        full_tailored_resume_text=data2.get("full_improved_resume"),
    )


def run_candidate_pipeline(
    resume_pdf_path: str,
    target_role: Optional[str] = None,
) -> Dict[str, Any]:
    raw_text = extract_text_from_pdf(resume_pdf_path)
    if not raw_text:
        raise ValueError("No selectable text could be extracted from the PDF.")

    structured = parse_resume_with_llm(raw_text)
    analysis = analyze_resume(structured, target_role)
    generated = generate_content_from_resume(structured, analysis, target_role)

    return {
        "structured_resume": structured,
        "resume_analysis": analysis,
        "generated_content": generated,
    }


# ---------------------------------------------------------------------------
# Recruiter-side: SQLite + JD analysis + matching
# ---------------------------------------------------------------------------

def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(CANDIDATE_DB_PATH)


def get_candidate_count() -> int:
    with get_connection() as conn:
        return int(conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])


def load_all_candidates() -> List[StructuredResume]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM candidates").fetchall()

    candidates: List[StructuredResume] = []

    for row in rows:
        try:
            candidate = StructuredResume(
                name=row[1] or "",
                email=row[2],
                phone=row[3],
                location=row[4],
                headline=row[5],
                summary=row[6],
                skills=Skills(**json.loads(row[7])) if row[7] else Skills(),
                experience=[Experience(**x) for x in json.loads(row[8] or "[]")],
                projects=[Project(**x) for x in json.loads(row[9] or "[]")],
                achievements=json.loads(row[10] or "[]"),
                certifications=json.loads(row[11] or "[]"),
                education=[Education(**x) for x in json.loads(row[12] or "[]")],
                raw_text=row[13],
            )
            candidates.append(candidate)
        except Exception:
            # Keep the behavior tolerant of one malformed DB row.
            continue

    return candidates


def parse_job_description(jd_text: str) -> JobRequirements:
    system_prompt = """You are an expert technical recruiter and job-description analyzer.
Extract key information accurately. Return only valid JSON."""

    user_prompt = f"""
Analyze this Job Description:

{jd_text[:12000]}

Return:
{{
  "job_title": "string",
  "company": "string or null",
  "location": "string or null",
  "required_skills": [],
  "preferred_skills": [],
  "experience_level": "string or null",
  "responsibilities": [],
  "keywords": [],
  "education_requirements": "string or null"
}}

Rules:
- required_skills = must-have skills
- preferred_skills = nice-to-have skills
- keywords = important technologies, domains, and soft skills
"""

    data = clean_llm_json(call_llm(user_prompt, system_prompt, max_tokens=1200))
    data["raw_text"] = jd_text
    return JobRequirements(**data)


def match_candidate_to_jd(
    candidate: StructuredResume,
    jd: JobRequirements,
) -> CandidateMatch:
    system_prompt = """You are an expert technical recruiter.
Compare the candidate to the job description honestly and realistically.
Return only valid JSON."""

    candidate_skills = ", ".join(
        (candidate.skills.technical + candidate.skills.tools)[:20]
    )
    experience = "; ".join(
        f"{e.title} at {e.company}" for e in candidate.experience[:3]
    ) or "None"

    prompt = f"""
Job Title: {jd.job_title}
Required Skills: {", ".join(jd.required_skills)}
Preferred Skills: {", ".join(jd.preferred_skills)}
Experience Level: {jd.experience_level}
Responsibilities: {", ".join(jd.responsibilities[:6])}

Candidate: {candidate.name}
Headline: {candidate.headline}
Skills: {candidate_skills}
Experience: {experience}
Projects: {", ".join(p.name for p in candidate.projects[:3]) or "None"}
Achievements: {", ".join(candidate.achievements[:3]) or "None"}
Certifications: {", ".join(candidate.certifications[:3]) or "None"}

Return:
{{
  "llm_score": 75.0,
  "strengths": [],
  "gaps": [],
  "summary": "2-sentence recruiter summary"
}}

Scoring:
90-100 Excellent
75-89 Strong
60-74 Moderate
40-59 Weak
Below 40 Poor
"""

    data = clean_llm_json(call_llm(prompt, system_prompt, max_tokens=1000))
    score = safe_float(data.get("llm_score"))

    return CandidateMatch(
        candidate_name=candidate.name,
        overall_score=score,
        llm_score=score,
        strengths=[str(x) for x in data.get("strengths", [])],
        gaps=[str(x) for x in data.get("gaps", [])],
        summary=str(data.get("summary", "")),
    )


def candidate_to_text(candidate: StructuredResume) -> str:
    skills = ", ".join(candidate.skills.technical + candidate.skills.tools)
    experience = " | ".join(
        f"{e.title} at {e.company}: {'; '.join(e.bullets[:2])}"
        for e in candidate.experience[:3]
    )
    projects = "; ".join(
        f"{p.name} ({p.technologies or ''})"
        for p in candidate.projects[:3]
    )

    return f"""
Name: {candidate.name}
Headline: {candidate.headline or ""}
Summary: {candidate.summary or ""}
Skills: {skills}
Experience: {experience}
Projects: {projects}
Achievements: {"; ".join(candidate.achievements[:3])}
Certifications: {"; ".join(candidate.certifications[:3])}
""".strip()


def build_vector_index(
    candidates: List[StructuredResume],
    reset: bool = False,
) -> int:
    global collection

    if reset:
        try:
            chroma_client.delete_collection("candidates")
        except Exception:
            pass
        collection = chroma_client.get_or_create_collection(
            name="candidates",
            metadata={"hnsw:space": "cosine"},
        )

    if collection.count() > 0 and not reset:
        return collection.count()

    documents = [candidate_to_text(c) for c in candidates]
    ids = [
        f"cand_{i}_{re.sub(r'[^A-Za-z0-9_-]+', '_', c.name)}"
        for i, c in enumerate(candidates)
    ]
    metadatas = [
        {
            "name": c.name or "",
            "headline": c.headline or "",
            "email": c.email or "",
        }
        for c in candidates
    ]

    if not documents:
        return 0

    embeddings = embedder.encode(documents).tolist()

    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )
    return len(documents)


def retrieve_candidates(jd_text: str, top_k: int = 8) -> List[dict]:
    if collection.count() == 0:
        candidates = load_all_candidates()
        build_vector_index(candidates)

    if collection.count() == 0:
        return []

    jd_embedding = embedder.encode([jd_text]).tolist()
    results = collection.query(
        query_embeddings=jd_embedding,
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    return [
        {
            "id": results["ids"][0][i],
            "name": results["metadatas"][0][i].get("name", ""),
            "document": results["documents"][0][i],
            "distance": results["distances"][0][i],
        }
        for i in range(len(results["ids"][0]))
    ]


def generate_interview_questions_for_candidate(
    candidate: StructuredResume,
    jd: JobRequirements,
    match: CandidateMatch,
) -> List[str]:
    system_prompt = "You are an expert technical interviewer. Return only valid JSON."

    prompt = f"""
Generate 6-8 interview questions for this candidate for this role.

Job: {jd.job_title}
Required Skills: {", ".join(jd.required_skills[:10])}

Candidate: {candidate.name}
Headline: {candidate.headline}
Strengths: {", ".join(match.strengths[:4])}
Gaps: {", ".join(match.gaps[:4])}

Return:
{{"interview_questions": ["Question 1", "Question 2", "Question 3", "Question 4", "Question 5", "Question 6"]}}
"""

    data = clean_llm_json(call_llm(prompt, system_prompt, max_tokens=1000))
    return [str(q) for q in data.get("interview_questions", [])]


def rank_candidates(
    candidates: List[StructuredResume],
    jd: JobRequirements,
    top_n: int = 5,
) -> List[CandidateMatch]:
    matches = [match_candidate_to_jd(c, jd) for c in candidates]
    matches.sort(key=lambda x: x.overall_score, reverse=True)
    return matches[:top_n]


def run_recruiter_pipeline_rag(
    jd_text: str,
    top_n: int = 5,
    retrieve_k: int = 8,
) -> Dict[str, Any]:
    jd = parse_job_description(jd_text)

    all_candidates = load_all_candidates()
    if not all_candidates:
        raise ValueError("No candidates found in the SQLite database.")

    # Build/reuse the semantic index.
    if collection.count() == 0:
        build_vector_index(all_candidates)

    retrieved = retrieve_candidates(jd_text, top_k=retrieve_k)
    name_map = {c.name.upper(): c for c in all_candidates}

    matches: List[CandidateMatch] = []
    for item in retrieved:
        candidate = name_map.get(item["name"].upper())
        if candidate:
            matches.append(match_candidate_to_jd(candidate, jd))

    matches.sort(key=lambda x: x.overall_score, reverse=True)
    top_matches = matches[:top_n]

    for match in top_matches:
        candidate = name_map.get(match.candidate_name.upper())
        if candidate:
            match.interview_questions = generate_interview_questions_for_candidate(
                candidate, jd, match
            )

    return {
        "job_requirements": jd,
        "retrieved": retrieved,
        "top_candidates": top_matches,
    }


# ---------------------------------------------------------------------------
# LangGraph wrappers
# ---------------------------------------------------------------------------

class CandidateState(TypedDict):
    resume_raw_text: str
    target_role: Optional[str]
    structured_resume: Optional[StructuredResume]
    resume_analysis: Optional[ResumeAnalysis]
    generated_content: Optional[GeneratedContent]
    error: Optional[str]


def resume_parser_node(state: CandidateState) -> CandidateState:
    try:
        state["structured_resume"] = parse_resume_with_llm(state["resume_raw_text"])
    except Exception as exc:
        state["error"] = str(exc)
    return state


def resume_analyzer_node(state: CandidateState) -> CandidateState:
    if state.get("error"):
        return state
    try:
        state["resume_analysis"] = analyze_resume(
            state["structured_resume"],
            state.get("target_role"),
        )
    except Exception as exc:
        state["error"] = str(exc)
    return state


def generator_node(state: CandidateState) -> CandidateState:
    if state.get("error"):
        return state
    try:
        state["generated_content"] = generate_content_from_resume(
            state["structured_resume"],
            state["resume_analysis"],
            state.get("target_role"),
        )
    except Exception as exc:
        state["error"] = str(exc)
    return state


def create_careerforge_graph():
    workflow = StateGraph(CandidateState)
    workflow.add_node("parse_resume", resume_parser_node)
    workflow.add_node("analyze_resume", resume_analyzer_node)
    workflow.add_node("generate_content", generator_node)
    workflow.set_entry_point("parse_resume")
    workflow.add_edge("parse_resume", "analyze_resume")
    workflow.add_edge("analyze_resume", "generate_content")
    workflow.add_edge("generate_content", END)
    return workflow.compile()


careerforge_agent = create_careerforge_graph()
