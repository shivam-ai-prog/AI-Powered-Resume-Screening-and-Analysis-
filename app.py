"""
CareerForge AI - Gradio application.

Run:
    python app.py

The app exposes the two pipelines found in the original notebook:
1. Candidate: upload a resume PDF + target role.
2. Recruiter: paste a job description and rank candidates from SQLite + Chroma.

Set GROQ_API_KEY before starting.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr

from model import run_candidate_pipeline, run_recruiter_pipeline_rag


def candidate_ui(resume_file: str, target_role: str):
    if not resume_file:
        return "Please upload a resume PDF.", "", "", ""

    try:
        result = run_candidate_pipeline(
            resume_file,
            target_role.strip() or None,
        )

        structured = result["structured_resume"]
        analysis = result["resume_analysis"]
        generated = result["generated_content"]

        parsed = structured.model_dump_json(indent=2)
        analysis_text = (
            f"## Score: {analysis.overall_score:.0f}/100\n\n"
            f"**Status:** {analysis.approval_status}\n\n"
            f"### Strengths\n"
            + "\n".join(f"- {x}" for x in analysis.strengths)
            + "\n\n### Weaknesses\n"
            + "\n".join(f"- {x}" for x in analysis.weaknesses)
            + "\n\n### Missing skills\n"
            + "\n".join(f"- {x}" for x in analysis.missing_skills_for_role)
            + "\n\n### Improvement areas\n"
            + "\n".join(f"- {x}" for x in analysis.improvement_areas)
            + f"\n\n### Feedback\n{analysis.feedback_summary}"
        )

        suggestions = "\n\n".join(
            f"**{i}.**\n\nOriginal: {s.original_bullet}\n\n"
            f"Improved: {s.improved_bullet}\n\nReason: {s.reason}"
            for i, s in enumerate(generated.resume_suggestions, 1)
        )

        questions = "\n".join(
            f"{i}. {q}" for i, q in enumerate(generated.interview_questions, 1)
        )

        return (
            parsed,
            analysis_text,
            generated.full_tailored_resume_text or "",
            suggestions + ("\n\n### Interview Questions\n" + questions if questions else ""),
        )

    except Exception as exc:
        return f"Error: {exc}", "", "", ""


def recruiter_ui(jd_text: str, top_n: int):
    if not jd_text or not jd_text.strip():
        return "Please paste a job description.", ""

    try:
        result = run_recruiter_pipeline_rag(
            jd_text,
            top_n=int(top_n),
            retrieve_k=max(int(top_n), 8),
        )

        jd = result["job_requirements"]
        report_lines = [
            f"# Recruiter Shortlist — {jd.job_title}",
            "",
            f"**Experience:** {jd.experience_level or 'Not specified'}",
            f"**Required skills:** {', '.join(jd.required_skills)}",
            "",
        ]

        for rank, match in enumerate(result["top_candidates"], 1):
            report_lines.extend(
                [
                    f"## #{rank} {match.candidate_name} — {match.overall_score:.0f}/100",
                    "",
                    "**Strengths**",
                    *[f"- {x}" for x in match.strengths],
                    "",
                    "**Gaps**",
                    *[f"- {x}" for x in match.gaps],
                    "",
                    f"**Summary:** {match.summary}",
                    "",
                    "**Interview questions**",
                    *[
                        f"{i}. {q}"
                        for i, q in enumerate(match.interview_questions, 1)
                    ],
                    "",
                ]
            )

        retrieval = "\n".join(
            f"- {x['name']} — vector distance {x['distance']:.3f}"
            for x in result["retrieved"]
        )

        return "\n".join(report_lines), retrieval

    except Exception as exc:
        return f"Error: {exc}", ""


with gr.Blocks(title="CareerForge AI") as demo:
    gr.Markdown(
        """
# CareerForge AI

AI-powered candidate-side resume coaching and recruiter-side candidate matching.
"""
    )

    with gr.Tab("Candidate"):
        resume = gr.File(
            label="Resume PDF",
            file_types=[".pdf"],
            type="filepath",
        )
        role = gr.Textbox(
            label="Target role",
            placeholder="e.g. Backend Engineer",
        )
        run_candidate = gr.Button("Analyze Resume", variant="primary")

        with gr.Row():
            parsed_output = gr.Code(label="Structured Resume", language="json")
            analysis_output = gr.Markdown(label="Resume Analysis")

        full_resume_output = gr.Textbox(
            label="Improved Resume + Career Growth",
            lines=24,
        )
        suggestions_output = gr.Markdown(label="Suggestions & Interview Questions")

        run_candidate.click(
            candidate_ui,
            inputs=[resume, role],
            outputs=[
                parsed_output,
                analysis_output,
                full_resume_output,
                suggestions_output,
            ],
        )

    with gr.Tab("Recruiter"):
        jd = gr.Textbox(
            label="Job Description",
            lines=14,
            placeholder="Paste the complete job description here...",
        )
        top_n = gr.Slider(
            minimum=1,
            maximum=10,
            value=5,
            step=1,
            label="Number of candidates",
        )
        run_recruiter = gr.Button("Find Candidates", variant="primary")

        recruiter_report = gr.Markdown(label="Recruiter Report")
        retrieval_output = gr.Markdown(label="Semantic Retrieval Candidates")

        run_recruiter.click(
            recruiter_ui,
            inputs=[jd, top_n],
            outputs=[recruiter_report, retrieval_output],
        )


if __name__ == "__main__":
    demo.launch()
