"""Multi-agent LinkedIn career portfolio optimizer powered by Gemini.

Run:
    python portfolio_optimizer.py

The default input is ``certifications.txt`` and the generated report is
``LINKEDIN_SKILL_MAP.md``. Both paths can be overridden with CLI flags.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types


DEFAULT_MODEL = "gemini-3.1-flash-lite"
FALLBACK_MODELS = ("gemini-flash-lite-latest", "gemini-3.5-flash")
DEFAULT_INPUT = Path("certifications.txt")
DEFAULT_OUTPUT = Path("LINKEDIN_SKILL_MAP.md")
AI_TAXONOMY = (
    "Generative AI & LLMs",
    "Deep Learning & Neural Networks",
    "Classic Machine Learning & Regression",
)
DATA_TAXONOMY = (
    "Data Analysis & Visualization",
    "Database & Querying",
    "Software Engineering & Python",
)
FULL_TAXONOMY = AI_TAXONOMY + DATA_TAXONOMY
REQUIRED_SKILL_ROWS = {
    "Generative AI & LLMs": (
        "Retrieval-Augmented Generation (RAG)",
        "Prompt Engineering",
        "Transformers",
        "Large Language Models",
        "Generative AI",
    ),
    "Deep Learning & Neural Networks": (
        "Deep Learning",
        "Neural Networks",
        "Autoencoders",
        "Encoder-Decoder Architecture",
    ),
    "Classic Machine Learning & Regression": (
        "Classic Machine Learning",
        "Regression Analysis",
        "Machine Learning",
    ),
    "Data Analysis & Visualization": (
        "Data Analysis",
        "Data Visualization",
        "Pandas",
        "Power BI",
        "Exploratory Data Analysis (EDA)",
    ),
    "Database & Querying": (
        "SQL",
        "Microsoft SQL Server",
        "Database Design",
    ),
    "Software Engineering & Python": (
        "Python",
        "Git/GitHub",
        "JavaScript",
        "Software Testing",
        "Operating Systems",
    ),
}


@dataclass(frozen=True)
class Certification:
    """A normalized certification extracted from the LinkedIn export."""

    name: str
    issuing_organization: str
    date: str | None
    linkedin_skills: tuple[str, ...]
    source_excerpt: str = ""


class GeminiAgent:
    """Small shared Gemini client used by each specialist agent."""

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is missing. Add it in Replit Secrets and run again."
            )
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def ask_json(self, agent_name: str, instructions: str, input_text: str) -> dict[str, Any]:
        prompt = f"""You are the {agent_name} in a career portfolio optimizer.

{instructions}

Return only valid JSON. Do not wrap it in Markdown fences. Preserve facts from
the source and use null when a field cannot be determined. Do not invent
certificates, employers, dates, or skills.

SOURCE DATA:
{input_text}
"""
        response = None
        models_to_try = [self.model, *FALLBACK_MODELS]
        last_error: Exception | None = None
        for model in dict.fromkeys(models_to_try):
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        response_mime_type="application/json",
                    ),
                )
                break
            except Exception as error:
                last_error = error
                if not _is_transient_capacity_error(error):
                    raise
        if response is None:
            raise RuntimeError(
                f"{agent_name} could not reach an available Gemini model: {last_error}"
            )
        if not response.text:
            raise RuntimeError(f"{agent_name} returned an empty response.")
        parsed = self._parse_json(response.text, agent_name)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"{agent_name} returned JSON in an unexpected format.")
        return parsed

    @staticmethod
    def _parse_json(text: str, agent_name: str) -> Any:
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Be tolerant if a model adds a short sentence despite the instruction.
            starts = [index for index in (cleaned.find("{"), cleaned.find("[")) if index >= 0]
            if starts:
                candidate = cleaned[min(starts) :]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError as error:
                    # Some responses contain one valid JSON object followed by
                    # an accidental duplicate or explanatory text. Decode the
                    # first complete value and ignore only the trailing text.
                    try:
                        parsed, _ = json.JSONDecoder().raw_decode(candidate)
                        return parsed
                    except json.JSONDecodeError:
                        pass
                    raise RuntimeError(
                        f"{agent_name} returned invalid JSON: {error}"
                    ) from error
            raise RuntimeError(f"{agent_name} returned invalid JSON.")


class IngestionAgent:
    """Turns noisy copied LinkedIn text into a clean certificate inventory."""

    def __init__(self, gemini: GeminiAgent) -> None:
        self.gemini = gemini

    def run(self, raw_text: str) -> list[Certification]:
        result = self.gemini.ask_json(
            "Ingestion Agent",
            """Parse the copied LinkedIn certifications text into a JSON object
with this exact shape:
{
  "certifications": [
    {
      "name": "Certificate Name",
      "issuing_organization": "Issuing Organization",
      "date": "Original month and year or null",
      "linkedin_skills": ["skills explicitly shown by LinkedIn"],
      "source_excerpt": "short useful evidence from the certificate description"
    }
  ]
}

Use the first clear certificate title followed by its organization and issued
date. Ignore logo labels, repeated thumbnail captions, credential buttons,
credential IDs, separators, social post captions, and unrelated prose. Keep
distinct certificates as separate records, including certificates without an
explicit date. Dates may be Turkish; preserve their original meaning in the
date field. Include an existing LinkedIn skill only when the source explicitly
shows it.""",
            raw_text,
        )
        raw_certificates = result.get("certifications")
        if not isinstance(raw_certificates, list):
            raise RuntimeError("Ingestion Agent did not return a certifications list.")

        certificates: list[Certification] = []
        for item in raw_certificates:
            if not isinstance(item, dict):
                continue
            name = self._text(item.get("name"))
            organization = self._text(item.get("issuing_organization"))
            if not name or not organization:
                continue
            skills = item.get("linkedin_skills")
            normalized_skills = tuple(
                skill.strip()
                for skill in skills
                if isinstance(skill, str) and skill.strip()
            ) if isinstance(skills, list) else ()
            certificates.append(
                Certification(
                    name=name,
                    issuing_organization=organization,
                    date=self._optional_text(item.get("date")),
                    linkedin_skills=normalized_skills,
                    source_excerpt=self._optional_text(item.get("source_excerpt")) or "",
                )
            )
        if not certificates:
            raise RuntimeError("Ingestion Agent found no usable certifications.")
        return certificates

    @staticmethod
    def _text(value: Any) -> str:
        return value.strip() if isinstance(value, str) else ""

    @classmethod
    def _optional_text(cls, value: Any) -> str | None:
        text = cls._text(value)
        return text or None


class AIAndMLSpecialistAgent:
    """Finds granular evidence across the AI and ML taxonomy."""

    def __init__(self, gemini: GeminiAgent) -> None:
        self.gemini = gemini

    def run(self, certificates: list[Certification]) -> dict[str, Any]:
        return self.gemini.ask_json(
            "AI & ML Specialist Agent",
            """Review the normalized certificate inventory and build a granular
taxonomy for exactly these three sub-categories:
1. "Generative AI & LLMs" with these required skill rows: RAG (Retrieval-
   Augmented Generation), Prompting (Prompt Engineering), Transformers, Large
   Language Models, and Generative AI
2. "Deep Learning & Neural Networks" with these required skill rows: Deep
   Learning, Neural Networks, Autoencoders, and Encoder-Decoder Architecture
3. "Classic Machine Learning & Regression" with these required skill rows:
   Classic Machine Learning, Regression Analysis, and Machine Learning

Return this exact shape:
{
  "taxonomy": [
    {
      "sub_category": "one of the three exact sub-category names",
      "skill_evidence": [
        {
          "skill": "one required skill row listed above",
          "evidence_certificates": ["EXACT certificate names from the input"],
          "evidence_type": "explicit_title|explicit_linkedin_skill|source_description|related_context",
          "evidence_summary": "what the named certificates document"
        }
      ],
      "domain_summary": "granular evidence-based summary",
      "evidence_status": "documented|adjacent|not_explicitly_named",
      "actionable_profile_use": [
        "specific way to use this taxonomy entry on LinkedIn"
      ]
    }
  ],
  "specialist_summary": "one evidence-based summary across these three domains"
}

Include all three sub-categories AND every required skill row listed above,
even when a specific technique has no direct evidence. Distinguish explicit
evidence from related context. Never turn a broad certificate into
proof of a narrower technique unless the title, LinkedIn skill, or source
description supports it. This is a skill taxonomy, not a judgment about the
candidate's level or hands-on experience. Never write "lacks experience",
"only theoretical", "foundational means no practical skill", or similar.

Every evidence_certificates item MUST be copied exactly from a certificate
name in the provided inventory. Do not abbreviate, paraphrase, correct spelling,
or invent certificate names. Use [] when there is no direct evidence.""",
            json.dumps([self._certificate_dict(cert) for cert in certificates], ensure_ascii=False),
        )

    @staticmethod
    def _certificate_dict(cert: Certification) -> dict[str, Any]:
        return {
            "name": cert.name,
            "issuing_organization": cert.issuing_organization,
            "date": cert.date,
            "linkedin_skills": list(cert.linkedin_skills),
            "source_excerpt": cert.source_excerpt,
        }


class DataScienceSpecialistAgent:
    """Finds granular evidence across the analytics and engineering taxonomy."""

    def __init__(self, gemini: GeminiAgent) -> None:
        self.gemini = gemini

    def run(self, certificates: list[Certification]) -> dict[str, Any]:
        return self.gemini.ask_json(
            "Data Science Specialist Agent",
            """Review the normalized certificate inventory and build a granular
taxonomy for exactly these three sub-categories:
1. "Data Analysis & Visualization" with these required skill rows: Data
   Analysis, Data Visualization, Pandas, Power BI, and EDA (Exploratory Data
   Analysis)
2. "Database & Querying" with these required skill rows: SQL, Microsoft SQL
   Server, and Database Design
3. "Software Engineering & Python" with these required skill rows: Python,
   Git/GitHub, JavaScript, Software Testing, and Operating Systems

Return this exact shape:
{
  "taxonomy": [
    {
      "sub_category": "one of the three exact sub-category names",
      "skill_evidence": [
        {
          "skill": "one required skill row listed above",
          "evidence_certificates": ["EXACT certificate names from the input"],
          "evidence_type": "explicit_title|explicit_linkedin_skill|source_description|related_context",
          "evidence_summary": "what the named certificates document"
        }
      ],
      "domain_summary": "granular evidence-based summary",
      "evidence_status": "documented|adjacent|not_explicitly_named",
      "actionable_profile_use": [
        "specific way to use this taxonomy entry on LinkedIn"
      ]
    }
  ],
  "specialist_summary": "one evidence-based summary across these three domains"
}

Include all three sub-categories AND every required skill row listed above,
even when a specific technique has no direct evidence. Treat an explicit
LinkedIn skill as valid evidence. Do not infer
Pandas, Power BI, EDA, Database Design, or another narrow technique merely
because a certificate mentions data or analytics. This is a skill taxonomy, not
a judgment about the candidate's level or hands-on experience. Never write
"lacks experience", "only theoretical", "foundational means no practical skill",
or similar.

Every evidence_certificates item MUST be copied exactly from a certificate
name in the provided inventory. Do not abbreviate, paraphrase, correct spelling,
or invent certificate names. Use [] when there is no direct evidence.""",
            json.dumps([self._certificate_dict(cert) for cert in certificates], ensure_ascii=False),
        )

    @staticmethod
    def _certificate_dict(cert: Certification) -> dict[str, Any]:
        return {
            "name": cert.name,
            "issuing_organization": cert.issuing_organization,
            "date": cert.date,
            "linkedin_skills": list(cert.linkedin_skills),
            "source_excerpt": cert.source_excerpt,
        }


class LinkedInSkillMappingAgent:
    """Maps specialist evidence into search-optimized LinkedIn skill tags."""

    def __init__(self, gemini: GeminiAgent) -> None:
        self.gemini = gemini

    def run(
        self,
        certificates: list[Certification],
        ai_ml_findings: dict[str, Any],
        data_science_findings: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "certifications": [
                {
                    "name": cert.name,
                    "issuing_organization": cert.issuing_organization,
                    "date": cert.date,
                    "linkedin_skills": list(cert.linkedin_skills),
                }
                for cert in certificates
            ],
            "ai_ml_findings": ai_ml_findings,
            "data_science_findings": data_science_findings,
        }
        return self.gemini.ask_json(
            "LinkedIn Skill Mapping Agent",
            """Create a granular, actionable LinkedIn skill map from the evidence.
Return this exact shape:
{
  "profile_positioning": "one concise positioning statement",
  "recommended_headline_keywords": ["specific search keywords"],
  "taxonomy": [
    {
      "sub_category": "one of the six exact taxonomy names",
      "linkedin_skill_tags": ["standard LinkedIn skill tags"],
      "skill_evidence": [
        {
          "skill_tag": "one LinkedIn skill tag",
          "evidence_certificates": ["EXACT certificate names from the input"],
          "evidence_summary": "why these exact certificates support the tag",
          "profile_application": "specific LinkedIn usage recommendation"
        }
      ],
      "domain_summary": "what this domain represents in the profile",
      "evidence_status": "documented|adjacent|not_explicitly_named",
      "actionable_profile_use": ["specific profile action"]
    }
  ],
  "skills": [
    {
      "skill": "standard LinkedIn skill tag",
      "category": "one of the six exact taxonomy names or Supporting",
      "confidence": "high|medium|emerging",
      "evidence_certificates": ["EXACT certificate names"],
      "rationale": "short factual mapping rationale",
      "search_variants": ["closely related search phrases"]
    }
  ],
  "priority_actions": ["three specific LinkedIn profile actions"]
}

The six exact taxonomy names are:
- Generative AI & LLMs
- Deep Learning & Neural Networks
- Classic Machine Learning & Regression
- Data Analysis & Visualization
- Database & Querying
- Software Engineering & Python

The final taxonomy MUST include every row below, with an empty
evidence_certificates list when the inventory does not directly document it:
- Generative AI & LLMs: Retrieval-Augmented Generation (RAG), Prompt
  Engineering, Transformers, Large Language Models, Generative AI
- Deep Learning & Neural Networks: Deep Learning, Neural Networks, Autoencoders,
  Encoder-Decoder Architecture
- Classic Machine Learning & Regression: Classic Machine Learning, Regression
  Analysis, Machine Learning
- Data Analysis & Visualization: Data Analysis, Data Visualization, Pandas,
  Power BI, Exploratory Data Analysis (EDA)
- Database & Querying: SQL, Microsoft SQL Server, Database Design
- Software Engineering & Python: Python, Git/GitHub, JavaScript, Software
  Testing, Operating Systems

Map narrow evidence to narrow LinkedIn tags. Examples include Generative AI,
Large Language Models, Prompt Engineering, Retrieval-Augmented Generation (RAG),
Transformers, Deep Learning, Neural Networks, Machine Learning, Regression
Analysis, Python, Data Analysis, Statistical Data Analysis, SQL, Database
Systems, Git, JavaScript, and Software Testing. Only use RAG, Transformers,
Pandas, Power BI, EDA, or Database Design if the specialist evidence supports
them; do not add them because they are common in the field.

For every taxonomy entry and every LinkedIn skill, list the EXACT certificate
names that serve as evidence. Certificate names must be copied character for
character from the inventory, including spelling and punctuation. Do not use
generic evidence phrases in place of names. Keep [] for a skill with no direct
evidence. Do not describe the candidate as lacking hands-on experience,
lacking professional experience, or having only theoretical knowledge. Use
"not_explicitly_named" only to label an evidence gap for a narrow sub-skill;
frame the overall output as a comprehensive skill taxonomy.

Return no "caveats" field. Do not invent modules, projects, employers, or
credentials.""",
            json.dumps(payload, ensure_ascii=False),
        )


def render_markdown(
    certificates: list[Certification],
    ai_ml_findings: dict[str, Any],
    data_science_findings: dict[str, Any],
    skill_map: dict[str, Any],
) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    valid_certificate_names = {certificate.name for certificate in certificates}
    lines = [
        "# LinkedIn Skill Map",
        "",
        "> Generated by the Multi-Agent Career Portfolio Optimizer.",
        "",
        f"Generated: `{generated_at}`  ",
        f"Certificates parsed: **{len(certificates)}**",
        "",
        "## Profile Positioning",
        "",
        str(skill_map.get("profile_positioning") or "No positioning statement returned."),
        "",
        "### Recommended Headline Keywords",
        "",
    ]
    lines.extend(_bullet_list(skill_map.get("recommended_headline_keywords")))
    lines.extend(["", "## Recommended LinkedIn Skills", ""])
    skills = skill_map.get("skills")
    if isinstance(skills, list) and skills:
        lines.extend(
            [
                "| Skill | Category | Confidence | Evidence |",
                "|---|---|---|---|",
            ]
        )
        for item in skills:
            if not isinstance(item, dict):
                continue
            evidence_names = _exact_certificate_names(
                item.get("evidence_certificates"), valid_certificate_names
            )
            evidence = ", ".join(evidence_names)
            rationale = _clean_cell(item.get("rationale"))
            evidence_text = _clean_cell(evidence or rationale or "Evidence recorded by agents")
            lines.append(
                f"| {_clean_cell(item.get('skill'))} | "
                f"{_clean_cell(item.get('category'))} | "
                f"{_clean_cell(item.get('confidence'))} | {evidence_text} |"
            )
    else:
        lines.append("_No skill recommendations returned._")

    lines.extend(["", "## Detailed Skill Taxonomy", ""])
    lines.append(
        "Each skill below is mapped to the exact certificate names that provide "
        "evidence in the parsed LinkedIn export."
    )
    lines.append("")
    lines.extend(
        _render_mapping_taxonomy(
            skill_map.get("taxonomy"), valid_certificate_names
        )
    )

    lines.extend(["", "## Specialist Agent Summaries", ""])
    lines.extend(_render_specialist_summary("AI & ML", ai_ml_findings))
    lines.extend(_render_specialist_summary("Data Science & Engineering", data_science_findings))

    lines.extend(["", "## Priority Actions", ""])
    lines.extend(_bullet_list(skill_map.get("priority_actions")))

    lines.extend(["", "## Parsed Certificate Inventory", ""])
    lines.extend(
        [
            "| # | Certificate | Issuing organization | Date | Existing LinkedIn skills |",
            "|---:|---|---|---|---|",
        ]
    )
    for index, cert in enumerate(certificates, start=1):
        lines.append(
            f"| {index} | {_clean_cell(cert.name)} | "
            f"{_clean_cell(cert.issuing_organization)} | "
            f"{_clean_cell(cert.date or 'Not stated')} | "
            f"{_clean_cell(', '.join(cert.linkedin_skills) or 'None shown')} |"
        )
    lines.append("")
    return "\n".join(lines)


def _render_mapping_taxonomy(
    taxonomy: Any, valid_certificate_names: set[str]
) -> list[str]:
    entries = {
        item.get("sub_category"): item
        for item in taxonomy
        if isinstance(item, dict) and isinstance(item.get("sub_category"), str)
    } if isinstance(taxonomy, list) else {}
    lines: list[str] = []
    for category in FULL_TAXONOMY:
        entry = entries.get(category, {})
        lines.extend([f"### {category}", ""])
        lines.append(
            f"**Evidence status:** {_clean_cell(entry.get('evidence_status', 'not returned'))}"
        )
        lines.append("")
        lines.append(
            str(
                entry.get("domain_summary")
                or "No domain summary was returned for this taxonomy category."
            )
        )
        lines.append("")

        skill_evidence = entry.get("skill_evidence")
        returned_skills = {
            (skill.get("skill_tag") or skill.get("skill")): skill
            for skill in skill_evidence
            if isinstance(skill, dict)
            and isinstance(skill.get("skill_tag") or skill.get("skill"), str)
        } if isinstance(skill_evidence, list) else {}
        for skill_name in REQUIRED_SKILL_ROWS[category]:
            skill = returned_skills.get(skill_name, {})
            lines.extend(
                [
                    f"#### {_clean_cell(skill_name)}",
                    "",
                    f"**Evidence type:** {_clean_cell(skill.get('evidence_type', 'not_explicitly_named'))}",
                    "",
                    "**Exact certificate evidence:**",
                ]
            )
            exact_names = _exact_certificate_names(
                skill.get("evidence_certificates"), valid_certificate_names
            )
            lines.extend(_bullet_list(exact_names))
            summary = skill.get("evidence_summary")
            if summary:
                lines.extend(["", f"**Evidence summary:** {summary}"])
            elif not exact_names:
                lines.extend(
                    [
                        "",
                        "**Evidence summary:** This sub-skill is not explicitly "
                        "named in the parsed certificate inventory.",
                    ]
                )
            application = skill.get("profile_application")
            if application:
                lines.extend(["", f"**LinkedIn use:** {application}"])
            lines.append("")

        lines.append("**Actionable profile use:**")
        lines.extend(_bullet_list(entry.get("actionable_profile_use")))
        lines.append("")
    return lines


def _render_specialist_summary(label: str, findings: dict[str, Any]) -> list[str]:
    summary = findings.get("specialist_summary")
    lines = [f"### {label} Specialist", ""]
    lines.append(str(summary or "No specialist summary returned."))
    lines.append("")
    taxonomy = findings.get("taxonomy")
    if isinstance(taxonomy, list):
        for entry in taxonomy:
            if not isinstance(entry, dict):
                continue
            category = entry.get("sub_category") or "Unnamed category"
            lines.append(
                f"- **{category}** — {_clean_cell(entry.get('evidence_status'))}: "
                f"{_clean_cell(entry.get('domain_summary'))}"
            )
    lines.append("")
    return lines


def _bullet_list(value: Any) -> list[str]:
    items = _string_list(value)
    return [f"- {item}" for item in items] or ["- None returned."]


def _string_list(value: Any) -> list[str]:
    return [item.strip() for item in value if isinstance(item, str) and item.strip()] if isinstance(value, list) else []


def _exact_certificate_names(value: Any, valid_names: set[str]) -> list[str]:
    """Keep only exact certificate names from the ingestion output."""
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(
        item for item in value if isinstance(item, str) and item in valid_names
    ))


def _clean_cell(value: Any) -> str:
    text = str(value or "").replace("|", "\\|").replace("\n", " ")
    return text.strip() or "—"


def _is_transient_capacity_error(error: Exception) -> bool:
    """Allow fallback for temporary model capacity failures, not bad requests."""
    message = str(error).upper()
    return "503" in message or "UNAVAILABLE" in message or "RESOURCE_EXHAUSTED" in message


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Raw LinkedIn text file")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Markdown report path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model name")
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()
    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 1

    raw_text = args.input.read_text(encoding="utf-8-sig")
    if not raw_text.strip():
        print(f"Input file is empty: {args.input}", file=sys.stderr)
        return 1

    try:
        gemini = GeminiAgent(model=args.model)
        certificates = IngestionAgent(gemini).run(raw_text)
        ai_ml_findings = AIAndMLSpecialistAgent(gemini).run(certificates)
        data_science_findings = DataScienceSpecialistAgent(gemini).run(certificates)
        skill_map = LinkedInSkillMappingAgent(gemini).run(
            certificates, ai_ml_findings, data_science_findings
        )
        report = render_markdown(
            certificates, ai_ml_findings, data_science_findings, skill_map
        )
        args.output.write_text(report, encoding="utf-8")
    except Exception as error:
        print(f"Optimizer failed: {error}", file=sys.stderr)
        return 1

    print(f"Parsed {len(certificates)} certificates.")
    print(f"Report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())