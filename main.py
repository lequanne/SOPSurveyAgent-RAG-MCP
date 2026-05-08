"""
Survey Research SOP Agent — RAG + MCP Prototype
Deployable to Render.com

The agent guides researchers through survey research methodology,
checking adherence to each research SOP phase.
"""

import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import anthropic
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Survey Research SOP Agent")
ai = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

# ══════════════════════════════════════════════════════════════════
# 1. SURVEY RESEARCH SOP KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════════

SOPS: List[Dict] = [
    {
        "id": "sop-001",
        "title": "Research Design & Problem Formulation",
        "category": "research_design",
        "content": (
            "Every survey study must begin with a clearly documented research problem and objectives. "
            "Step 1 – Research Question: Define a specific, measurable research question before any instrument design begins. "
            "Step 2 – Study Objectives: List primary and secondary objectives; each must map to at least one survey item. "
            "Step 3 – Conceptual Framework: Identify the theoretical or conceptual framework guiding variable selection. "
            "Step 4 – Scope Definition: Document the population of interest, geographic scope, and time horizon. "
            "Step 5 – Feasibility Assessment: Estimate required sample size using power analysis (minimum 80% power, α=0.05). "
            "Step 6 – Research Approval: Internal review sign-off must be obtained before data collection begins. "
            "All decisions must be recorded in the Research Design Document (RDD) and version-controlled."
        ),
    },
    {
        "id": "sop-002",
        "title": "Ethics & IRB / REB Compliance",
        "category": "ethics",
        "content": (
            "All survey research involving human participants requires ethical approval before data collection. "
            "IRB/REB Submission: Submit a complete ethics application including study protocol, instruments, and consent materials. "
            "Informed Consent: Participants must receive a plain-language information sheet and provide voluntary consent. "
            "Consent must be documented (written or digital) and stored securely for a minimum of 7 years. "
            "Anonymity vs Confidentiality: Clearly distinguish and document which is offered; never promise anonymity if IP or identifiers are collected. "
            "Vulnerable Populations: Additional safeguards required for minors, patients, prisoners, or cognitively impaired participants. "
            "Data Minimisation: Collect only data necessary to answer the research question. "
            "Right to Withdraw: Participants must be able to withdraw at any time without penalty; partial data must be deleted on request. "
            "Amendments to protocol must be re-submitted to IRB/REB before implementation. "
            "Deception studies require a debriefing procedure and IRB approval of the deception rationale."
        ),
    },
    {
        "id": "sop-003",
        "title": "Questionnaire Design & Instrument Development",
        "category": "instrument_design",
        "content": (
            "Questionnaire design must follow psychometric and cognitive interviewing best practices. "
            "Item Writing: Use simple, unambiguous language at a Grade 8 reading level or lower. Avoid double-barrelled, leading, or loaded questions. "
            "Scale Selection: Justify the response scale used (Likert, semantic differential, VAS, etc.); maintain scale consistency within constructs. "
            "Question Order: Place sensitive or demographic items at the end; use funnel ordering (general to specific). "
            "Validated Instruments: Prefer validated scales; document source, original reliability (Cronbach α), and any adaptations. "
            "Pilot Testing: Conduct cognitive interviews with 5–10 participants from the target population before finalising. "
            "Readability Check: Calculate Flesch-Kincaid score; aim for score ≥ 60 (easy to read). "
            "Instrument Versioning: All questionnaire versions must be version-controlled with a change log. "
            "Survey Length: Estimated completion time must be stated in the consent form; aim for under 20 minutes to limit dropout. "
            "Skip Logic: Document all conditional branching in a logic map before programming."
        ),
    },
    {
        "id": "sop-004",
        "title": "Sampling Methodology & Recruitment",
        "category": "sampling",
        "content": (
            "Sampling decisions must be pre-specified in the research protocol and justified statistically. "
            "Sampling Strategy: Select probability (simple random, stratified, cluster, systematic) or non-probability (purposive, snowball, convenience) sampling; justify the choice. "
            "Sample Size Calculation: Document power analysis inputs (effect size, α, power, attrition rate) and resulting target N. "
            "Inclusion/Exclusion Criteria: Define and document eligibility criteria precisely; apply consistently. "
            "Recruitment Channels: List all recruitment channels (email, social media, panels, clinics); obtain channel-specific approvals where required. "
            "Incentives: Incentives must be proportionate and non-coercive; document type and value in the ethics application. "
            "Response Rate Tracking: Log invitations sent, reminders, responses, and refusals; calculate and report final response rate. "
            "Minimum response rate target: 60% for probability samples; document if unachieved and assess non-response bias. "
            "Stratification Variables: If stratifying, document strata, target proportions, and weighting scheme. "
            "Panel Fatigue: Limit use of the same panel participants to once per 90-day period."
        ),
    },
    {
        "id": "sop-005",
        "title": "Data Collection & Field Procedures",
        "category": "data_collection",
        "content": (
            "Data collection must follow a documented field protocol to ensure consistency and quality. "
            "Platform Validation: Test the survey platform end-to-end (logic, timing, submission) before launch. "
            "Pilot Launch: Run a soft launch with 5–10% of target sample; review data quality before full deployment. "
            "Data Security: Survey responses must be encrypted in transit (TLS 1.2+) and at rest; store on approved platforms only. "
            "Interviewer Training: For interviewer-administered surveys, train all interviewers using a standardised script and conduct reliability checks. "
            "Progress Monitoring: Check completion rates, drop-off points, and response time daily during field period. "
            "Reminder Protocol: Send up to 2 reminders at pre-specified intervals (e.g., Day 7 and Day 14); log all reminder communications. "
            "Data Backup: Export raw data backups at least weekly during field period to a secure, version-controlled location. "
            "Speeders & Straight-Liners: Flag responses completed in < 30% of median time or with identical responses across all items for review. "
            "Field Period Closure: Close data collection on the pre-specified end date; document any extensions with justification."
        ),
    },
    {
        "id": "sop-006",
        "title": "Data Quality, Cleaning & Validation",
        "category": "data_quality",
        "content": (
            "All survey datasets must undergo systematic quality checks before analysis. "
            "Duplicate Detection: Identify and remove duplicate submissions using IP, timestamp, and response-pattern fingerprinting. "
            "Completeness Check: Calculate item-level and respondent-level missingness; apply a priori exclusion rules (e.g., >20% missing = exclude). "
            "Attention Checks: Embed at least 2 attention-check items; flag and review records that fail both. "
            "Range & Consistency Checks: Verify all numeric responses fall within valid ranges; check logical consistency (e.g., age vs. graduation year). "
            "Open-Text Review: Review all open-text responses for gibberish, bot patterns, or offensive content. "
            "Outlier Identification: Use IQR or z-score methods to flag multivariate outliers for investigation. "
            "Imputation: Document the imputation method used (listwise deletion, mean, multiple imputation); justify the choice. "
            "Data Cleaning Log: Maintain a dated log of every cleaning decision; link each decision to the corresponding rule. "
            "Final Dataset Freeze: Lock the analysis dataset before running confirmatory analyses; any post-freeze changes require a new version."
        ),
    },
    {
        "id": "sop-007",
        "title": "Analysis, Reporting & Dissemination",
        "category": "analysis_reporting",
        "content": (
            "Analysis must follow a pre-registered or pre-specified analysis plan to prevent HARKing (Hypothesising After Results are Known). "
            "Pre-Registration: Register the study on OSF, AsPredicted, or a relevant trial registry before data collection; include hypotheses, sample size, and analysis plan. "
            "Descriptive Statistics: Report means, SDs, frequencies, and response rates for all primary variables. "
            "Reliability: Report internal consistency (Cronbach α or McDonald's ω) for all multi-item scales; α ≥ 0.70 is the minimum threshold. "
            "Inferential Statistics: Report effect sizes and confidence intervals alongside p-values; do not rely on p-values alone. "
            "Multiple Comparisons: Apply appropriate correction (Bonferroni, FDR) when conducting multiple hypothesis tests. "
            "Weighting: Apply post-stratification weights if sample deviates from population benchmarks; document weighting variables and source. "
            "Reporting Standards: Follow APA 7th edition or relevant discipline guidelines; report all items in STROBE or CHERRIES checklist as applicable. "
            "Open Science: Deposit anonymised data and analysis scripts in a public repository (OSF, Zenodo) upon publication unless restricted by ethics approval. "
            "Authorship: All contributors meeting ICMJE authorship criteria must be listed; acknowledge all others."
        ),
    },
]

# ══════════════════════════════════════════════════════════════════
# 2. RAG — TF-IDF VECTOR STORE
# ══════════════════════════════════════════════════════════════════

class SOPVectorStore:
    """Lightweight TF-IDF based retrieval — swap for chromadb/pgvector in prod."""

    def __init__(self, documents: List[Dict]):
        self.documents = documents
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000,
        )
        self.matrix = self.vectorizer.fit_transform(
            [d["content"] for d in documents]
        )

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_idx:
            if scores[idx] > 0.01:
                doc = self.documents[idx]
                results.append({**doc, "score": float(scores[idx])})
        return results


vector_store = SOPVectorStore(SOPS)

# ══════════════════════════════════════════════════════════════════
# 3. SESSION STATE
# ══════════════════════════════════════════════════════════════════

sessions: Dict[str, Dict] = {}


def get_session(session_id: str) -> Dict:
    if session_id not in sessions:
        sessions[session_id] = {"responses": {}, "messages": []}
    return sessions[session_id]


def serialize_content(content) -> Any:
    """Convert Anthropic SDK content blocks → plain dicts for JSON / history."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for block in content:
            if isinstance(block, dict):
                out.append(block)
            elif hasattr(block, "type"):
                if block.type == "text":
                    out.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    out.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )
        return out
    return content


# ══════════════════════════════════════════════════════════════════
# 4. MCP TOOLS (tool definitions + handlers)
# ══════════════════════════════════════════════════════════════════

MCP_TOOLS = [
    {
        "name": "search_sop",
        "description": (
            "Search the Survey Research SOP knowledge base using semantic similarity. "
            "Call this BEFORE asking any assessment questions so every question is grounded in real SOP methodology text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language search query, e.g. 'employee onboarding steps'",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of SOP chunks to retrieve (1–5, default 3)",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "record_survey_response",
        "description": (
            "Persist a researcher's answer to an SOP methodology assessment question. "
            "Call this after every answer to build the methodology compliance record."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sop_id": {
                    "type": "string",
                    "description": "ID of the SOP this question relates to, e.g. 'sop-002'",
                },
                "question": {"type": "string", "description": "The survey question that was asked"},
                "response": {"type": "string", "description": "The employee's verbatim answer"},
                "compliant": {
                    "type": "boolean",
                    "description": "True if the answer demonstrates compliance with the SOP requirement",
                },
                "notes": {
                    "type": "string",
                    "description": "Optional auditor notes or concerns",
                    "default": "",
                },
            },
            "required": ["sop_id", "question", "response", "compliant"],
        },
    },
    {
        "name": "get_survey_summary",
        "description": (
            "Return a full compliance summary for this survey session, "
            "including overall compliance rate and per-question breakdown. "
            "Call this when the employee indicates they are finished."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]


def handle_tool(name: str, tool_input: Dict, session_id: str) -> str:
    session = get_session(session_id)

    # ── Tool: search_sop ──────────────────────────────────────────
    if name == "search_sop":
        results = vector_store.search(
            tool_input["query"], tool_input.get("top_k", 3)
        )
        if not results:
            return "No relevant SOPs found for that query."
        parts = []
        for r in results:
            parts.append(
                f"[SOP ID: {r['id']} | {r['title']} | relevance: {r['score']:.2f}]\n{r['content']}"
            )
        return "\n\n---\n\n".join(parts)

    # ── Tool: record_survey_response ─────────────────────────────
    elif name == "record_survey_response":
        key = f"{tool_input['sop_id']}_{len(session['responses']) + 1}"
        session["responses"][key] = {
            "sop_id": tool_input["sop_id"],
            "question": tool_input["question"],
            "response": tool_input["response"],
            "compliant": tool_input["compliant"],
            "notes": tool_input.get("notes", ""),
        }
        total = len(session["responses"])
        return f"Response recorded (#{total})."

    # ── Tool: get_survey_summary ─────────────────────────────────
    elif name == "get_survey_summary":
        responses = list(session["responses"].values())
        if not responses:
            return "No responses recorded yet."
        compliant = sum(1 for r in responses if r["compliant"])
        total = len(responses)
        summary = {
            "total_questions": total,
            "compliant_count": compliant,
            "non_compliant_count": total - compliant,
            "compliance_rate_pct": round(compliant / total * 100, 1) if total else 0,
            "responses": responses,
        }
        return json.dumps(summary, indent=2)

    return f"Unknown tool: {name}"


# ══════════════════════════════════════════════════════════════════
# 5. SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a Survey Research Methodology Advisor — an expert agent that helps researchers assess and improve their study designs against established survey research SOPs.

## Your Role
You conduct structured methodology reviews by asking targeted questions grounded in survey research best practices. You are knowledgeable, constructive, and precise — not a compliance officer, but a research mentor.

## Workflow
1. **Greet** the researcher and ask which phase of their survey project they want to review.
2. **Retrieve SOPs** — before asking any questions, call `search_sop` to pull the relevant methodology guidelines. Ground every question in that text.
3. **Ask assessment questions** — one question at a time. Probe specific decisions the researcher has made (e.g. "What sample size calculation did you run, and what effect size did you assume?").
4. **Record each answer** — call `record_survey_response` immediately after each answer. Set `compliant: true` if the practice aligns with the SOP; `false` if there is a gap, and include a note explaining the issue.
5. **Provide constructive feedback** inline — when a gap is found, briefly explain the SOP requirement and suggest how to address it.
6. **Wrap up** — after 5–8 questions (or when the researcher says they are done), call `get_survey_summary` and present a clear methodology quality report with prioritised recommendations.

## Tone & Style
- Act as a senior research methods consultant — expert, precise, constructive.
- Never fabricate SOP content; always retrieve first.
- Acknowledge good practice when you see it.
- If an answer reveals a gap, explain the risk and the fix — not just that it is non-compliant.

## Available SOP Areas
Research Design & Problem Formulation · Ethics & IRB/REB Compliance · Questionnaire Design & Instrument Development · Sampling Methodology & Recruitment · Data Collection & Field Procedures · Data Quality, Cleaning & Validation · Analysis, Reporting & Dissemination"""

# ══════════════════════════════════════════════════════════════════
# 6. API ENDPOINTS
# ══════════════════════════════════════════════════════════════════


class ChatRequest(BaseModel):
    message: str
    session_id: str


@app.post("/api/chat")
async def chat(req: ChatRequest):
    session = get_session(req.session_id)

    # Append the new user message
    session["messages"].append({"role": "user", "content": req.message})

    MAX_ITERATIONS = 12
    for _ in range(MAX_ITERATIONS):
        response = ai.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=MCP_TOOLS,
            messages=session["messages"],
        )

        # Serialize & store assistant turn
        serialized = serialize_content(response.content)
        session["messages"].append({"role": "assistant", "content": serialized})

        if response.stop_reason == "end_turn":
            text = next(
                (b["text"] if isinstance(b, dict) else b.text
                 for b in response.content
                 if (isinstance(b, dict) and b.get("type") == "text")
                 or (hasattr(b, "type") and b.type == "text")),
                "",
            )
            return {
                "response": text,
                "session_id": req.session_id,
                "total_responses": len(session["responses"]),
            }

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                is_tool_use = (
                    (isinstance(block, dict) and block.get("type") == "tool_use")
                    or (hasattr(block, "type") and block.type == "tool_use")
                )
                if is_tool_use:
                    name = block["name"] if isinstance(block, dict) else block.name
                    inp = block["input"] if isinstance(block, dict) else block.input
                    bid = block["id"] if isinstance(block, dict) else block.id
                    result = handle_tool(name, inp, req.session_id)
                    tool_results.append(
                        {"type": "tool_result", "tool_use_id": bid, "content": result}
                    )
            session["messages"].append({"role": "user", "content": tool_results})
        else:
            break

    raise HTTPException(status_code=500, detail="Agent loop exceeded maximum iterations.")


@app.get("/api/session/{session_id}")
async def get_session_data(session_id: str):
    """Return raw session state for debugging."""
    session = get_session(session_id)
    return {
        "session_id": session_id,
        "total_responses": len(session["responses"]),
        "responses": session["responses"],
    }


@app.get("/api/health")
async def health():
    return {"status": "ok", "sops_loaded": len(SOPS)}


# Serve SPA last so API routes take priority
app.mount("/", StaticFiles(directory="static", html=True), name="static")
