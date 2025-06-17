"""
prompts/metrics.py
──────────────────
LangChain prompt templates for evaluating LLM-generated insights
on structured data.  Each template returns a 1-to-5 `score`
and a brief `rationale` in a JSON object.

Metrics included
----------------
1. Correctness     — factual alignment with data
2. Helpfulness     — usefulness / actionability for the target user
3. Coherence       — clarity and logical flow
4. Complexity      — depth / novelty of reasoning
5. Verbosity       — appropriate level of detail (conciseness)

Usage
-----
from langchain.chat_models import ChatOpenAI
from prompts.metrics import metric_prompts

llm = ChatOpenAI(model_name="gpt-4o-mini")

prompt = metric_prompts["correctness"]
messages = prompt.format_messages(
    claim=claim,
    dataset_summary=eda_summary,
    task_description=task_desc or "N/A"
)
result = llm(messages)
"""

from langchain.prompts import PromptTemplate

# ───────────────────────────────────────────────────────────────────────────────
# 1. Correctness (data-backed factual accuracy)
# ───────────────────────────────────────────────────────────────────────────────
_correctness_template = """
### SYSTEM
You are a data-savvy evaluator. Rate the factual correctness of the claim on a 1-to-5 scale:

5 – Fully supported by data (strong evidence)
4 – Mostly supported (minor caveats)
3 – Unclear or weak support
2 – Likely unsupported (contradicted by data)
1 – Clearly false or unverifiable

Think step-by-step internally, but return ONLY a JSON object:
{{"score": <int 1-5>, "rationale": "<one sentence>"}}

### USER
Claim: {claim}

Dataset Summary: {dataset_summary}

Task Description: {task_description}
""".strip()

correctness_prompt = PromptTemplate(
    input_variables=["claim", "dataset_summary", "task_description"],
    template=_correctness_template,
)

# ───────────────────────────────────────────────────────────────────────────────
# 2. Helpfulness (usefulness / actionability)
# ───────────────────────────────────────────────────────────────────────────────
_helpfulness_template = """
### SYSTEM
Evaluate how helpful or actionable this insight is for {audience_description}.
Rate 1-to-5:

5 – Highly actionable; directly informs decisions
4 – Useful; clear guidance
3 – Somewhat helpful; limited impact
2 – Slightly helpful; mostly trivia/obvious
1 – Not helpful; irrelevant or misleading

Return JSON: {{ "score": ..., "rationale": "..." }}

### USER
Claim: {claim}

Dataset Summary: {dataset_summary}
""".strip()

helpfulness_prompt = PromptTemplate(
    input_variables=["claim", "dataset_summary", "audience_description"],
    template=_helpfulness_template,
)

# ───────────────────────────────────────────────────────────────────────────────
# 3. Coherence (clarity & logical flow)
# ───────────────────────────────────────────────────────────────────────────────
_coherence_template = """
### SYSTEM
Rate the clarity and logical flow of the insight (1-to-5):

5 – Crystal-clear, well-structured, logically consistent
4 – Mostly clear; minor wording issues
3 – Understandable but some gaps/awkwardness
2 – Hard to follow; poor structure
1 – Incoherent or contradictory

Return JSON: {{ "score": ..., "rationale": "..." }}

### USER
Claim: {claim}
""".strip()

coherence_prompt = PromptTemplate(
    input_variables=["claim"],
    template=_coherence_template,
)

# ───────────────────────────────────────────────────────────────────────────────
# 4. Complexity (depth / novelty)
# ───────────────────────────────────────────────────────────────────────────────
_complexity_template = """
### SYSTEM
Assess the intellectual depth or novelty of the insight (1-to-5):

5 – Deep, non-obvious reasoning or novel pattern
4 – Above-average depth
3 – Moderate; basic but correct
2 – Shallow; obvious summary
1 – Trivial; no real analysis

Return JSON: {{ "score": ..., "rationale": "..." }}

### USER
Claim: {claim}

Dataset Summary: {dataset_summary}
""".strip()

complexity_prompt = PromptTemplate(
    input_variables=["claim", "dataset_summary"],
    template=_complexity_template,
)

# ───────────────────────────────────────────────────────────────────────────────
# 5. Verbosity (conciseness vs. rambling)
# ───────────────────────────────────────────────────────────────────────────────
_verbosity_template = """
### SYSTEM
Rate how the level of detail matches the prompt’s needs (1-to-5):

5 – Concise yet complete; no fluff
4 – Slightly wordy or brief but acceptable
3 – Noticeably over- or under-detailed
2 – Excessively verbose or missing key info
1 – Severely verbose or extremely terse

Return JSON: {{ "score": ..., "rationale": "..." }}

### USER
Claim: {claim}
""".strip()

verbosity_prompt = PromptTemplate(
    input_variables=["claim"],
    template=_verbosity_template,
)

# ───────────────────────────────────────────────────────────────────────────────
# Registry for easy access in LangGraph / LangChain workflows
# ───────────────────────────────────────────────────────────────────────────────
metric_prompts = {
    "correctness": correctness_prompt,
    "helpfulness": helpfulness_prompt,
    "coherence": coherence_prompt,
    "complexity": complexity_prompt,
    "verbosity": verbosity_prompt,
}

__all__ = [
    "correctness_prompt",
    "helpfulness_prompt",
    "coherence_prompt",
    "complexity_prompt",
    "verbosity_prompt",
    "metric_prompts",
]