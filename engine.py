"""Scheme Saathi engine: local Gemma 4 (via Ollama) + deterministic eligibility rules.
No cloud APIs. The LLM never decides eligibility - plain code does."""
import json
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma4:e4b"  # use "gemma4:e2b" if your GPU is slow

SCHEMES = json.loads((Path(__file__).parent / "schemes.json").read_text(encoding="utf-8"))

LANGS = {"en": "English", "hi": "हिन्दी", "mr": "मराठी"}
LANG_NAMES = {"en": "English", "hi": "Hindi", "mr": "Marathi"}

FIELD_ORDER = ["occupation", "state", "annual_family_income", "category", "owns_agri_land"]

QUESTIONS = {
    "occupation": {
        "en": "Are you a student, a farmer, or something else?",
        "hi": "आप विद्यार्थी हैं, किसान हैं, या कुछ और?",
        "mr": "तुम्ही विद्यार्थी आहात, शेतकरी आहात की इतर?",
    },
    "state": {
        "en": "Which state do you live in?",
        "hi": "आप किस राज्य में रहते हैं?",
        "mr": "तुम्ही कोणत्या राज्यात राहता?",
    },
    "annual_family_income": {
        "en": "What is your family's total yearly income (in rupees)?",
        "hi": "आपके परिवार की सालाना कुल आमदनी कितनी है (रुपये में)?",
        "mr": "तुमच्या कुटुंबाचे वार्षिक एकूण उत्पन्न किती आहे (रुपयांत)?",
    },
    "category": {
        "en": "Which category are you in (General / OBC / SC / ST)?",
        "hi": "आपकी श्रेणी क्या है (सामान्य / OBC / SC / ST)?",
        "mr": "तुमचा प्रवर्ग कोणता (खुला / OBC / SC / ST)?",
    },
    "owns_agri_land": {
        "en": "Does your family own agricultural land?",
        "hi": "क्या आपके परिवार के पास खेती की जमीन है?",
        "mr": "तुमच्या कुटुंबाकडे शेतजमीन आहे का?",
    },
}

DONE = {
    "en": "I have enough information. Open the schemes on the right to see eligibility, documents and a draft.",
    "hi": "मेरे पास पर्याप्त जानकारी है। दाईं ओर योजनाएँ खोलकर पात्रता, दस्तावेज़ और ड्राफ्ट देखें।",
    "mr": "माझ्याकडे पुरेशी माहिती आहे. उजवीकडील योजना उघडून पात्रता, कागदपत्रे आणि मसुदा पहा.",
}

PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        "occupation": {"type": "string", "enum": ["student", "farmer", "other"]},
        "state": {"type": "string"},
        "annual_family_income": {"type": "number"},
        "category": {"type": "string", "enum": ["general", "obc", "sc", "st"]},
        "owns_agri_land": {"type": "boolean"},
    },
}


def llm(messages, schema=None, temperature=0.0):
    """One call to the local Ollama server. Raises if Ollama is not running."""
    body = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": 8192},
    }
    if schema:
        body["format"] = schema
    r = requests.post(OLLAMA_URL, json=body, timeout=600)
    r.raise_for_status()
    return r.json()["message"]["content"]


def extract_profile(text, asked_field=None):
    """Turn free text into profile facts. Only facts the user stated."""
    hint = ""
    if asked_field:
        hint = (f"\nThe assistant just asked about: {asked_field}. "
                "A short answer such as a number or yes/no refers to that field.")
    system = (
        "Extract facts the user explicitly stated about themselves. Return JSON with only the keys that were stated: "
        "occupation (student, farmer or other), state (Indian state, lowercase English), "
        "annual_family_income (rupees per year as a number; convert lakh to rupees, 2.5 lakh = 250000), "
        "category (general, obc, sc or st), owns_agri_land (true or false). "
        "Omit anything not stated. Never guess." + hint
    )
    raw = llm([{"role": "system", "content": system}, {"role": "user", "content": text}], schema=PROFILE_SCHEMA)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    out = {}
    for key in FIELD_ORDER:
        val = data.get(key)
        if val in (None, "", []):
            continue
        if key in ("state", "occupation", "category"):
            val = str(val).strip().lower()
        if key == "annual_family_income":
            try:
                val = float(val)
            except (TypeError, ValueError):
                continue
        out[key] = val
    return out


OPS = {
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "in": lambda a, b: a in b,
}


def check(profile, scheme):
    """Deterministic validator: pass / fail / unknown for every rule. Never guesses."""
    results, missing = [], []
    for rule in scheme["rules"]:
        have = profile.get(rule["field"])
        if have is None:
            results.append({**rule, "status": "unknown", "have": None})
            missing.append(rule["field"])
            continue
        ok = OPS[rule["op"]](have, rule["value"])
        results.append({**rule, "status": "pass" if ok else "fail", "have": have})
    if any(r["status"] == "fail" for r in results):
        status = "not_eligible"
    elif missing:
        status = "need_info"
    else:
        status = "eligible"
    return {"scheme": scheme, "status": status, "results": results, "missing": sorted(set(missing))}


def evaluate_all(profile):
    return [check(profile, s) for s in SCHEMES]


def next_question_field(evals):
    """Ask about the field that unlocks the most still-possible schemes."""
    counts = {}
    for e in evals:
        if e["status"] == "need_info":
            for f in e["missing"]:
                counts[f] = counts.get(f, 0) + 1
    if not counts:
        return None
    return sorted(counts, key=lambda f: (-counts[f], FIELD_ORDER.index(f)))[0]


def build_plan(scheme, profile, have_docs, lang):
    """Steps + missing documents + a draft statement, grounded only in the scheme facts."""
    missing_docs = [d for d in scheme["documents"] if d not in have_docs]
    facts = {k: scheme.get(k) for k in ("name", "benefit", "deadline", "documents", "notes", "source_url")}
    system = (
        f"You are Scheme Saathi, a careful assistant. Write in {LANG_NAMES[lang]} using simple words. "
        "Use ONLY the SCHEME facts and the USER_PROFILE. Never invent amounts, dates, links or rules; "
        "if something is not in SCHEME, say 'check the official portal'. "
        "Output three short sections: 1) Steps to apply (numbered), 2) Documents still missing, "
        "3) A short draft statement the user can paste into the form, using placeholders like [NAME] for unknown details. "
        "End with the line: Source: <source_url>."
    )
    user = json.dumps(
        {"SCHEME": facts, "USER_PROFILE": profile,
         "DOCUMENTS_ALREADY_HAVE": have_docs, "DOCUMENTS_MISSING": missing_docs},
        ensure_ascii=False,
    )
    return llm([{"role": "system", "content": system}, {"role": "user", "content": user}], temperature=0.2)
