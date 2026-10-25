# Scheme Saathi

Local AI agent that finds government schemes a person may qualify for, checks the rules, lists missing documents and drafts the application. Runs fully offline on Gemma 4 E4B via Ollama. No API, no cloud, no cost.

## Run
1. Install Ollama, then: `ollama pull gemma4:e4b` (use `gemma4:e2b` and edit MODEL in engine.py if your GPU is slow)
2. `pip install -r requirements.txt`
3. `streamlit run app.py`

## How it works
- Planner/extractor: Gemma turns what the user says into profile facts (JSON schema output).
- Validator: plain Python checks every scheme rule. It returns pass / fail / unknown and never guesses. Unknown facts trigger a clarifying question.
- Action: Gemma writes the plan and draft using only the scheme facts. The user must confirm before downloading.
- Memory: the profile and ticked documents live in the session only.

## Before the demo (important)
`schemes.json` is a starter set marked "verified": false. Open each official source, correct every rule, amount and deadline, then set "verified": true. Add the state scheme your demo persona needs.
