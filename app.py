import requests
import streamlit as st

from engine import (DONE, LANGS, QUESTIONS, build_plan, evaluate_all,
                    extract_profile, next_question_field)

st.set_page_config(page_title="Scheme Saathi", layout="wide")
S = st.session_state
S.setdefault("profile", {})
S.setdefault("chat", [])
S.setdefault("trace", [])
S.setdefault("asked", None)

lang = st.sidebar.selectbox("Language", list(LANGS), format_func=lambda k: LANGS[k])
st.sidebar.subheader("What I know (this session only)")
st.sidebar.json(S.profile)
st.sidebar.caption("Nothing is saved or sent anywhere. Everything runs on this laptop.")
if st.sidebar.button("Reset session"):
    S.clear()
    st.rerun()

st.title("Scheme Saathi")
st.caption("A local AI agent that finds government schemes you may qualify for, checks the rules, and prepares your application.")

msg = st.chat_input("Tell me about yourself and what you need...")
if msg:
    S.chat.append(("user", msg))
    try:
        with st.spinner("Thinking locally..."):
            S.trace.append("Planner: reading your message and updating your profile")
            found = extract_profile(msg, S.asked)
            S.profile.update(found)
            S.trace.append(f"Extractor: found {found if found else 'nothing new'}")
            evals = evaluate_all(S.profile)
            field = next_question_field(evals)
        if field:
            S.asked = field
            S.trace.append(f"Validator: some schemes still need '{field}' - asking you")
            S.chat.append(("assistant", QUESTIONS[field][lang]))
        else:
            S.asked = None
            S.trace.append("Validator: all possible schemes checked")
            S.chat.append(("assistant", DONE[lang]))
    except requests.exceptions.RequestException:
        S.chat.pop()
        st.error("Cannot reach Ollama. Start it, then run: ollama pull gemma4:e4b")

left, right = st.columns([1, 1.3])

with left:
    for role, text in S.chat:
        st.chat_message(role).write(text)
    with st.expander("Agent trace (what the agent did)"):
        for i, step in enumerate(S.trace, 1):
            st.write(f"{i}. {step}")

ICON = {"pass": "✅", "fail": "❌", "unknown": "❓"}
BADGE = {"eligible": "✅ Eligible", "need_info": "❓ Need more info", "not_eligible": "❌ Not eligible"}

with right:
    st.subheader("Schemes")
    if not S.profile:
        st.info("Tell me about yourself on the left to see matching schemes.")
    for e in evaluate_all(S.profile) if S.profile else []:
        sc = e["scheme"]
        with st.expander(f"{sc['name']} - {BADGE[e['status']]}", expanded=(e["status"] == "eligible")):
            if not sc.get("verified"):
                st.warning("Facts not yet verified against the official source. Check the portal before relying on them.")
            for r in e["results"]:
                have = "not told yet" if r["have"] is None else r["have"]
                st.write(f"{ICON[r['status']]} {r['label']} (you: {have})")
            if e["status"] == "not_eligible":
                continue
            st.write(f"**Deadline:** {sc['deadline']}")
            st.write(f"**Notes:** {sc['notes']}")
            st.markdown(f"[Official source]({sc['source_url']})")
            st.write("**Tick the documents you already have:**")
            have_docs = [d for d in sc["documents"] if st.checkbox(d, key=f"{sc['id']}:{d}")]
            missing = [d for d in sc["documents"] if d not in have_docs]
            st.write(f"Still missing: {', '.join(missing) if missing else 'nothing'}")
            if e["status"] == "eligible" and st.button("Generate plan and draft", key=f"plan:{sc['id']}"):
                try:
                    with st.spinner("Writing your plan locally..."):
                        S[f"plan_{sc['id']}"] = build_plan(sc, S.profile, have_docs, lang)
                except requests.exceptions.RequestException:
                    st.error("Cannot reach Ollama.")
            plan = S.get(f"plan_{sc['id']}")
            if plan:
                st.markdown(plan)
                ok = st.checkbox("I have read this draft and confirm it is accurate", key=f"ok:{sc['id']}")
                st.download_button("Download draft", plan, file_name=f"{sc['id']}_draft.txt", disabled=not ok)
