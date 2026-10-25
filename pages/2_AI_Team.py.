"""Papoine AI Team — a multi-agent system (supervisor + specialist sub-agents).

Pattern: a supervisor ("Manager") agent plans the work and delegates sub-tasks to
specialist sub-agents, passing each one the findings of the agents before it.
  - Data Analyst     -> the existing ReAct agent with pandas tools (from app.py)
  - Marketing Writer -> writes Instagram posts, captions and email campaigns
  - Customer Care    -> drafts friendly messages and replies to customers
Author: Shreya Sandeep Jagtap — Programming for Data Science, RNU
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import SYSTEM_PROMPT, TOOL_LABELS, make_chart, parse_json, run_agent  # noqa: E402

# =====================================================================
# 1. THE TEAM
# =====================================================================

PAPOINE_INFO = """Papoine business info (sample information for this prototype):
- Flower subscription business in Riga, Latvia.
- Plans: One-off, Weekly, Bi-weekly, Monthly. Subscription orders get 10% off.
- Bouquets: Riga Rose Classic, Tulip Spring, Peony Blush, Sunflower Glow, Wild Meadow, Seasonal Surprise. Sizes: Small, Medium, Large.
- Delivery areas: Centrs, Āgenskalns, Teika, Purvciems, Jūrmala, Mežaparks, Imanta.
- Customers can pause or cancel a subscription before their next delivery.
- Sales channels: website, Instagram, referrals, gift vouchers.
- Brand voice: warm, elegant, local and a little playful."""

AGENTS = {
    "analyst": {"name": "Data Analyst", "icon": "📊",
                "job": "Answers questions with real numbers by running pandas tools on the order data."},
    "marketing": {"name": "Marketing Writer", "icon": "📣",
                  "job": "Writes Instagram posts, captions and email campaigns."},
    "care": {"name": "Customer Care", "icon": "💌",
             "job": "Drafts friendly replies and messages to customers."},
}

SUPERVISOR_PROMPT = f"""You are the Papoine Manager, the supervisor of a small AI team for Papoine.
{PAPOINE_INFO}

Your team:
- analyst: {AGENTS['analyst']['job']} Use it whenever the request needs numbers or facts from the order data.
- marketing: {AGENTS['marketing']['job']}
- care: {AGENTS['care']['job']}

Plan which team members to use, in order, and write each one a clear, specific task.
Rules:
- Use only the members that are needed (1 to 3), each at most once.
- Put analyst first when numbers are needed, so the others can build on its findings.
- Do not invent numbers yourself.

Reply with ONE JSON object and nothing else:
{{"goal":"<one sentence>","plan":[{{"agent":"analyst|marketing|care","task":"<specific instruction>"}}]}}"""

WRITER_RULES = f"""{PAPOINE_INFO}

Rules:
- Use only facts from the task, the teammates' findings and the business info above.
- Never invent numbers, prices, discounts, dates or policies that are not given. If something
  is unknown, write it as a clear placeholder like [discount %] or say the team will check.
- Write in Papoine's brand voice.

Reply with ONE JSON object and nothing else:
{{"title":"<short title>","content":"<the finished text, with line breaks>","notes":"<one sentence on why this works>"}}"""

MARKETING_PROMPT = "You are Papoine's Marketing Writer. You write Instagram posts, captions and email campaigns.\n" + WRITER_RULES
CARE_PROMPT = "You are Papoine's Customer Care agent. You write warm, clear messages and replies to customers.\n" + WRITER_RULES


# =====================================================================
# 2. MODEL ACCESS with automatic fallback between models
# =====================================================================

def make_caller(client, types, models, notify, counter):
    """Returns call(system, prompt, temperature) that retries busy models and
    switches to the next model when one is unavailable or out of quota."""
    state = {"i": 0}

    def call(system, prompt, temperature=0.3):
        while state["i"] < len(models):
            model = models[state["i"]]
            for attempt in range(3):
                try:
                    resp = client.models.generate_content(
                        model=model, contents=prompt,
                        config=types.GenerateContentConfig(system_instruction=system,
                                                           response_mime_type="application/json",
                                                           temperature=temperature))
                    counter["calls"] += 1
                    counter["model"] = model
                    return resp.text or ""
                except Exception as e:
                    msg = str(e)
                    if any(k in msg for k in ("404", "NOT_FOUND", "429", "RESOURCE_EXHAUSTED")):
                        break  # model retired or out of quota -> try the next model
                    if any(k in msg for k in ("503", "UNAVAILABLE")) and attempt < 2:
                        notify(f"{model} is busy, retrying in {3 * (attempt + 1)} seconds…")
                        time.sleep(3 * (attempt + 1))
                        continue
                    if any(k in msg for k in ("503", "UNAVAILABLE")):
                        break
                    raise
            state["i"] += 1
            if state["i"] < len(models):
                notify(f"Switching from {model} to {models[state['i']]}…")
        raise RuntimeError("All models are busy or out of today's free quota. "
                           "Try again later or add another model name in the sidebar.")

    return call


# =====================================================================
# 3. THE SUPERVISOR LOOP
# =====================================================================

def clean_plan(raw):
    plan, seen = [], set()
    for step in (raw or [])[:3]:
        agent = str(step.get("agent", "")).strip().lower()
        task = str(step.get("task", "")).strip()
        if agent in AGENTS and agent not in seen and task:
            plan.append({"agent": agent, "task": task})
            seen.add(agent)
    return plan


def run_team(request, df, call, on_event=lambda kind, data: None):
    """Supervisor plans -> each sub-agent works in turn, receiving earlier findings."""
    log = []

    def record(agent, action, detail=""):
        log.append({"time": datetime.now().strftime("%H:%M:%S"), "agent": agent,
                    "action": action, "detail": detail[:300]})

    record("Manager", "Received request", request)
    plan_reply = parse_json(call(SUPERVISOR_PROMPT, f"Owner's request: {request}", 0.2))
    plan = clean_plan(plan_reply.get("plan"))
    if not plan:
        raise RuntimeError("The Manager could not make a plan. Try rephrasing the request.")
    goal = plan_reply.get("goal", "")
    record("Manager", "Made a plan", " → ".join(AGENTS[p["agent"]]["name"] for p in plan))
    on_event("plan", {"goal": goal, "plan": plan})

    findings, results = [], []
    for step in plan:
        agent, task = step["agent"], step["task"]
        name = AGENTS[agent]["name"]
        record("Manager", f"Delegated to {name}", task)
        on_event("start", {"agent": agent, "task": task})
        context = "\n".join(findings) or "None yet."

        if agent == "analyst":
            final, tool_steps = run_agent(task, df, [], lambda p: call(SYSTEM_PROMPT, p, 0.2),
                                          lambda s: on_event("tool", s))
            for s in tool_steps:
                record(name, TOOL_LABELS.get(s["tool"], s["tool"]),
                       json.dumps(s["args"], ensure_ascii=False))
            out = {"agent": agent, "task": task, "answer": final.get("answer", ""),
                   "recommendation": final.get("recommendation", ""),
                   "final": final, "steps": tool_steps}
            findings.append(f"Data Analyst found: {out['answer']} "
                            f"Recommendation: {out['recommendation']}")
        else:
            system = MARKETING_PROMPT if agent == "marketing" else CARE_PROMPT
            reply = parse_json(call(system, f"Task from the Manager: {task}\n\n"
                                            f"Findings from teammates:\n{context}", 0.6))
            out = {"agent": agent, "task": task, "title": reply.get("title", ""),
                   "content": reply.get("content", ""), "notes": reply.get("notes", "")}
            findings.append(f"{name} wrote '{out['title']}': {out['content'][:400]}")
        record(name, "Finished task", out.get("answer") or out.get("title", ""))
        results.append(out)
        on_event("done", out)

    record("Manager", "Delivered results to the owner")
    return {"request": request, "goal": goal, "plan": plan, "results": results, "log": log}


# =====================================================================
# 4. WEB PAGE (Streamlit)
# =====================================================================

EXAMPLES = [
    "Plan a Mother's Day campaign around our best-selling bouquets",
    "Why do Weekly customers cancel, and what should we send them?",
    "Find our busiest delivery area and write an Instagram post for it",
    "Draft a reply to a customer asking if we deliver to Jūrmala on Sundays",
]


def main():
    import streamlit as st
    from google import genai
    from google.genai import types

    st.set_page_config(page_title="Papoine AI Team", page_icon="🌸")

    def secret(name, default=""):
        try:
            return st.secrets.get(name, default)
        except Exception:
            return default

    st.title("🌸 Papoine AI Team")
    st.write("A **Manager** agent plans the work and delegates tasks to three specialist "
             "agents. Each agent builds on the findings of the one before it.")

    cols = st.columns(4)
    team = [("🧭", "Manager", "Plans and delegates")] + \
           [(a["icon"], a["name"], a["job"].split(" by ")[0]) for a in AGENTS.values()]
    for col, (icon, name, job) in zip(cols, team):
        with col.container(border=True):
            st.markdown(f"### {icon}\n**{name}**")
            st.caption(job)

    with st.sidebar:
        st.header("Settings")
        api_key = secret("GEMINI_API_KEY")
        if not api_key:
            api_key = st.text_input("Gemini API key", type="password")
        default_models = secret("GEMINI_MODELS", "") or ", ".join(
            dict.fromkeys([secret("GEMINI_MODEL", "gemini-3.8-flash"), "gemini-2.5-flash-lite"]))
        models_text = st.text_input("Models (tried in order)", value=default_models,
                                    help="If one model is busy or out of quota, the next is used.")
        st.caption("Each request uses about 4–6 AI calls.")

    csvs = sorted(Path(__file__).resolve().parent.parent.glob("*.csv"))
    if not csvs:
        st.error("No CSV file found in the repository.")
        return
    df = pd.read_csv(csvs[0])

    st.subheader("Give the team a task")
    clicked = None
    ex_cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if ex_cols[i % 2].button(ex, use_container_width=True):
            clicked = ex
    typed = st.chat_input("Describe a task for the team")
    request = clicked or typed

    if request:
        if not api_key:
            st.error("Add a Gemini API key in the sidebar or the app's Secrets first.")
            return
        models = [m.strip() for m in models_text.split(",") if m.strip()]
        counter = {"calls": 0, "model": ""}
        st.markdown(f"#### 📝 {request}")
        with st.status("The team is working…", expanded=True) as status:
            def on_event(kind, data):
                if kind == "plan":
                    st.markdown(f"🧭 **Manager:** {data['goal']}")
                    for n, p in enumerate(data["plan"], 1):
                        st.markdown(f"{n}. {AGENTS[p['agent']]['icon']} "
                                    f"**{AGENTS[p['agent']]['name']}**: {p['task']}")
                elif kind == "start":
                    st.markdown(f"{AGENTS[data['agent']]['icon']} "
                                f"**{AGENTS[data['agent']]['name']}** is working…")
                elif kind == "tool" and "tool" in data:
                    label = TOOL_LABELS.get(data["tool"], data["tool"])
                    st.caption(f"🌿 {label}" if data["ok"] else f"🥀 Error, retrying: {label}")
            try:
                call = make_caller(genai.Client(api_key=api_key), types, models,
                                   st.caption, counter)
                result = run_team(request, df, call, on_event)
                result["calls"], result["model"] = counter["calls"], counter["model"]
                st.session_state["team_result"] = result
                status.update(label=f"Done: {len(result['results'])} agents, "
                                    f"{counter['calls']} AI calls", state="complete",
                              expanded=False)
            except Exception as e:
                status.update(label="Something went wrong", state="error")
                st.error(str(e))
                return

    result = st.session_state.get("team_result")
    if not result:
        return

    if not request:
        st.markdown(f"#### 📝 {result['request']}")
    st.info(f"🧭 **Manager's goal:** {result['goal']}")
    for out in result["results"]:
        meta = AGENTS[out["agent"]]
        with st.container(border=True):
            st.markdown(f"### {meta['icon']} {meta['name']}")
            st.caption(f"Task: {out['task']}")
            if out["agent"] == "analyst":
                st.write(out["answer"])
                fig = make_chart(out["final"], out["steps"])
                if fig:
                    st.pyplot(fig)
                if out["recommendation"]:
                    st.success(f"**Recommendation:** {out['recommendation']}")
            else:
                st.markdown(f"**{out['title']}**")
                st.text(out["content"])
                if out["notes"]:
                    st.caption(f"💡 {out['notes']}")

    with st.expander("📜 Team activity log"):
        log_df = pd.DataFrame(result["log"])
        st.dataframe(log_df, use_container_width=True, hide_index=True)
        st.download_button("Download log (CSV)", log_df.to_csv(index=False),
                           file_name="papoine_team_log.csv", mime="text/csv")
        st.caption(f"{result.get('calls', 0)} AI calls · model: {result.get('model', '')}")
    st.caption("Sample data and sample business info. Check content before publishing.")


if __name__ == "__main__":
    main()
