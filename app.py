"""Papoine Insights Agent — an AI data-analysis agent (Streamlit + Gemini).

The agent follows the ReAct loop: Reason -> Act (call a pandas tool) -> Observe -> repeat,
then writes a plain-English answer, a chart and a business recommendation.
Author: Shreya Sandeep Jagtap — Programming for Data Science, RNU
"""
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# =====================================================================
# 1. TOOLS (plain pandas functions the agent can call)
# =====================================================================

OPS = ["==", "!=", ">", "<", ">=", "<=", "contains", "in"]


def check_column(df, col, label):
    if col not in df.columns:
        raise ValueError(f'Unknown {label} "{col}". Columns are: {", ".join(df.columns)}')


def describe_data(df, args=None):
    """Summary of the dataset: columns, types and example values."""
    cols = []
    for c in df.columns:
        s = df[c]
        missing = int(s.isna().sum())
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().sum() >= 0.9 * s.notna().sum() and s.notna().any():
            cols.append({"name": c, "type": "number", "min": float(num.min()),
                         "max": float(num.max()), "mean": round(float(num.mean()), 2),
                         "missing": missing})
        else:
            counts = s.dropna().astype(str).value_counts()
            info = {"name": c, "type": "text", "distinct": int(len(counts)), "missing": missing}
            if len(counts) <= 15:
                info["values"] = sorted(counts.index.tolist())
            else:
                ordered = sorted(counts.index.tolist())
                info.update(first=ordered[0], last=ordered[-1], examples=counts.index[:5].tolist())
            cols.append(info)
    return {"row_count": int(len(df)), "columns": cols}


def _mask(series, op, value):
    if op not in OPS:
        raise ValueError(f'Unknown filter op "{op}". Use one of: {", ".join(OPS)}')
    text = series.astype(str).str.lower()
    if op == "in":
        if not isinstance(value, list):
            raise ValueError('The value for "in" must be a list.')
        return text.isin([str(v).lower() for v in value])
    if op == "contains":
        return text.str.contains(str(value).lower(), regex=False)
    num = pd.to_numeric(series, errors="coerce")
    try:
        target = float(value)
        use_num = num.notna().any()
    except (TypeError, ValueError):
        use_num = False
    left, right = (num, target) if use_num else (text, str(value).lower())
    return {"==": lambda: left == right, "!=": lambda: left != right,
            ">": lambda: left > right, "<": lambda: left < right,
            ">=": lambda: left >= right, "<=": lambda: left <= right}[op]()


def apply_filters(df, filters):
    if not filters:
        return df
    if not isinstance(filters, list):
        raise ValueError('"filters" must be a list of {"column","op","value"} objects.')
    mask = pd.Series(True, index=df.index)
    for f in filters:
        check_column(df, f.get("column"), "filter column")
        mask &= _mask(df[f["column"]], f.get("op", "=="), f.get("value"))
    return df[mask]


def _sort_limit(s, sort, limit):
    if sort == "key":
        s = s.sort_index()
    else:
        s = s.sort_values(ascending=(sort == "asc"), na_position="last")
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 15
    return s.head(max(1, min(limit, 40)))


def _rows(s):
    return [{"group": str(k), "value": None if pd.isna(v) else round(float(v), 2)}
            for k, v in s.items()]


def _keys(df, group_by):
    check_column(df, group_by, "group_by column")
    col = df[group_by]
    if pd.api.types.is_float_dtype(col) and (col.dropna() % 1 == 0).all():
        col = col.astype("Int64")  # show 5 instead of 5.0
    return col.astype("string").fillna("(blank)")


def aggregate(df, args):
    """count / sum / mean / median of a column, optionally grouped and filtered."""
    metric, group_by, field = args.get("metric") or "count", args.get("group_by"), args.get("field")
    if metric not in ("count", "sum", "mean", "median"):
        raise ValueError(f'Unknown metric "{metric}". Use count, sum, mean or median.')
    if metric != "count":
        if not field:
            raise ValueError(f'Metric "{metric}" needs a numeric "field".')
        check_column(df, field, "field")
    data = apply_filters(df, args.get("filters"))
    if group_by:
        keys = _keys(data, group_by)
        if metric == "count":
            s = keys.value_counts()
        else:
            s = pd.to_numeric(data[field], errors="coerce").groupby(keys).agg(metric)
    else:
        value = len(data) if metric == "count" else getattr(
            pd.to_numeric(data[field], errors="coerce"), metric)()
        s = pd.Series({"All rows": value})
    s = _sort_limit(s, args.get("sort"), args.get("limit"))
    return {"matching_rows": int(len(data)), "rows": _rows(s)}


def percentage(df, args):
    """% of base rows that meet a condition, optionally grouped."""
    cond = args.get("condition")
    if not isinstance(cond, list) or not cond:
        raise ValueError('"condition" must be a non-empty list of filters.')
    base = apply_filters(df, args.get("filters"))
    flag = pd.Series(base.index.isin(apply_filters(base, cond).index), index=base.index)
    if args.get("group_by"):
        s = flag.groupby(_keys(base, args["group_by"])).mean() * 100
    else:
        s = pd.Series({"All rows": flag.mean() * 100})
    s = _sort_limit(s, args.get("sort"), args.get("limit"))
    return {"base_rows": int(len(base)), "unit": "percent", "rows": _rows(s)}


TOOLS = {"describe_data": describe_data, "aggregate": aggregate, "percentage": percentage}
TOOL_LABELS = {"describe_data": "Checked the data structure",
               "aggregate": "Calculated totals", "percentage": "Calculated a percentage"}

# =====================================================================
# 2. THE AGENT (instructions + ReAct loop)
# =====================================================================

SYSTEM_PROMPT = """You are the Papoine Insights Agent, a data analyst for Papoine, a flower subscription business in Riga, Latvia. You answer business questions ONLY by calling tools on the loaded dataset. Never invent numbers: every number in your answer must come from a tool result.

Reply with ONE JSON object and nothing else.

To use a tool:
{"type":"tool","thought":"<one short sentence on what you need and why>","tool":"<name>","args":{...}}

To finish:
{"type":"final","answer":"<2-4 plain-English sentences with the key numbers>","recommendation":"<one practical business action, or empty string>","chart":{"step":<number of the tool step to chart, or null>,"kind":"bar" or "line","title":"<short title>"}}

Tools:
1. describe_data - args {}. Returns columns, types and example values.
2. aggregate - args {"group_by": column or null, "metric": "count"|"sum"|"mean"|"median", "field": numeric column (required unless metric is count), "filters": [{"column","op","value"}], "sort": "desc"|"asc"|"key", "limit": number}. Use sort "key" for time columns such as month.
3. percentage - args {"condition": [filters defining the rows to count], "filters": [filters for the base rows], "group_by": column or null, "sort", "limit"}. Returns the % of base rows that meet the condition.
Filter ops: "==", "!=", ">", "<", ">=", "<=", "contains", "in" (value must be a list for "in").

Rules:
- Use exact column names and values from the dataset summary.
- If a tool returns an error, read it, fix the arguments and try again.
- Use at most 4 tool calls.
- If the data cannot answer the question, say so honestly.
- Use a "line" chart for trends over time and "bar" for comparing categories. Money is in EUR."""

MAX_TOOL_CALLS = 4


def parse_json(text):
    clean = text.replace("```json", "").replace("```", "").strip()
    start, end = clean.find("{"), clean.rfind("}")
    return json.loads(clean[start:end + 1])


def build_prompt(question, schema, history, steps, force_final):
    steps_text = "\n\n".join(
        f"Step {i}: {s['tool']} {json.dumps(s['args'], ensure_ascii=False)}\n"
        f"Result: {json.dumps(s['result'], ensure_ascii=False)[:2500]}"
        for i, s in enumerate(steps, 1)) or "None yet."
    hist = "\n".join(f"Q: {h['q']}\nA: {h['answer']}" for h in history[-3:]) or "None."
    last = ("You have no tool calls left. Reply with the final JSON now."
            if force_final else "Reply with your next JSON object.")
    return (f"Dataset summary:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
            f"Earlier questions in this session (for follow-ups):\n{hist}\n\n"
            f"Current question: {question}\n\nTool steps so far:\n{steps_text}\n\n{last}")


def run_agent​​​​​​​​​​​​​​​​
