import io
import re
import json
import base64
import difflib
import traceback
import contextlib
import xml.etree.ElementTree as ET
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ====== SECURITY: keep Gemini key on server via env var ======
import os
GEMINI_URL_DEFAULT = os.environ.get(
    "GEMINI_URL",
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"
)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")  # set in Render dashboard

# ====== FastAPI app + CORS (allow your Fiori host origin) ======
app = FastAPI(title="SAP OData ChatBot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your FLP domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Utility Functions
# -----------------------------
def normalize_col(c):
    return re.sub(r"[^0-9a-z_]", "_", c.strip().lower())

def fuzzy_column_map(columns):
    mapping = {}
    for c in columns:
        mapping[c.lower()] = c
        for token in c.lower().split("_"):
            mapping[token] = c
    return mapping

def extract_json_from_response(resp):
    try:
        if isinstance(resp, dict):
            if "candidates" in resp:
                text = resp["candidates"][0]["content"]["parts"][0]["text"]
            elif "text" in resp:
                text = resp["text"]
            else:
                text = json.dumps(resp)
        else:
            text = str(resp)
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except:
        pass
    return None

def validate_expr(expr):
    forbidden = ["subprocess", "os.", "sys.", "open(", "eval(", "exec(", "__import__", "input("]
    if any(f in expr for f in forbidden):
        raise ValueError("Unsafe code detected.")
    return True

def fuzzy_filter(df, col, value):
    col_values = df[col].dropna().astype(str).unique()
    closest = difflib.get_close_matches(str(value), col_values, n=1, cutoff=0.6)
    if closest:
        return df[df[col].fillna('').str.contains(closest[0], case=False, na=False)]
    else:
        return df[df[col].fillna('').str.contains(str(value), case=False, na=False)]

def parse_odata_xml(xml_text):
    ns = {
        'atom': 'http://www.w3.org/2005/Atom',
        'm': 'http://schemas.microsoft.com/ado/2007/08/dataservices/metadata',
        'd': 'http://schemas.microsoft.com/ado/2007/08/dataservices'
    }
    root = ET.fromstring(xml_text)
    entries = root.findall('.//atom:entry', ns)
    data = []

    for entry in entries:
        props = entry.find('.//m:properties', ns)
        if props is None:
            continue
        record = {}
        for child in props:
            tag = re.sub(r'^{.*}', '', child.tag)
            record[tag] = child.text
        data.append(record)

    if not data:
        raise ValueError("No valid data entries found in OData response.")
    return pd.DataFrame(data)

def call_gemini_json(url, key, prompt, timeout=40):
    headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    try:
        return r.json()
    except:
        return {"text": r.text}

def safe_exec(expr, df):
    """
    Headless safe execution. If a matplotlib figure is produced, return it.
    """
    local_env = {"df": df, "pd": pd, "np": np, "plt": plt, "re": re, "fuzzy_filter": fuzzy_filter}

    validate_expr(expr)

    result = None
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        try:
            try:
                result = eval(expr, {}, local_env)
            except:
                exec(expr, {}, local_env)
                # Try to find a meaningful object
                for k, v in list(local_env.items())[::-1]:
                    if isinstance(v, (pd.DataFrame, pd.Series, plt.Figure)):
                        result = v
                        break
                if result is None:
                    result = "OK"
        except Exception:
            raise

    # If nothing returned but a plot exists
    if not isinstance(result, plt.Figure):
        fig = plt.gcf()
        if fig.get_axes():
            result = fig

    return result

# -----------------------------
# Pydantic models
# -----------------------------
class QueryBody(BaseModel):
    odata_url: str
    username: str | None = None
    password: str | None = None
    question: str
    timeout_sec: int | None = 30
    # optional override for gemini (not recommended from UI):
    gemini_url: str | None = None
    gemini_key: str | None = None

# -----------------------------
# Health
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -----------------------------
# Core endpoint
# -----------------------------
@app.post("/query")
def query(body: QueryBody):
    if not body.odata_url or not body.question:
        raise HTTPException(status_code=400, detail="odata_url and question are required")

    # 1) Fetch OData
    try:
        auth = (body.username, body.password) if (body.username and body.password) else None
        resp = requests.get(body.odata_url, auth=auth, headers={"Accept": "application/atom+xml"}, timeout=body.timeout_sec or 30)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"OData fetch failed: {resp.status_code} {resp.text[:500]}")
        df = parse_odata_xml(resp.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch/parse OData: {e}")

    # 2) Prepare DF
    orig_cols = df.columns.tolist()
    norm_map = {c: normalize_col(c) for c in orig_cols}
    df.columns = [norm_map[c] for c in orig_cols]
    fuzzy_map = fuzzy_column_map(df.columns)

    # numeric conversion
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='ignore')

    # 3) Build schema + prompt
    schema = []
    for c in df.columns:
        sample = str(df[c].dropna().iloc[0]) if df[c].dropna().shape[0] > 0 else ""
        schema.append({"name": c, "dtype": str(df[c].dtype), "sample": sample})
    schema_json = json.dumps(schema, indent=2)
    aliases = ", ".join(list(fuzzy_map.keys()))

    PROMPT_PANDAS_TRANSLATE = f"""
You are an expert data reasoning assistant.
DataFrame 'df' schema:
{schema_json}

Column aliases (fuzzy matches allowed): {aliases}

Return ONLY JSON:
  "explain": brief description
  "expr": valid pandas one-liner

Rules:
1. Use closest matching column names
2. String comparisons are case-insensitive and fuzzy (handled automatically)
3. Numeric operations safe
4. Never hallucinate columns/values
5. No loops/imports/prints
6. Always valid Python one-liner
7. When grouping numeric columns, use aggregation (sum, mean, count)
8. When a name came up keep in mind that its not full name only part of name
9. Do not answer general knowledge questions (outside dataset); reply with "only ask questions related to data please".
10. Always handle NaN values safely:
   - For string filters: use str.contains(..., na=False)
   - For numeric operations: safely handle empty sequences
"""
    gemini_url = body.gemini_url or GEMINI_URL_DEFAULT
    gemini_key = body.gemini_key or GEMINI_API_KEY
    if not gemini_key:
        raise HTTPException(status_code=500, detail="Server missing GEMINI_API_KEY (set it in environment).")

    # 4) Ask Gemini for pandas expression
    resp = call_gemini_json(gemini_url, gemini_key, PROMPT_PANDAS_TRANSLATE + "\nQuestion: " + body.question, body.timeout_sec or 30)
    js = extract_json_from_response(resp)

    if not js or "expr" not in js:
        msg = ""
        try:
            msg = resp["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            pass
        raise HTTPException(status_code=400, detail=f"Gemini didn't return expr. Raw: {msg or str(resp)[:1000]}")

    expr = js["expr"]
    explain = js.get("explain", "")

    # 5) Execute
    try:
        result_obj = safe_exec(expr, df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing expr:\n{traceback.format_exc()}")

    # 6) Package response
    result_table = None
    result_series = None
    result_chart_base64 = None

    if isinstance(result_obj, pd.DataFrame):
        result_table = {
            "columns": list(result_obj.columns),
            "rows": result_obj.fillna("").astype(str).values.tolist()
        }
    elif isinstance(result_obj, pd.Series):
        result_series = {
            "name": str(result_obj.name),
            "index": [str(x) for x in result_obj.index.tolist()],
            "values": [str(x) for x in result_obj.fillna("").astype(str).tolist()]
        }
    elif isinstance(result_obj, plt.Figure):
        buf = io.BytesIO()
        result_obj.savefig(buf, format="png", bbox_inches="tight")
        plt.close(result_obj)
        result_chart_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    else:
        # check active figure
        fig = plt.gcf()
        if fig.get_axes():
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            result_chart_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # 7) Natural language answer
    PROMPT_ENGLISH = f"""
You are a helpful assistant. 
Question: {body.question}
The result is: {repr(result_obj)[:4000]}
Give the answer with explanation, in natural English.
"""
    resp2 = call_gemini_json(gemini_url, gemini_key, PROMPT_ENGLISH, body.timeout_sec or 30)
    try:
        answer_text = resp2["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        answer_text = str(resp2)

    return {
        "explain": explain,
        "expr": expr,
        "answer_text": answer_text,
        "result_table": result_table,
        "result_series": result_series,
        "result_chart_base64": result_chart_base64
    }

# ---- Local dev runner (Render runs via start command) ----
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
