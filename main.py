import io
import re
import json
import base64
import difflib
import traceback
import contextlib
import xml.etree.ElementTree as ET
from urllib.parse import urlencode
import time
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import ssl
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager
import urllib3

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

# ========== ENV / CONFIG ==========
GEMINI_URL_DEFAULT = os.environ.get(
    "GEMINI_URL",
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"
)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
CACHE_TTL_MINUTES = int(os.environ.get("CACHE_TTL_MINUTES", "30"))  # 🆕 default 30 min

# ==================================
app = FastAPI(title="SAP OData ChatBot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# -----------------------------
# SSL Adapter (Legacy SAP Fix)
# -----------------------------
class SSLAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.options |= 0x4  # SSL_OP_LEGACY_SERVER_CONNECT
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        kwargs["ssl_context"] = ctx
        return super(SSLAdapter, self).init_poolmanager(*args, **kwargs)

session = requests.Session()
session.mount("https://", SSLAdapter())

# ================================================================
# 🆕 CACHING LAYER FOR ODATA
# ================================================================
_odata_cache = {}  # {(url, username, password): (timestamp, dataframe)}

def _cache_key(url, username, password):
    return f"{url}::{username or ''}::{password or ''}"

def _is_cache_valid(entry):
    ts, _ = entry
    return (datetime.now() - ts) < timedelta(minutes=CACHE_TTL_MINUTES)

def clear_odata_cache():
    _odata_cache.clear()

def get_cached_odata(url, username, password):
    key = _cache_key(url, username, password)
    entry = _odata_cache.get(key)
    if entry and _is_cache_valid(entry):
        return entry[1]
    return None

def set_cached_odata(url, username, password, df):
    key = _cache_key(url, username, password)
    _odata_cache[key] = (datetime.now(), df.copy())
# ================================================================


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
        return df[df[col].fillna("").str.contains(closest[0], case=False, na=False)]
    else:
        return df[df[col].fillna("").str.contains(str(value), case=False, na=False)]

# ----- Flexible OData Parser -----
def parse_odata_any(resp):
    ctype = resp.headers.get("Content-Type", "").lower()
    if "json" in ctype:
        js = resp.json()
        if "d" in js:
            results = js["d"].get("results", [])
        else:
            results = js
        df = pd.DataFrame(results)
    else:
        df = parse_odata_xml(resp.text)

    if "__metadata" in df.columns:
        df["metadata_id"] = df["__metadata"].apply(
            lambda x: x.get("id") if isinstance(x, dict) else None
        )
        df.drop(columns=["__metadata"], inplace=True)
    return df

def parse_odata_xml(xml_text):
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "m": "http://schemas.microsoft.com/ado/2007/08/dataservices/metadata",
        "d": "http://schemas.microsoft.com/ado/2007/08/dataservices",
    }
    root = ET.fromstring(xml_text)
    entries = root.findall(".//atom:entry", ns)
    data = []
    for entry in entries:
        props = entry.find(".//m:properties", ns)
        if props is None:
            continue
        record = {}
        for child in props:
            tag = re.sub(r"^{.*}", "", child.tag)
            record[tag] = child.text
        data.append(record)
    if not data:
        raise ValueError("No valid data entries found in OData response.")
    return pd.DataFrame(data)

def call_gemini_json(url, key, prompt, timeout=40):
    headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    r = session.post(url, headers=headers, json=payload, timeout=timeout, verify=False)
    try:
        return r.json()
    except:
        return {"text": r.text}

def safe_exec(expr, df):
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
                for k, v in list(local_env.items())[::-1]:
                    if isinstance(v, (pd.DataFrame, pd.Series, plt.Figure)):
                        result = v
                        break
                if result is None:
                    result = "OK"
        except Exception:
            raise
    if not isinstance(result, plt.Figure):
        fig = plt.gcf()
        if fig.get_axes():
            result = fig
    return result

# -----------------------------
# Pydantic Models
# -----------------------------
class QueryBody(BaseModel):
    odata_url: str
    username: str | None = None
    password: str | None = None
    question: str
    timeout_sec: int | None = 30
    gemini_url: str | None = None
    gemini_key: str | None = None

# -----------------------------
# Health
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok", "cache_entries": len(_odata_cache)}

# 🆕 Endpoint to clear cache manually
@app.get("/clear-cache")
def clear_cache():
    clear_odata_cache()
    return {"status": "cache cleared"}

# -----------------------------
# Core /query endpoint
# -----------------------------
@app.post("/query")
def query(body: QueryBody):
    if not body.odata_url or not body.question:
        raise HTTPException(status_code=400, detail="odata_url and question are required")

    # 🆕 Try cache first
    df = get_cached_odata(body.odata_url, body.username, body.password)
    if df is None:
        # --- Fetch OData if not cached ---
        try:
            auth = (body.username, body.password) if (body.username and body.password) else None
            if "$format" not in body.odata_url:
                sep = "&" if "?" in body.odata_url else "?"
                body.odata_url += f"{sep}$format=json"
            resp = session.get(
                body.odata_url,
                auth=auth,
                headers={"Accept": "application/json"},
                timeout=body.timeout_sec or 30,
                verify=False
            )
            if resp.status_code != 200:
                raise HTTPException(status_code=502, detail=f"OData fetch failed: {resp.status_code} {resp.text[:500]}")
            df = parse_odata_any(resp)
            set_cached_odata(body.odata_url, body.username, body.password, df)  # 🆕 store in cache
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch/parse OData: {e}")

    # --- Normalize DF ---
    orig_cols = df.columns.tolist()
    norm_map = {c: normalize_col(c) for c in orig_cols}
    df.columns = [norm_map[c] for c in orig_cols]
    fuzzy_map = fuzzy_column_map(df.columns)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")

    # --- Gemini + Execution ---
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
Column aliases: {aliases}
Return ONLY JSON with "explain" and "expr".
Rules:
1. Use closest matching column names
2. String comparisons are case-insensitive and fuzzy (handled automatically)
3. Numeric operations safe
4. Never hallucinate columns/values
5. No loops/imports/prints
6. Never use print() or display()
7. Always RETURN the final result (DataFrame, Series, numeric, dict, or plt.Figure)
8. Always handle NaN values safely:
   - For string filters: use str.contains(..., na=False)
   - For numeric operations: safely handle empty sequences
9. Always assume the expression will be executed inside a safe environment that automatically displays the result.
10. Do not print anything — simply return the result or expression output.
11. Prefer one-liners that evaluate to a result directly (no variables unless necessary).
12. When multiple values are logically related (like total count + list), return a dictionary.
13. If visualization is the best answer, generate a matplotlib figure object (plt.figure()) and plot accordingly.
14. When grouping numeric columns, use aggregation (sum, mean, count).
15. Do not answer general knowledge questions (outside dataset); reply with "only ask questions related to data please".
"""
    gemini_url = body.gemini_url or GEMINI_URL_DEFAULT
    gemini_key = body.gemini_key or GEMINI_API_KEY
    if not gemini_key:
        raise HTTPException(status_code=500, detail="Missing GEMINI_API_KEY.")

    resp = call_gemini_json(
        gemini_url, gemini_key, PROMPT_PANDAS_TRANSLATE + "\nQuestion: " + body.question, body.timeout_sec or 30
    )
    js = extract_json_from_response(resp)
    if not js or "expr" not in js:
        raise HTTPException(status_code=400, detail="Gemini didn't return expr.")
    expr = js["expr"]
    explain = js.get("explain", "")

    try:
        result_obj = safe_exec(expr, df)
    except Exception:
        raise HTTPException(status_code=500, detail=f"Error executing expr:\n{traceback.format_exc()}")

    # Serialize results
    result_table = result_series = result_chart_base64 = None
    if isinstance(result_obj, pd.DataFrame):
        result_table = {"columns": list(result_obj.columns), "rows": result_obj.fillna("").astype(str).values.tolist()}
    elif isinstance(result_obj, pd.Series):
        result_series = {
            "name": str(result_obj.name),
            "index": [str(x) for x in result_obj.index.tolist()],
            "values": [str(x) for x in result_obj.fillna("").astype(str).tolist()],
        }
    elif isinstance(result_obj, plt.Figure):
        buf = io.BytesIO()
        result_obj.savefig(buf, format="png", bbox_inches="tight")
        plt.close(result_obj)
        result_chart_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    PROMPT_ENGLISH = f"Question: {body.question}\nResult: {repr(result_obj)[:2000]}\nExplain in plain English."
    resp2 = call_gemini_json(gemini_url, gemini_key, PROMPT_ENGLISH, body.timeout_sec or 30)
    try:
        answer_text = resp2["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        answer_text = str(resp2)

    return {
        "explain": explain,
        "expr": expr,
        "answer_text": answer_text,
        "cached": df is not None,
        "result_table": result_table,
        "result_series": result_series,
        "result_chart_base64": result_chart_base64,
    }

# ---- Local dev runner ----
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
