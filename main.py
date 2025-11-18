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
import threading

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
CACHE_TTL_MINUTES = int(os.environ.get("CACHE_TTL_MINUTES", "30"))  # default 30 min
DEFAULT_TIMEOUT = int(os.environ.get("DEFAULT_TIMEOUT_SEC", "30"))

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

# global session only used for Gemini calls (safe-ish) - we still use per-request sessions for OData fetch
session = requests.Session()
session.mount("https://", SSLAdapter())

# ================================================================
# CACHING LAYER FOR ODATA (raw + processed)
# ================================================================
_odata_cache = {}         # {(url, username, password): (timestamp, dataframe)}
_processed_cache = {}    # {(url, username, password): (timestamp, processed_bundle)}
_fetch_locks = {}        # {(key): threading.Lock()}

def _cache_key(url, username, password):
    # Do not include trailing $format differences -> normalize
    u = url.split("$format")[0]
    return f"{u}::{username or ''}::{password or ''}"

def _is_cache_valid(entry):
    ts, _ = entry
    return (datetime.now() - ts) < timedelta(minutes=CACHE_TTL_MINUTES)

def clear_odata_cache():
    _odata_cache.clear()

def clear_processed_cache():
    _processed_cache.clear()

def clear_all_caches():
    clear_odata_cache()
    clear_processed_cache()

def get_cached_odata(url, username, password):
    key = _cache_key(url, username, password)
    entry = _odata_cache.get(key)
    if entry and _is_cache_valid(entry):
        return entry[1]
    return None

def set_cached_odata(url, username, password, df):
    key = _cache_key(url, username, password)
    _odata_cache[key] = (datetime.now(), df.copy())

def get_cached_processed(url, username, password):
    key = _cache_key(url, username, password)
    entry = _processed_cache.get(key)
    if entry and _is_cache_valid(entry):
        return entry[1]
    return None

def set_cached_processed(url, username, password, processed_bundle):
    key = _cache_key(url, username, password)
    # store a shallow copy for safety
    _processed_cache[key] = (datetime.now(), {
        "df": processed_bundle["df"].copy(),
        "fuzzy": processed_bundle["fuzzy"],
        "schema_json": processed_bundle["schema_json"],
        "aliases": processed_bundle["aliases"],
    })

def _get_lock_for_key(key):
    lock = _fetch_locks.get(key)
    if lock is None:
        lock = threading.Lock()
        _fetch_locks[key] = lock
    return lock
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
    # reuse global SSL-enabled session for Gemini (safe), but allow per-call timeout
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
    timeout_sec: int | None = DEFAULT_TIMEOUT
    gemini_url: str | None = None
    gemini_key: str | None = None

# -----------------------------
# Health
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok", "raw_cache_entries": len(_odata_cache), "processed_cache_entries": len(_processed_cache)}

# Endpoint to clear cache manually (clears both raw & processed)
@app.get("/clear-cache")
def clear_cache():
    clear_all_caches()
    return {"status": "cache cleared"}

# -----------------------------
# Helper: fetch raw OData now (per-request session, thread-safe)
# -----------------------------
def fetch_odata_now(url, username=None, password=None, timeout=30):
    auth = (username, password) if (username and password) else None
    # ensure $format=json present (but don't mutate caller's string)
    if "$format" not in url:
        sep = "&" if "?" in url else "?"
        url_with_format = url + f"{sep}$format=json"
    else:
        url_with_format = url

    with requests.Session() as s:
        s.mount("https://", SSLAdapter())
        try:
            resp = s.get(
                url_with_format,
                auth=auth,
                headers={"Accept": "application/json"},
                timeout=timeout,
                verify=False
            )
        except requests.exceptions.ConnectTimeout:
            raise HTTPException(status_code=504, detail="OData connection timed out (SAP server not responding).")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OData connection error: {e}")

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"OData fetch failed: {resp.status_code} {resp.text[:500]}")
    return resp

# -----------------------------
# Helper: get processed bundle (df normalized, fuzzy map, schema_json, aliases)
# -----------------------------
def get_processed_bundle(odata_url, username, password, timeout=30):
    key = _cache_key(odata_url, username, password)

    # 1) Try processed cache
    proc = get_cached_processed(odata_url, username, password)
    if proc is not None:
        return proc, True  # processed cache hit

    # Acquire per-key lock so only one thread fetches/processes at a time
    lock = _get_lock_for_key(key)
    with lock:
        # double-check after acquiring lock
        proc = get_cached_processed(odata_url, username, password)
        if proc is not None:
            return proc, True

        # 2) Try raw cache
        raw_df = get_cached_odata(odata_url, username, password)
        if raw_df is None:
            # fetch fresh using per-request session
            resp = fetch_odata_now(odata_url, username, password, timeout=timeout)
            raw_df = parse_odata_any(resp)
            set_cached_odata(odata_url, username, password, raw_df)

        # 3) Process (this is the expensive step we now cache)
        orig_cols = raw_df.columns.tolist()
        norm_map = {c: normalize_col(c) for c in orig_cols}
        df_proc = raw_df.copy()
        df_proc.columns = [norm_map[c] for c in orig_cols]
        fuzzy_map = fuzzy_column_map(df_proc.columns)

        for c in df_proc.columns:
            df_proc[c] = pd.to_numeric(df_proc[c], errors="ignore")

        schema = []
        for c in df_proc.columns:
            sample = str(df_proc[c].dropna().iloc[0]) if df_proc[c].dropna().shape[0] > 0 else ""
            schema.append({"name": c, "dtype": str(df_proc[c].dtype), "sample": sample})
        schema_json = json.dumps(schema, indent=2)
        aliases = ", ".join(list(fuzzy_map.keys()))

        processed_bundle = {
            "df": df_proc,
            "fuzzy": fuzzy_map,
            "schema_json": schema_json,
            "aliases": aliases,
        }

        # store processed cache
        set_cached_processed(odata_url, username, password, processed_bundle)

        return processed_bundle, False

# -----------------------------
# Core /query endpoint
# -----------------------------
@app.post("/query")
def query(body: QueryBody):
    if not body.odata_url or not body.question:
        raise HTTPException(status_code=400, detail="odata_url and question are required")

    gemini_url = body.gemini_url or GEMINI_URL_DEFAULT
    gemini_key = body.gemini_key or GEMINI_API_KEY
    if not gemini_key:
        raise HTTPException(status_code=500, detail="Missing GEMINI_API_KEY.")

    # 1) Get processed bundle (may use cache) - fast after first run
    try:
        processed, used_processed_cache = get_processed_bundle(
            body.odata_url, body.username, body.password, timeout=body.timeout_sec or DEFAULT_TIMEOUT
        )
    except HTTPException as e:
        # rethrow HTTPExceptions
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to prepare data: {e}")

    df = processed["df"]
    fuzzy_map = processed["fuzzy"]
    schema_json = processed["schema_json"]
    aliases = processed["aliases"]

    # 2) Build Gemini prompt quickly using cached schema_json & aliases
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

    # 3) Call Gemini for pandas expression
    resp = call_gemini_json(
        gemini_url, gemini_key, PROMPT_PANDAS_TRANSLATE + "\nQuestion: " + body.question, body.timeout_sec or DEFAULT_TIMEOUT
    )
    js = extract_json_from_response(resp)
    if not js or "expr" not in js:
        # provide raw Gemini text in error for debugging
        msg = ""
        try:
            msg = resp["candidates"][0]["content"][0]["text"]
        except Exception:
            pass
        raise HTTPException(status_code=400, detail=f"Gemini didn't return expr. Raw: {msg or str(resp)[:1000]}")

    expr = js["expr"]
    explain = js.get("explain", "")

    # 4) Execute expression safely (on processed df)
    try:
        result_obj = safe_exec(expr, df)
    except Exception:
        raise HTTPException(status_code=500, detail=f"Error executing expr:\n{traceback.format_exc()}")

    # 5) Serialize results
    result_table = result_series = result_chart_base64 = None

    if isinstance(result_obj, pd.DataFrame):
        result_table = {
            "columns": list(result_obj.columns),
            "rows": result_obj.fillna("").astype(str).values.tolist()
        }

    elif isinstance(result_obj, pd.Series):
        import ast

        idx_list = result_obj.index.tolist()
        val_list = result_obj.values.tolist()

        parsed_rows = []
        is_tuple_index = False

        for idx in idx_list:
            idx_str = str(idx)

            # Detect tuple-like string index
            try:
                parsed = ast.literal_eval(idx_str)
                if isinstance(parsed, tuple):
                    is_tuple_index = True
                    parsed_rows.append(list(parsed))
                else:
                    parsed_rows.append([idx_str])
            except:
                parsed_rows.append([idx_str])

        # If tuple index → convert to DataFrame dynamically (no hardcoding)
        if is_tuple_index:
            max_len = max(len(r) for r in parsed_rows)
            col_names = [f"level_{i}" for i in range(max_len)]

            df_temp = pd.DataFrame(parsed_rows, columns=col_names)
            df_temp[result_obj.name] = val_list

            result_table = {
                "columns": list(df_temp.columns),
                "rows": df_temp.fillna("").astype(str).values.tolist()
            }
            result_series = None

        else:
            # Normal Series
            result_series = {
                "name": str(result_obj.name),
                "index": [str(x) for x in idx_list],
                "values": [str(x) for x in result_obj.fillna("").astype(str).tolist()],
            }

    elif isinstance(result_obj, plt.Figure):
        buf = io.BytesIO()
        result_obj.savefig(buf, format="png", bbox_inches="tight")
        plt.close(result_obj)
        result_chart_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    else:
        # if there is a current matplotlib figure
        fig = plt.gcf()
        if fig.get_axes():
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            result_chart_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            
    # 6) Ask Gemini for natural-language answer (shorter repr)
    PROMPT_ENGLISH = f"""
You are a helpful assistant. 
Question: {body.question}
The result is: {repr(result_obj)[:2000]}
Give the answer with explanation, in natural English.
"""
    resp2 = call_gemini_json(gemini_url, gemini_key, PROMPT_ENGLISH, body.timeout_sec or DEFAULT_TIMEOUT)
    try:
        answer_text = resp2["candidates"][0]["content"][0]["text"]
    except Exception:
        # fallback
        try:
            answer_text = resp2["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            answer_text = str(resp2)

    return {
        "explain": explain,
        "expr": expr,
        "answer_text": answer_text,
        "used_processed_cache": used_processed_cache,
        "result_table": result_table,
        "result_series": result_series,
        "result_chart_base64": result_chart_base64,
    }

# ---- Local dev runner ----
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
