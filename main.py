from __future__ import annotations

import io
import os
import re
import json
import time
import base64
import hashlib
import threading
import traceback
import contextlib
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import ssl
from requests.adapters import HTTPAdapter
import urllib3

from fastapi import FastAPI, HTTPException, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Optional duckdb
try:
    import duckdb
    HAS_DUCKDB = True
except Exception:
    HAS_DUCKDB = False

# ========== ENV / CONFIG ==========
GEMINI_URL_DEFAULT = os.environ.get(
    "GEMINI_URL",
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"
)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
CACHE_TTL_MINUTES = int(os.environ.get("CACHE_TTL_MINUTES", "30"))
DEFAULT_TIMEOUT = int(os.environ.get("DEFAULT_TIMEOUT_SEC", "30"))
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ========== APP SETUP ==========
app = FastAPI(title="SAP OData ChatBot API (FastAPI conversion)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== SSL Adapter (Legacy SAP Fix) ==========
class SSLAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.options |= 0x4  # SSL_OP_LEGACY_SERVER_CONNECT
        # Intentionally disable hostname check & cert verify for legacy servers
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        kwargs["ssl_context"] = ctx
        return super(SSLAdapter, self).init_poolmanager(*args, **kwargs)

# Global session for Gemini (re-used)
_gemini_session = requests.Session()
_gemini_session.mount("https://", SSLAdapter())

# ========== In-memory caches & locks ==========
_odata_cache: Dict[str, Tuple[datetime, pd.DataFrame]] = {}
_processed_cache: Dict[str, Tuple[datetime, dict]] = {}
_fetch_locks: Dict[str, threading.Lock] = {}
_cache_lock = threading.Lock()


def _cache_key(url: str, username: Optional[str], password: Optional[str]) -> str:
    # Normalize by stripping $format suffix for cache key stability
    base = url.split("$format")[0]
    return f"{base}::{username or ''}::{password or ''}"


def _is_cache_valid(ts: datetime) -> bool:
    return (datetime.now() - ts) < timedelta(minutes=CACHE_TTL_MINUTES)


def clear_odata_cache():
    with _cache_lock:
        _odata_cache.clear()


def clear_processed_cache():
    with _cache_lock:
        _processed_cache.clear()


def clear_all_caches():
    clear_odata_cache()
    clear_processed_cache()


def _get_lock_for_key(key: str) -> threading.Lock:
    # ensure one lock per dataset
    with _cache_lock:
        lock = _fetch_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _fetch_locks[key] = lock
        return lock

# ========== Utilities (from streamlit agent) ==========
def normalize_col(c: str) -> str:
    return re.sub(r"[^0-9a-z_]", "_", c.strip().lower())


def fuzzy_column_map(columns: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for c in columns:
        lc = c.lower()
        mapping[lc] = c
        for token in lc.split("_"):
            mapping[token] = c
    return mapping


def extract_json_from_response(resp: Any) -> Optional[dict]:
    try:
        if isinstance(resp, dict):
            # common shapes observed
            if "candidates" in resp:
                # gemini-like
                try:
                    text = resp["candidates"][0]["content"][0]["parts"][0]["text"]
                except Exception:
                    # alternative indexing
                    text = resp["candidates"][0]["content"][0].get("text", "")
            elif "text" in resp:
                text = resp["text"]
            else:
                text = json.dumps(resp)
        else:
            text = str(resp)
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass
    return None


def validate_expr(expr: str) -> bool:
    forbidden = ["subprocess", "os.", "sys.", "open(", "eval(", "exec(", "__import__", "input(", "print("]
    for f in forbidden:
        if f in expr:
            raise ValueError(f"Unsafe code detected (forbidden token: {f})")
    return True


def fuzzy_filter(df: pd.DataFrame, col: str, value: str) -> pd.DataFrame:
    col_values = df[col].dropna().astype(str).unique()
    import difflib
    closest = difflib.get_close_matches(str(value), col_values, n=1, cutoff=0.6)
    if closest:
        return df[df[col].fillna('').str.contains(closest[0], case=False, na=False)]
    else:
        return df[df[col].fillna('').str.contains(str(value), case=False, na=False)]


# OData XML parser (kept as-is)
def parse_odata_xml(xml_text: str) -> pd.DataFrame:
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


def parse_odata_any(resp: requests.Response) -> pd.DataFrame:
    ctype = resp.headers.get("Content-Type", "").lower()
    if "json" in ctype:
        js = resp.json()
        if isinstance(js, dict) and "d" in js:
            results = js["d"].get("results", [])
        elif isinstance(js, dict) and "value" in js:
            results = js.get("value", [])
        else:
            results = js
        df = pd.DataFrame(results)
    else:
        df = parse_odata_xml(resp.text)

    if "__metadata" in df.columns:
        # safe attempt to extract id
        try:
            df["metadata_id"] = df["__metadata"].apply(lambda x: x.get("id") if isinstance(x, dict) else None)
            df.drop(columns=["__metadata"], inplace=True)
        except Exception:
            pass
    return df


# Gemini REST wrapper
def call_gemini_json(url: str, key: str, prompt: str, timeout: int = 40) -> dict:
    headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        r = _gemini_session.post(url, headers=headers, json=payload, timeout=timeout, verify=False)
        try:
            return r.json()
        except Exception:
            return {"text": r.text}
    except Exception as e:
        # bubble up as HTTPException by caller if needed
        raise RuntimeError(f"Gemini call failed: {e}")


# safe_exec (executes pandas expressions in limited local_env)
def safe_exec(expr: str, df: pd.DataFrame):
    local_env = {"df": df, "pd": pd, "np": np, "plt": plt, "re": re, "fuzzy_filter": fuzzy_filter}
    validate_expr(expr)
    result = None
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        try:
            # try eval first (one-liner)
            try:
                result = eval(expr, {}, local_env)
            except Exception:
                # fallback exec (multi-line)
                exec(expr, {}, local_env)
                # find last DataFrame/Series/Figure variable in local_env (reverse)
                for k, v in reversed(list(local_env.items())):
                    if isinstance(v, (pd.DataFrame, pd.Series, plt.Figure)):
                        result = v
                        break
                if result is None:
                    result = "✅ Code executed successfully (no direct result returned)"
        except Exception:
            # re-raise for caller to produce HTTP 500
            raise
    # If a plt figure was created implicitly, return that
    if not isinstance(result, plt.Figure):
        fig = plt.gcf()
        if fig.get_axes():
            result = fig
    return result


# ========== Processing pipeline - cache-aware ==========
def get_cached_odata(url: str, username: Optional[str], password: Optional[str]) -> Optional[pd.DataFrame]:
    key = _cache_key(url, username, password)
    entry = _odata_cache.get(key)
    if entry and _is_cache_valid(entry[0]):
        return entry[1].copy()
    return None


def set_cached_odata(url: str, username: Optional[str], password: Optional[str], df: pd.DataFrame):
    key = _cache_key(url, username, password)
    _odata_cache[key] = (datetime.now(), df.copy())


def get_cached_processed(url: str, username: Optional[str], password: Optional[str]) -> Optional[dict]:
    key = _cache_key(url, username, password)
    entry = _processed_cache.get(key)
    if entry and _is_cache_valid(entry[0]):
        # return a shallow copy to avoid mutation
        bundle = entry[1].copy()
        bundle["df"] = bundle["df"].copy()
        return bundle
    return None


def set_cached_processed(url: str, username: Optional[str], password: Optional[str], processed_bundle: dict):
    key = _cache_key(url, username, password)
    # store essential fields
    to_store = {
        "df": processed_bundle["df"].copy(),
        "fuzzy": processed_bundle["fuzzy"],
        "schema_json": processed_bundle["schema_json"],
        "aliases": processed_bundle["aliases"],
    }
    _processed_cache[key] = (datetime.now(), to_store)


def fetch_odata_now(url: str, username: Optional[str], password: Optional[str], timeout: int = 30) -> requests.Response:
    # ensure $format=json for easier parsing
    if "$format" not in url:
        sep = "&" if "?" in url else "?"
        url_with_format = url + f"{sep}$format=json"
    else:
        url_with_format = url

    auth = (username, password) if (username and password) else None
    with requests.Session() as s:
        s.mount("https://", SSLAdapter())
        try:
            resp = s.get(
                url_with_format,
                auth=auth,
                headers={"Accept": "application/json,application/atom+xml"},
                timeout=timeout,
                verify=False
            )
        except requests.exceptions.ConnectTimeout:
            raise HTTPException(status_code=504, detail="OData connection timed out (SAP server not responding).")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OData connection error: {e}")

    if resp.status_code == 401:
        raise HTTPException(status_code=401, detail="Unauthorized (401). Check username/password.")
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"OData fetch failed: {resp.status_code}: {resp.text[:1000]}")
    return resp


def process_raw_df(raw_df: pd.DataFrame) -> dict:
    # normalize columns
    orig_cols = raw_df.columns.tolist()
    norm_map = {c: normalize_col(c) for c in orig_cols}
    df_proc = raw_df.copy()
    df_proc.columns = [norm_map[c] for c in orig_cols]
    fuzzy_map = fuzzy_column_map(df_proc.columns)

    # try to coerce numerics
    for c in df_proc.columns:
        try:
            df_proc[c] = pd.to_numeric(df_proc[c], errors='ignore')
        except Exception:
            pass

    # minimal schema json
    schema = []
    for c in df_proc.columns:
        sample = ""
        try:
            s = df_proc[c].dropna()
            if s.shape[0] > 0:
                sample = str(s.iloc[0])[:25]
        except Exception:
            sample = ""
        schema.append({"name": c, "dtype": str(df_proc[c].dtype), "sample": sample})
    schema_json = json.dumps(schema, separators=(",", ":"), ensure_ascii=False)
    aliases = ", ".join(list(fuzzy_map.keys()))

    processed_bundle = {
        "df": df_proc,
        "fuzzy": fuzzy_map,
        "schema_json": schema_json,
        "aliases": aliases,
    }
    return processed_bundle


def get_processed_bundle(odata_url: str, username: Optional[str], password: Optional[str], timeout: int = DEFAULT_TIMEOUT) -> Tuple[dict, bool]:
    """
    Returns (processed_bundle, used_processed_cache_bool)
    """
    key = _cache_key(odata_url, username, password)

    # try processed cache first
    cached_proc = get_cached_processed(odata_url, username, password)
    if cached_proc is not None:
        return cached_proc, True

    lock = _get_lock_for_key(key)
    with lock:
        # double-check
        cached_proc = get_cached_processed(odata_url, username, password)
        if cached_proc is not None:
            return cached_proc, True

        # try raw cache
        raw_df = get_cached_odata(odata_url, username, password)
        if raw_df is None:
            resp = fetch_odata_now(odata_url, username, password, timeout=timeout)
            raw_df = parse_odata_any(resp)
            set_cached_odata(odata_url, username, password, raw_df)

        processed_bundle = process_raw_df(raw_df)
        set_cached_processed(odata_url, username, password, processed_bundle)

        return processed_bundle, False


# ========== Pydantic Models ==========
class QueryBody(BaseModel):
    odata_url: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    question: str
    timeout_sec: Optional[int] = DEFAULT_TIMEOUT
    gemini_url: Optional[str] = None
    gemini_key: Optional[str] = None
    use_duckdb: Optional[bool] = True


# ========== Prompt builder (compact) ==========
def build_prompt_cached(schema_json: str, aliases: str) -> str:
    # similar to original prompt but compact
    PROMPT_PREAMBLE = f"""
You are a data analysis agent. The dataframe is available as a pandas DataFrame named `df`.
Schema (columns and types): {schema_json}
Column aliases (comma separated): {aliases}
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
16. SELF-CHECK COLUMN VALIDATION RULE:

When choosing the column to filter for a user value (e.g., "Professional Tax"),
perform the following steps:

1. Select the best matching string column using fuzzy matching.
2. Apply the filter on that column.
3. If the filter returns ZERO matching rows, do NOT return this result.
4. Instead, automatically try the NEXT BEST matching string column.
5. Continue checking all string columns in order of similarity.
6. Stop when you find a column that produces at least one matching row.
7. If no column produces any result, return:
     "No matching data found in the dataset."
8. Never assume that the first matched column is correct; always apply
   this self-check loop before finalizing the expression.
17.If the question logically implies a value search across multiple columns
(e.g., 'Professional Tax', 'GST', 'Jackson'),
never restrict search to a single guessed column.
Always use full multi-column search.
18.Automatically treat the search term as a generic value search.
Search it inside all string columns using OR across all columns.
Return all matching rows.
19.When working with dates, always convert using:
pd.to_datetime(df[column], dayfirst=True, errors='coerce')

Never assume the exact date format.
Always use dayfirst=True for dd-mm-yyyy formats.
Always handle invalid dates using errors='coerce'.
20.Avoid df.apply(..., axis=1) unless absolutely required.
Prefer vectorized operations because they are more reliable and efficient.
Only use axis=1 for complex row-level logic.
Always handle NaN safely in such cases.
21.SQL MODE (MANDATORY):
- If the user message begins with “sql:”, then you MUST return SQL and NOTHING else.
- NEVER convert the SQL request to pandas or python.
- NEVER rewrite SQL into python.
- ALWAYS return an expression starting with “sql:” exactly like this (lowercase):
      sql: SELECT ... FROM odata ...
- ALWAYS query the table named `odata`.
- Write SQL in ONE line only.
- Valid example:
      sql: SELECT COUNT(*) FROM odata
- If the user writes “sql:”, pandas expressions like df.shape[0], df[...] are NOT allowed.

"""
    return PROMPT_PREAMBLE


# ========== Serialization helpers ==========
def dataframe_to_serializable(df: pd.DataFrame) -> dict:
    return {"columns": list(df.columns), "rows": df.fillna("").astype(str).values.tolist()}


def series_to_serializable(s: pd.Series) -> dict:
    return {"name": str(s.name), "index": [str(x) for x in s.index.tolist()], "values": [str(x) for x in s.fillna("").tolist()]}


def figure_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ========== Endpoints ==========
@app.get("/health")
def health():
    return {
        "status": "ok",
        "raw_cache_entries": len(_odata_cache),
        "processed_cache_entries": len(_processed_cache),
        "cache_ttl_minutes": CACHE_TTL_MINUTES,
        "duckdb_available": HAS_DUCKDB,
    }


@app.get("/clear-cache")
def clear_cache():
    clear_all_caches()
    return {"status": "cache cleared"}


@app.post("/query")
def query(body: QueryBody):
    """
    Main endpoint. Use either:
      - Provide body.odata_url (and optionally credentials) to fetch OData and query.
      - Or, use /upload-query for file uploads (this endpoint expects odata_url).
    """
    if not body.question:
        raise HTTPException(status_code=400, detail="Missing 'question' in request body.")

    # determine gemini settings
    gemini_url = body.gemini_url or GEMINI_URL_DEFAULT
    gemini_key = body.gemini_key or GEMINI_API_KEY
    if not gemini_key:
        raise HTTPException(status_code=500, detail="Missing Gemini API key (GEMINI_API_KEY or gemini_key).")

    if not body.odata_url:
        raise HTTPException(status_code=400, detail="Provide 'odata_url' or use /upload-query for upload-based queries.")

    # 1) Prepare processed bundle (cached)
    try:
        processed, used_cache = get_processed_bundle(body.odata_url, body.username, body.password, timeout=body.timeout_sec or DEFAULT_TIMEOUT)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to prepare data: {e}")

    df = processed["df"]
    fuzzy_map = processed["fuzzy"]
    schema_json = processed["schema_json"]
    aliases = processed["aliases"]

    # 2) Build prompt and call Gemini to produce JSON {explain, expr}
    PROMPT = build_prompt_cached(schema_json, aliases) + "\nQuestion: " + body.question + "\nRespond ONLY with a JSON object containing keys: explain and expr."
    try:
        resp = call_gemini_json(gemini_url, gemini_key, PROMPT, timeout=int(body.timeout_sec or DEFAULT_TIMEOUT))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    js = extract_json_from_response(resp)
    if not js or "expr" not in js:
        # try raw fallback text for debugging
        raw_text = ""
        try:
            raw_text = resp.get("candidates", [])[0].get("content", [])[0].get("parts", [])[0].get("text", "")
        except Exception:
            raw_text = str(resp)[:1000]
        raise HTTPException(status_code=400, detail=f"Gemini didn't return 'expr'. Raw response: {raw_text}")

    expr = js["expr"]
    explain = js.get("explain", "")

    # 3) Execute expr
    result_obj = None
    exec_error = None
    t0 = time.time()
    try:
        if isinstance(expr, str) and expr.strip().lower().startswith("sql:"):
            sql_text = expr.strip()[4:].strip()
            if body.use_duckdb and HAS_DUCKDB:
                try:
                    con = duckdb.connect(database=":memory:")
                    con.register("odata", df)
                    result_df = con.execute(sql_text).df()
                    result_obj = result_df
                    con.close()
                except Exception as e:
                    exec_error = f"DuckDB execution failed: {e}"
                    result_obj = None
            else:
                exec_error = "SQL requested but DuckDB not available or disabled."
                result_obj = None
        else:
            # treat as pandas expression
            try:
                validate_expr(expr)
            except Exception as e:
                exec_error = f"Expression validation failed: {e}"
                result_obj = None
            else:
                try:
                    result_obj = safe_exec(expr, df)
                except Exception:
                    raise
    except HTTPException:
        raise
    except Exception as e:
        exec_error = f"Execution exception: {traceback.format_exc()}"
        result_obj = None
    t_exec = int((time.time() - t0) * 1000)

    # 4) Serialize result
    result_table = None
    result_series = None
    result_chart_base64 = None
    if isinstance(result_obj, pd.DataFrame):
        result_table = dataframe_to_serializable(result_obj)
    elif isinstance(result_obj, pd.Series):
        result_series = series_to_serializable(result_obj)
    elif isinstance(result_obj, plt.Figure):
        result_chart_base64 = figure_to_base64(result_obj)
    else:
        # if a matplotlib figure exists implicitly
        fig = plt.gcf()
        if fig.get_axes():
            result_chart_base64 = figure_to_base64(fig)

    # 5) Ask Gemini to produce a natural language answer (short explanation)
    PROMPT_ENGLISH = f"""You are a helpful assistant.
Question: {body.question}
The result (repr): {repr(result_obj)[:2000]}
Provide a concise natural-language explanation for the user.
"""
    try:
        resp2 = call_gemini_json(gemini_url, gemini_key, PROMPT_ENGLISH, timeout=int(body.timeout_sec or DEFAULT_TIMEOUT))
        try:
            answer_text = resp2["candidates"][0]["content"][0]["parts"][0]["text"]
        except Exception:
            answer_text = str(resp2)
    except Exception:
        answer_text = explain or (str(result_obj) if result_obj is not None else exec_error or "No answer produced.")

    return {
        "explain": explain,
        "expr": expr,
        "answer_text": answer_text,
        "used_processed_cache": used_cache,
        "execution_ms": t_exec,
        "exec_error": exec_error,
        "result_table": result_table,
        "result_series": result_series,
        "result_chart_base64": result_chart_base64,
    }


@app.post("/upload-query")
def upload_query(file: UploadFile = File(...), question: str = Body(..., embed=True), gemini_key: Optional[str] = Body(None), timeout_sec: Optional[int] = Body(DEFAULT_TIMEOUT), use_duckdb: Optional[bool] = Body(True)):
    """
    Upload a CSV/XLSX and ask a question. This bypasses OData.
    """
    content = file.file.read()
    try:
        if file.filename.lower().endswith(".csv"):
            df_raw = pd.read_csv(io.BytesIO(content))
        else:
            df_raw = pd.read_excel(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse uploaded file: {e}")

    # compute a pseudo-cache key so repeated uploads with same content reuse caches
    h = hashlib.md5(content).hexdigest()
    pseudo_url = f"upload://{h}"

    # try processed cache
    proc = get_cached_processed(pseudo_url, None, None)
    if proc is None:
        processed_bundle = process_raw_df(df_raw)
        set_cached_processed(pseudo_url, None, None, processed_bundle)
        processed = processed_bundle
        used_cache = False
    else:
        processed = proc
        used_cache = True

    # call main logic by assembling QueryBody-like parameters
    body = QueryBody(
        odata_url=pseudo_url,
        username=None,
        password=None,
        question=question,
        timeout_sec=timeout_sec,
        gemini_url=None,
        gemini_key=gemini_key or GEMINI_API_KEY,
        use_duckdb=use_duckdb
    )

    # reuse get_processed_bundle retrieval logic by bypassing fetch (we already saved processed cache)
    try:
        # call the same internal query logic by invoking get_processed_bundle then constructing response manually
        processed_bundle, used_cache_flag = get_processed_bundle(body.odata_url, None, None, timeout=timeout_sec or DEFAULT_TIMEOUT)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process uploaded data: {e}")

    # build prompt and call Gemini for expr
    schema_json = processed_bundle["schema_json"]
    aliases = processed_bundle["aliases"]
    gemini_url = body.gemini_url or GEMINI_URL_DEFAULT
    gemini_key = body.gemini_key or GEMINI_API_KEY
    if not gemini_key:
        raise HTTPException(status_code=500, detail="Missing Gemini API key.")

    PROMPT = build_prompt_cached(schema_json, aliases) + "\nQuestion: " + body.question + "\nRespond ONLY with a JSON object containing keys: explain and expr."
    try:
        resp = call_gemini_json(gemini_url, gemini_key, PROMPT, timeout=int(body.timeout_sec or DEFAULT_TIMEOUT))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini call failed: {e}")

    js = extract_json_from_response(resp)
    if not js or "expr" not in js:
        raw_text = ""
        try:
            raw_text = resp.get("candidates", [])[0].get("content", [])[0].get("parts", [])[0].get("text", "")
        except Exception:
            raw_text = str(resp)[:1000]
        raise HTTPException(status_code=400, detail=f"Gemini didn't return 'expr'. Raw response: {raw_text}")

    expr = js["expr"]
    explain = js.get("explain", "")

    # Execute
    try:
        df = processed_bundle["df"]
        if isinstance(expr, str) and expr.strip().lower().startswith("sql:"):
            sql_text = expr.strip()[4:].strip()
            if use_duckdb and HAS_DUCKDB:
                con = duckdb.connect(database=":memory:")
                con.register("odata", df)
                result_df = con.execute(sql_text).df()
                result_obj = result_df
                con.close()
            else:
                raise HTTPException(status_code=400, detail="SQL requested but duckdb not available or disabled.")
        else:
            validate_expr(expr)
            result_obj = safe_exec(expr, df)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail=f"Error executing expression:\n{traceback.format_exc()}")

    # serialize result
    result_table = None
    result_series = None
    result_chart_base64 = None
    if isinstance(result_obj, pd.DataFrame):
        result_table = dataframe_to_serializable(result_obj)
    elif isinstance(result_obj, pd.Series):
        result_series = series_to_serializable(result_obj)
    elif isinstance(result_obj, plt.Figure):
        result_chart_base64 = figure_to_base64(result_obj)

    # short english answer from Gemini
    PROMPT_ENGLISH = f"""
You are a helpful assistant.
Question: {question}
The result (repr): {repr(result_obj)[:2000]}
Provide a concise explanation for the user.
"""
    try:
        resp2 = call_gemini_json(gemini_url, gemini_key, PROMPT_ENGLISH, timeout=int(body.timeout_sec or DEFAULT_TIMEOUT))
        try:
            answer_text = resp2["candidates"][0]["content"][0]["parts"][0]["text"]
        except Exception:
            answer_text = str(resp2)
    except Exception:
        answer_text = explain or (str(result_obj) if result_obj is not None else "No answer produced.")

    return {
        "explain": explain,
        "expr": expr,
        "answer_text": answer_text,
        "used_processed_cache": used_cache,
        "result_table": result_table,
        "result_series": result_series,
        "result_chart_base64": result_chart_base64,
    }


# ========== Local runner ==========
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
