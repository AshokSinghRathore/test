import os
import re
import json
import html
import asyncio
import torch
import numpy as np
import pandas as pd
from io import BytesIO
from typing import Dict, List, Set

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from openai import AsyncOpenAI
import docx

try:
    import fitz
except Exception:
    fitz = None
try:
    from pdfminer_high_level import extract_text as pdfminer_extract_text
except Exception:
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract_text
    except Exception:
        pdfminer_extract_text = None
try:
    import PyPDF2
except Exception:
    PyPDF2 = None
try:
    import pytesseract
    from PIL import Image
except Exception:
    pytesseract = None
    Image = None

torch.set_num_threads(1)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
CLS_MODEL_NAME = os.getenv("CLS_MODEL_NAME", "Jaymin123321/Rem-Classifier")

TOP_K = int(os.getenv("TOP_K", "20"))
AGAINST_THRESHOLD = float(os.getenv("AGAINST_THRESHOLD", "0.01"))
FLIP_LABELS = os.getenv("FLIP_LABELS", "1").strip() not in {"0", "false", "False", "no", "No"}
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "50"))

AGAINST_LABEL = 0
FOR_LABEL = 1

device = "cuda" if torch.cuda.is_available() else "cpu"

emb_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
emb_model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(device).eval()

cls_tokenizer = AutoTokenizer.from_pretrained(CLS_MODEL_NAME)
classifier_model = AutoModelForSequenceClassification.from_pretrained(CLS_MODEL_NAME).to(device).eval()
NUM_LABELS = classifier_model.config.num_labels

FOR_INDEX, AGAINST_INDEX = 1, 0
if getattr(classifier_model.config, "id2label", None):
    for k, v in classifier_model.config.id2label.items():
        if "FOR" in str(v).upper():
            FOR_INDEX = int(k)
        if "AGAINST" in str(v).upper():
            AGAINST_INDEX = int(k)

client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

DATA_DIR = "/app/data"
df = pd.read_csv(os.getenv("POLICY_CSV", f"{DATA_DIR}/investor_rem_policies.csv"))
investor_policies: Dict[str, str] = dict(zip(df["Investor"], df["RemunerationPolicy"]))

INVESTOR_NAMES = list(investor_policies.keys())
INVESTOR_POLICY_TEXTS = list(investor_policies.values())

CSV_MAP = {
    "autotrader": f"{DATA_DIR}/autotrader_against_votes.csv",
    "unilever": f"{DATA_DIR}/unilever_against_votes.csv",
    "sainsbury": f"{DATA_DIR}/sainsbury_against_votes.csv",
    "leg": f"{DATA_DIR}/leg_against_votes.csv",
}

def _tokenize_name(s: str):
    return [t for t in re.findall(r"[a-z0-9]+", s.lower()) if t]

def _prefix_key(tokens):
    if not tokens:
        return ""
    return " ".join(tokens[:2]) if len(tokens) >= 2 else tokens[0]

INVESTOR_PREFIX_INDEX: Dict[str, Set[str]] = {}
for inv in INVESTOR_NAMES:
    toks = _tokenize_name(inv)
    for k in {_prefix_key(toks), toks[0] if toks else ""}:
        if k:
            INVESTOR_PREFIX_INDEX.setdefault(k, set()).add(inv)

def load_company_against_investors_from_csv(path: str) -> Set[str]:
    out = set()
    try:
        dfc = pd.read_csv(path)
    except Exception:
        return out
    col = next(iter(dfc.columns), None)
    if not col:
        return out
    for raw in dfc[col].dropna().astype(str):
        toks = _tokenize_name(raw)
        for k in {_prefix_key(toks), toks[0] if toks else ""}:
            if k in INVESTOR_PREFIX_INDEX:
                out.update(INVESTOR_PREFIX_INDEX[k])
                break
    return out

def extract_text_from_docx_bytes(data: bytes) -> str:
    d = docx.Document(BytesIO(data))
    out = [p.text for p in d.paragraphs if p.text.strip()]
    for table in getattr(d, "tables", []):
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                out.append("\t".join(cells))
    return "\n".join(out)

def extract_text_from_pdf_bytes(data: bytes) -> str:
    if fitz:
        try:
            with fitz.open(stream=data, filetype="pdf") as doc:
                txt = "\n".join(p.get_text() for p in doc)
                if txt.strip():
                    return txt
        except Exception:
            pass
    if pdfminer_extract_text:
        try:
            txt = pdfminer_extract_text(BytesIO(data))
            if txt.strip():
                return txt
        except Exception:
            pass
    if PyPDF2:
        reader = PyPDF2.PdfReader(BytesIO(data))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    raise RuntimeError("PDF extraction failed")

@torch.no_grad()
def embed(texts: List[str]) -> np.ndarray:
    enc = emb_tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    out = emb_model(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1)
    pooled = (out * mask).sum(1) / mask.sum(1)
    return torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy()

INVESTOR_EMBEDDINGS = embed(INVESTOR_POLICY_TEXTS)

def chunk_text(text: str) -> List[str]:
    ids = emb_tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(ids), 256):
        win = ids[i:i+512]
        if len(win) < 16:
            continue
        chunks.append(emb_tokenizer.decode(win))
        if len(chunks) >= MAX_CHUNKS:
            break
    return chunks

@torch.no_grad()
def predict_votes_batch(policy: str, chunks: List[str]):
    pairs = [[policy, c] for c in chunks]
    enc = cls_tokenizer(pairs, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    logits = classifier_model(**enc).logits
    if NUM_LABELS == 1:
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        out = [(AGAINST_LABEL if p >= 0.5 else FOR_LABEL, float(p)) for p in probs]
    else:
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        out = []
        for p in probs:
            pa, pf = p[AGAINST_INDEX], p[FOR_INDEX]
            out.append((AGAINST_LABEL if pa >= pf else FOR_LABEL, float(pa)))
    if FLIP_LABELS:
        out = [(FOR_LABEL if v == AGAINST_LABEL else AGAINST_LABEL, 1 - p) for v, p in out]
    return out

def weighted_decision(scored, sims):
    votes = np.array([v for _, v, _ in scored])
    w = sims / (sims.sum() + 1e-8)
    frac = float((votes == AGAINST_LABEL).astype(float) @ w)
    return AGAINST_LABEL if frac >= AGAINST_THRESHOLD else FOR_LABEL

async def gpt_stream(investor, policy, chunks, queue):
    await queue.put({"type": "reason-start", "investor": investor})
    if not client:
        await queue.put({"type": "reason-end", "investor": investor})
        return
    prompt = (
        "An investor policy states:\n\n"
        + policy
        + "\n\nThe company has disclosed:\n\n"
        + "\n".join("- " + c for c in chunks[:TOP_K])
    )
    stream = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    async for ch in stream:
        delta = ch.choices[0].delta
        if delta and delta.content:
            await queue.put({
                "type": "reason-chunk",
                "investor": investor,
                "token": delta.content
            })
    await queue.put({"type": "reason-end", "investor": investor})

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/healthz")
def healthz():
    return {"status": "okii", "device": device}

@app.get("/investors")
def investors():
    return INVESTOR_NAMES

@app.post("/analyze-stream")
async def analyze_stream(file: UploadFile = File(...)):
    raw = await file.read()
    loop = asyncio.get_running_loop()
    if file.filename.lower().endswith(".docx"):
        text = await loop.run_in_executor(None, extract_text_from_docx_bytes, raw)
    else:
        text = await loop.run_in_executor(None, extract_text_from_pdf_bytes, raw)

    chunks = chunk_text(text)
    chunk_embeddings = embed(chunks)

    queue = asyncio.Queue()
    tasks = []

    async def generator():
        yield json.dumps({"type": "meta", "filename": file.filename}) + "\n"
        for idx, investor in enumerate(INVESTOR_NAMES):
            policy = INVESTOR_POLICY_TEXTS[idx]
            sims = chunk_embeddings @ INVESTOR_EMBEDDINGS[idx]
            top_idx = np.argsort(sims)[-TOP_K:][::-1]
            top_chunks = [chunks[i] for i in top_idx]
            preds = predict_votes_batch(policy, top_chunks)
            scored = [(top_chunks[i], preds[i][0], preds[i][1]) for i in range(len(top_chunks))]
            maj = weighted_decision(scored, sims[top_idx])
            verdict = "AGAINST" if maj == AGAINST_LABEL else "FOR"
            yield json.dumps({"type": "result", "data": {"investor": investor, "verdict": verdict}}) + "\n"
            if verdict == "AGAINST":
                tasks.append(asyncio.create_task(gpt_stream(investor, policy, top_chunks, queue)))
        pending = len(tasks)
        while pending:
            msg = await queue.get()
            yield json.dumps(msg) + "\n"
            if msg["type"] == "reason-end":
                pending -= 1
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(generator(), media_type="application/json")
