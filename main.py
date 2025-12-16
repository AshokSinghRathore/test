import torch
torch.set_num_threads(1)

import os
import json
import html
import asyncio
import secrets
import re
import numpy as np
import pandas as pd
from io import BytesIO
from typing import List, Dict, Set

from fastapi import FastAPI, UploadFile, File, Form, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, RedirectResponse

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from openai import AsyncOpenAI
import docx

try:
    import fitz
except:
    fitz = None
try:
    import pytesseract
    from PIL import Image
except:
    pytesseract = None
    Image = None
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
except:
    pdfminer_extract_text = None
try:
    import PyPDF2
except:
    PyPDF2 = None


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo")

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CLS_MODEL_NAME = "Jaymin123321/Rem-Classifier"

TOP_K = 20
AGAINST_THRESHOLD = 0.01
MAX_CHUNKS = 10

AGAINST_LABEL = 0
FOR_LABEL = 1

device = "cuda" if torch.cuda.is_available() else "cpu"

client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


emb_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
emb_model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(device).eval()

cls_tokenizer = AutoTokenizer.from_pretrained(CLS_MODEL_NAME)
cls_model = AutoModelForSequenceClassification.from_pretrained(CLS_MODEL_NAME).to(device).eval()

NUM_LABELS = cls_model.config.num_labels

df = pd.read_csv("investor_rem_policies.csv")
investor_policies: Dict[str, str] = dict(zip(df["Investor"], df["RemunerationPolicy"]))

INVESTOR_NAMES = list(investor_policies.keys())
INVESTOR_POLICIES = [investor_policies[n] for n in INVESTOR_NAMES]

def normalize_name(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())

INVESTOR_MAP = {normalize_name(k): k for k in INVESTOR_NAMES}

@torch.no_grad()
def embed_texts(texts: List[str]):
    enc = emb_tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    out = emb_model(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1)
    emb = (out * mask).sum(1) / mask.sum(1)
    return torch.nn.functional.normalize(emb, dim=1).cpu().numpy()

POLICY_EMBS = embed_texts(INVESTOR_POLICIES)

def extract_docx(data: bytes) -> str:
    doc = docx.Document(BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

def extract_pdf(data: bytes) -> str:
    if fitz:
        try:
            text = []
            with fitz.open(stream=data, filetype="pdf") as doc:
                for page in doc:
                    text.append(page.get_text("text"))
            joined = "\n".join(text)
            if joined.strip():
                return joined
        except:
            pass

    if pdfminer_extract_text:
        try:
            txt = pdfminer_extract_text(BytesIO(data))
            if txt.strip():
                return txt
        except:
            pass

    if PyPDF2:
        try:
            reader = PyPDF2.PdfReader(BytesIO(data))
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        except:
            pass

    if pytesseract and Image and fitz:
        out = []
        with fitz.open(stream=data, filetype="pdf") as doc:
            for page in doc:
                pix = page.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                out.append(pytesseract.image_to_string(img))
        return "\n".join(out)

    return ""

def chunk_text(text: str):
    ids = emb_tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(ids), 256):
        win = ids[i:i+512]
        if len(win) >= 16:
            chunks.append(emb_tokenizer.decode(win))
        if len(chunks) >= MAX_CHUNKS:
            break
    return chunks

@torch.no_grad()
def predict_votes_batch(policy: str, chunks: List[str]):
    pairs = [(policy, c) for c in chunks]
    enc = cls_tokenizer(
        [p for p, c in pairs],
        [c for p, c in pairs],
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    logits = cls_model(**enc).logits

    if NUM_LABELS == 1:
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        return [(AGAINST_LABEL if p >= 0.5 else FOR_LABEL, float(p)) for p in probs]

    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return [
        (AGAINST_LABEL if p[0] >= p[1] else FOR_LABEL, float(p[0]))
        for p in probs
    ]

def weighted_decision(scored, sims):
    votes = np.array([v for _, v, _ in scored], dtype=float)
    probs = np.array([p for _, _, p in scored], dtype=float)
    w = sims + 1e-8
    w /= w.sum()
    frac = float(((votes == AGAINST_LABEL) * w).sum())
    mean = float((probs * w).sum())
    maj = AGAINST_LABEL if frac >= AGAINST_THRESHOLD else FOR_LABEL
    conf = abs(mean - 0.5)
    return maj, conf


async def stream_reason(policy, chunks):
    if not client:
        return
    prompt = (
        "An investor policy states:\n\n"
        + policy
        + "\n\nCompany disclosure:\n\n"
        + "\n".join(f"- {c}" for c in chunks)
    )
    async for chunk in client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    ):
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz")
def healthz():
    return {"status": "ok", "device": device}

@app.get("/investors")
def investors():
    return INVESTOR_NAMES

@app.post("/analyze-stream")
async def analyze_stream(file: UploadFile = File(...), investors: str = Form("all")):
    data = await file.read()
    loop = asyncio.get_running_loop()

    if file.filename.endswith(".docx"):
        text = await loop.run_in_executor(None, extract_docx, data)
    else:
        text = await loop.run_in_executor(None, extract_pdf, data)

    chunks = chunk_text(text)
    chunk_embs = embed_texts(chunks)

    if investors == "all":
        names = INVESTOR_NAMES
    else:
        names = [INVESTOR_MAP[normalize_name(x)] for x in investors.split("@")]

    async def gen():
        yield json.dumps({"type": "meta", "data": {"investors": names}}) + "\n"

        for name in names:
            idx = INVESTOR_NAMES.index(name)
            policy = investor_policies[name]
            policy_emb = POLICY_EMBS[idx]

            sims = chunk_embs @ policy_emb
            top_idx = np.argsort(sims)[-TOP_K:]
            top_chunks = [chunks[i] for i in top_idx]
            top_sims = sims[top_idx]

            preds = predict_votes_batch(policy, top_chunks)
            scored = [(top_chunks[i], preds[i][0], preds[i][1]) for i in range(len(preds))]
            maj, conf = weighted_decision(scored, top_sims)

            verdict = "AGAINST" if maj == AGAINST_LABEL else "FOR"
            yield json.dumps({"type": "result", "data": {"investor": name, "verdict": verdict, "confidence": conf}}) + "\n"

            if verdict == "AGAINST":
                yield json.dumps({"type": "reason-start", "investor": name}) + "\n"
                async for token in stream_reason(policy, top_chunks):
                    yield json.dumps({"type": "reason-chunk", "investor": name, "token": token}) + "\n"
                yield json.dumps({"type": "reason-end", "investor": name}) + "\n"

        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(gen(), media_type="application/json")
