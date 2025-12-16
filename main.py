import torch
torch.set_num_threads(1)

import os
import json
import html
import asyncio
import numpy as np
import pandas as pd
from io import BytesIO
from typing import List, Dict, Set

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from openai import AsyncOpenAI
import docx
import re

try:
    import fitz
except:
    fitz = None

device = "cuda" if torch.cuda.is_available() else "cpu"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
CLS_MODEL_NAME = os.getenv("CLS_MODEL_NAME", "Jaymin123321/Rem-Classifier")

TOP_K = int(os.getenv("TOP_K", "20"))
AGAINST_THRESHOLD = float(os.getenv("AGAINST_THRESHOLD", "0.01"))
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "10"))

AGAINST_LABEL = 0
FOR_LABEL = 1

client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

emb_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
emb_model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(device).eval()

cls_tokenizer = AutoTokenizer.from_pretrained(CLS_MODEL_NAME)
cls_model = AutoModelForSequenceClassification.from_pretrained(CLS_MODEL_NAME).to(device).eval()

DATA_DIR = "/workspace/data"
df = pd.read_csv(f"{DATA_DIR}/investor_rem_policies.csv")
investor_policies = dict(zip(df["Investor"], df["RemunerationPolicy"]))

def normalize_name(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())

INVESTOR_MAP = {normalize_name(k): k for k in investor_policies.keys()}

@torch.no_grad()
def embed_texts(texts):
    enc = emb_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    out = emb_model(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1)
    emb = (out * mask).sum(1) / mask.sum(1)
    return torch.nn.functional.normalize(emb, dim=1).cpu().numpy()

INVESTOR_EMBS = embed_texts(list(investor_policies.values()))

def extract_docx(data):
    doc = docx.Document(BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

def extract_pdf(data):
    if not fitz:
        return ""
    text = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            text.append(page.get_text())
    return "\n".join(text)

def chunk_text(text):
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
def batched_classify(policies, chunks):
    pairs = []
    for p in policies:
        for c in chunks:
            pairs.append((p, c))
    enc = cls_tokenizer(
        [p for p, c in pairs],
        [c for p, c in pairs],
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    logits = cls_model(**enc).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs.reshape(len(policies), len(chunks), -1)

async def stream_reason(policy, chunks):
    if not client:
        return
    prompt = policy + "\n\n" + "\n".join(chunks)
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
    return list(investor_policies.keys())

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
        names = list(investor_policies.keys())
    else:
        names = [INVESTOR_MAP[normalize_name(x)] for x in investors.split("@")]

    policies = [investor_policies[n] for n in names]
    policy_embs = INVESTOR_EMBS[[list(investor_policies.keys()).index(n) for n in names]]

    sims = chunk_embs @ policy_embs.T
    top_idx = np.argsort(sims, axis=0)[-TOP_K:]

    probs = batched_classify(policies, chunks)

    async def gen():
        yield json.dumps({"type": "meta", "data": {"investors": names}}) + "\n"

        for i, name in enumerate(names):
            against_prob = probs[i, top_idx[:, i], 0].mean()
            verdict = "AGAINST" if against_prob >= AGAINST_THRESHOLD else "FOR"
            yield json.dumps({"type": "result", "data": {"investor": name, "verdict": verdict, "confidence": float(against_prob)}}) + "\n"

            if verdict == "AGAINST":
                yield json.dumps({"type": "reason-start", "investor": name}) + "\n"
                async for token in stream_reason(policies[i], [chunks[j] for j in top_idx[:, i]]):
                    yield json.dumps({"type": "reason-chunk", "investor": name, "token": token}) + "\n"
                yield json.dumps({"type": "reason-end", "investor": name}) + "\n"

        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(gen(), media_type="application/json")
