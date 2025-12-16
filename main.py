import os
import re
import json
import html
import torch
import numpy as np
import pandas as pd
from io import BytesIO
from typing import Dict, List, Set
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.concurrency import run_in_threadpool
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from openai import AsyncOpenAI
import docx

try:
    import fitz
except:
    fitz = None
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
except:
    pdfminer_extract_text = None
try:
    import PyPDF2
except:
    PyPDF2 = None

USERNAME = os.getenv("APP_USERNAME", "JayminShah")
PASSWORD = os.getenv("APP_PASSWORD", "Password1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
CLS_MODEL_NAME = os.getenv("CLS_MODEL_NAME", "Jaymin123321/Rem-Classifier")

TOP_K = int(os.getenv("TOP_K", "10"))
AGAINST_THRESHOLD = float(os.getenv("AGAINST_THRESHOLD", "0.01"))
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "10"))

AGAINST_LABEL = 0
FOR_LABEL = 1

device = "cuda" if torch.cuda.is_available() else "cpu"

emb_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
emb_model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(device).eval()

cls_tokenizer = AutoTokenizer.from_pretrained(CLS_MODEL_NAME)
cls_model = AutoModelForSequenceClassification.from_pretrained(CLS_MODEL_NAME).to(device).eval()

client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

df = pd.read_csv("/app/data/investor_rem_policies.csv")
investor_policies = dict(zip(df["Investor"], df["RemunerationPolicy"]))

POLICY_EMB_CACHE = {}
with torch.no_grad():
    for k, v in investor_policies.items():
        enc = emb_tokenizer(v, return_tensors="pt", truncation=True, padding=True).to(device)
        out = emb_model(**enc).last_hidden_state.mean(dim=1)
        POLICY_EMB_CACHE[k] = torch.nn.functional.normalize(out, p=2, dim=1).cpu().numpy()[0]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_docx(data: bytes):
    d = docx.Document(BytesIO(data))
    out = []
    for p in d.paragraphs:
        if p.text.strip():
            out.append(p.text)
    return "\n".join(out)

def extract_pdf(data: bytes):
    if fitz:
        with fitz.open(stream=data, filetype="pdf") as doc:
            return "\n".join(p.get_text() for p in doc)
    if PyPDF2:
        r = PyPDF2.PdfReader(BytesIO(data))
        return "\n".join(p.extract_text() or "" for p in r.pages)
    if pdfminer_extract_text:
        return pdfminer_extract_text(BytesIO(data))
    raise RuntimeError("No PDF parser")

def chunk_text(text: str):
    ids = emb_tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(ids), 256):
        if len(chunks) >= MAX_CHUNKS:
            break
        window = ids[i:i+512]
        if len(window) >= 16:
            chunks.append(emb_tokenizer.decode(window))
    return chunks

@torch.no_grad()
def embed_chunks(chunks: List[str]):
    enc = emb_tokenizer(chunks, return_tensors="pt", truncation=True, padding=True).to(device)
    out = emb_model(**enc).last_hidden_state.mean(dim=1)
    return torch.nn.functional.normalize(out, p=2, dim=1).cpu().numpy()

@torch.no_grad()
def classify(policy: str, chunks: List[str]):
    pairs = [[policy, c] for c in chunks]
    enc = cls_tokenizer(pairs, return_tensors="pt", truncation=True, padding=True).to(device)
    logits = cls_model(**enc).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs[:, AGAINST_LABEL]

async def stream_reason(policy: str, chunks: List[str]):
    if not client:
        yield "GPT disabled"
        return
    prompt = policy + "\n\n" + "\n".join(chunks)
    async for ev in client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role":"user","content":prompt}],
        stream=True
    ):
        delta = ev.choices[0].delta
        if delta and delta.content:
            yield delta.content

@app.post("/analyze-stream")
async def analyze_stream(file: UploadFile = File(...), policies: str = Form("all")):
    data = await file.read()
    name = file.filename.lower()

    if name.endswith(".docx"):
        text = await run_in_threadpool(extract_docx, data)
    else:
        text = await run_in_threadpool(extract_pdf, data)

    chunks = chunk_text(text)
    chunk_embs = embed_chunks(chunks)

    investors = list(investor_policies.keys()) if policies == "all" else policies.split("@")

    async def gen():
        yield json.dumps({"type":"meta","data":{"filename":file.filename,"investors":investors}}) + "\n"

        for inv in investors:
            pol = investor_policies[inv]
            sims = chunk_embs @ POLICY_EMB_CACHE[inv]
            idx = np.argsort(sims)[-TOP_K:][::-1]
            top_chunks = [chunks[i] for i in idx]
            top_sims = sims[idx]

            probs = classify(pol, top_chunks)
            frac = float((probs >= 0.5).mean())
            verdict = "AGAINST" if frac >= AGAINST_THRESHOLD else "FOR"

            yield json.dumps({"type":"result","data":{"investor":inv,"verdict":verdict,"confidence":frac}}) + "\n"

            if verdict == "AGAINST":
                yield json.dumps({"type":"reason-start","investor":inv}) + "\n"
                async for t in stream_reason(pol, top_chunks):
                    yield json.dumps({"type":"reason-chunk","investor":inv,"token":t}) + "\n"
                yield json.dumps({"type":"reason-end","investor":inv}) + "\n"

        yield json.dumps({"type":"done"}) + "\n"

    return StreamingResponse(gen(), media_type="application/json")
