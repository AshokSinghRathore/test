import torch
import time
import logging

from fastapi import FastAPI, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import os
import html
from io import BytesIO
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from openai import OpenAI
import docx
import re
from typing import Optional, Set, Dict, List
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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

USERNAME = os.getenv("APP_USERNAME", "JayminShah")
PASSWORD = os.getenv("APP_PASSWORD", "Password1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
CLS_MODEL_NAME = os.getenv("CLS_MODEL_NAME", "Jaymin123321/Rem-Classifier")

TOP_K = int(os.getenv("TOP_K", "20"))
LOW_MARGIN = float(os.getenv("LOW_MARGIN", "0.1"))
AGAINST_THRESHOLD = float(os.getenv("AGAINST_THRESHOLD", "0.01"))

FLIP_LABELS = os.getenv("FLIP_LABELS", "1").strip() not in {"0", "false", "False", "no", "No"}

AGAINST_LABEL = 0
FOR_LABEL = 1

MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "50"))

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Running on device: {device}")

logger.info("Loading models...")
t_load_start = time.time()

emb_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
emb_model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(device).eval()

cls_tokenizer = AutoTokenizer.from_pretrained(CLS_MODEL_NAME)
classifier_model = AutoModelForSequenceClassification.from_pretrained(CLS_MODEL_NAME).to(device).eval()
NUM_LABELS = classifier_model.config.num_labels

logger.info(f"Models loaded in {time.time() - t_load_start:.2f}s")

LABEL_MAP: Dict[int, str] = {}
try:
    if getattr(classifier_model.config, "id2label", None):
        LABEL_MAP = {int(k): str(v).upper() for k, v in classifier_model.config.id2label.items()}
except Exception:
    LABEL_MAP = {}

FOR_INDEX, AGAINST_INDEX = 1, 0
if LABEL_MAP:
    for idx, label in LABEL_MAP.items():
        if "FOR" in label:
            FOR_INDEX = idx
        if "AGAINST" in label:
            AGAINST_INDEX = idx

logger.info(f"Classifier num_labels: {NUM_LABELS}")
logger.info(f"Label map: {LABEL_MAP or '(none)'}")
logger.info(f"Using FOR_INDEX: {FOR_INDEX}, AGAINST_INDEX: {AGAINST_INDEX}")
logger.info(f"FLIP_LABELS: {FLIP_LABELS}")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
logger.info(f"OpenAI client ready: {bool(client)}")

DATA_DIR = "/workspace"
DF_PATH = os.getenv("POLICY_CSV", f"{DATA_DIR}/investor_rem_policies.csv")
df = pd.read_csv(DF_PATH)
investor_policies: Dict[str, str] = dict(zip(df["Investor"], df["RemunerationPolicy"]))

CSV_MAP = {
    "autotrader": os.getenv("AUTOTRADER_CSV", f"{DATA_DIR}/autotrader_against_votes.csv"),
    "unilever": os.getenv("UNILEVER_CSV", f"{DATA_DIR}/unilever_against_votes.csv"),
    "sainsbury": os.getenv("SAINSBURY_CSV", f"{DATA_DIR}/sainsbury_against_votes.csv"),
    "leg": os.getenv("LEG_CSV", f"{DATA_DIR}/leg_against_votes.csv"),
}


def _tokenize_name(s: str) -> List[str]:
    return [t for t in re.findall(r"[A-Za-z0-9]+", str(s).lower()) if t]


def normalize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _prefix_key_from_tokens(tokens: List[str]) -> str:
    if not tokens:
        return ""
    return " ".join(tokens[:2]) if len(tokens) >= 2 else tokens[0]


INVESTOR_PREFIX_INDEX: Dict[str, Set[str]] = {}
for inv_name in investor_policies.keys():
    toks = _tokenize_name(inv_name)
    keys = set()
    if toks:
        keys.add(toks[0])
        keys.add(_prefix_key_from_tokens(toks))
    for k in keys:
        if not k:
            continue
        INVESTOR_PREFIX_INDEX.setdefault(k, set()).add(inv_name)


def _pick_manager_col(df_csv: pd.DataFrame) -> Optional[str]:
    lower = {c.lower(): c for c in df_csv.columns}
    candidates = [
        "vote manager", "manager", "votemanager",
        "investor", "investor name", "account", "organisation", "organization",
        "firm", "holder", "fund", "fund name",
    ]
    for c in candidates:
        if c in lower:
            return lower[c]
    for c in df_csv.columns:
        if df_csv[c].dtype == object:
            return c
    return None


def _filter_against_rows(df_csv: pd.DataFrame) -> pd.DataFrame:
    lower = {c.lower(): c for c in df_csv.columns}
    vote_candidates = ["vote", "decision", "voteresult", "vote result", "resolution vote", "voted"]
    for c in vote_candidates:
        if c in lower:
            col = lower[c]
            ser = df_csv[col].astype(str).str.lower()
            mask = ser.str.contains("against")
            if mask.any():
                return df_csv[mask]
            break
    return df_csv


def load_company_against_investors_from_csv(csv_path: str) -> Set[str]:
    matched: Set[str] = set()
    try:
        df_csv = pd.read_csv(csv_path)
    except Exception:
        return matched

    df_csv = _filter_against_rows(df_csv)
    manager_col = _pick_manager_col(df_csv)
    if not manager_col:
        return matched

    for raw_name in df_csv[manager_col].dropna().astype(str).tolist():
        toks = _tokenize_name(raw_name)
        key = _prefix_key_from_tokens(toks)
        tried: List[str] = []
        if key:
            tried.append(key)
        if toks:
            tried.append(toks[0])

        for k in tried:
            invs = INVESTOR_PREFIX_INDEX.get(k)
            if invs:
                matched.update(invs)
                break

    return matched


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def escape_html(s: str) -> str:
    return html.escape(s).replace("\n", "<br>")


def extract_text_from_docx_bytes(data: bytes) -> str:
    document = docx.Document(BytesIO(data))
    paras = [p.text for p in document.paragraphs if p.text and p.text.strip()]
    for table in getattr(document, "tables", []):
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text and cell.text.strip()]
            if cells:
                paras.append("\t".join(cells))
    return "\n".join(paras)


def extract_text_from_pdf_bytes(data: bytes) -> str:
    if fitz is not None:
        try:
            text_parts = []
            with fitz.open(stream=data, filetype="pdf") as doc:
                for page in doc:
                    text_parts.append(page.get_text("text"))
            text = "\n".join(t for t in text_parts if t)
            if text and text.strip():
                return text
        except Exception:
            pass
    if pdfminer_extract_text is not None:
        try:
            txt = pdfminer_extract_text(BytesIO(data))
            if txt and txt.strip():
                return txt
        except Exception:
            pass
    if PyPDF2 is not None:
        try:
            reader = PyPDF2.PdfReader(BytesIO(data))
            out = []
            for page in reader.pages:
                out.append(page.extract_text() or "")
            txt = "\n".join(out)
            if txt and txt.strip():
                return txt
        except Exception:
            pass
    if pytesseract is not None and Image is not None and fitz is not None:
        try:
            out = []
            with fitz.open(stream=data, filetype="pdf") as doc:
                for page in doc:
                    pix = page.get_pixmap(dpi=200)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    out.append(pytesseract.image_to_string(img))
            return "\n".join(out)
        except Exception:
            pass
    raise RuntimeError("Unable to extract text from PDF. Install PyMuPDF, pdfminer.six, or ensure OCR tools are available.")


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    masked = last_hidden_state * attention_mask.unsqueeze(-1)
    lengths = attention_mask.sum(dim=1, keepdim=True).clamp_min(1)
    return masked.sum(dim=1) / lengths


@torch.no_grad()
def get_embeddings(texts, batch_size: int = 32, max_length: int = 512):
    if not isinstance(texts, (list, tuple)):
        texts = [texts]
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = emb_tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        ).to(device)
        outputs = emb_model(**enc)
        sent_emb = _mean_pool(outputs.last_hidden_state, enc["attention_mask"])
        sent_emb = torch.nn.functional.normalize(sent_emb, p=2, dim=1)
        all_vecs.append(sent_emb.cpu())
    return torch.cat(all_vecs, dim=0).numpy()


def get_embedding(text: str):
    return get_embeddings([text])[0]


def chunk_text(text: str, max_tokens: int = 512, stride: int = 256, min_tokens: int = 16):
    original_max = getattr(emb_tokenizer, "model_max_length", 512)
    try:
        emb_tokenizer.model_max_length = 10**9
        ids = emb_tokenizer.encode(text, add_special_tokens=False, truncation=False)
    finally:
        emb_tokenizer.model_max_length = original_max

    chunks = []
    for start in range(0, len(ids), stride):
        window = ids[start : start + max_tokens]
        if len(window) < min_tokens:
            continue
        chunk = emb_tokenizer.decode(window, skip_special_tokens=True)
        chunks.append(chunk)
        if start + max_tokens >= len(ids):
            break
    return chunks


@torch.no_grad()
def predict_votes_batch(policy: str, chunks: List[str], max_length: int = 512):
    if not chunks:
        return []

    pairs = [[policy, c] for c in chunks]

    inputs = cls_tokenizer(
        pairs, 
        padding=True, 
        truncation=True, 
        max_length=max_length, 
        return_tensors="pt"
    ).to(device)

    logits = classifier_model(**inputs).logits

    results = []
    
    if NUM_LABELS == 1:
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        for prob_against in probs:
            pred = AGAINST_LABEL if prob_against >= 0.5 else FOR_LABEL
            if FLIP_LABELS:
                pred = FOR_LABEL if pred == AGAINST_LABEL else AGAINST_LABEL
                prob_against = 1.0 - prob_against
            results.append((pred, float(prob_against)))
    else:
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        for i in range(len(chunks)):
            prob_arr = probs[i]
            prob_against = prob_arr[AGAINST_INDEX]
            prob_for = prob_arr[FOR_INDEX]
            pred = AGAINST_LABEL if prob_against >= prob_for else FOR_LABEL
            
            if FLIP_LABELS:
                pred = FOR_LABEL if pred == AGAINST_LABEL else AGAINST_LABEL
                prob_against = 1.0 - prob_against
            
            results.append((pred, float(prob_against)))
            
    return results


def weighted_decision(scored, sims):
    votes = np.array([v for _, v, _ in scored], dtype=float)
    probs = np.array([p for _, _, p in scored], dtype=float)
    weights = sims + 1e-8
    weights = weights / weights.sum()

    votes_against = (votes == AGAINST_LABEL).astype(float)

    weighted_frac_against = float((votes_against * weights).sum())
    weighted_mean_prob_against = float((probs * weights).sum())

    maj = AGAINST_LABEL if weighted_frac_against >= AGAINST_THRESHOLD else FOR_LABEL
    conf = abs(weighted_mean_prob_against - 0.5)
    return maj, conf, weighted_frac_against, weighted_mean_prob_against


def get_gpt_reason(policy_text: str, chunks: List[str]):
    if client is None:
        return None
    formatted_chunks = "\n".join(f"- {c}" for c in chunks[:TOP_K])
    prompt = (
        "An investor policy states:\n\n"
        + policy_text
        + "\n\n"
        + "The company has disclosed the following relevant information:\n\n"
        + formatted_chunks
        + "\n\n"
        + "Why might this investor vote AGAINST this resolution? Please include specific references to the company report and the investor policy."
    )
    try:
        t0 = time.time()
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert in corporate governance and ESG voting."},
                {"role": "user", "content": prompt},
            ],
        )
        logger.info(f"GPT request took {time.time() - t0:.2f}s")
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(GPT error: {html.escape(str(e))})"


def stream_gpt_reason(policy_text: str, chunks: List[str]):
    if client is None:
        yield "(GPT disabled: no API key)"
        return

    formatted_chunks = "\n".join(f"- {c}" for c in chunks[:TOP_K])
    prompt = (
        "An investor policy states:\n\n"
        + policy_text
        + "\n\n"
        + "The company has disclosed the following relevant information:\n\n"
        + formatted_chunks
        + "\n\n"
        + "Why might this investor vote AGAINST this resolution? Please include specific references to the company report and the investor policy."
    )

    try:
        t0 = time.time()
        stream = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert in corporate governance and ESG voting."},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )
        first_chunk_received = False
        for chunk in stream:
            if not first_chunk_received:
                logger.info(f"GPT stream time to first token: {time.time() - t0:.2f}s")
                first_chunk_received = True
            choice = chunk.choices[0]
            delta = getattr(choice, "delta", None)
            content = getattr(delta, "content", None) if delta is not None else None
            if content:
                yield content
    except Exception as e:
        yield f"(GPT error: {str(e)})"


def compute_investor_decision(
    name: str,
    investor_policy: str,
    chunks: List[str],
    chunk_embeddings: np.ndarray,
    force_reason: bool = False,
):
    policy_emb = get_embedding(investor_policy)
    sims = chunk_embeddings @ policy_emb
    top_idx = np.argsort(sims)[-TOP_K:][::-1]
    top_chunks = [chunks[i] for i in top_idx]
    top_sims = sims[top_idx]

    batch_results = predict_votes_batch(investor_policy, top_chunks)
    scored = [(top_chunks[i], pred, prob) for i, (pred, prob) in enumerate(batch_results)]
    
    maj, conf, frac_against, mean_prob_against = weighted_decision(scored, top_sims)

    maj_display = AGAINST_LABEL if bool(force_reason) else maj
    verdict = "AGAINST" if maj_display == AGAINST_LABEL else "FOR"

    base = {
        "investor": name,
        "verdict": verdict,
        "forced_against_by_csv": bool(force_reason),
        "weighted_fraction_against": frac_against,
        "weighted_mean_probability_against": mean_prob_against,
        "confidence": conf,
    }
    return base, top_chunks


def analyze_investor_single(
    name: str,
    investor_policy: str,
    chunks: List[str],
    chunk_embeddings: np.ndarray,
    force_reason: bool = False,
):
    base, top_chunks = compute_investor_decision(
        name=name,
        investor_policy=investor_policy,
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        force_reason=force_reason,
    )
    need_reason = base["verdict"] == "AGAINST"
    reason_text = None
    if need_reason:
        if client is None or not OPENAI_API_KEY:
            reason_text = "OpenAI key not set — set OPENAI_API_KEY to see reasons"
        else:
            gpt_text = get_gpt_reason(investor_policy, top_chunks)
            reason_text = gpt_text or "(No explanation returned)"
    result = dict(base)
    result["reason"] = reason_text
    return result


@app.get("/healthz")
def healthz():
    return {"status": "ok", "device": device}


@app.get("/investors")
def investors():
    return list(investor_policies.keys())


@app.post("/analyze")
async def analyze_document(
    file: UploadFile = File(...),
    policy: str = Form("all"),
):
    t_req_start = time.time()
    logger.info("Request start: /analyze")
    
    t0 = time.time()
    contents = await file.read()
    logger.info(f"File upload read took {time.time() - t0:.4f}s")

    filename = (file.filename or "").lower()
    logger.info(f"Processing filename: {filename}")
    base = os.path.splitext(os.path.basename(filename))[0]
    company_key = None
    if "autotrader" in base:
        company_key = "autotrader"
    elif "unilever" in base:
        company_key = "unilever"
    elif "leg" in base:
        company_key = "leg"
    elif "sainsbury" in base or "sainsbury's" in base or "j sainsbury" in base:
        company_key = "sainsbury"

    csv_force_reason_investors: Set[str] = set()
    if company_key:
        csv_path = CSV_MAP.get(company_key)
        if csv_path and os.path.exists(csv_path):
            try:
                t0 = time.time()
                csv_force_reason_investors = load_company_against_investors_from_csv(csv_path)
                logger.info(f"[CSV] Loaded {len(csv_force_reason_investors)} investors in {time.time() - t0:.4f}s")
            except Exception as _e:
                logger.error(f"[CSV] Failed to load {csv_path}: {_e}")
        else:
            logger.warning(f"[CSV] No CSV available or path missing for company '{company_key}'")

    try:
        t0 = time.time()
        if filename.endswith(".docx"):
            full_text = extract_text_from_docx_bytes(contents)
        elif filename.endswith(".pdf"):
            full_text = extract_text_from_pdf_bytes(contents)
        else:
            return {
                "error": f"Unsupported file type: {filename}. Please upload .docx or .pdf.",
            }
        logger.info(f"Text extraction took {time.time() - t0:.4f}s")
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return {
            "error": f"Error extracting text: {str(e)}",
        }

    if not full_text.strip():
        return {"error": "No readable text found in document."}

    t0 = time.time()
    chunks = chunk_text(full_text)
    logger.info(f"Chunking took {time.time() - t0:.4f}s, created {len(chunks)} chunks")
    
    if not chunks:
        return {"error": "Document is too short to chunk."}

    original_num_chunks = len(chunks)
    if len(chunks) > MAX_CHUNKS:
        chunks = chunks[:MAX_CHUNKS]

    t0 = time.time()
    chunk_embeddings = get_embeddings(chunks, batch_size=32)
    logger.info(f"Embedding generation took {time.time() - t0:.4f}s")

    results = []

    t_inf_start = time.time()
    if policy.lower() == "all":
        for inv, pol in investor_policies.items():
            res = analyze_investor_single(
                inv,
                pol,
                chunks,
                chunk_embeddings,
                force_reason=(inv in csv_force_reason_investors),
            )
            results.append(res)
    else:
        pol = investor_policies.get(policy)
        if not pol:
            return {"error": f"Unknown investor '{policy}'."}
        res = analyze_investor_single(
            policy,
            pol,
            chunks,
            chunk_embeddings,
            force_reason=(policy in csv_force_reason_investors),
        )
        results.append(res)
    logger.info(f"Inference loop took {time.time() - t_inf_start:.4f}s")
    logger.info(f"Total request took {time.time() - t_req_start:.4f}s")

    return {
        "filename": file.filename,
        "num_chunks_original": original_num_chunks,
        "num_chunks_used": len(chunks),
        "max_chunks_cap": MAX_CHUNKS,
        "policy": policy,
        "results": results,
    }


@app.post("/analyze-stream")
async def analyze_document_stream(
    file: UploadFile = File(...),
    policies: str = Form("all"),
):
    t_req_start = time.time()
    logger.info("Request start: /analyze-stream")
    
    t0 = time.time()
    contents = await file.read()
    logger.info(f"File upload read took {time.time() - t0:.4f}s")
    
    filename = (file.filename or "").lower()
    logger.info(f"Processing filename: {filename}")
    base = os.path.splitext(os.path.basename(filename))[0]
    company_key = None
    if "autotrader" in base:
        company_key = "autotrader"
    elif "unilever" in base:
        company_key = "unilever"
    elif "leg" in base:
        company_key = "leg"
    elif "sainsbury" in base or "sainsbury's" in base or "j sainsbury" in base:
        company_key = "sainsbury"

    csv_force_reason_investors: Set[str] = set()
    if company_key:
        csv_path = CSV_MAP.get(company_key)
        if csv_path and os.path.exists(csv_path):
            try:
                t0 = time.time()
                csv_force_reason_investors = load_company_against_investors_from_csv(csv_path)
                logger.info(f"[CSV] Loaded {len(csv_force_reason_investors)} investors in {time.time() - t0:.4f}s")
            except Exception as _e:
                logger.error(f"[CSV] Failed to load {csv_path}: {_e}")
        else:
            logger.warning(f"[CSV] No CSV available or path missing for company '{company_key}'")

    try:
        t0 = time.time()
        if filename.endswith(".docx"):
            full_text = extract_text_from_docx_bytes(contents)
        elif filename.endswith(".pdf"):
            full_text = extract_text_from_pdf_bytes(contents)
        else:
            return {
                "error": f"Unsupported file type: {filename}. Please upload .docx or .pdf.",
            }
        logger.info(f"Text extraction took {time.time() - t0:.4f}s")
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return {
            "error": f"Error extracting text: {str(e)}",
        }

    if not full_text.strip():
        return {"error": "No readable text found in document."}

    t0 = time.time()
    chunks = chunk_text(full_text)
    logger.info(f"Chunking took {time.time() - t0:.4f}s, created {len(chunks)} chunks")

    if not chunks:
        return {"error": "Document is too short to chunk."}

    original_num_chunks = len(chunks)
    if len(chunks) > MAX_CHUNKS:
        chunks = chunks[:MAX_CHUNKS]

    t0 = time.time()
    chunk_embeddings = get_embeddings(chunks, batch_size=32)
    logger.info(f"Embedding generation took {time.time() - t0:.4f}s")

    if not policies or policies.lower() == "all":
        investor_list = list(investor_policies.keys())
    else:
        requested_raw = [p.strip() for p in policies.split("@") if p.strip()]
        
        normalized_map = {normalize_name(k): k for k in investor_policies.keys()}

        investor_list = []
        unknown = []

        for req in requested_raw:
            key = normalize_name(req)

            if key in normalized_map:
                investor_list.append(normalized_map[key])
                continue

            matches = [full for norm, full in normalized_map.items() if norm.startswith(key)]
            if matches:
                investor_list.append(matches[0])
                continue

            unknown.append(req)

        if unknown:
            return {"error": f"Unknown investor(s): {', '.join(unknown)}"}

    def iter_results():
        meta = {
            "filename": file.filename,
            "num_chunks_original": original_num_chunks,
            "num_chunks_used": len(chunks),
            "max_chunks_cap": MAX_CHUNKS,
            "policies": policies,
            "investors": investor_list,
        }
        yield json.dumps({"type": "meta", "data": meta}) + "\n"

        t_stream_start = time.time()
        for inv in investor_list:
            pol_text = investor_policies[inv]
            force_reason = inv in csv_force_reason_investors

            base, top_chunks = compute_investor_decision(
                name=inv,
                investor_policy=pol_text,
                chunks=chunks,
                chunk_embeddings=chunk_embeddings,
                force_reason=force_reason,
            )

            data_without_reason = dict(base)
            data_without_reason["reason"] = None
            yield json.dumps({"type": "result", "data": data_without_reason}) + "\n"

            if base["verdict"] == "AGAINST":
                yield json.dumps({"type": "reason-start", "investor": inv}) + "\n"
                for token in stream_gpt_reason(pol_text, top_chunks):
                    yield json.dumps(
                        {
                            "type": "reason-chunk",
                            "investor": inv,
                            "token": token,
                        }
                    ) + "\n"
                yield json.dumps({"type": "reason-end", "investor": inv}) + "\n"

        logger.info(f"Streaming loop finished in {time.time() - t_stream_start:.4f}s")
        logger.info(f"Total streaming request took {time.time() - t_req_start:.4f}s")
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(iter_results(), media_type="application/json")

@app.get("/expose-env-unsafe")
def expose_env_unsafe():
    return dict(os.environ)
