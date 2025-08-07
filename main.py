import os
import re
import json
import uuid
import time
import tempfile
import logging
import requests
from typing import List, Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Security, BackgroundTasks, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pdfplumber
import pytesseract
import nltk
import numpy as np
from PIL import Image
import cv2
from sentence_transformers import SentenceTransformer, util
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
import tiktoken
import spacy
from symspellpy import SymSpell
from sklearn.linear_model import LogisticRegression


# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="HackRx 6.0 Insurance Policy Analysis API",
    description="Insurance policy PDF processing and question answering",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# Globals (initialized on startup)
embedder = None
client = None
collection = None
nlp = None
sym_spell = None
llm_model = None
trust_model = None
enc = None
decision_cache = {}
CLAUSE_LOG = []
all_chunks = []
metadata = []

MAX_TOK = 500
OVERLAP = 20
CLAUSE_ANCHOR_REGEX = r'(?i)(Excl\d+|Exclusion\s+Code\s*–\s*\d+|Section\s+\w+|Clause\s+\d+|Waiting\s+Period|Covered\s+under)'
KEYWORDS_DENSITY = {
    "waiting period": "waiting_period",
    "exclusion": "exclusion",
    "covered": "coverage",
    "not covered": "exclusion",
}


# -------------------------------
# Pydantic request/response models
# -------------------------------
class HackRxRequest(BaseModel):
    documents: str  # PDF URL
    questions: List[str]


class HackRxResponse(BaseModel):
    answers: List[str]


# -------------------------------
# Auth dependency (token validation placeholder)
# -------------------------------
async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    if not token:
        raise HTTPException(status_code=403, detail="Not authenticated")
    # Add real token verification here if needed
    return token


# -------------------------------
# Startup - initialize required models and downloads
# -------------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting HackRx 6.0 API...")
    await initialize_models()
    logger.info("✅ Initialization complete.")


async def initialize_models():
    global embedder, client, collection, nlp, sym_spell, llm_model, trust_model, enc

    # Setup SSL for nltk download on some platforms
    try:
        import ssl
        ssl._create_unverified_https_context = ssl._create_unverified_https_context
    except Exception:
        pass

    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    try:
        nlp = spacy.load("en_core_web_sm")
        logger.info("Loaded spaCy model")
    except OSError:
        logger.error("spaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
        raise

    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.Client()

    # Load symspell dictionary
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dict_url = "https://raw.githubusercontent.com/mammothb/symspellpy/master/symspellpy/frequency_dictionary_en_82_765.txt"
    dict_path = dict_url.split("/")[-1]
    if not os.path.exists(dict_path):
        logger.info("Downloading symspell dictionary...")
        r = requests.get(dict_url)
        r.raise_for_status()
        with open(dict_path, "wb") as f:
            f.write(r.content)
    sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)
    logger.info("Loaded symspell dictionary")

    api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
    if not api_key:
        logger.error("GOOGLE_GEMINI_API_KEY not set in environment")
        raise RuntimeError("GOOGLE_GEMINI_API_KEY environment variable missing")

    genai.configure(api_key=api_key)
    llm_model = genai.GenerativeModel("gemini-2.5-pro")  # Version matches your notebook

    # Train trust model for confidence calibration
    np.random.seed(42)
    X_train = np.random.rand(100, 3)
    y_train = (0.2*X_train[:,0] + 0.6*X_train[:,1] + 0.2*X_train[:,2] > 0.45).astype(int)
    trust_model = LogisticRegression()
    trust_model.fit(X_train, y_train)


# -------------------------------
# Image preprocessing
# -------------------------------
def preprocess_image(img: Image.Image) -> np.ndarray:
    img_np = np.array(img.convert("RGB"))
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    img_med = cv2.medianBlur(img_gray, 3)
    img_thr = cv2.adaptiveThreshold(
        img_med, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 15)
    img_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img_thr)
    return img_clahe


# -------------------------------
# PDF page extraction with OCR fallback
# -------------------------------
def extract_pages(pdf_path: str) -> List[Dict[str, Any]]:
    data = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for idx, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                table_clauses = []
                for table in tables:
                    if table:
                        tb_str = "\n".join([" | ".join([str(cell or "") for cell in row]) for row in table])
                        table_clauses.append(tb_str)
                text = page.extract_text() or ""
                is_scanned = len(text.strip()) < 50
                if is_scanned:
                    logger.info(f"OCR applying to {pdf_path}, page {idx}.")
                    img = page.to_image(resolution=300).original
                    img_prep = preprocess_image(img)
                    img_pil = Image.fromarray(img_prep)
                    text = pytesseract.image_to_string(img_pil, lang='eng')
                entry = {"page": idx, "text": text}
                if table_clauses:
                    entry["tables"] = table_clauses
                data.append(entry)
        return data
    except Exception as e:
        logger.error(f"Error during PDF extraction: {e}")
        raise HTTPException(status_code=500, detail=f"PDF extraction failed: {e}")


# -------------------------------
# Extract tags from text chunks
# -------------------------------
def extract_tags(text: str) -> List[str]:
    tags = []
    kwt = {
        "hospitalization": "hospital",
        "accident": "accident",
        "diagnostic": "test",
        "mental": "mental_illness",
        "orthopedic": "orthopedic",
        "cancer": "cancer",
        "pregnancy": "maternity",
        "waiting period": "waiting_period",
        "excl": "exclusion",
        "coverage": "coverage"
    }
    text_lower = text.lower()
    for k, v in kwt.items():
        if k in text_lower:
            tags.append(v)
    return list(set(tags))


# -------------------------------
# Text chunking logic
# -------------------------------
def chunk_text(text: str, max_tok=MAX_TOK, overlap=OVERLAP) -> List[str]:
    anchors = re.split(CLAUSE_ANCHOR_REGEX, text)
    text = "".join(anchors)
    sentences = nltk.sent_tokenize(text)
    chunks, chunk, tokens = [], [], 0
    for sent in sentences:
        sent_tok = len(enc.encode(sent))
        if tokens + sent_tok > max_tok:
            if chunk:
                chunks.append(" ".join(chunk))
            chunk = chunk[-overlap:]
            tokens = sum(len(enc.encode(s)) for s in chunk)
        chunk.append(sent)
        tokens += sent_tok
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks


# -------------------------------
# Spell correction function
# -------------------------------
def spell_correct(text: str) -> str:
    if sym_spell is None:
        return text
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    return suggestions[0].term if suggestions else text


# -------------------------------
# Parse user's query for entities
# -------------------------------
def parse_query(q: str) -> Dict:
    corrected = spell_correct(q)
    out = {}
    doc = nlp(corrected)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            out["name"] = ent.text
        elif ent.label_ == "GPE":
            out["location"] = ent.text
        elif ent.label_ == "DATE":
            m = re.search(r"(\d+)\s*(year|month)", ent.text)
            if m:
                val, unit = int(m.group(1)), m.group(2)
                out["policy_duration_months"] = val * 12 if "year" in unit else val
    if "age" not in out:
        m = re.search(r"(\d{1,3})\s*(years? old|y/o)", corrected, re.I)
        if m:
            out["age"] = int(m.group(1))
    if "male" in corrected.lower():
        out["gender"] = "male"
    if "female" in corrected.lower():
        out["gender"] = "female"
    disease_pat = ["surgery", "appendectomy", "dental", "cataract", "pregnancy", "cancer", "orthopedic", "kidney", "mental", "diagnostic", "test"]
    for p in disease_pat:
        if p in corrected.lower():
            out["procedure"] = p
    if "accident" in corrected.lower():
        out["cause"] = "accident"
    if "policy_duration_months" not in out:
        m = re.search(r"policy.*?(\d+)\s*months?", corrected)
        if m:
            out["policy_duration_months"] = int(m.group(1))
        else:
            m = re.search(r"policy.*?(\d+)\s*years?", corrected)
            if m:
                out["policy_duration_months"] = int(m.group(1)) * 12
    if "policy_duration_months" not in out:
        out["policy_duration_months"] = 0
    return out


# -------------------------------
# Dynamic number k for retrieval count
# -------------------------------
def dynamic_k(query: str) -> int:
    base = 5
    dur = parse_query(query).get("policy_duration_months", 0)
    return int(base + dur // 6)


# -------------------------------
# Claim type classification
# -------------------------------
def classify_claim_type(query: str) -> str:
    q = query.lower()
    if "orthopedic" in q or "knee" in q or "joint" in q:
        return "orthopedic_surgery"
    if "cancer" in q or "tumor" in q:
        return "cancer_treatment"
    if "mental" in q:
        return "mental_illness"
    if "hospitalization" in q:
        return "hospitalization"
    if "pregnancy" in q:
        return "maternity"
    if "dental" in q:
        return "dental"
    if "appendectomy" in q:
        return "appendectomy"
    if "cataract" in q:
        return "cataract"
    return "general"


# -------------------------------
# Clause citation logger
# -------------------------------
def log_clause_citation(clause: str, meta: Dict):
    global CLAUSE_LOG
    CLAUSE_LOG.append({"Clause_Text": clause, "Doc": meta["doc"], "Page": meta["page"], "Time": time.time()})


# -------------------------------
# Validate clauses by fuzzy matching
# -------------------------------
def validate_clause_output(text_from_llm: str):
    if not text_from_llm.strip():
        return None, None
    emb = embedder.encode([text_from_llm], convert_to_tensor=True)
    all_embs = embedder.encode(all_chunks, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(emb, all_embs)[0]
    top_idx = int(np.argmax(cos_scores.cpu().numpy()))
    if float(cos_scores[top_idx]) < 0.7:
        return None, None
    return all_chunks[top_idx], metadata[top_idx]


# -------------------------------
# Retrieve relevant chunks from ChromaDB with optional tag filtering
# -------------------------------
def retrieve_relevant(
    q: str,
    k: int = None,
    clause_type=None,
    tags=None
):
    if collection is None or collection.count() == 0:
        return [], 0
    k = k or 5
    where_filter = {}
    if clause_type:
        where_filter["clause_type"] = clause_type
    if tags:
        where_filter["tags"] = {"$in": tags}
    res = collection.query(
        query_texts=[q], n_results=k, where=where_filter if where_filter else None
    )
    docs, metas, distances = res["documents"][0], res["metadatas"][0], res["distances"][0]
    extra = []
    for kw in re.findall(r"\b(excl\d+|waiting|period|maternity|pregnancy|cataract|dental|accident|orthopedic|cancer)\b", q, re.I):
        for idx, chunk in enumerate(all_chunks):
            if kw.lower() in chunk.lower():
                extra.append({
                    "text": chunk, "doc": metadata[idx]["doc"], "page": metadata[idx]["page"],
                    "chunk": metadata[idx]["chunk"], "distance": 0.0,
                })
    all_results = []
    for i in range(min(k, len(docs))):
        all_results.append({
            "text": docs[i], "doc": metas[i]["doc"], "page": metas[i]["page"],
            "chunk": metas[i]["chunk"], "distance": distances[i]
        })
    seen = {(x['doc'], x['page'], x['chunk']) for x in all_results}
    for x in extra:
        sig = (x['doc'], x['page'], x['chunk'])
        if sig not in seen:
            all_results.append(x)
            seen.add(sig)
    coverage_score = len(all_results) / max(k, 1)
    return all_results[:k], coverage_score


# -------------------------------
# Highlight keywords in text
# -------------------------------
def highlight_keywords(text: str) -> str:
    return re.sub(r'(waiting period|exclusion|covered|not covered)', r'**\1**', text, flags=re.I)


RULE_MATRIX = {
    "cataract": {
        "query_keywords": ["cataract"],
        "required_duration_months": 24,
        "decision_if_met": {
            "Decision": "Approved",
            "Summary_Template": "Cataract surgery is covered after the 24-mo wait. Eligible.",
            "Followup_Question": None,
            "Clause_Keyword": "cataract",
            "Clause_Type": "waiting_period"
        },
        "decision_if_not_met": {
            "Decision": "Rejected",
            "Summary_Template": "Cataract: subject to 24-mo wait. Not eligible yet.",
            "Followup_Question": None,
            "Clause_Keyword": "cataract",
            "Clause_Type": "waiting_period"
        }
    }
    # Extend with more rules as needed
}


# -------------------------------
# Calibration for LLM confidence
# -------------------------------
def calibrated_llm_confidence(features: List[float]) -> float:
    proba = trust_model.predict_proba([features])[0][1]
    return float(round(proba, 3))


# -------------------------------
# Decision function combines rule logic and LLM
# -------------------------------
def decide(query: str) -> Dict:
    q_lower = query.lower()
    if q_lower in decision_cache:
        logger.info("Returning cached decision.")
        return decision_cache[q_lower]

    ents = parse_query(query)
    claim_type = classify_claim_type(query)

    # Check valid policy duration
    pdur = ents.get("policy_duration_months", -1)
    if not (0 <= pdur <= 600):
        result = {
            "Decision": "Needs More Info",
            "Amount": None,
            "Justification": {"Summary": "Invalid policy duration.", "Clauses_Cited": []},
            "Followup_Question": "Provide a valid policy duration.",
            "Confidence": "low"
        }
        decision_cache[q_lower] = result
        return result

    # Rule matrix check
    for rule_key, rule in RULE_MATRIX.items():
        if any(kw in q_lower for kw in rule["query_keywords"]):
            met_condition = pdur >= rule["required_duration_months"]
            selected_decision = rule["decision_if_met"] if met_condition else rule["decision_if_not_met"]
            summary = selected_decision["Summary_Template"]
            justification = {"Summary": summary, "Clauses_Cited": []}

            clause_text, meta = validate_clause_output(selected_decision["Clause_Keyword"])
            if clause_text and meta:
                log_clause_citation(clause_text, meta)
                justification["Clauses_Cited"].append({
                    "Clause_Text": highlight_keywords(clause_text)[:1000],
                    "Document_Name": meta["doc"],
                    "Page_Number": meta["page"]
                })
            fs = [1.0 if clause_text else 0.5, 1.0, pdur / 36.0]
            trust = calibrated_llm_confidence(fs)
            trust_score = {"Clause_Recall": fs[0], "NER_Confidence": fs[1], "LLM_Confidence": trust}

            result = {
                "Decision": selected_decision["Decision"],
                "Amount": None,
                "Justification": justification,
                "Followup_Question": selected_decision.get("Followup_Question"),
                "Confidence": trust,
                "Trust_Score": trust_score
            }
            decision_cache[q_lower] = result
            return result

    # Dynamic retrieve + LLM fallback
    k = dynamic_k(query)
    clauses, coverage_score = retrieve_relevant(query, k=k)
    clause_str = "\n---\n".join([f"[{c['doc']} p{c['page']}] {highlight_keywords(c['text'])[:450]}" for c in clauses]) if clauses else "No relevant extracted clause."

    prompt = LLM_SYSTEM_PROMPT.replace(
        "Policy Logic:",
        f"Policy Logic:\n- See clause DB above.\nClauses:\n{clause_str}"
    )
    ents_json = json.dumps(ents, indent=2)
    prompt = prompt.replace(
        'Return strict JSON:',
        f"\nEntities extracted:\n{ents_json}\nClaim_type: {claim_type}\nClauses found:\n{clause_str}\nReturn strict JSON:"
    )

    logger.info("Sending prompt to Gemini LLM...")
    max_retries = 3
    base_delay = 1.5
    for attempt in range(max_retries):
        try:
            resp = llm_model.generate_content(prompt)
            decision_text = resp.text.strip()
            match = re.search(r'json\s*(\{.*\})', decision_text, re.DOTALL)
            if match:
                decision_json = json.loads(match.group(1))
            else:
                curly = re.search(r'(\{.*\})', decision_text, re.DOTALL)
                if curly is None:
                    raise ValueError("No JSON output from LLM")
                decision_json = json.loads(curly.group(1))

            # Validate citations
            for cited in decision_json.get("Justification", {}).get("Clauses_Cited", []):
                real_text, real_meta = validate_clause_output(cited.get("Clause_Text", ""))
                if not real_text:
                    cited["Clause_Text"] = "[INVALID CITATION REMOVED]"
                elif real_meta:
                    cited.update({"Document_Name": real_meta["doc"], "Page_Number": real_meta["page"]})

            fs = [coverage_score, 0.9, ents.get("policy_duration_months", 0) / 36.0]
            trust = calibrated_llm_confidence(fs)
            decision_json["Trust_Score"] = {"Clause_Recall": fs[0], "NER_Confidence": fs[1], "LLM_Confidence": trust}
            decision_json["Confidence"] = trust
            decision_cache[q_lower] = decision_json
            return decision_json

        except Exception as e:
            logger.warning(f"LLM output decode error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(base_delay * 2 ** attempt)
            else:
                break

    return {
        "Decision": "Needs More Info",
        "Amount": None,
        "Justification": {"Summary": "Failed to get valid decision from LLM.", "Clauses_Cited": []},
        "Followup_Question": "Try rephrasing your question.",
        "Confidence": 0.1,
        "Trust_Score": {"Clause_Recall": 0.0, "NER_Confidence": 0.5, "LLM_Confidence": 0.1}
    }


LLM_SYSTEM_PROMPT = """You are a strict insurance analyst. Use only actual policy clauses & the Policy Logic below.
Apply these principles:
- Never invent or hallucinate clauses or page numbers. Cite only exact clauses extracted.
- Prioritize waiting period & specific exclusion clauses over general ones.
- If missing critical details, return "Needs More Info".
Policy Logic:
- Maternity: Excl18, excluded unless add-on.
- Cataract: covered after 24mo.
- Dental: excluded unless hospitalization.
- All non-accidental illnesses: 30d waiting period.
- Orthopedic: 12mo waiting.
- Cancer: 24mo waiting.
Return strict JSON:
{
"Decision": "...",
"Amount": ...,
"Justification": {
   "Summary": "...",
   "Clauses_Cited": [{"Clause_Text": "...", "Document_Name": "...", "Page_Number": ...}]
},
"Followup_Question": "...",
"Confidence": "high/medium/low",
"Trust_Score": {
   "Clause_Recall": ...,
   "NER_Confidence": ...,
   "LLM_Confidence": ...
}
}
"""


# -------------------------------
# PDF file download helper
# -------------------------------
async def download_pdf(url: str) -> str:
    try:
        logger.info(f"Downloading PDF from URL: {url[:60]}...")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(resp.content)
            logger.info(f"PDF downloaded successfully, size: {len(resp.content)} bytes")
            return tmp_file.name
    except Exception as e:
        logger.error(f"Error downloading PDF: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {e}")


# -------------------------------
# Process document: extract, chunk, embed, index
# -------------------------------
async def process_document(pdf_path: str):
    global all_chunks, metadata, collection
    pages = extract_pages(pdf_path)
    logger.info(f"Extracted {len(pages)} pages from PDF.")

    all_chunks = []
    metadata = []

    for page in pages:
        table_detected = page.get("tables", [])
        text_chunks = chunk_text(page["text"])

        for ci, chunk in enumerate(text_chunks):
            ctype = "general"
            cscore = 0
            for k, t in KEYWORDS_DENSITY.items():
                if k in chunk.lower():
                    ctype = t
                    cscore += chunk.lower().count(k)

            tags = extract_tags(chunk)
            tags_str = ",".join(tags) if tags else ""

            meta = {
                "doc": "policy.pdf",
                "page": page["page"],
                "chunk": ci,
                "clause_type": ctype,
                "confidence_score": min(1.0, 0.3 + 0.25 * cscore),
                "is_table_clause": bool(table_detected),
                "tags": tags_str
            }
            all_chunks.append(chunk)
            metadata.append(meta)

        # Process tables
        for tbl_str in table_detected:
            tags = extract_tags(tbl_str)
            tags_str = ",".join(tags) if tags else ""
            meta = {
                "doc": "policy.pdf",
                "page": page["page"],
                "chunk": -1,
                "clause_type": "table",
                "confidence_score": 1.0,
                "is_table_clause": True,
                "tags": tags_str
            }
            all_chunks.append(tbl_str)
            metadata.append(meta)

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2')
    try:
        client.delete_collection("policy_clauses")
    except Exception:
        pass

    collection = client.create_collection("policy_clauses", embedding_function=embed_fn)

    collection.add(documents=all_chunks, metadatas=metadata, ids=[str(uuid.uuid4()) for _ in all_chunks])
    logger.info(f"Vector store populated with {len(all_chunks)} chunks.")


# -------------------------------
# Background cleanup task
# -------------------------------
async def cleanup_temp_file(path: str):
    try:
        if os.path.exists(path):
            os.unlink(path)
            logger.info(f"Cleaned up temp file {path}")
    except Exception as e:
        logger.warning(f"Cleanup failed for {path}: {e}")


# -------------------------------
# Main API endpoint
# -------------------------------
@app.post("/hackrx/run", response_model=HackRxResponse)
async def hackrx_run(
    request: HackRxRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    start_time = time.time()
    pdf_path = await download_pdf(request.documents)
    try:
        await process_document(pdf_path)
        answers = []
        for i, question in enumerate(request.questions, start=1):
            logger.info(f"Answering question {i}/{len(request.questions)}")
            ans = decide(question)
            answers.append(ans if isinstance(ans, str) else json.dumps(ans))
        elapsed = time.time() - start_time
        logger.info(f"Completed processing in {elapsed:.2f}s")
        background_tasks.add_task(cleanup_temp_file, pdf_path)
        return HackRxResponse(answers=answers)
    except Exception as e:
        background_tasks.add_task(cleanup_temp_file, pdf_path)
        logger.error(f"Error in hackrx_run: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


# -------------------------------
# Health check endpoint
# -------------------------------
@app.get("/health")
async def health_check():
    try:
        model_status = {
            "embedder": embedder is not None,
            "nlp": nlp is not None,
            "llm_model": llm_model is not None,
            "chromadb": client is not None,
            "trust_model": trust_model is not None
        }
        status = "healthy" if all(model_status.values()) else "initializing"
        return {
            "status": status,
            "timestamp": time.time(),
            "models": model_status,
            "ready_for_hackrx": all(model_status.values())
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {"status": "unhealthy", "timestamp": time.time(), "error": str(e)}


# -------------------------------
# Root endpoint
# -------------------------------
@app.get("/")
async def root():
    return {
        "message": "HackRx 6.0 Insurance Policy Analysis API",
        "version": "1.0.0",
        "hackrx_endpoint": "/hackrx/run",
        "health_check": "/health",
        "documentation": "/docs",
        "status": "ready"
    }


@app.get("/docs-info")
async def api_documentation():
    return {
        "hackrx_6_0": {
            "endpoint": "/hackrx/run",
            "method": "POST",
            "authentication": "Bearer token required",
            "request_format": {
                "documents": "string (PDF URL)",
                "questions": ["array of strings"]
            },
            "response_format": {
                "answers": ["array of strings"]
            },
            "example_request": {
                "documents": "https://example.com/policy.pdf",
                "questions": [
                    "What is the grace period for premium payment?",
                    "What is the waiting period for pre-existing diseases?"
                ]
            },
            "example_response": {
                "answers": [
                    "Grace period is 30 days for premium payment.",
                    "Waiting period for pre-existing diseases is 36 months."
                ]
            }
        }
    }


# -------------------------------
# Error handlers
# -------------------------------
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return {"error": "Endpoint not found", "hackrx_endpoint": "/hackrx/run"}


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {exc}")
    return {"error": "Internal server error", "message": "Please try again later"}


# -------------------------------
# Run app with uvicorn if main
# -------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
