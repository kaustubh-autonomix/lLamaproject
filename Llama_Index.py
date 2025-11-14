"""

this is Production-ready ( atleast i think that is) RAG + KG service using 2 main components (Weaviate + Gemini 2.5 Flash)

Single-file implementation that:

- it has a vector RAG flow (so it ingests PDFs, chunk them, embed the chunks with gemini api, upsert vectors that are derived)
- this file has a one-time KnowledgeGraph extraction flow (KG_Metadata, KG_Entity, KG_Relation in Weaviate)
- also this exposes endpoints to trigger KG extraction and sessioned queries (continued conversation)
- uses checksum to avoid re-extraction of data from the pdf

Environment variables required ( atleast what we have abhi filhaal ke liye ):
  GEMINI_API_KEY,
  WEAVIATE_URL,
  WEAVIATE_API_KEY (optional),
  WEAVIATE_CLASS (optional)
  CHUNK_SIZE,
  CHUNK_OVERLAP

also if any changes done other than what is mentioned above will be noted down here ☟☟☟


"""

import os
import uuid
import time
import hashlib
import json
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
import pdfplumber
import requests
import weaviate
import asyncio

from config import (
    WEAVIATE_URL,
    WEAVIATE_API_KEY,
    GEMINI_API_KEY,
    WEAVIATE_CLASS,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from logger import logger


# weaviate client
client = weaviate.Client(
    url=WEAVIATE_URL,
    additional_headers=(
        {"Authorization": f"Bearer {WEAVIATE_API_KEY}"} if WEAVIATE_API_KEY else None
    ),
)

# Ensure core classes exist (Document) and KG classes
def ensure_core_schema():
    schema = client.schema.get()
    classes = [c['class'] for c in schema.get('classes', [])]
    # Document class
    if WEAVIATE_CLASS not in classes:
        class_obj = {
            "class": WEAVIATE_CLASS,
            "properties": [
                {"name": "text", "dataType": ["text"]},
                {"name": "source", "dataType": ["string"]},
                {"name": "meta", "dataType": ["text"]},
                {"name": "chunk_id", "dataType": ["string"]}
            ]
        }
        client.schema.create_class(class_obj)
        logger.info(f"Created Weaviate class {WEAVIATE_CLASS}")
    # KG classes
    kg_classes = {"KG_Entity": [
                    {"name": "name", "dataType": ["string"]},
                    {"name": "type", "dataType": ["string"]},
                    {"name": "canonical_id", "dataType": ["string"]},
                    {"name": "aliases", "dataType": ["string[]"]},
                    {"name": "source_chunks", "dataType": ["string[]"]}
                ],
                "KG_Relation": [
                    {"name": "relation_type", "dataType": ["string"]},
                    {"name": "evidence_chunks", "dataType": ["string[]"]},
                    {"name": "confidence", "dataType": ["number"]},
                    {"name": "from", "dataType": ["KG_Entity"]},
                    {"name": "to", "dataType": ["KG_Entity"]}
                ],
                "KG_Metadata": [
                    {"name": "doc_name", "dataType": ["string"]},
                    {"name": "checksum", "dataType": ["string"]},
                    {"name": "extraction_status", "dataType": ["string"]},
                    {"name": "extraction_timestamp", "dataType": ["string"]}
                ]
               }
    for cls, props in kg_classes.items():
        if cls not in classes:
            client.schema.create_class({"class": cls, "properties": props})
            logger.info(f"Created KG class {cls}")

# PDF extraction
def extract_text_from_pdf(path: str) -> str:
    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                texts.append(txt)
    return "\n".join(texts)

# chunker
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+size]
        chunks.append(" ".join(chunk))
        i += size - overlap
    return chunks

# helper: compute checksum
def file_checksum(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# Gemini embedding (same placeholder)
def embed_texts(texts: List[str]) -> List[List[float]]:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")
    url = "https://api.generative.googleapis.com/v1/embeddings:embed"
    headers = {"Authorization": f"Bearer {GEMINI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "gemini-embeddings-1.0", "input": texts}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    embeddings = [item['embedding'] for item in data['data']]
    return embeddings

# Gemini generator wrapper (used for KG extraction and answer generation)
def generate_from_gemini(prompt: str, max_tokens: int = 1024) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")
    url = "https://api.generative.googleapis.com/v1/answers:generate"
    headers = {"Authorization": f"Bearer {GEMINI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "gemini-2.5-flash",
        "prompt": prompt,
        "maxOutputTokens": max_tokens,
        "temperature": 0.0
    }
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    if 'candidates' in data:
        return data['candidates'][0].get('content', '')
    if 'output' in data:
        return "\n".join([o.get('content', '') for o in data['output']])
    return str(data)

# upsert vectors (same as before)
def upsert_documents(source: str, chunks: List[str]):
    embeddings = embed_texts(chunks)
    with client.batch as batch:
        batch.batch_size = 50
        for i, text in enumerate(chunks):
            vec = embeddings[i]
            chunk_id = f"{uuid.uuid4().hex}"
            obj = {
                "text": text,
                "source": source,
                "meta": "{}",
                "chunk_id": chunk_id
            }
            batch.add_data_object(obj, WEAVIATE_CLASS, chunk_id, vector=vec)
    logger.info(f"Upserted {len(chunks)} chunks from {source}")

# query vectors
def query_weaviate_vectors(query: str, top_k: int = 5):
    q_emb = embed_texts([query])[0]
    near_vector = {"vector": q_emb}
    res = (
        client.query
        .get(WEAVIATE_CLASS, ["text", "source", "chunk_id"])
        .with_near_vector(near_vector)
        .with_limit(top_k)
        .do()
    )
    hits = []
    for item in (res.get('data', {}).get('Get', {}).get(WEAVIATE_CLASS, []) or []):
        hits.append({"text": item.get('text'), "source": item.get('source'), "chunk_id": item.get('chunk_id')})
    return hits

# KG metadata check
def kg_metadata_exists(checksum: str) -> bool:
    res = client.query.get("KG_Metadata", ["checksum"]) \
        .with_where({"path": ["checksum"], "operator": "Equal", "valueString": checksum}) \
        .do()
    found = (res.get('data', {}).get('Get', {}).get('KG_Metadata') or [])
    return len(found) > 0

def upsert_kg_metadata(doc_name: str, checksum: str, status: str = "completed"):
    obj = {"doc_name": doc_name, "checksum": checksum, "extraction_status": status, "extraction_timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ')}
    client.data_object.create(obj, "KG_Metadata")

# KG upsert helpers
def upsert_entity(name: str, ent_type: str, canonical_id: str, aliases: List[str], source_chunks: List[str]):
    # create or update by canonical_id
    res = client.query.get("KG_Entity", ["canonical_id"]).with_where({"path": ["canonical_id"], "operator": "Equal", "valueString": canonical_id}).do()
    existing = (res.get('data', {}).get('Get', {}).get('KG_Entity') or [])
    if existing:
        # update source_chunks and aliases
        obj_id = existing[0]['_additional']['id']
        client.data_object.merge({"aliases": aliases, "source_chunks": source_chunks}, "KG_Entity", obj_id)
        return obj_id
    else:
        obj = {"name": name, "type": ent_type, "canonical_id": canonical_id, "aliases": aliases, "source_chunks": source_chunks}
        oid = client.data_object.create(obj, "KG_Entity")
        return oid

def upsert_relation(from_canonical: str, to_canonical: str, relation_type: str, evidence_chunks: List[str], confidence: float):
    # resolve entity ids
    from_res = client.query.get("KG_Entity", ["canonical_id"]).with_where({"path": ["canonical_id"], "operator": "Equal", "valueString": from_canonical}).do()
    to_res = client.query.get("KG_Entity", ["canonical_id"]).with_where({"path": ["canonical_id"], "operator": "Equal", "valueString": to_canonical}).do()
    from_objs = (from_res.get('data', {}).get('Get', {}).get('KG_Entity') or [])
    to_objs = (to_res.get('data', {}).get('Get', {}).get('KG_Entity') or [])
    if not from_objs or not to_objs:
        logger.warning(f"Cannot create relation, missing entities: {from_canonical} or {to_canonical}")
        return None
    from_id = from_objs[0]['_additional']['id']
    to_id = to_objs[0]['_additional']['id']
    relation_obj = {"relation_type": relation_type, "evidence_chunks": evidence_chunks, "confidence": confidence, "from": {"beacon": f"weaviate://localhost/{from_id}"}, "to": {"beacon": f"weaviate://localhost/{to_id}"}}
    oid = client.data_object.create(relation_obj, "KG_Relation")
    return oid

# KG extraction: call LLM to extract entities+relations in JSON
KG_ENTITY_EXTRACTION_PROMPT = """
You will receive a text block. Extract named entities and concepts into JSON with fields: id (suggested), name, type (PERSON/ORG/DATE/CONCEPT/LOCATION/OTHER), aliases (list), source_chunks (list of chunk_ids where this appears).
Output strictly as JSON: {"entities": [{...}], "relations": [{"from":"id","to":"id","type":"relation","evidence_chunks":[...],"confidence":0.0}]}
Do not include explanation text.
"""

async def extract_and_store_kg(doc_name: str, path: str):
    # compute checksum and skip if exists
    checksum = file_checksum(path)
    if kg_metadata_exists(checksum):
        logger.info("KG already exists for this document, skipping extraction.")
        return {"status": "skipped", "checksum": checksum}
    # extract text & chunks
    text = extract_text_from_pdf(path)
    chunks = chunk_text(text)
    # iterate in batches to control LLM cost
    batch_size = 100
    entities_index = {}
    relations_accum = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        # prepare prompt with batch and chunk ids
        payload = []
        for j, c in enumerate(batch):
            cid = f"{doc_name}_chunk_{i+j}"
            payload.append({"chunk_id": cid, "text": c})
        prompt_body = {"doc_name": doc_name, "chunks": payload}
        prompt = KG_ENTITY_EXTRACTION_PROMPT + "\nDATA:\n" + json.dumps(prompt_body)
        try:
            out = generate_from_gemini(prompt, max_tokens=1024)
            parsed = json.loads(out)
        except Exception as e:
            logger.exception("KG extraction LLM failed")
            continue
        # upsert entities
        for ent in parsed.get('entities', []):
            cid_list = ent.get('source_chunks', [])
            canonical = ent.get('canonical_id') or ent.get('id') or ent.get('name')
            eid = upsert_entity(ent.get('name'), ent.get('type', 'OTHER'), canonical, ent.get('aliases', []), cid_list)
            entities_index[canonical] = eid
        # collect relations
        for rel in parsed.get('relations', []):
            relations_accum.append(rel)
    # upsert relations after entities exist
    for rel in relations_accum:
        from_id = rel.get('from')
        to_id = rel.get('to')
        upsert_relation(from_id, to_id, rel.get('type'), rel.get('evidence_chunks', []), float(rel.get('confidence', 0.9)))
    # mark metadata
    upsert_kg_metadata(doc_name, checksum, status="completed")
    return {"status": "completed", "checksum": checksum}

# Session manager (simple in-memory)
SESSIONS: Dict[str, Dict[str, Any]] = {}

# API models
class IngestResponse(BaseModel):
    status: str
    inserted: int

class KGResponse(BaseModel):
    status: str
    checksum: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

# FastAPI app
app = FastAPI(title="RAG-KG-Service")

@app.on_event("startup")
async def startup_event():
    ensure_core_schema()

@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...), source: str = Form(None)):
    filename = file.filename
    dest = f"/tmp/{uuid.uuid4().hex}_{filename}"
    contents = await file.read()
    with open(dest, "wb") as f:
        f.write(contents)
    text = extract_text_from_pdf(dest)
    chunks = chunk_text(text)
    upsert_documents(source or filename, chunks)
    return {"status": "ok", "inserted": len(chunks)}

@app.post("/ingest/path", response_model=IngestResponse)
def ingest_path(path: str, source: str = None):
    text = extract_text_from_pdf(path)
    chunks = chunk_text(text)
    upsert_documents(source or path, chunks)
    return {"status": "ok", "inserted": len(chunks)}

@app.post("/ingest/kg-trigger", response_model=KGResponse)
async def kg_trigger(path: str = Form(...), doc_name: str = Form(...)):
    # asynchronous background extraction
    task = asyncio.create_task(extract_and_store_kg(doc_name, path))
    res = await task
    return {"status": res.get('status'), "checksum": res.get('checksum')}

@app.post("/session/start")
def session_start():
    sid = uuid.uuid4().hex
    SESSIONS[sid] = {"history": []}
    return {"session_id": sid}

@app.post("/session/{sid}/query", response_model=QueryResponse)
def session_query(sid: str, q: str, mode: str = Form('hybrid'), top_k: int = Form(5), max_hops: int = Form(2)):
    if sid not in SESSIONS:
        raise HTTPException(status_code=404, detail="session not found")
    # append to history
    SESSIONS[sid]['history'].append({"user": q})
    # hybrid retrieval
    # 1) graph seed entities
    # simple entity lookup by name
    graph_contexts = []
    try:
        gql = {
            "query": "{ Get { KG_Entity(where: {path:[\"name\"], operator: Equal, valueString: \"%s\"}) { name canonical_id source_chunks outgoing: KG_Relation { relation_type confidence to { name canonical_id source_chunks } evidence_chunks } } } }" % q
        }
        # use client.query.raw to run custom GraphQL
        raw = client.query.raw(gql['query'])
        # parse entities
        data = raw.get('data', {}).get('Get', {}).get('KG_Entity') or []
        for ent in data:
            # collect outgoing relations up to max_hops (simplified to depth 1 here)
            for rel in ent.get('outgoing', []) or []:
                graph_contexts.append({"path": f"{ent.get('name')} -> {rel.get('relation_type')} -> {rel.get('to', {}).get('name')}", "evidence": rel.get('evidence_chunks', [])})
    except Exception:
        graph_contexts = []
    # 2) vector retrieval
    vec_hits = query_weaviate_vectors(q, top_k=top_k)
    # 3) fuse
    contexts = []
    sources = []
    for g in graph_contexts:
        contexts.append(f"GRAPH: {g['path']} | evidence: {','.join(g['evidence'])}")
    for h in vec_hits:
        contexts.append(h['text'])
        sources.append(h['chunk_id'])
    context_block = "\n---\n".join(contexts[:10])
    prompt = f"You are an assistant. Use only the following contexts (GRAPH paths followed by text chunks).\nCONTEXTS:\n{context_block}\n\nQUESTION:\n{q}\n\nAnswer concisely and cite chunk ids. If unknown, say 'I don't know'."
    answer = generate_from_gemini(prompt)
    SESSIONS[sid]['history'].append({"assistant": answer})
    return {"answer": answer, "sources": sources}

@app.post("/session/{sid}/end")
def session_end(sid: str):
    if sid in SESSIONS:
        del SESSIONS[sid]
    return {"status": "ended"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("trying_Llama_Index:app", host="0.0.0.0", port=8000, reload=False)