from dotenv import load_dotenv
load_dotenv()

import PyPDF2
import google.generativeai as genai
import numpy as np
import os

EMBED_MODEL = "models/text-embedding-004"
LLM_MODEL = "models/gemini-2.5-flash"

genai.configure(api_key=os.getenv("LLM_API_KEY"))

model = genai.GenerativeModel(LLM_MODEL)

def load_pdf(path):
    reader = PyPDF2.PdfReader(path)
    for page in reader.pages:
        t = page.extract_text()
        if t:
            yield t

def embed(text):
    r = genai.embed_content(model=EMBED_MODEL, content=text)
    return np.array(r["embedding"])

def chunk(text_iter, max_chars=2000):
    buf = []
    length = 0
    for text in text_iter:
        words = text.split()
        for w in words:
            buf.append(w)
            length += len(w) + 1
            if length >= max_chars:
                yield " ".join(buf)
                buf = []
                length = 0
    if buf:
        yield " ".join(buf)

def search(query_emb, chunks, chunk_embs):
    scores = [np.dot(query_emb, e) for e in chunk_embs]
    idx = int(np.argmax(scores))
    return chunks[idx]

def ask(context, query):
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer in one clear paragraph:"
    r = model.generate_content(prompt)
    return r.text

PDF_PATH = "/Users/apple/Downloads/Downloadreceipt.pdf"

text_iter = load_pdf(PDF_PATH)
chunks = list(chunk(text_iter))
chunk_embs = [embed(c) for c in chunks]

while True:
    query = input("Query (or type 'exit'): ")
    if query.lower() == "exit":
        break

    query_emb = embed(query)
    best = search(query_emb, chunks, chunk_embs)
    answer = ask(best, query)

    print("\nAnswer:")
    words = answer.split()
    for i in range(0, len(words), 8):
        print(" ".join(words[i:i+8]))
    print()