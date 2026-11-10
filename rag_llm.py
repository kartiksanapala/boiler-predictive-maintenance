from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "documents"
OUT = ROOT / "outputs" / "rag"
INDEX = OUT / "boiler_faiss.index"
CHUNKS = OUT / "chunks.json"
EPISODES = ROOT / "outputs" / "episodes_for_llm.csv"
EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"
LLM_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Cache the model for the lifetime of this Python process.
_LOCAL_MODEL = None
_LOCAL_TOKENIZER = None

def dependencies():
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise SystemExit("Install dependencies first: pip install sentence-transformers faiss-cpu") from e
    return faiss, SentenceTransformer

def extract_docs():
    from pypdf import PdfReader
    docs = []
    for p in sorted(DOCS.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() == ".pdf":
            try:
                for n, page in enumerate(PdfReader(str(p)).pages, 1):
                    t = (page.extract_text() or "").strip()
                    if t:
                        docs.append({"source": p.name, "page": n, "text": t})
            except Exception as e:
                print(f"Skipping {p.name}: {e}")
        elif p.suffix.lower() in {".txt", ".md"}:
            t = p.read_text(encoding="utf-8", errors="ignore").strip()
            if t:
                docs.append({"source": p.name, "page": None, "text": t})
    return docs

def chunk_docs(docs, size=220, overlap=40):
    chunks = []
    for d in docs:
        words = d["text"].split()
        for start in range(0, len(words), size - overlap):
            piece = " ".join(words[start:start + size]).strip()
            if len(piece) > 80:
                chunks.append({"source": d["source"], "page": d["page"], "text": piece})
    return chunks

def build():
    faiss, SentenceTransformer = dependencies()
    docs = extract_docs()
    if not docs:
        raise SystemExit("Add permitted PDF/TXT/MD maintenance references to ./documents before building.")
    chunks = chunk_docs(docs)
    if not chunks:
        raise SystemExit("No usable text chunks were extracted. Scanned PDFs may require OCR.")
    print("Loading embedding model...")
    enc = SentenceTransformer(EMBEDDING)
    vec = enc.encode([c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=True)
    index = faiss.IndexFlatIP(vec.shape[1])
    index.add(np.asarray(vec, dtype="float32"))
    OUT.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX))
    CHUNKS.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Indexed {len(chunks)} chunks from {len(docs)} document sections.")

def retrieve(query, top_k=5):
    faiss, SentenceTransformer = dependencies()
    if not INDEX.exists() or not CHUNKS.exists():
        raise SystemExit("Build the knowledge base first: python rag_llm.py build")
    enc = SentenceTransformer(EMBEDDING)
    q = enc.encode([query], normalize_embeddings=True)
    ix = faiss.read_index(str(INDEX))
    scores, ids = ix.search(np.asarray(q, dtype="float32"), top_k)
    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))
    results = []
    seen = set()
    for score, i in zip(scores[0], ids[0]):
        if i >= 0:
            c = chunks[int(i)].copy()
            # Duplicate PDFs or repeated chunks can enter the corpus under different filenames.
            fingerprint = " ".join(c["text"].lower().split())
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            c["similarity"] = float(score)
            results.append(c)
    return results

def episode_query(row):
    return (
        f"Boiler operating anomaly from {row.get('start', 'unknown')} to {row.get('end', 'unknown')}. "
        f"Duration: {row.get('duration_min', 'unknown')} minutes. "
        f"Top variables: {row.get('top_variables', 'not supplied')}. "
        f"Operating context: " +
        ", ".join(f"{k}={v}" for k, v in row.items() if str(k).startswith("ctx_"))
    )

def load_local_llm():
    global _LOCAL_MODEL, _LOCAL_TOKENIZER
    if _LOCAL_MODEL is not None:
        return _LOCAL_TOKENIZER, _LOCAL_MODEL

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading local model: {LLM_NAME}")
    print("The first download may take a few minutes.")
    _LOCAL_TOKENIZER = AutoTokenizer.from_pretrained(LLM_NAME)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    _LOCAL_MODEL = AutoModelForCausalLM.from_pretrained(
        LLM_NAME,
        torch_dtype=dtype,
        device_map="auto"
    )
    _LOCAL_MODEL.eval()
    return _LOCAL_TOKENIZER, _LOCAL_MODEL

def ask_llm(query, passages, episode=None):
    import torch
    tokenizer, model = load_local_llm()

    evidence = "\n\n".join(
        f"[{i+1}] Source: {p['source']}, page: {p.get('page')}, "
        f"retrieval similarity: {p['similarity']:.3f}\n{p['text']}"
        for i, p in enumerate(passages)
    )

    citation_key = "\\n".join(
        f"[{i+1}] {p['source']} | page {p.get('page') if p.get('page') is not None else 'not specified'}"
        for i, p in enumerate(passages)
    )

    system_prompt = (
        "You are an academically cautious assistant supporting review of boiler sensor anomalies. "
        "The sensor tags in this task have NOT been verified against an authoritative tag dictionary. "
        "Therefore, NEVER infer or name their physical measurements, equipment locations, units, "
        "or process meanings from their codes. Refer to each variable only by its exact tag. "
        "The only observed facts are the exact tags, high/low direction, z-scores, timestamps, "
        "duration, and context values explicitly supplied in the episode summary. Do not invent "
        "additional readings or operating conditions. Do not diagnose a fault or assert a cause. "
        "If the physical meaning of a tag is unknown, explicitly say it is unverified and that "
        "physical interpretation is not possible from the supplied data. Retrieved passages are "
        "general documentary context, not proof that their described conditions occurred in this episode. "
        "Do not convert a general maintenance checklist into a claim that a particular alarm, leak, "
        "water-quality issue, or equipment problem exists. Do not recommend control changes, setpoint "
        "changes, or bypassing safety procedures. "
        "CITATION RULE: every factual statement about a retrieved document must include at least one "
        "valid bracketed passage citation such as [1] or [2]. Use ONLY passage IDs in the citation key. "
        "Never write document page numbers in the report. If no passage directly supports a statement, "
        "do not make that statement. The sensor observations from the episode summary do not need citations."
    )
    user_prompt = f"""Write a concise academic maintenance-support report using exactly these headings:

**1. Observed data evidence**
List the supplied sensor tags, direction and z-scores exactly. State that the tags' physical meanings are unverified.

**2. Possible interpretations**
Do NOT assign physical meanings or causes to the tags. State that no physical interpretation or root cause can be established without a verified tag dictionary and engineering context. Do not invent hypotheses based on tag names.

**3. Relevant document evidence**
Summarise only what the retrieved passages literally support. Cite each document-based statement using [1], [2], etc. Do not claim the document proves that any condition occurred in this episode. If passages are generic or weakly relevant, say so.

**4. Inspection questions for qualified personnel**
Give only general questions explicitly supported by the retrieved maintenance passages. Do not imply that a fault is present. Cite each question or its supporting checklist statement. If no specific check is justified, say so.

**5. Uncertainty and human-review note**
State that this is an anomaly flag, not a confirmed fault; tag meanings and operating limits are unverified; and qualified plant personnel must interpret the readings using plant documentation.

Episode summary:
{query}

Authoritative citation key (the only permitted citation IDs):
{citation_key}

Retrieved document passages:
{evidence}

Important: Never infer that a sensor is an oxygen, pressure, temperature, flow, exhaust, or air-intake measurement unless the episode summary explicitly verifies that meaning. Never state that a leak, low-water event, contamination, combustion problem, or equipment fault exists based only on general manual text. If evidence is insufficient, say so. Keep the report complete and concise.
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=6000)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=900,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def generate_episode(episode_id):
    if not EPISODES.exists():
        raise SystemExit(f"Missing {EPISODES}; run boiler_pipeline.py --stage scenarios first.")
    df = pd.read_csv(EPISODES)
    row = df.loc[df["episode_id"].astype(str) == str(episode_id)]
    if row.empty:
        raise SystemExit(f"Episode {episode_id} not found.")
    r = row.iloc[0]
    cls = str(r.get("episode_class", ""))
    if cls == "injected_missed" or (
        str(r.get("kind")) == "injected" and not bool(r.get("detected"))
    ):
        raise SystemExit(
            "This injected event was missed by the detector. Keep it in benchmark evaluation; "
            "do not present it as a detected alert."
        )
    if str(r.get("kind")) == "injected":
        print("NOTICE: This is a synthetic injected event, not a historical plant fault.")

    q = episode_query(r)
    print("Retrieving relevant manual passages...")
    passages = retrieve(q, 5)
    print("Generating report with local Qwen model (no API key required)...")
    answer = ask_llm(q, passages, r)
    # Programmatic citation audit: checks whether the report uses only supplied
    # passage IDs. This cannot prove semantic correctness; it flags invalid IDs.
    import re
    cited_ids = sorted({int(x) for x in re.findall(r"\[(\d+)\]", answer)})
    valid_ids = list(range(1, len(passages) + 1))
    invalid_ids = [x for x in cited_ids if x not in valid_ids]
    citation_audit = {
        "cited_passage_ids": cited_ids,
        "valid_passage_ids": valid_ids,
        "invalid_passage_ids": invalid_ids,
        "status": (
            "no passage citations generated; revise report"
            if not cited_ids else
            "invalid passage IDs found; review report"
            if invalid_ids else
            "passage IDs are valid; human relevance review still required"
        ),
        "note": "This checks citation numbering only. It does not verify that each cited passage semantically supports the associated claim."
    }
    source_key = [
        {
            "citation_id": i + 1,
            "source": p["source"],
            "page": p.get("page"),
            "similarity": p.get("similarity")
        }
        for i, p in enumerate(passages)
    ]
    result = {
        "episode_id": str(episode_id),
        "episode_class": cls,
        "query": q,
        "retrieved": passages,
        "citation_key": source_key,
        "citation_audit": citation_audit,
        "llm_report": answer,
        "llm_model": LLM_NAME,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"report_{episode_id}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + answer)
    print("\nCitation audit:", json.dumps(citation_audit, ensure_ascii=False))
    print(f"\nSaved {path}")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    q = sub.add_parser("query")
    q.add_argument("--text", required=True)
    q.add_argument("--top-k", type=int, default=5)
    e = sub.add_parser("episode")
    e.add_argument("--id", required=True)
    a = ap.parse_args()

    if a.cmd == "build":
        build()
    elif a.cmd == "query":
        for i, p in enumerate(retrieve(a.text, a.top_k), 1):
            print(f"\n[{i}] {p['source']} page={p.get('page')} "
                  f"score={p['similarity']:.3f}\n{p['text']}")
    else:
        generate_episode(a.id)

if __name__ == "__main__":
    main()
