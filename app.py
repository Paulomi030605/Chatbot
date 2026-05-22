"""
Knowledge Assistant Pro
Hugging Face Spaces - Professional RAG Output
Optimized for Qwen2.5 1.5B Instruct GGUF
"""

import os
import json
import time
import urllib.request
import hashlib
import re
import shutil
from pathlib import Path
from datetime import datetime
from typing import List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_PATH = os.path.join(BASE_DIR, "docs")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
MODELS_PATH = os.path.join(BASE_DIR, "models")
LOGS_PATH = os.path.join(BASE_DIR, "logs")

for path in [DOCS_PATH, CHROMA_PATH, MODELS_PATH, LOGS_PATH]:
    os.makedirs(path, exist_ok=True)


def download_if_missing(filename: str, url: str):
    path = os.path.join(MODELS_PATH, filename)
    if not os.path.exists(path):
        print(f"📥 Downloading {filename}...")
        urllib.request.urlretrieve(url, path)
        size = os.path.getsize(path) / (1024 * 1024)
        print(f"✅ Downloaded {filename}: {size:.1f} MB")
    else:
        print(f"✅ {filename} already exists")


print("📦 Checking models...")

download_if_missing(
    "tokenizer.json",
    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json"
)

download_if_missing(
    "model.onnx",
    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"
)

download_if_missing(
    "qwen2.5-1.5b-instruct-q4.gguf",
    "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
)


import numpy as np
from tokenizers import Tokenizer
import onnxruntime as ort
import chromadb
from chromadb.config import Settings
from llama_cpp import Llama
import gradio as gr


class Embedder:
    def __init__(self):
        self.tokenizer = Tokenizer.from_file(os.path.join(MODELS_PATH, "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=512)
        self.tokenizer.enable_padding(
            direction="right",
            pad_id=0,
            pad_type_id=0,
            pad_token=""
        )

        self.session = ort.InferenceSession(
            os.path.join(MODELS_PATH, "model.onnx"),
            providers=["CPUExecutionProvider"]
        )

    def embed(self, text: str) -> np.ndarray:
        encoding = self.tokenizer.encode(text)

        input_ids = np.array([encoding.ids], dtype=np.int64)
        attention_mask = np.array([encoding.attention_mask], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

        outputs = self.session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        })

        token_embeddings = outputs[0][0]
        mask = attention_mask[0].astype(float)
        mask_sum = mask.sum()

        if mask_sum > 0:
            return (token_embeddings * mask[:, np.newaxis]).sum(axis=0) / mask_sum

        return token_embeddings.mean(axis=0)


class VectorDB:
    def __init__(self):
        self.client = chromadb.Client(Settings(
            persist_directory=CHROMA_PATH,
            is_persistent=True,
            anonymized_telemetry=False
        ))
        self.collection = self.client.get_or_create_collection(name="docs")
        self.embedder = Embedder()

    def reset_collection(self):
        """Clear and recreate the Chroma collection before rebuilding the docs index."""
        try:
            self.client.delete_collection(name="docs")
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(name="docs")

    def build_index(self, force_rebuild: bool = False):
        import glob

        if force_rebuild:
            self.reset_collection()
        elif self.collection.count() > 0:
            return self.collection.count()

        documents = []
        metadatas = []
        ids = []

        for filepath in glob.glob(f"{DOCS_PATH}/**/*.md", recursive=True):
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            filename = os.path.basename(filepath)
            file_hash = hashlib.md5(filepath.encode("utf-8")).hexdigest()[:10]

            for chunk_no, i in enumerate(range(0, len(content), 700)):
                chunk = content[i:i + 900]

                if len(chunk.strip()) > 100:
                    documents.append(chunk)
                    metadatas.append({"source": filename, "path": filepath})
                    ids.append(f"{file_hash}_{chunk_no}")

        if not documents:
            return 0

        embeddings = [self.embedder.embed(doc) for doc in documents]

        for i in range(0, len(documents), 100):
            end = min(i + 100, len(documents))
            self.collection.add(
                ids=ids[i:end],
                documents=documents[i:end],
                embeddings=np.array(embeddings[i:end]).tolist(),
                metadatas=metadatas[i:end]
            )

        return len(documents)

    def search(self, query: str, top_k: int = 30):
        count = self.collection.count()

        if count == 0:
            return {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]]
            }

        query_emb = self.embedder.embed(query).reshape(1, -1)

        return self.collection.query(
            query_embeddings=query_emb.tolist(),
            n_results=min(top_k, count)
        )


class IntentDetector:
    @staticmethod
    def detect(query: str) -> str:
        q = query.lower()

        if any(x in q for x in ["list", "names", "all", "extract", "full names"]):
            return "extraction"

        if any(x in q for x in ["insight", "insights", "problem statement", "context", "source"]):
            return "insights"

        if any(x in q for x in ["compare", "difference", "vs", "versus"]):
            return "compare"

        if any(x in q for x in ["step", "how to", "process", "guide"]):
            return "steps"

        if any(x in q for x in ["summary", "summarize", "overview", "key points"]):
            return "summary"

        if any(x in q for x in ["code", "python", "script", "function"]):
            return "code"

        return "answer"


class PromptBuilder:
    @staticmethod
    def build(query: str, context: str, intent: str) -> str:
        return f"""
You are a strict document QA assistant.

Use ONLY the CONTEXT below.
Do not guess.
Do not use outside knowledge.
Do not infer missing details.
Do not repeat yourself.
Do not add explanations that were not asked.

Answer style rules:
- Answer exactly what the user asked.
- For a name-only question, return only the name.
- For a simple factual question, return one short sentence.
- For a list request, return only the list.
- For summary/details/explanation, use maximum 3 short paragraphs.
- If the answer is not clearly present in CONTEXT, say exactly:
The uploaded documents do not contain this information.

QUESTION:
{query}

CONTEXT:
{context}

ANSWER:
"""

class Formatter:
    @staticmethod
    def clean(text: str) -> str:
        text = text.strip()
        text = re.sub(r"^(ANSWER:\s*)+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", text)
        return text.strip()

    @staticmethod
    def add_sources(answer: str, sources: List[str]) -> str:
        clean_sources = []
        for s in sources:
            if s not in clean_sources:
                clean_sources.append(s)

        source_text = "\n".join([f"- 📄 {s}" for s in clean_sources[:8]])

        return f"""{answer}

---

## 📚 Sources
{source_text if source_text else "- No relevant source found"}

---
*Generated from your document knowledge base*
"""


class Generator:
    def __init__(self):
        self.llm = Llama(
            model_path=os.path.join(MODELS_PATH, "qwen2.5-1.5b-instruct-q4.gguf"),
            n_ctx=2048,
            verbose=False,
            n_gpu_layers=0,
            n_threads=max(2, os.cpu_count() or 2),
            n_batch=128
        )
        self.intent_detector = IntentDetector()

    @staticmethod
    def is_simple_query(question: str) -> bool:
        q = question.lower().strip()
        simple_patterns = [
            "name", "candidate name", "name of candidate", "name of the candidate",
            "who is", "what is", "email", "phone", "mobile", "location", "designation",
            "role", "company", "college", "degree"
        ]
        if len(q.split()) <= 8 and any(p in q for p in simple_patterns):
            return True
        return False

    @staticmethod
    def extract_name_from_context(question: str, context: str):
        """Deterministic answer for common resume-style name questions."""
        q = question.lower()
        if "name" not in q and "candidate" not in q and "who is" not in q:
            return None

        patterns = [
            r"(?im)^\s*(?:candidate\s*name|name)\s*[:\-]\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,3})\b",
            r"(?im)^\s*#\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,3})\b",
            r"(?im)^\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,3})\s*$",
        ]
        blocked = {
            "Original file", "Converted on", "Professional Experience", "Technical Skills",
            "Work Experience", "Project Experience", "Contact Information", "Uploaded Document"
        }
        for pattern in patterns:
            for match in re.finditer(pattern, context):
                name = match.group(1).strip()
                if name not in blocked and not any(x in name.lower() for x in ["source", "resume", "page"]):
                    return name

        # Fallback: infer from resume filename/source such as Pallab_Resume.md only when context also contains that name.
        source_match = re.search(r"\[SOURCE:\s*([^\]]+)\]", context)
        if source_match:
            source = os.path.splitext(source_match.group(1))[0]
            source = re.sub(r"(?i)(resume|cv|profile|updated|final|copy|pdf|docx|md)", " ", source)
            source = re.sub(r"[^A-Za-z]+", " ", source).strip()
            parts = [p for p in source.split() if len(p) > 1]
            if len(parts) >= 2:
                return " ".join(p.capitalize() for p in parts[:3])
        return None

    def filter_sources(self, question: str, sources):
        q = question.lower()
        keywords = [w for w in re.findall(r"[a-zA-Z0-9]{3,}", q)]
        stopwords = {"what", "who", "the", "and", "with", "from", "give", "tell", "show", "candidate", "name"}
        keywords = [kw for kw in keywords if kw not in stopwords]

        filtered = []
        for doc, meta, dist in sources:
            text = (doc or "").lower()
            source = meta.get("source", "").lower()
            score = 0

            for kw in keywords:
                if kw in text:
                    score += 3
                if kw in source:
                    score += 2

            # Chroma returns smaller distance for closer results. Keep that signal.
            try:
                score += max(0, 2 - float(dist))
            except Exception:
                pass

            if score > 0 or not keywords:
                filtered.append((score, doc, meta, dist))

        if filtered:
            filtered.sort(key=lambda x: x[0], reverse=True)
            return [(doc, meta, dist) for score, doc, meta, dist in filtered[:5]]

        return sources[:5]

    def build_context(self, sources, max_chars: int = 1200):
        context_parts = []
        source_names = []
        total_chars = 0

        for doc, meta, dist in sources[:5]:
            source = meta.get("source", "Unknown source")
            if source not in source_names:
                source_names.append(source)

            chunk = (doc or "").strip()
            if len(chunk) > 500:
                chunk = chunk[:500]

            block = f"[SOURCE: {source}]\n{chunk}"
            if total_chars + len(block) <= max_chars:
                context_parts.append(block)
                total_chars += len(block)
            else:
                remaining = max_chars - total_chars
                if remaining > 250:
                    context_parts.append(block[:remaining])
                break

        return "\n\n".join(context_parts), source_names

    def generate(self, question: str, sources):
        intent = self.intent_detector.detect(question)
        print(f"🎯 Intent: {intent}")

        sources = self.filter_sources(question, sources)
        context, source_names = self.build_context(sources)

        if not context.strip():
            return "The uploaded documents do not contain this information.", intent, []

        direct_name = self.extract_name_from_context(question, context)
        if direct_name:
            return direct_name, intent, source_names

        prompt = PromptBuilder.build(question, context, intent)

        response = self.llm(
            prompt,
            max_tokens=80,
            temperature=0.0,
            top_p=0.2,
            repeat_penalty=1.4,
            stop=["QUESTION:", "CONTEXT:", "User question:", "\n\n\n"]
        )

        raw_answer = response["choices"][0]["text"]
        clean_answer = Formatter.clean(raw_answer)

        if not clean_answer or clean_answer.lower() in {"none", "unknown", "not found"}:
            clean_answer = "The uploaded documents do not contain this information."

        if self.is_simple_query(question):
            return clean_answer, intent, source_names

        return clean_answer, intent, source_names

class ChatLogger:
    def __init__(self):
        self.sessions_file = os.path.join(LOGS_PATH, "sessions.jsonl")
        self.messages_file = os.path.join(LOGS_PATH, "messages.jsonl")

    def log_session(self, session_id: str, first_question: str, intent: str):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "first_question": first_question[:200],
            "intent": intent
        }
        with open(self.sessions_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def log_message(self, session_id: str, question: str, answer: str, sources: List[str], response_time: float, intent: str):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "question": question[:500],
            "answer": answer[:1000],
            "sources": sources,
            "response_time": round(response_time, 2),
            "intent": intent
        }
        with open(self.messages_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def get_stats(self):
        messages = []

        if os.path.exists(self.messages_file):
            with open(self.messages_file, "r", encoding="utf-8") as f:
                messages = [json.loads(line) for line in f if line.strip()]

        intent_counts = {}
        for m in messages:
            intent = m.get("intent", "unknown")
            intent_counts[intent] = intent_counts.get(intent, 0) + 1

        return {
            "total_messages": len(messages),
            "intent_distribution": intent_counts,
            "recent": messages[-5:] if messages else []
        }

    def export_logs(self):
        sessions = []
        messages = []

        if os.path.exists(self.sessions_file):
            with open(self.sessions_file, "r", encoding="utf-8") as f:
                sessions = [json.loads(line) for line in f if line.strip()]

        if os.path.exists(self.messages_file):
            with open(self.messages_file, "r", encoding="utf-8") as f:
                messages = [json.loads(line) for line in f if line.strip()]

        return json.dumps({
            "export_time": datetime.now().isoformat(),
            "sessions": sessions,
            "messages": messages
        }, indent=2)


print("=" * 60)
print("🚀 Knowledge Assistant Pro")
print("Optimized for Qwen2.5 1.5B Instruct")
print("=" * 60)

db = VectorDB()
generator = Generator()
logger = ChatLogger()

chunk_count = db.build_index()
print(f"✅ Indexed {chunk_count} chunks")


DOCS_CACHE = {
    "text": "",
    "source_files": [],
    "loaded_at": 0.0
}


def refresh_docs_cache():
    """Load all Markdown docs once into memory for instant direct answers."""
    texts = []
    source_files = []

    for path in Path(DOCS_PATH).glob("*.md"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                texts.append(f"\n\n[SOURCE: {path.name}]\n{text}")
                source_files.append(path.name)
        except Exception:
            pass

    DOCS_CACHE["text"] = "\n".join(texts)
    DOCS_CACHE["source_files"] = source_files
    DOCS_CACHE["loaded_at"] = time.time()


def get_docs_text():
    """Return cached docs text. Refresh only if cache is empty."""
    if not DOCS_CACHE["text"]:
        refresh_docs_cache()
    return DOCS_CACHE["text"], DOCS_CACHE["source_files"]


def clean_extracted_value(value: str) -> str:
    value = re.sub(r"\s+", " ", value or "").strip(" :-|")
    value = re.sub(r"(?i)\s*(email|phone|mobile|contact|location)\s*:.*$", "", value).strip()
    return value


def extract_section(text: str, headings):
    heading_pattern = "|".join([re.escape(h) for h in headings])
    stop_headings = (
        "experience|work experience|professional experience|education|projects|project experience|"
        "certifications|achievements|summary|profile|objective|contact|personal information|"
        "technical skills|core skills|skills|tools|languages"
    )

    pattern = rf"(?is)(?:^|\n)\s*(?:#+\s*)?(?:{heading_pattern})\s*[:\-]?\s*(.*?)(?=\n\s*(?:#+\s*)?(?:{stop_headings})\s*[:\-]?\s*\n|\Z)"
    match = re.search(pattern, text)
    if not match:
        return None

    value = match.group(1).strip()
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip(" :-")


def direct_fast_answer(message):
    """
    Fast deterministic answers for common resume/document fields.
    This skips Chroma and skips the LLM, so it is much faster and reduces hallucination.
    """
    q = message.lower().strip()
    all_text, source_files = get_docs_text()
    not_found = "The uploaded documents do not contain this information."

    if not all_text.strip():
        return not_found

    if q in ["file", "file name", "filename", "name of the file", "document", "document name"] or (
        "file" in q and "name" in q
    ):
        return source_files[-1] if source_files else not_found

    if ("candidate" in q and "name" in q) or q in ["name", "candidate name", "name of candidate", "name of the candidate"]:
        patterns = [
            r"(?im)^\s*(?:candidate\s*name|name)\s*[:\-]\s*([A-Z][A-Za-z .'-]{2,80})\s*$",
            r"(?im)^\s*#\s*([A-Z][A-Za-z .'-]{2,80})\s*$",
            r"(?im)^\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,3})\s*$",
        ]

        blocked = {
            "Core Skills", "Technical Skills", "Professional Experience", "Work Experience",
            "Education", "Projects", "Project Experience", "Certifications", "Achievements",
            "Summary", "Profile", "Contact", "Contact Information", "Original File", "Converted On"
        }

        for pattern in patterns:
            for match in re.finditer(pattern, all_text):
                name = clean_extracted_value(match.group(1))
                if name and name not in blocked and not any(x in name.lower() for x in ["resume", "curriculum", "source"]):
                    return name

        if source_files:
            source = Path(source_files[-1]).stem
            source = re.sub(r"(?i)(resume|cv|profile|updated|final|copy|pdf|docx|md)", " ", source)
            source = re.sub(r"[^A-Za-z]+", " ", source).strip()
            parts = [p.capitalize() for p in source.split() if len(p) > 1]
            if len(parts) >= 2:
                return " ".join(parts[:3])

        return not_found

    if q in ["email", "email id", "mail", "mail id"] or "email" in q:
        match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", all_text)
        return match.group(0) if match else not_found

    if any(x in q for x in ["phone", "mobile", "contact number", "phone number", "mobile number"]):
        match = re.search(r"(\+?\d[\d\s\-()]{8,}\d)", all_text)
        return clean_extracted_value(match.group(1)) if match else not_found

    if q in ["location", "address", "current location"] or ("location" in q and len(q.split()) <= 6):
        patterns = [
            r"(?im)^\s*(?:location|address)\s*[:\-]\s*(.+)$",
            r"(?im)\b(?:based in|located in)\s+([A-Z][A-Za-z ,.-]{2,80})",
        ]
        for pattern in patterns:
            match = re.search(pattern, all_text)
            if match:
                return clean_extracted_value(match.group(1))
        return not_found

    if "skill" in q:
        skills = extract_section(all_text, ["core skills", "technical skills", "skills"])
        if skills:
            skills = re.sub(r"\n+", ", ", skills)
            skills = re.sub(r"\s*,\s*", ", ", skills)
            return clean_extracted_value(skills)[:900]
        return not_found

    return None


def chat_response(message, history, depth):
    start = time.time()
    session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]

    intent = generator.intent_detector.detect(message)
    logger.log_session(session_id, message, intent)

    direct_answer = direct_fast_answer(message)
    if direct_answer:
        elapsed = time.time() - start
        logger.log_message(session_id, message, direct_answer, [], elapsed, intent)
        return direct_answer

    results = db.search(message, top_k=3)

    sources_list = list(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ))

    answer, detected_intent, source_files = generator.generate(message, sources_list)

    elapsed = time.time() - start
    logger.log_message(session_id, message, answer, source_files, elapsed, detected_intent)

    return answer


def get_stats():
    return logger.get_stats()


def export_logs():
    return logger.export_logs()

ALLOWED_UPLOAD_EXTENSIONS = {
    ".md", ".markdown", ".txt", ".csv", ".json", ".py", ".js", ".html", ".css",
    ".pdf", ".docx"
}


def safe_filename(filename: str) -> str:
    """Return a simple safe filename while preserving the extension."""
    name = os.path.basename(filename or "uploaded_file")
    stem, ext = os.path.splitext(name)
    stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", stem).strip("._-") or "uploaded_file"
    return f"{stem}{ext.lower()}"


def unique_docs_path(filename: str) -> str:
    """Avoid overwriting existing Markdown files in docs/."""
    target = Path(DOCS_PATH) / filename
    if not target.exists():
        return str(target)

    stem = target.stem
    suffix = target.suffix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(target.with_name(f"{stem}_{timestamp}{suffix}"))


def read_pdf_text(filepath: str) -> str:
    """Extract text from PDF if pypdf or PyPDF2 is available."""
    try:
        from pypdf import PdfReader
    except Exception:
        try:
            from PyPDF2 import PdfReader
        except Exception:
            return "PDF text extraction requires pypdf or PyPDF2. Add one of them to requirements.txt."

    reader = PdfReader(filepath)
    pages = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"\n\n## Page {page_no}\n\n{text.strip()}")
    return "".join(pages).strip()


def read_docx_text(filepath: str) -> str:
    """Extract text from DOCX if python-docx is available."""
    try:
        import docx
    except Exception:
        return "DOCX text extraction requires python-docx. Add python-docx to requirements.txt."

    document = docx.Document(filepath)
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def convert_file_to_markdown(uploaded_file) -> str:
    """Convert an uploaded file to Markdown, save it into docs/, and return the saved path."""
    source_path = getattr(uploaded_file, "name", None) or str(uploaded_file)
    original_name = safe_filename(os.path.basename(source_path))
    stem, ext = os.path.splitext(original_name)

    if ext.lower() not in ALLOWED_UPLOAD_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {', '.join(sorted(ALLOWED_UPLOAD_EXTENSIONS))}")

    md_filename = f"{stem}.md"
    md_path = unique_docs_path(md_filename)

    if ext.lower() in [".md", ".markdown"]:
        shutil.copyfile(source_path, md_path)
        return md_path

    if ext.lower() in [".txt", ".csv", ".json", ".py", ".js", ".html", ".css"]:
        with open(source_path, "r", encoding="utf-8", errors="ignore") as f:
            body = f.read()
    elif ext.lower() == ".pdf":
        body = read_pdf_text(source_path)
    elif ext.lower() == ".docx":
        body = read_docx_text(source_path)
    else:
        body = ""

    markdown = f"""# {stem}

**Original file:** `{original_name}`  
**Converted on:** {datetime.now().isoformat(timespec="seconds")}

---

{body.strip()}
"""

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    return md_path


def upload_files_to_docs(files):
    """Gradio callback: convert uploaded files to Markdown and rebuild the vector index."""
    if not files:
        return "Please upload at least one file."

    converted = []
    errors = []

    for file_obj in files:
        try:
            md_path = convert_file_to_markdown(file_obj)
            converted.append(os.path.basename(md_path))
        except Exception as exc:
            source_name = os.path.basename(getattr(file_obj, "name", str(file_obj)))
            errors.append(f"{source_name}: {exc}")

    chunk_total = db.build_index(force_rebuild=True)
    refresh_docs_cache()

    lines = []
    if converted:
        lines.append("✅ Converted and saved to docs/:")
        lines.extend([f"- {name}" for name in converted])
    if errors:
        lines.append("\n⚠️ Some files were not converted:")
        lines.extend([f"- {err}" for err in errors])
    lines.append(f"\n🔄 Vector index rebuilt with {chunk_total} chunks.")

    return "\n".join(lines)


with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Knowledge Assistant Pro")
    gr.Markdown("Professional document-grounded assistant for insights, extraction, summaries, and source-backed answers.")

    with gr.Row():
        with gr.Column(scale=3):
            gr.ChatInterface(
                fn=chat_response,
                additional_inputs=[
                    gr.Dropdown(
                        choices=["Quick", "Standard", "Deep", "Maximum"],
                        value="Standard",
                        label="Response Depth"
                    )
                ],
                examples=[
                    ["Name of the Candidate", "Standard"],
                ],
                cache_examples=False
            )

        with gr.Column(scale=1):
            gr.Markdown("### 📁 Upload Documents")
            gr.Markdown("Upload files here. They are converted to `.md`, saved in `docs/`, and added to the chatbot knowledge base.")

            file_upload = gr.File(
                label="Upload files",
                file_count="multiple",
                type="filepath"
            )
            upload_btn = gr.Button("➕ Convert & Add to Docs")
            upload_status = gr.Markdown()
            upload_btn.click(upload_files_to_docs, inputs=file_upload, outputs=upload_status)

            gr.Markdown("### 📊 Admin")

            stats_btn = gr.Button("🔄 Refresh Stats")
            stats_output = gr.JSON(label="Stats")
            stats_btn.click(get_stats, outputs=stats_output)

            export_btn = gr.Button("📥 Export Logs")
            export_output = gr.Textbox(label="Logs JSON", lines=10)
            export_btn.click(export_logs, outputs=export_output)

            gr.Markdown("""
---
### Output Improvements
- User file uploads beside chatbot interface
- Uploaded files auto-convert to Markdown
- Converted files saved in `docs/`
- Index rebuilds after upload
- Strict document grounding
- Source-backed answers
""")


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False
    )
