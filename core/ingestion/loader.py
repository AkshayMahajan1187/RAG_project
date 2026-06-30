from pathlib import Path
from typing import List
from langchain_core.documents import Document
import fitz
import hashlib


def get_file_hash(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_pdf(file_path: str) -> List[Document]:
    docs = []
    file_hash = get_file_hash(file_path)
    with fitz.open(file_path) as pdf:
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text = page.get_text()
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source": file_path,
                        "filename": Path(file_path).name,
                        "page": page_num + 1,
                        "type": "pdf",
                        "file_hash": file_hash
                    }
                ))
    return docs


def load_txt(file_path: str) -> List[Document]:
    file_hash = get_file_hash(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return [Document(
        page_content=text,
        metadata={
            "source": file_path,
            "filename": Path(file_path).name,
            "page": 1,
            "type": "txt",
            "file_hash": file_hash
        }
    )]


def load_docx(file_path: str) -> List[Document]:
    from docx import Document as DocxDocument
    file_hash = get_file_hash(file_path)
    doc = DocxDocument(file_path)
    text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    return [Document(
        page_content=text,
        metadata={
            "source": file_path,
            "filename": Path(file_path).name,
            "page": 1,
            "type": "docx",
            "file_hash": file_hash
        }
    )]


LOADERS = {
    ".pdf": load_pdf,
    ".txt": load_txt,
    ".docx": load_docx
}


def load_document(file_path: str) -> List[Document]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()

    if ext not in LOADERS:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {list(LOADERS.keys())}")

    try:
        return LOADERS[ext](file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load {path.name}: {e}")