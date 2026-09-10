"""Turn a folder of reStructuredText docs into retrieval chunks.

Each chunk is one section (or a slice of a long section) with its heading path kept as
context, because support questions are usually answered by a single section and the
heading tells the model where it is in the product.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

HEADING_CHARS = set("=-~^\"'`#*+")
ADMONITIONS = ("note", "tip", "important", "warning", "seealso", "caution", "danger", "attention", "example", "exercise")


@dataclass
class Chunk:
    id: str
    doc: str
    title: str
    text: str

    @property
    def words(self) -> int:
        return len(self.text.split())


def clean_inline(line: str) -> str:
    """Strip RST roles and markup but keep the words a user would see."""
    line = re.sub(r":menuselection:`([^`]+)`", lambda m: m.group(1).replace("-->", ">"), line)
    line = re.sub(r":(?:ref|doc|term):`([^`<]+?)\s*<[^`]+>`", r"\1", line)
    line = re.sub(r":(?:ref|doc|term):`([^`]+)`", lambda m: m.group(1).split("/")[-1].replace("_", " "), line)
    line = re.sub(r":\w+:`([^`]+)`", r"\1", line)
    line = re.sub(r"`([^`<]+?)\s*<[^`]+>`_+", r"\1", line)
    line = re.sub(r"``([^`]+)``", r"\1", line)
    line = re.sub(r"`([^`]+)`", r"\1", line)
    line = re.sub(r"\*\*([^*]+)\*\*|\*([^*]+)\*", lambda m: m.group(1) or m.group(2), line)
    line = re.sub(r"\|[\w-]+\|", "", line)
    return re.sub(r"[ \t]{2,}", " ", line).rstrip()


def _is_underline(line: str) -> bool:
    s = line.strip()
    return len(s) >= 3 and len(set(s)) == 1 and s[0] in HEADING_CHARS


def parse_rst(raw: str) -> list[tuple[list[str], str]]:
    """Return (heading_path, body) for every section, in order."""
    lines = raw.splitlines()
    sections: list[tuple[list[str], list[str]]] = []
    levels: list[str] = []  # underline char per depth, in order of first appearance
    path: list[str] = []
    body: list[str] = []
    skip_indent: int | None = None
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        # overline + title + underline
        if _is_underline(line) and i + 2 < len(lines) and _is_underline(lines[i + 2]) and lines[i + 1].strip():
            line, nxt = lines[i + 1], lines[i + 2]
            i += 1
        if line.strip() and not line.startswith(" ") and _is_underline(nxt) and len(nxt.strip()) >= len(line.strip()) - 2:
            sections.append((path.copy(), body))
            ch = nxt.strip()[0]
            if ch not in levels:
                levels.append(ch)
            depth = levels.index(ch)
            path = path[:depth] + [clean_inline(line.strip())]
            body = []
            skip_indent = None
            i += 2
            continue
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if skip_indent is not None:
            if stripped and indent <= skip_indent:
                skip_indent = None
            else:
                i += 1
                continue
        m = re.match(r"\.\.\s+([\w-]+)::", stripped)
        if m:
            if m.group(1) not in ADMONITIONS:
                skip_indent = indent  # images, toctree, code, etc.: drop the whole block
            i += 1
            continue
        if stripped.startswith(".. ") or re.match(r":[\w-]+:", stripped) and indent > 0:
            i += 1
            continue
        body.append(stripped)
        i += 1
    sections.append((path.copy(), body))
    out = []
    for p, b in sections:
        text = re.sub(r"\n{3,}", "\n\n", "\n".join(b)).strip()
        # RST wraps paragraphs at ~100 columns: rejoin wrapped lines, keep list items apart
        text = re.sub(r"\n(?!\n|[-*+] |#\. |\d+\. )", " ", text)
        text = clean_inline(text)
        if text:
            out.append((p, text))
    return out


def chunk_document(doc: str, raw: str, min_words: int = 40, max_words: int = 380) -> list[Chunk]:
    pieces: list[tuple[str, str]] = []
    for heading, text in parse_rst(raw):
        title = " > ".join(heading) or doc
        paras = [p for p in text.split("\n\n") if p.strip()]
        buf: list[str] = []
        for p in paras:
            if buf and len(" ".join(buf + [p]).split()) > max_words:
                pieces.append((title, "\n\n".join(buf)))
                buf = []
            buf.append(p)
        if buf:
            pieces.append((title, "\n\n".join(buf)))
    # a tiny section reads better glued to the one after it
    merged: list[tuple[str, str]] = []
    carry: tuple[str, str] | None = None
    for title, text in pieces:
        if carry:
            title, text = carry[0], f"{carry[1]}\n\n{title.split(' > ')[-1]}\n{text}"
            carry = None
        if len(text.split()) < min_words:
            carry = (title, text)
            continue
        merged.append((title, text))
    if carry:
        if merged:
            t, x = merged[-1]
            merged[-1] = (t, f"{x}\n\n{carry[1]}")
        else:
            merged.append(carry)
    return [Chunk(id=f"{doc}#{n}", doc=doc, title=t, text=x) for n, (t, x) in enumerate(merged)]


def load_corpus(root: Path, only: list[str] | None = None) -> list[Chunk]:
    """Chunk every .rst under root. `only` keeps docs whose path starts with one of the prefixes."""
    root = Path(root)
    chunks: list[Chunk] = []
    for path in sorted(root.rglob("*.rst")):
        doc = path.relative_to(root).with_suffix("").as_posix()
        if only and not any(doc == p or doc.startswith(p.rstrip("/") + "/") for p in only):
            continue
        chunks.extend(chunk_document(doc, path.read_text(encoding="utf-8")))
    return chunks


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def read_chunks(path: Path) -> list[Chunk]:
    with open(path, encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]
