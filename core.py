"""Offline-friendly text preprocessing engine for the virtual laboratory."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from math import sqrt
import re
import unicodedata


DEFAULT_TEXT = (
    "Knowledge graphs are connecting entities, and the systems were indexing "
    "documents efficiently. Researchers don't keep noisy TEXT; they normalize it!"
)

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "for",
    "from", "had", "has", "have", "he", "her", "hers", "him", "his", "i", "in",
    "is", "it", "its", "me", "my", "of", "on", "or", "our", "ours", "she", "that",
    "the", "their", "theirs", "them", "they", "this", "those", "to", "was", "we",
    "were", "will", "with", "you", "your", "yours",
}

CONTRACTIONS = {
    "can't": "cannot", "couldn't": "could not", "didn't": "did not",
    "doesn't": "does not", "don't": "do not", "hadn't": "had not",
    "hasn't": "has not", "haven't": "have not", "isn't": "is not",
    "shouldn't": "should not", "wasn't": "was not", "weren't": "were not",
    "won't": "will not", "wouldn't": "would not", "i'm": "i am",
    "it's": "it is", "they're": "they are", "we're": "we are", "you're": "you are",
}

IRREGULAR_LEMMAS = {
    "am": "be", "are": "be", "is": "be", "was": "be", "were": "be",
    "been": "be", "children": "child", "feet": "foot", "geese": "goose",
    "men": "man", "mice": "mouse", "teeth": "tooth", "women": "woman",
    "better": "good", "best": "good", "worse": "bad", "worst": "bad",
    "went": "go", "gone": "go", "ran": "run", "written": "write",
}


@dataclass(frozen=True)
class PipelineOptions:
    lowercase: bool = True
    expand_contractions: bool = True
    remove_accents: bool = True
    remove_punctuation: bool = True
    remove_numbers: bool = False
    remove_stopwords: bool = True
    morphology: str = "Lemmatization"


@dataclass(frozen=True)
class PipelineResult:
    original: str
    normalized: str
    tokens: list[str]
    filtered_tokens: list[str]
    final_tokens: list[str]
    output_text: str
    removed_stopwords: list[str]
    original_token_count: int
    final_token_count: int
    vocabulary_size: int
    reduction_percent: float
    frequencies: dict[str, int]

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_text(text: str, options: PipelineOptions) -> str:
    """Apply configurable surface-level normalization."""
    value = unicodedata.normalize("NFKC", text)
    value = value.replace("\u2018", "'").replace("\u2019", "'")

    if options.lowercase:
        value = value.lower()

    if options.expand_contractions:
        for contraction, expanded in sorted(
            CONTRACTIONS.items(), key=lambda item: len(item[0]), reverse=True
        ):
            value = re.sub(rf"\b{re.escape(contraction)}\b", expanded, value, flags=re.I)

    if options.remove_accents:
        value = "".join(
            char
            for char in unicodedata.normalize("NFKD", value)
            if not unicodedata.combining(char)
        )

    if options.remove_numbers:
        value = re.sub(r"\b\d+(?:[.,]\d+)?\b", " ", value)

    if options.remove_punctuation:
        value = re.sub(r"[^\w\s'-]", " ", value)
        value = value.replace("_", " ").replace("'", " ").replace("-", " ")

    return re.sub(r"\s+", " ", value).strip()


def tokenize(text: str) -> list[str]:
    """Split text into words and numbers while retaining internal apostrophes."""
    return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+(?:'[A-Za-zÀ-ÖØ-öø-ÿ]+)?", text)


def stem_word(word: str) -> str:
    """Apply a compact suffix stemmer whose rules students can inspect."""
    value = word.lower()
    if len(value) <= 3:
        return value
    if value.endswith("ies") and len(value) > 4:
        return value[:-3] + "y"
    if value.endswith("ing") and len(value) > 5:
        base = value[:-3]
        if len(base) > 2 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if value.endswith("ed") and len(value) > 4:
        base = value[:-2]
        if len(base) > 2 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    for suffix in ("ization", "ational", "fulness", "ousness", "iveness", "ment", "ness", "ly"):
        if value.endswith(suffix) and len(value) - len(suffix) >= 3:
            return value[: -len(suffix)] + ("ize" if suffix == "ization" else "")
    if value.endswith("es") and len(value) > 4:
        return value[:-2]
    if value.endswith("s") and not value.endswith("ss") and len(value) > 3:
        return value[:-1]
    return value


def lemmatize_word(word: str) -> str:
    """Return a dictionary/rule-based base form for a transparent offline demo."""
    value = word.lower()
    if value in IRREGULAR_LEMMAS:
        return IRREGULAR_LEMMAS[value]
    if value.endswith("ies") and len(value) > 4:
        return value[:-3] + "y"
    if value.endswith("ves") and len(value) > 4:
        return value[:-3] + "f"
    if value.endswith("ing") and len(value) > 5:
        base = value[:-3]
        if len(base) > 2 and base[-1] == base[-2]:
            base = base[:-1]
        if base.endswith(("at", "iz", "bl")):
            base += "e"
        return base
    if value.endswith("ed") and len(value) > 4:
        base = value[:-2]
        if len(base) > 2 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if value.endswith("es") and len(value) > 4 and value[-3] in "sxz":
        return value[:-2]
    if value.endswith("s") and not value.endswith(("ss", "us", "is")) and len(value) > 3:
        return value[:-1]
    return value


def run_pipeline(text: str, options: PipelineOptions) -> PipelineResult:
    """Execute the selected stages and expose every intermediate representation."""
    normalized = normalize_text(text, options)
    tokens = tokenize(normalized)
    removed = [token for token in tokens if options.remove_stopwords and token.lower() in STOP_WORDS]
    filtered = [
        token for token in tokens
        if not options.remove_stopwords or token.lower() not in STOP_WORDS
    ]

    if options.morphology == "Stemming":
        final_tokens = [stem_word(token) for token in filtered]
    elif options.morphology == "Lemmatization":
        final_tokens = [lemmatize_word(token) for token in filtered]
    else:
        final_tokens = filtered.copy()

    original_count = len(tokenize(text))
    final_count = len(final_tokens)
    reduction = ((original_count - final_count) / original_count * 100) if original_count else 0.0

    return PipelineResult(
        original=text,
        normalized=normalized,
        tokens=tokens,
        filtered_tokens=filtered,
        final_tokens=final_tokens,
        output_text=" ".join(final_tokens),
        removed_stopwords=removed,
        original_token_count=original_count,
        final_token_count=final_count,
        vocabulary_size=len(set(final_tokens)),
        reduction_percent=round(reduction, 2),
        frequencies=dict(Counter(final_tokens).most_common()),
    )


def split_documents(text: str) -> list[str]:
    """Treat non-empty lines, or sentences on a single line, as mini documents."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]


def build_inverted_index(documents: list[str], options: PipelineOptions) -> dict[str, list[int]]:
    """Map every processed term to the one-based document IDs containing it."""
    index: dict[str, list[int]] = {}
    for document_id, document in enumerate(documents, start=1):
        for token in sorted(set(run_pipeline(document, options).final_tokens)):
            index.setdefault(token, []).append(document_id)
    return dict(sorted(index.items()))


def search_documents(
    query: str,
    documents: list[str],
    options: PipelineOptions,
) -> list[dict[str, object]]:
    """Rank documents with cosine similarity after applying the same pipeline."""
    query_tokens = run_pipeline(query, options).final_tokens
    query_counts = Counter(query_tokens)
    query_norm = sqrt(sum(value * value for value in query_counts.values()))
    rows = []

    for document_id, document in enumerate(documents, start=1):
        document_tokens = run_pipeline(document, options).final_tokens
        document_counts = Counter(document_tokens)
        document_norm = sqrt(sum(value * value for value in document_counts.values()))
        dot_product = sum(
            query_counts[token] * document_counts.get(token, 0)
            for token in query_counts
        )
        denominator = query_norm * document_norm
        score = dot_product / denominator if denominator else 0.0
        matched_terms = sorted(set(query_counts).intersection(document_counts))
        rows.append(
            {
                "Document": f"D{document_id}",
                "Similarity": round(score, 4),
                "Matched Terms": ", ".join(matched_terms) or "—",
                "Text": document,
            }
        )

    return sorted(rows, key=lambda row: (-float(row["Similarity"]), str(row["Document"])))
