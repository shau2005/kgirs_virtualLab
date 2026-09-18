# Text Preprocessing and Normalization — Virtual Laboratory

Experiment 2 for the Knowledge Graph & Information Retrieval laboratory.

**Assigned operations:** tokenization, stop-word removal, stemming, lemmatization, and text normalization.

**Expected outcome:** a clean and standardized text corpus ready for indexing and analysis.

## Virtual Lab structure

The Streamlit application follows the supplied four-section template and the learning flow used by IIT Kharagpur Virtual Lab experiment pages:

1. **Theory** — aim, objectives, concepts, procedure, terminology, and expected outcome
2. **Simulation** — interactive pipeline, intermediate results, visualizations, downloads, and trial logging
3. **Quiz** — ten self-grading questions with immediate explanatory feedback
4. **Report Generation** — student information, recorded trials, observations, references, and PDF/JSON export

The simulator uses a compact pipeline graph and requires no Neo4j implementation.

## Features

- Paste text or upload a `.txt`/`.csv` corpus
- Toggle case folding, contraction expansion, accent removal, punctuation removal, number removal, and stop-word removal
- Compare stemming, lemmatization, and no word-form reduction
- Inspect every intermediate representation
- View token metrics and a frequency chart
- Build and inspect a real inverted search index
- Compare document-query similarity before and after preprocessing
- Download the cleaned corpus and trial log
- Record multiple experimental trials
- Complete a self-grading conceptual quiz
- Generate a downloadable PDF laboratory report
- Export the complete session as JSON
- Native Streamlit components only, so light and dark themes both work

## Run in Antigravity or a terminal

Open this repository folder, then run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The browser normally opens at `http://localhost:8501`.

## Project files

| File | Purpose |
|---|---|
| `app.py` | Streamlit user interface, quiz, trial logger, charts, and report generator |
| `core.py` | Deterministic preprocessing functions and intermediate pipeline results |
| `test_core.py` | Unit tests for normalization, tokenization, stemming, and lemmatization |
| `requirements.txt` | Python dependencies |

## Educational note

The stemmer and lemmatizer are deliberately transparent, offline-friendly teaching implementations. Their rules can be inspected directly in `core.py`. Production NLP systems should use validated language-specific libraries or models.

## References

- Daniel Jurafsky and James H. Martin, *Speech and Language Processing*
- Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze, *Introduction to Information Retrieval*
- [Virtual Labs, IIT Kharagpur](https://vlabs.iitkgp.ac.in/)
