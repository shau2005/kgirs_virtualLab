"""Streamlit Virtual Lab: Text Preprocessing and Normalization."""

from __future__ import annotations

from datetime import date, datetime
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF

from core import (
    DEFAULT_TEXT,
    PipelineOptions,
    build_inverted_index,
    run_pipeline,
    search_documents,
    split_documents,
)


EXPERIMENT = {
    "number": 2,
    "title": "Text Preprocessing and Normalization",
    "estimated_minutes": "6–10 minutes",
    "aim": (
        "To perform tokenization, stop-word removal, stemming, lemmatization, "
        "and text normalization on a small text corpus."
    ),
    "expected_outcome": "Clean and standardized text corpus ready for indexing and analysis.",
    "objectives": [
        "Explain why raw text must be standardized before indexing and analysis.",
        "Apply tokenization, case folding, punctuation handling, and stop-word removal.",
        "Compare stemming and lemmatization and identify their trade-offs.",
        "Generate a clean corpus suitable for information retrieval and knowledge-graph workflows.",
    ],
}


QUIZ_QUESTIONS = [
    {
        "id": 1,
        "question": "Which operation divides a sentence into smaller units such as words?",
        "options": ["Tokenization", "Lemmatization", "Indexing", "Graph traversal"],
        "answer": 0,
        "explanation": "Tokenization creates the units that later preprocessing stages operate on.",
    },
    {
        "id": 2,
        "question": "Why are stop words sometimes removed?",
        "options": [
            "To convert all words into numbers",
            "To reduce frequent terms that may carry little discriminative information",
            "To correct every spelling error",
            "To create relationships in Neo4j",
        ],
        "answer": 1,
        "explanation": "Words such as ‘the’ and ‘is’ may add little value to some retrieval tasks.",
    },
    {
        "id": 3,
        "question": "Which statement best distinguishes lemmatization from stemming?",
        "options": [
            "Lemmatization only removes punctuation",
            "Stemming always needs a dictionary",
            "Lemmatization aims for a meaningful base word; a stem may be truncated",
            "They are identical operations",
        ],
        "answer": 2,
        "explanation": "A lemma is a linguistic base form, whereas a stem is mainly produced by affix rules.",
    },
    {
        "id": 4,
        "question": "What is case folding?",
        "options": [
            "Converting text to a consistent letter case",
            "Splitting a dataset into folds",
            "Removing duplicate documents",
            "Finding named entities",
        ],
        "answer": 0,
        "explanation": "Case folding usually converts all text to lowercase so case variants match.",
    },
    {
        "id": 5,
        "question": "Why should contractions be expanded before punctuation is removed?",
        "options": [
            "It preserves meanings such as ‘do not’ from ‘don't’",
            "It makes every token longer",
            "It creates graph edges",
            "The order never matters",
        ],
        "answer": 0,
        "explanation": "Removing the apostrophe first can destroy the structure needed to expand a contraction.",
    },
    {
        "id": 6,
        "question": "When might stop-word removal be harmful?",
        "options": [
            "When negation or function words carry task-specific meaning",
            "Only when text contains numbers",
            "Whenever lowercase is used",
            "It can never be harmful",
        ],
        "answer": 0,
        "explanation": "Tasks such as sentiment analysis may depend on words such as ‘not’.",
    },
    {
        "id": 7,
        "question": "What does vocabulary size measure in this simulation?",
        "options": [
            "The number of characters",
            "The number of distinct final tokens",
            "The number of sentences",
            "The number of uploaded files",
        ],
        "answer": 1,
        "explanation": "Vocabulary size is the count of unique tokens in the processed output.",
    },
    {
        "id": 8,
        "question": "Which form is a likely lemma of ‘were’?",
        "options": ["wer", "were", "be", "beingness"],
        "answer": 2,
        "explanation": "The irregular verb form ‘were’ maps to the base form ‘be’.",
    },
    {
        "id": 9,
        "question": "Why record multiple trials with different settings?",
        "options": [
            "To compare how configuration choices change the output",
            "To make every result identical",
            "To avoid inspecting intermediate stages",
            "To train a language model",
        ],
        "answer": 0,
        "explanation": "Controlled comparisons reveal the effects and limitations of each operation.",
    },
    {
        "id": 10,
        "question": "What is the expected result of this experiment?",
        "options": [
            "A Neo4j server",
            "An image dataset",
            "A clean, standardized corpus ready for indexing and analysis",
            "A trained neural network",
        ],
        "answer": 2,
        "explanation": "The specified outcome is a standardized corpus for downstream use.",
    },
]


def initialize_state() -> None:
    defaults = {
        "trials": [],
        "quiz_answers": {},
        "quiz_submitted": False,
        "quiz_score": 0,
        "student_info": {"name": "", "roll": "", "date": str(date.today())},
        "student_notes": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def configuration_label(options: PipelineOptions) -> str:
    enabled = []
    if options.lowercase:
        enabled.append("case folding")
    if options.expand_contractions:
        enabled.append("contraction expansion")
    if options.remove_accents:
        enabled.append("accent removal")
    if options.remove_punctuation:
        enabled.append("punctuation removal")
    if options.remove_numbers:
        enabled.append("number removal")
    if options.remove_stopwords:
        enabled.append("stop-word removal")
    enabled.append(options.morphology.lower())
    return ", ".join(enabled)


def make_pipeline_figure(morphology: str) -> go.Figure:
    labels = ["Raw text", "Normalize", "Tokenize", "Stop-word filter", morphology, "Clean corpus"]
    x_values = list(range(len(labels)))
    figure = go.Figure()
    for index in range(len(labels) - 1):
        figure.add_trace(
            go.Scatter(
                x=[x_values[index], x_values[index + 1]],
                y=[0, 0],
                mode="lines",
                line={"color": "#64748b", "width": 3},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=[0] * len(labels),
            mode="markers+text",
            marker={
                "size": 38,
                "color": ["#f59e0b", "#2563eb", "#2563eb", "#2563eb", "#7c3aed", "#16a34a"],
            },
            text=labels,
            textposition="bottom center",
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )
    figure.update_layout(
        height=180,
        margin={"l": 20, "r": 20, "t": 15, "b": 55},
        xaxis={"visible": False, "range": [-0.45, len(labels) - 0.55]},
        yaxis={"visible": False, "range": [-0.45, 0.35]},
    )
    return figure


def render_theory_section() -> None:
    st.header("Theoretical Framework & Background")
    st.info(f"**Aim:** {EXPERIMENT['aim']}")

    st.subheader("Learning Objectives")
    for index, objective in enumerate(EXPERIMENT["objectives"], start=1):
        st.markdown(f"- **Goal {index}:** {objective}")

    st.divider()
    st.subheader("Overview")
    st.write(
        "Real-world text is noisy: the same concept can appear with different capitalization, "
        "punctuation, spelling, or grammatical form. A preprocessing pipeline reduces these "
        "surface differences so that search engines, analytical models, and knowledge-graph "
        "pipelines can compare text consistently."
    )
    st.write(
        "The order of operations matters. In this laboratory, contractions are expanded before "
        "punctuation removal, text is normalized before tokenization, optional stop words are "
        "filtered next, and word-form reduction is applied last."
    )

    concept_rows = [
        ["Text normalization", "Standardizes case, accents, spacing, punctuation, and numbers.", "Café!! → cafe"],
        ["Tokenization", "Segments text into processable units.", "graphs connect data → [graphs, connect, data]"],
        ["Stop-word removal", "Optionally removes frequent low-information function words.", "the graph is useful → [graph, useful]"],
        ["Stemming", "Uses suffix-removal rules; output need not be a dictionary word.", "connected → connect"],
        ["Lemmatization", "Maps an inflected form to a meaningful base form.", "were → be"],
    ]
    st.dataframe(
        pd.DataFrame(concept_rows, columns=["Concept", "Purpose", "Example"]),
        hide_index=True,
        width="stretch",
    )

    st.subheader("Stemming versus Lemmatization")
    comparison = pd.DataFrame(
        [
            ["Method", "Rule-based affix removal", "Dictionary and grammatical base-form mapping"],
            ["Speed", "Usually faster", "Usually more computationally expensive"],
            ["Output", "May be an incomplete word", "Usually a valid word"],
            ["Use", "Recall-focused retrieval", "Readable, meaning-sensitive analysis"],
        ],
        columns=["Aspect", "Stemming", "Lemmatization"],
    )
    st.table(comparison)

    st.divider()
    st.subheader("Experimental Procedure")
    procedure = [
        "Review the aim, theory, and key terminology.",
        "Open the Simulation section and enter or upload a small text corpus.",
        "Choose the required normalization and filtering controls.",
        "Select Stemming, Lemmatization, or None as the word-form reduction method.",
        "Inspect the stage-by-stage transformations, metrics, and token frequencies.",
        "Record the trial and repeat with at least two different configurations.",
        "Compare results, complete the quiz, and generate the PDF report.",
    ]
    for number, step in enumerate(procedure, start=1):
        st.markdown(f"**Step {number}.** {step}")

    st.subheader("Expected Outcome")
    st.success(EXPERIMENT["expected_outcome"])
    st.caption(
        "The compact stemming and lemmatization rules are intentionally transparent and offline-friendly. "
        "Production systems should use validated, language-specific NLP libraries or models."
    )


def read_uploaded_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    if uploaded_file.name.lower().endswith(".txt"):
        return uploaded_file.getvalue().decode("utf-8", errors="replace")
    dataframe = pd.read_csv(uploaded_file)
    text_columns = dataframe.select_dtypes(include="object").columns.tolist()
    if not text_columns:
        return ""
    return " ".join(dataframe[text_columns[0]].dropna().astype(str).tolist())


def render_simulation_section() -> None:
    st.header("Interactive Simulation Sandbox")
    st.info("Configure the pipeline, inspect each transformation, and record comparative trials.")

    source_mode = st.radio("Text source", ["Use text box", "Upload TXT or CSV"], horizontal=True)
    uploaded_text = ""
    if source_mode == "Upload TXT or CSV":
        upload = st.file_uploader("Upload a UTF-8 text file or a CSV with a text column", type=["txt", "csv"])
        uploaded_text = read_uploaded_text(upload)
        if upload and not uploaded_text:
            st.warning("No usable text column was found in the uploaded file.")

    raw_text = st.text_area(
        "Input text corpus",
        value=uploaded_text or DEFAULT_TEXT,
        height=150,
        help="A short corpus keeps every transformation easy to inspect.",
    )

    st.subheader("Pipeline Controls")
    column_1, column_2, column_3 = st.columns(3)
    with column_1:
        lowercase = st.checkbox("Convert to lowercase", value=True)
        contractions = st.checkbox("Expand contractions", value=True)
        accents = st.checkbox("Remove accents", value=True)
    with column_2:
        punctuation = st.checkbox("Remove punctuation", value=True)
        numbers = st.checkbox("Remove numbers", value=False)
        stop_words = st.checkbox("Remove stop words", value=True)
    with column_3:
        morphology = st.radio("Word-form reduction", ["Lemmatization", "Stemming", "None"])

    options = PipelineOptions(
        lowercase=lowercase,
        expand_contractions=contractions,
        remove_accents=accents,
        remove_punctuation=punctuation,
        remove_numbers=numbers,
        remove_stopwords=stop_words,
        morphology=morphology,
    )
    result = run_pipeline(raw_text, options)

    st.plotly_chart(
        make_pipeline_figure(morphology),
        width="stretch",
        config={"displayModeBar": False},
    )

    if not raw_text.strip():
        st.warning("Enter at least one word to run the experiment.")
        return

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Input tokens", result.original_token_count)
    metric_2.metric("Final tokens", result.final_token_count)
    metric_3.metric("Vocabulary size", result.vocabulary_size)
    metric_4.metric("Token reduction", f"{result.reduction_percent:.1f}%")

    st.subheader("Stage-by-Stage Results")
    stages = pd.DataFrame(
        [
            [1, "Normalized text", result.normalized or "—"],
            [2, "Tokenization", " | ".join(result.tokens) or "—"],
            [3, "Stop-word filtering", " | ".join(result.filtered_tokens) or "—"],
            [4, morphology, " | ".join(result.final_tokens) or "—"],
            [5, "Clean standardized corpus", result.output_text or "—"],
        ],
        columns=["Stage", "Operation", "Output"],
    )
    st.dataframe(stages, hide_index=True, width="stretch")

    if result.removed_stopwords:
        st.caption(f"Removed stop words: {', '.join(result.removed_stopwords)}")

    if result.frequencies:
        frequency_data = pd.DataFrame(
            list(result.frequencies.items()), columns=["Token", "Frequency"]
        ).head(15)
        frequency_figure = go.Figure(
            go.Bar(
                x=frequency_data["Token"],
                y=frequency_data["Frequency"],
                marker_color="#2563eb",
                hovertemplate="%{x}: %{y}<extra></extra>",
            )
        )
        frequency_figure.update_layout(
            title="Final Token Frequency (Top 15)",
            xaxis_title="Token",
            yaxis_title="Frequency",
            height=340,
            margin={"l": 20, "r": 20, "t": 50, "b": 20},
        )
        st.plotly_chart(frequency_figure, width="stretch")

    st.divider()
    st.subheader("Real-World Use: Search Indexing Demonstration")
    st.write(
        "Search engines preprocess both stored documents and the user's query. This demonstration "
        "treats each sentence (or each non-empty line) as a document, builds a small inverted index, "
        "and shows how normalization improves matching between different word forms."
    )
    search_query = st.text_input(
        "Search query",
        value="connect entities and index documents",
        help="Try changing ‘connect’ to ‘connecting’ or ‘indexed’ and compare the scores.",
    )
    documents = split_documents(raw_text)
    raw_search_options = PipelineOptions(
        lowercase=True,
        expand_contractions=False,
        remove_accents=False,
        remove_punctuation=False,
        remove_numbers=False,
        remove_stopwords=False,
        morphology="None",
    )
    raw_rankings = search_documents(search_query, documents, raw_search_options)
    processed_rankings = search_documents(search_query, documents, options)
    raw_by_document = {row["Document"]: row for row in raw_rankings}
    processed_by_document = {row["Document"]: row for row in processed_rankings}

    comparison_rows = []
    for document_id, document in enumerate(documents, start=1):
        key = f"D{document_id}"
        comparison_rows.append(
            {
                "Document": key,
                "Raw Similarity": raw_by_document[key]["Similarity"],
                "Processed Similarity": processed_by_document[key]["Similarity"],
                "Processed Matches": processed_by_document[key]["Matched Terms"],
                "Text": document,
            }
        )
    comparison_dataframe = pd.DataFrame(comparison_rows)

    index_data = build_inverted_index(documents, options)
    raw_vocabulary = build_inverted_index(documents, raw_search_options)
    search_metric_1, search_metric_2, search_metric_3 = st.columns(3)
    search_metric_1.metric("Indexed documents", len(documents))
    search_metric_2.metric("Raw vocabulary", len(raw_vocabulary))
    search_metric_3.metric(
        "Processed vocabulary",
        len(index_data),
        delta=len(index_data) - len(raw_vocabulary),
        delta_color="inverse",
    )

    st.dataframe(
        comparison_dataframe.sort_values("Processed Similarity", ascending=False),
        hide_index=True,
        width="stretch",
    )
    comparison_figure = go.Figure()
    comparison_figure.add_trace(
        go.Bar(
            name="Raw text",
            x=comparison_dataframe["Document"],
            y=comparison_dataframe["Raw Similarity"],
            marker_color="#94a3b8",
        )
    )
    comparison_figure.add_trace(
        go.Bar(
            name="After preprocessing",
            x=comparison_dataframe["Document"],
            y=comparison_dataframe["Processed Similarity"],
            marker_color="#16a34a",
        )
    )
    comparison_figure.update_layout(
        title="Search Similarity: Raw vs Preprocessed Text",
        xaxis_title="Document",
        yaxis_title="Cosine similarity",
        barmode="group",
        height=340,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    st.plotly_chart(comparison_figure, width="stretch")

    with st.expander("Inspect the Generated Inverted Index"):
        index_rows = [
            {"Term": term, "Appears in Documents": ", ".join(f"D{doc_id}" for doc_id in document_ids)}
            for term, document_ids in index_data.items()
        ]
        st.dataframe(pd.DataFrame(index_rows), hide_index=True, width="stretch")
        st.caption(
            "An inverted index is the core lookup structure used by many information-retrieval systems: "
            "each term points to the documents in which it occurs."
        )

    st.download_button(
        "Download Clean Corpus (.txt)",
        data=result.output_text.encode("utf-8"),
        file_name="clean_standardized_corpus.txt",
        mime="text/plain",
        width="stretch",
    )

    st.divider()
    st.subheader("Experimental Data Log Book")
    log_column, clear_column = st.columns(2)
    with log_column:
        if st.button("Record Current Trial", type="primary", width="stretch"):
            best_processed_result = processed_rankings[0] if processed_rankings else None
            best_document = best_processed_result["Document"] if best_processed_result else "—"
            processed_similarity = (
                float(best_processed_result["Similarity"]) if best_processed_result else 0.0
            )
            raw_similarity = (
                float(raw_by_document[best_document]["Similarity"])
                if best_document in raw_by_document
                else 0.0
            )
            trial = {
                "Trial #": len(st.session_state["trials"]) + 1,
                "Timestamp": datetime.now().strftime("%H:%M:%S"),
                "Configuration": configuration_label(options),
                "Input Tokens": result.original_token_count,
                "Final Tokens": result.final_token_count,
                "Vocabulary": result.vocabulary_size,
                "Reduction (%)": result.reduction_percent,
                "Indexed Documents": len(documents),
                "Raw Vocabulary": len(raw_vocabulary),
                "Processed Vocabulary": len(index_data),
                "Top Search Result": best_document,
                "Raw Similarity": round(raw_similarity, 4),
                "Processed Similarity": round(processed_similarity, 4),
                "Clean Corpus": result.output_text,
            }
            st.session_state["trials"].append(trial)
            st.success(f"Trial #{trial['Trial #']} recorded successfully.")
    with clear_column:
        if st.button("Clear Logged Trials", width="stretch"):
            st.session_state["trials"] = []
            st.rerun()

    if st.session_state["trials"]:
        trial_dataframe = pd.DataFrame(st.session_state["trials"])
        st.dataframe(trial_dataframe, hide_index=True, width="stretch")
        st.download_button(
            "Download Trials as CSV",
            data=trial_dataframe.to_csv(index=False).encode("utf-8"),
            file_name="text_preprocessing_trials.csv",
            mime="text/csv",
        )
    else:
        st.info("No trials recorded. Record at least two configurations for comparison.")


def render_quiz_section() -> None:
    st.header("Concept Assessment Quiz")
    st.write("Answer all ten questions and submit the quiz for immediate feedback.")

    with st.form("concept_quiz"):
        responses = {}
        for question in QUIZ_QUESTIONS:
            st.markdown(f"**Question {question['id']}**")
            st.write(question["question"])
            selected = st.radio(
                f"Options for Question {question['id']}",
                question["options"],
                index=None,
                key=f"quiz_{question['id']}",
                label_visibility="collapsed",
            )
            responses[question["id"]] = (
                question["options"].index(selected) if selected is not None else None
            )
        submitted = st.form_submit_button("Submit Quiz for Grading", type="primary")

    if submitted:
        score = 0
        st.session_state["quiz_answers"] = responses
        st.session_state["quiz_submitted"] = True
        st.divider()
        st.subheader("Evaluation Results and Feedback")

        for question in QUIZ_QUESTIONS:
            response = responses[question["id"]]
            if response == question["answer"]:
                score += 1
                st.success(f"Question {question['id']}: Correct — {question['explanation']}")
            elif response is None:
                st.warning(f"Question {question['id']}: Not answered — {question['explanation']}")
            else:
                correct = question["options"][question["answer"]]
                st.error(
                    f"Question {question['id']}: Incorrect. Correct answer: {correct}. "
                    f"{question['explanation']}"
                )

        st.session_state["quiz_score"] = score
        percentage = score / len(QUIZ_QUESTIONS) * 100
        st.info(f"Final Score: **{score}/{len(QUIZ_QUESTIONS)} ({percentage:.0f}%)**")
    elif st.session_state["quiz_submitted"]:
        st.info(
            f"Latest score: {st.session_state['quiz_score']}/{len(QUIZ_QUESTIONS)}. "
            "Resubmit to record a new score."
        )


def build_trial_analysis(trials: list[dict]) -> tuple[list[str], str]:
    """Create evidence-based analysis and a conclusion from recorded trials."""
    if not trials:
        return (
            ["No experimental trials were recorded; perform at least two trials for comparison."],
            "The experiment cannot be concluded until preprocessing trials are recorded.",
        )

    best_reduction = max(trials, key=lambda trial: float(trial.get("Reduction (%)", 0)))
    smallest_vocabulary = min(trials, key=lambda trial: int(trial.get("Vocabulary", 0)))
    configurations = {str(trial.get("Configuration", "")) for trial in trials}
    analysis = [
        f"{len(trials)} trial(s) were performed using {len(configurations)} distinct preprocessing configuration(s).",
        (
            f"Trial {best_reduction.get('Trial #', '—')} achieved the highest token reduction: "
            f"{best_reduction.get('Input Tokens', 0)} input tokens became "
            f"{best_reduction.get('Final Tokens', 0)} final tokens "
            f"({float(best_reduction.get('Reduction (%)', 0)):.2f}% reduction)."
        ),
        (
            f"The smallest final vocabulary was {smallest_vocabulary.get('Vocabulary', 0)} terms "
            f"in Trial {smallest_vocabulary.get('Trial #', '—')}, showing how normalization can "
            "combine word variants and reduce index size."
        ),
    ]

    searchable_trials = [
        trial
        for trial in trials
        if "Processed Similarity" in trial and "Raw Similarity" in trial
    ]
    if searchable_trials:
        strongest_search = max(
            searchable_trials,
            key=lambda trial: float(trial.get("Processed Similarity", 0)),
        )
        improvement = float(strongest_search.get("Processed Similarity", 0)) - float(
            strongest_search.get("Raw Similarity", 0)
        )
        direction = "increased" if improvement >= 0 else "changed"
        analysis.append(
            f"For the top processed result in Trial {strongest_search.get('Trial #', '—')} "
            f"({strongest_search.get('Top Search Result', '—')}), similarity {direction} from "
            f"{float(strongest_search.get('Raw Similarity', 0)):.4f} to "
            f"{float(strongest_search.get('Processed Similarity', 0)):.4f}."
        )
        analysis.append(
            f"The corpus was converted from {strongest_search.get('Raw Vocabulary', '—')} raw terms "
            f"to {strongest_search.get('Processed Vocabulary', '—')} indexed terms across "
            f"{strongest_search.get('Indexed Documents', '—')} documents."
        )

    conclusion = (
        "The experiment successfully applied normalization, tokenization, stop-word handling, "
        "and word-form reduction. The recorded results show that preprocessing reduced surface "
        "variation and produced a compact, consistent vocabulary. The resulting corpus and "
        "inverted index are suitable for efficient document searching, indexing, and further "
        "text analysis."
    )
    return analysis, conclusion


def make_trial_comparison_figure(trials: list[dict]) -> go.Figure:
    """Compare input/final token counts and reduction across recorded trials."""
    labels = [f"Trial {trial.get('Trial #', index)}" for index, trial in enumerate(trials, start=1)]
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            name="Input Tokens",
            x=labels,
            y=[trial.get("Input Tokens", 0) for trial in trials],
            marker_color="#94a3b8",
        )
    )
    figure.add_trace(
        go.Bar(
            name="Final Tokens",
            x=labels,
            y=[trial.get("Final Tokens", 0) for trial in trials],
            marker_color="#2563eb",
        )
    )
    figure.update_layout(
        title="Trial Comparison: Input vs Final Tokens",
        xaxis_title="Recorded Trial",
        yaxis_title="Number of Tokens",
        barmode="group",
        height=360,
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
    )
    return figure


class LabReportPDF(FPDF):
    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 116, 139)
        self.cell(0, 8, f"Page {self.page_no()}/{{nb}} | Virtual Laboratory Report", align="C")


def pdf_safe(value: object) -> str:
    return str(value).encode("latin-1", "replace").decode("latin-1")


def pdf_section_title(pdf: LabReportPDF, number: int, title: str) -> None:
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 64, 175)
    pdf.cell(0, 7, f"{number}. {title}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(31, 41, 55)


def draw_pdf_trial_chart(pdf: LabReportPDF, trials: list[dict]) -> None:
    """Draw a dependency-free grouped bar chart directly in the PDF."""
    if not trials:
        return
    chart_trials = trials[:8]
    max_tokens = max(
        max(float(trial.get("Input Tokens", 0)), float(trial.get("Final Tokens", 0)))
        for trial in chart_trials
    ) or 1
    chart_x = 24
    chart_y = pdf.get_y() + 5
    chart_width = 160
    chart_height = 45
    group_width = chart_width / len(chart_trials)

    pdf.set_draw_color(148, 163, 184)
    pdf.line(chart_x, chart_y, chart_x, chart_y + chart_height)
    pdf.line(chart_x, chart_y + chart_height, chart_x + chart_width, chart_y + chart_height)

    for index, trial in enumerate(chart_trials):
        input_value = float(trial.get("Input Tokens", 0))
        final_value = float(trial.get("Final Tokens", 0))
        input_height = input_value / max_tokens * (chart_height - 5)
        final_height = final_value / max_tokens * (chart_height - 5)
        group_x = chart_x + index * group_width + group_width * 0.18
        bar_width = max(3, group_width * 0.25)

        pdf.set_fill_color(148, 163, 184)
        pdf.rect(group_x, chart_y + chart_height - input_height, bar_width, input_height, "F")
        pdf.set_fill_color(37, 99, 235)
        pdf.rect(
            group_x + bar_width + 1,
            chart_y + chart_height - final_height,
            bar_width,
            final_height,
            "F",
        )
        pdf.set_xy(group_x - 1, chart_y + chart_height + 1)
        pdf.set_font("Helvetica", "", 7)
        pdf.cell(group_width * 0.8, 4, f"T{trial.get('Trial #', index + 1)}", align="C")

    pdf.set_xy(chart_x, chart_y + chart_height + 8)
    pdf.set_font("Helvetica", "", 7)
    pdf.set_fill_color(148, 163, 184)
    pdf.rect(chart_x, pdf.get_y() + 1, 3, 3, "F")
    pdf.cell(5, 5, "")
    pdf.cell(28, 5, "Input tokens")
    pdf.set_fill_color(37, 99, 235)
    pdf.rect(chart_x + 36, pdf.get_y() + 1, 3, 3, "F")
    pdf.cell(5, 5, "")
    pdf.cell(28, 5, "Final tokens", new_x="LMARGIN", new_y="NEXT")
    pdf.set_y(chart_y + chart_height + 15)


def generate_pdf_report(
    student_name: str,
    roll_number: str,
    experiment_date: str,
    trials: list[dict],
    notes: str,
    quiz_score: int = 0,
) -> bytes:
    pdf = LabReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(30, 64, 175)
    pdf.multi_cell(0, 9, EXPERIMENT["title"], new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(31, 41, 55)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, pdf_safe(f"Experiment No.: {EXPERIMENT['number']}"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, pdf_safe(f"Student: {student_name or 'N/A'}"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, pdf_safe(f"Roll number: {roll_number or 'N/A'}"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, pdf_safe(f"Date: {experiment_date}"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(
        0,
        6,
        pdf_safe(f"Quiz score: {quiz_score}/{len(QUIZ_QUESTIONS)}"),
        new_x="LMARGIN",
        new_y="NEXT",
    )

    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 64, 175)
    pdf.cell(0, 7, "1. Aim", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(31, 41, 55)
    pdf.multi_cell(0, 5, pdf_safe(EXPERIMENT["aim"]), new_x="LMARGIN", new_y="NEXT")

    pdf_section_title(pdf, 2, "Methodology / Operations Performed")
    pdf.set_font("Helvetica", "", 9)
    methodology = [
        "The input corpus was normalized using the selected case, contraction, accent, punctuation, and number settings.",
        "The normalized text was tokenized and optional stop words were removed.",
        "Stemming, lemmatization, or no morphology was applied according to each trial configuration.",
        "The cleaned corpus was used to build an inverted index and compare raw versus processed search similarity.",
    ]
    for step_number, step in enumerate(methodology, start=1):
        pdf.multi_cell(
            0,
            5,
            pdf_safe(f"{step_number}. {step}"),
            new_x="LMARGIN",
            new_y="NEXT",
        )

    pdf_section_title(pdf, 3, "Experimental Results")
    pdf.set_font("Helvetica", "", 8)
    if not trials:
        pdf.multi_cell(0, 5, "No trials were recorded.", new_x="LMARGIN", new_y="NEXT")
    for trial in trials:
        summary = (
            f"Trial {trial.get('Trial #', '—')}: {trial.get('Configuration', '—')} | "
            f"tokens {trial.get('Input Tokens', 0)} -> {trial.get('Final Tokens', 0)} | "
            f"vocabulary {trial.get('Vocabulary', 0)} | "
            f"reduction {float(trial.get('Reduction (%)', 0)):.2f}%"
        )
        pdf.set_font("Helvetica", "B", 8)
        pdf.multi_cell(0, 5, pdf_safe(summary), new_x="LMARGIN", new_y="NEXT")
        if "Processed Similarity" in trial:
            search_summary = (
                f"Index: {trial.get('Indexed Documents', '—')} documents, "
                f"{trial.get('Raw Vocabulary', '—')} -> {trial.get('Processed Vocabulary', '—')} terms | "
                f"Top result: {trial.get('Top Search Result', '—')} | "
                f"Similarity: {float(trial.get('Raw Similarity', 0)):.4f} -> "
                f"{float(trial.get('Processed Similarity', 0)):.4f}"
            )
            pdf.set_font("Helvetica", "", 8)
            pdf.multi_cell(0, 5, pdf_safe(search_summary), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "I", 7)
        output_preview = str(trial.get("Clean Corpus", ""))[:300]
        pdf.multi_cell(
            0,
            4,
            pdf_safe(f"Clean corpus preview: {output_preview}"),
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.ln(2)

    if trials:
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 6, "Trial Comparison Graph", new_x="LMARGIN", new_y="NEXT")
        draw_pdf_trial_chart(pdf, trials)

    analysis_points, conclusion = build_trial_analysis(trials)
    pdf_section_title(pdf, 4, "Automatic Comparative Analysis")
    pdf.set_font("Helvetica", "", 9)
    for point in analysis_points:
        pdf.multi_cell(
            0,
            5,
            pdf_safe(f"- {point}"),
            new_x="LMARGIN",
            new_y="NEXT",
        )

    pdf_section_title(pdf, 5, "Student Observations")
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(
        0,
        5,
        pdf_safe(notes.strip() or "No additional student observation was entered."),
        new_x="LMARGIN",
        new_y="NEXT",
    )

    pdf_section_title(pdf, 6, "Conclusion")
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, pdf_safe(conclusion), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


def render_report_section() -> None:
    st.header("Report Generation")
    st.write(
        "Generate a complete laboratory report containing the performed methodology, trial results, "
        "comparison graphs, automatic analysis, student observations, and conclusion."
    )

    name_column, roll_column, date_column = st.columns(3)
    with name_column:
        student_name = st.text_input("Student Name", value=st.session_state["student_info"]["name"])
    with roll_column:
        roll_number = st.text_input("Student Roll Number", value=st.session_state["student_info"]["roll"])
    with date_column:
        experiment_date = st.date_input("Experiment Date", value=date.today())

    st.session_state["student_info"] = {
        "name": student_name,
        "roll": roll_number,
        "date": str(experiment_date),
    }

    default_notes = (
        "I observed that stop-word removal reduced the token count. Lemmatization produced more "
        "readable base forms, while stemming was faster but sometimes produced incomplete words."
    )
    notes = st.text_area(
        "Student Observations (write what you noticed during the experiment)",
        value=st.session_state["student_notes"] or default_notes,
        height=130,
    )
    st.session_state["student_notes"] = notes

    trial_dataframe = pd.DataFrame(st.session_state["trials"])
    summary_1, summary_2 = st.columns(2)
    summary_1.metric("Recorded Trials", len(trial_dataframe))
    quiz_display = (
        f"{st.session_state['quiz_score']}/{len(QUIZ_QUESTIONS)}"
        if st.session_state["quiz_submitted"]
        else "Pending"
    )
    summary_2.metric("Quiz Score", quiz_display)

    if trial_dataframe.empty:
        st.warning("Record at least two simulation trials for a meaningful report comparison.")
    else:
        st.dataframe(trial_dataframe, hide_index=True, width="stretch")
        st.plotly_chart(make_trial_comparison_figure(st.session_state["trials"]), width="stretch")

        reduction_figure = go.Figure(
            go.Bar(
                x=[f"Trial {trial.get('Trial #', index)}" for index, trial in enumerate(st.session_state["trials"], start=1)],
                y=[trial.get("Reduction (%)", 0) for trial in st.session_state["trials"]],
                marker_color="#16a34a",
                text=[f"{float(trial.get('Reduction (%)', 0)):.2f}%" for trial in st.session_state["trials"]],
                textposition="outside",
            )
        )
        reduction_figure.update_layout(
            title="Token Reduction Achieved in Each Trial",
            xaxis_title="Recorded Trial",
            yaxis_title="Reduction (%)",
            height=330,
            margin={"l": 20, "r": 20, "t": 55, "b": 20},
        )
        st.plotly_chart(reduction_figure, width="stretch")

    analysis_points, conclusion = build_trial_analysis(st.session_state["trials"])
    st.subheader("Automatic Analysis Preview")
    for point in analysis_points:
        st.markdown(f"- {point}")

    st.subheader("Conclusion Preview")
    st.success(conclusion)

    report_bytes = generate_pdf_report(
        student_name,
        roll_number,
        str(experiment_date),
        st.session_state["trials"],
        notes,
        st.session_state["quiz_score"],
    )
    st.download_button(
        "Download Official Lab Report (.pdf)",
        data=report_bytes,
        file_name="text_preprocessing_virtual_lab_report.pdf",
        mime="application/pdf",
        type="primary",
        width="stretch",
    )

    session_data = {
        "experiment": EXPERIMENT,
        "student": st.session_state["student_info"],
        "trials": st.session_state["trials"],
        "quiz_score": st.session_state["quiz_score"],
        "quiz_total": len(QUIZ_QUESTIONS),
        "notes": notes,
        "automatic_analysis": analysis_points,
        "conclusion": conclusion,
    }
    st.download_button(
        "Download Session Data (.json)",
        data=json.dumps(session_data, indent=2).encode("utf-8"),
        file_name="text_preprocessing_session.json",
        mime="application/json",
        width="stretch",
    )

    st.divider()
    st.subheader("References")
    st.markdown(
        "1. Daniel Jurafsky and James H. Martin, *Speech and Language Processing*, text normalization chapters.\n"
        "2. Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze, *Introduction to Information Retrieval*.\n"
        "3. [Virtual Labs, IIT Kharagpur](https://vlabs.iitkgp.ac.in/) — experiment learning flow and content structure."
    )


def main() -> None:
    st.set_page_config(
        page_title=EXPERIMENT["title"],
        page_icon="🧪",
        layout="wide",
    )
    initialize_state()

    st.title(EXPERIMENT["title"])
    st.caption(
        f"Knowledge Graph & Information Retrieval Virtual Laboratory · "
        f"Experiment {EXPERIMENT['number']} · Estimated time: {EXPERIMENT['estimated_minutes']}"
    )

    section = st.sidebar.radio(
        "Lab Navigator",
        ["Theory", "Simulation", "Quiz", "Report Generation"],
    )
    st.sidebar.divider()
    st.sidebar.subheader("Progress Tracker")
    st.sidebar.write(f"- **Trials recorded:** {len(st.session_state['trials'])}")
    quiz_status = "Done" if st.session_state["quiz_submitted"] else "Pending"
    st.sidebar.write(f"- **Quiz status:** {quiz_status}")
    if st.session_state["quiz_submitted"]:
        st.sidebar.write(
            f"- **Quiz score:** {st.session_state['quiz_score']}/{len(QUIZ_QUESTIONS)}"
        )
    st.sidebar.divider()
    st.sidebar.caption(f"Expected outcome: {EXPERIMENT['expected_outcome']}")
    st.sidebar.caption("Neo4j is not required for this experiment.")

    if section == "Theory":
        render_theory_section()
    elif section == "Simulation":
        render_simulation_section()
    elif section == "Quiz":
        render_quiz_section()
    else:
        render_report_section()


if __name__ == "__main__":
    main()
