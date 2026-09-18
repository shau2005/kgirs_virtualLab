import unittest

from core import (
    PipelineOptions,
    build_inverted_index,
    lemmatize_word,
    run_pipeline,
    search_documents,
    split_documents,
    stem_word,
    tokenize,
)


class PipelineTests(unittest.TestCase):
    def test_pipeline_normalizes_filters_and_lemmatizes(self):
        result = run_pipeline(
            "The CAFÉ was connecting 2 systems!",
            PipelineOptions(remove_numbers=True, morphology="Lemmatization"),
        )
        self.assertEqual(result.normalized, "the cafe was connecting systems")
        self.assertEqual(result.final_tokens, ["cafe", "connect", "system"])
        self.assertIn("the", result.removed_stopwords)
        self.assertIn("was", result.removed_stopwords)

    def test_contraction_is_expanded_before_punctuation(self):
        result = run_pipeline("Researchers don't stop.", PipelineOptions(remove_stopwords=False))
        self.assertEqual(result.tokens, ["researchers", "do", "not", "stop"])

    def test_irregular_lemmas(self):
        self.assertEqual(lemmatize_word("were"), "be")
        self.assertEqual(lemmatize_word("children"), "child")

    def test_stemmer(self):
        self.assertEqual(stem_word("connecting"), "connect")
        self.assertEqual(stem_word("studies"), "study")

    def test_tokenizer_retains_internal_apostrophe(self):
        self.assertEqual(tokenize("don't split 2026"), ["don't", "split", "2026"])

    def test_empty_input(self):
        result = run_pipeline("", PipelineOptions())
        self.assertEqual(result.final_token_count, 0)
        self.assertEqual(result.reduction_percent, 0.0)

    def test_inverted_index_consolidates_inflected_forms(self):
        documents = ["Graphs connect entities.", "A graph connects data."]
        options = PipelineOptions(morphology="Lemmatization")
        index = build_inverted_index(documents, options)
        self.assertEqual(index["graph"], [1, 2])
        self.assertEqual(index["connect"], [1, 2])

    def test_processed_search_matches_word_variation(self):
        documents = split_documents("Systems are connecting entities. A report describes weather.")
        options = PipelineOptions(morphology="Lemmatization")
        results = search_documents("connect entity", documents, options)
        self.assertEqual(results[0]["Document"], "D1")
        self.assertGreater(results[0]["Similarity"], 0)


if __name__ == "__main__":
    unittest.main()
