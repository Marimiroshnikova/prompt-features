"""Tests for the prompt feature engine.

    python -m unittest discover tests

The headline test walks the registry and checks that every documented example
produces the documented value, so FEATURES.md can never claim something the
code does not do.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features import extract_features, extract_top_features  # noqa: E402
from promptfeat import REGISTRY, TOP30_FEATURES, explain_prompt  # noqa: E402
from promptfeat.doc import PromptDoc  # noqa: E402
from promptfeat.engine import compute  # noqa: E402
from promptfeat.registry import (  # noqa: E402
    NOT_APPLICABLE,
    OK,
    UNAVAILABLE,
    UNDEFINED,
    UNRELIABLE,
)

ADVERSARIAL = [
    "",
    " ",
    "\n\n\n",
    "?",
    "a",
    "Who wrote The Hobbit?",
    "I don\u2019t want NSAIDs",  # curly apostrophe
    "1,000 people in the 1990s paid $2.5 million",
    "COVID-19 and v1.2.3 and JIRA-4821",
    "WHY IS EVERYTHING IN CAPS",
    "\U0001F600 \U0001F4DA \U0001F680",
    "```python\ndef f(x):\n    return x\n```\nWhat does this do?",
    "\u00bfQui\u00e9n escribi\u00f3 El Hobbit?",
    "Compare " + "very " * 400 + "long prompt about nothing in particular.",
    "Example 1: a\nExample 2: b\nExample 3: c\nWhat is next?",
    "the the the the the the",
    "\t\ttabs\tand\tspaces\t",
    "email me at a@b.co or see https://example.com/x?y=1#z",
]


def value_of(prompt: str, name: str):
    _, ctx = compute(prompt)
    return ctx.results[name].value


def result_of(prompt: str, name: str):
    _, ctx = compute(prompt)
    return ctx.results[name]


class TestDocumentedExamples(unittest.TestCase):
    """Every example in FEATURES.md must match what the code returns."""

    def test_examples_match(self):
        checked = 0
        failures = []
        for name, feature in REGISTRY.items():
            if not feature.has_expected:
                continue
            checked += 1
            actual = value_of(feature.example, name)
            if actual != feature.expected:
                failures.append(
                    f"{name}: documented {feature.expected!r}, got {actual!r} "
                    f"for {feature.example!r}"
                )
        self.assertGreater(checked, 100, "most features should document an example")
        self.assertEqual(failures, [], "\n".join(failures))

    def test_every_feature_is_documented(self):
        for name, feature in REGISTRY.items():
            with self.subTest(feature=name):
                self.assertTrue(feature.summary, f"{name} has no summary")
                self.assertTrue(feature.formula, f"{name} has no formula")
                self.assertTrue(feature.why, f"{name} has no retrieval rationale")
                self.assertTrue(feature.example, f"{name} has no example")
                self.assertIn(feature.dtype, ("int", "float", "bool", "label"))


class TestTopThirty(unittest.TestCase):
    def test_exactly_thirty_ranked_features(self):
        self.assertEqual(len(TOP30_FEATURES), 30)

    def test_ranks_are_one_to_thirty_without_gaps(self):
        ranks = sorted(f.rank for f in REGISTRY.values() if f.tier == 1)
        self.assertEqual(ranks, list(range(1, 31)))

    def test_each_ranked_feature_explains_its_rank(self):
        for feature in REGISTRY.values():
            if feature.tier == 1:
                with self.subTest(feature=feature.name):
                    self.assertTrue(feature.rank_reason)

    def test_extract_top_features_returns_only_those(self):
        top = extract_top_features("Who wrote The Hobbit?")
        self.assertEqual(list(top), TOP30_FEATURES)


class TestRegressionsFromTheOldImplementation(unittest.TestCase):
    """Each of these was a measured bug in the original features.py."""

    def test_named_entity_hint_ignores_boilerplate_and_sentence_starts(self):
        prompt = (
            "Instructions:\nUse the docs only.\n"
            "Compare ibuprofen and aspirin for fever in children after 2020. "
            "What dose is safe? What should be avoided?"
        )
        # The old heuristic counted Instructions, Use, Compare and What as four
        # "entities". There are no real named entities in this prompt.
        self.assertEqual(value_of(prompt, "named_entity_hint"), 0)

    def test_numeral_count_handles_thousands_separators(self):
        self.assertEqual(value_of("1,000 people attended", "numeral_count"), 1)

    def test_numeral_count_sees_decades(self):
        self.assertEqual(value_of("music from the 1990s", "numeral_count"), 1)

    def test_numeral_count_ignores_digits_glued_to_names(self):
        self.assertEqual(value_of("COVID-19 cases in 2020", "numeral_count"), 1)

    def test_numeral_count_ignores_version_strings(self):
        self.assertEqual(value_of("upgrade to v1.2.3 today", "numeral_count"), 0)

    def test_percent_is_counted_separately(self):
        self.assertEqual(value_of("revenue grew 12%", "percent_count"), 1)

    def test_negation_survives_a_curly_apostrophe(self):
        self.assertTrue(value_of("I don\u2019t want NSAIDs", "contains_negation"))

    def test_specific_short_question_is_not_ambiguous(self):
        self.assertFalse(value_of("Who wrote The Hobbit?", "is_ambiguous"))

    def test_vague_longer_question_is_ambiguous(self):
        self.assertTrue(
            value_of("Can you tell me more about that thing we discussed?", "is_ambiguous")
        )

    def test_fact_question_is_categorised_as_fact_retrieval(self):
        self.assertEqual(value_of("Who wrote The Hobbit?", "question_category"), "Fact Retrieval")

    def test_creative_beats_coding_when_evidence_is_stronger(self):
        self.assertEqual(
            value_of("Write a story about a python programmer", "question_category"),
            "Creative",
        )

    def test_multi_part_detects_coordinated_imperatives(self):
        prompt = "Summarize the 2023 filing and compare it to 2022."
        self.assertTrue(value_of(prompt, "is_multi_part"))
        self.assertEqual(prompt.count("?"), 0, "no question marks, so the old rule failed")

    def test_relative_recency_is_a_temporal_constraint(self):
        self.assertTrue(value_of("What are the latest treatments?", "has_temporal_constraint"))
        self.assertTrue(value_of("What are the latest treatments?", "has_relative_recency"))

    def test_task_verb_is_not_taken_from_a_question_body(self):
        # "wrote" lemmatises to "write", but this prompt asks for no task.
        result = result_of("Who wrote The Hobbit?", "primary_task_verb")
        self.assertEqual(result.status, NOT_APPLICABLE)

    def test_parse_depth_is_bounded(self):
        depth = value_of("Who wrote The Hobbit?", "max_parse_depth")
        self.assertLess(depth, 10, "walking to the root must stop at the root")

    def test_english_from_and_return_are_not_code(self):
        prompt = "Think step by step: why did the 2008 crisis differ from 1929?"
        self.assertFalse(value_of(prompt, "has_code_span"))
        self.assertEqual(value_of(prompt, "question_category"), "Reasoning")

    def test_real_code_is_still_detected(self):
        self.assertTrue(value_of("Why does my_function(x) raise a TypeError?", "has_code_span"))
        self.assertTrue(value_of("import pandas as pd\nWhat does this do?", "has_code_span"))

    def test_inline_instruction_header_does_not_swallow_the_question(self):
        prompt = "Instructions: use only the docs. Compare aspirin and ibuprofen."
        self.assertIn("Compare aspirin and ibuprofen", PromptDoc(prompt).core_text)
        ratio = value_of(prompt, "core_question_ratio")
        self.assertGreater(ratio, 0.0, "the question must not be lost as boilerplate")
        self.assertLess(ratio, 1.0, "the instruction must still be recognised")

    def test_politeness_padding_does_not_make_a_prompt_look_specific(self):
        bare = "Who wrote The Hobbit?"
        polite = "Hi, could you please tell me who wrote The Hobbit? Thanks!"
        self.assertGreater(
            value_of(polite, "retrieval_difficulty_score"),
            value_of(bare, "retrieval_difficulty_score"),
            "courtesy filler dilutes the query, so it cannot lower the risk",
        )
        self.assertNotIn("Thanks", [w.text for w in PromptDoc(polite).content_words])

    def test_non_english_prompt_is_high_risk(self):
        spanish = "\u00bfQui\u00e9n escribi\u00f3 El Hobbit y en qu\u00e9 a\u00f1o?"
        self.assertFalse(value_of(spanish, "is_english"))
        self.assertGreater(value_of(spanish, "retrieval_difficulty_score"), 0.6)

    def test_item_count_covers_output_shape_nouns(self):
        prompt = "Compare the two vendors and summarise in 3 bullets."
        self.assertEqual(value_of(prompt, "requested_item_count"), 3)
        self.assertTrue(value_of(prompt, "has_enumeration_request"))

    def test_numeric_length_budget_is_a_length_limit(self):
        self.assertTrue(value_of("Explain the CAP theorem in 200 words", "has_length_limit"))
        self.assertTrue(value_of("Summarise this in two sentences.", "has_length_limit"))
        self.assertFalse(value_of("Who wrote The Hobbit?", "has_length_limit"))

    def test_length_units_are_not_counted_as_requested_items(self):
        for prompt in ("Explain the CAP theorem in 200 words", "Summarise this in two sentences."):
            with self.subTest(prompt=prompt):
                self.assertEqual(result_of(prompt, "requested_item_count").status, NOT_APPLICABLE)


class TestStatusRules(unittest.TestCase):
    """A feature that cannot be measured must say so, not return zero."""

    def test_year_span_is_not_applicable_without_a_year(self):
        result = result_of("Who wrote The Hobbit?", "year_span")
        self.assertEqual(result.status, NOT_APPLICABLE)
        self.assertIsNone(result.value)
        self.assertIn("no year", result.reason)

    def test_year_span_is_zero_for_a_single_year(self):
        result = result_of("What happened in 2020?", "year_span")
        self.assertEqual(result.status, OK)
        self.assertEqual(result.value, 0)

    def test_readability_is_unreliable_on_short_prompts(self):
        self.assertEqual(result_of("a", "question_complexity_score").status, UNRELIABLE)

    def test_readability_is_ok_on_longer_prompts(self):
        prompt = (
            "Compare the pharmacokinetics of ibuprofen and acetylsalicylic acid "
            "in paediatric patients presenting with fever."
        )
        self.assertEqual(result_of(prompt, "question_complexity_score").status, OK)

    def test_ratio_is_undefined_on_an_empty_prompt(self):
        result = result_of("", "avg_words_per_sentence")
        self.assertEqual(result.status, UNDEFINED)
        self.assertIsNone(result.value)

    def test_mtld_is_unreliable_below_fifty_tokens(self):
        self.assertEqual(result_of("Who wrote The Hobbit?", "mtld").status, UNRELIABLE)

    def test_language_confidence_is_unreliable_on_very_short_text(self):
        self.assertEqual(result_of("Hi?", "language_confidence").status, UNRELIABLE)

    def test_not_applicable_labels_still_carry_a_fallback_value(self):
        result = result_of("Summarize this.", "question_type")
        self.assertEqual(result.status, NOT_APPLICABLE)
        self.assertEqual(result.value, "other")

    def test_with_status_exposes_reasons(self):
        row = extract_features("Who wrote The Hobbit?", with_status=True)
        self.assertEqual(row["year_span__status"], NOT_APPLICABLE)
        self.assertTrue(row["year_span__reason"])
        self.assertIsNone(row["year_span"])

    def test_statuses_are_from_the_known_set(self):
        allowed = {OK, NOT_APPLICABLE, UNDEFINED, UNRELIABLE, UNAVAILABLE}
        for prompt in ADVERSARIAL:
            _, ctx = compute(prompt)
            for name, result in ctx.results.items():
                with self.subTest(prompt=prompt[:24], feature=name):
                    self.assertIn(result.status, allowed)

    def test_no_feature_ever_raises(self):
        for prompt in ADVERSARIAL:
            _, ctx = compute(prompt)
            for name, result in ctx.results.items():
                if result.status == UNDEFINED and "raised" in result.reason:
                    self.fail(f"{name} raised on {prompt[:40]!r}: {result.reason}")

    def test_a_non_ok_status_always_has_a_reason(self):
        for prompt in ADVERSARIAL:
            _, ctx = compute(prompt)
            for name, result in ctx.results.items():
                if result.status != OK:
                    with self.subTest(prompt=prompt[:24], feature=name):
                        self.assertTrue(result.reason, f"{name} gave no reason")


class TestPromptDoc(unittest.TestCase):
    def test_normalisation_preserves_offsets(self):
        for prompt in ADVERSARIAL:
            with self.subTest(prompt=prompt[:24]):
                doc = PromptDoc(prompt)
                self.assertEqual(len(doc.text), len(doc.raw))

    def test_spans_stay_inside_the_prompt(self):
        for prompt in ADVERSARIAL:
            _, ctx = compute(prompt)
            for name, result in ctx.results.items():
                for span in result.spans:
                    with self.subTest(prompt=prompt[:24], feature=name):
                        self.assertGreaterEqual(span["start"], 0)
                        self.assertLessEqual(span["end"], len(prompt))

    def test_core_question_strips_instruction_scaffolding(self):
        doc = PromptDoc(
            "Instructions:\nUse the docs only.\nWho wrote The Hobbit?"
        )
        self.assertEqual(doc.core_text, "Who wrote The Hobbit?")

    def test_core_question_strips_few_shot_examples(self):
        doc = PromptDoc("Example 1: Paris. What is the capital of France?")
        self.assertIn("capital of France", doc.core_text)
        self.assertNotIn("Example 1", doc.core_text)

    def test_backward_compatible_raw_counts(self):
        prompt = "Who wrote The Hobbit?"
        self.assertEqual(value_of(prompt, "question_length_words"), len(prompt.split()))
        self.assertEqual(value_of(prompt, "question_length_chars"), len(prompt))


class TestPublicApi(unittest.TestCase):
    def test_extract_features_returns_every_registered_name(self):
        row = extract_features("Who wrote The Hobbit?")
        self.assertEqual(list(row), list(REGISTRY))

    def test_extract_features_is_json_safe(self):
        import json

        for prompt in ADVERSARIAL:
            with self.subTest(prompt=prompt[:24]):
                json.dumps(extract_features(prompt))

    def test_explain_prompt_shape(self):
        report = explain_prompt("Who wrote The Hobbit?")
        self.assertEqual(len(report["features"]), len(REGISTRY))
        self.assertEqual(len(report["top"]), 30)
        self.assertTrue(report["groups"])
        for item in report["features"]:
            self.assertIn("formula", item)
            self.assertIn("steps", item)

    def test_difficulty_score_ranks_prompts_sensibly(self):
        vague = value_of("What about it?", "retrieval_difficulty_score")
        specific = value_of("Who wrote The Hobbit?", "retrieval_difficulty_score")
        self.assertGreater(vague, specific)

    def test_difficulty_score_stays_in_range(self):
        for prompt in ADVERSARIAL:
            value = value_of(prompt, "retrieval_difficulty_score")
            with self.subTest(prompt=prompt[:24]):
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
