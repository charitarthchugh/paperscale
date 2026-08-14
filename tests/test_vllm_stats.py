"""Tests for the vLLM /metrics scraper."""

import pathlib
import unittest

from paperscale.vllm_stats import parse_metrics

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "vllm_metrics.txt"


class ParseMetricsTest(unittest.TestCase):
    def setUp(self):
        self.parsed = parse_metrics(FIXTURE.read_text())

    def test_skips_comments(self):
        self.assertNotIn("# HELP", self.parsed)
        for name in self.parsed:
            self.assertFalse(name.startswith("#"))

    def test_groups_samples_by_metric_name(self):
        samples = self.parsed["vllm:generation_tokens_total"]
        self.assertEqual(len(samples), 2)
        self.assertEqual({s.value for s in samples}, {200.0, 100.0})

    def test_parses_labels(self):
        samples = self.parsed["vllm:num_requests_running"]
        engines = sorted(s.labels["engine"] for s in samples)
        self.assertEqual(engines, ["0", "1"])

    def test_created_series_is_a_separate_metric(self):
        # _created siblings must not contaminate the counter they accompany.
        self.assertEqual(len(self.parsed["vllm:prefix_cache_queries_total"]), 2)
        self.assertIn("vllm:prefix_cache_queries_created", self.parsed)

    def test_multi_label_series_parsed(self):
        samples = self.parsed["vllm:num_requests_waiting_by_reason"]
        self.assertEqual(samples[0].labels["reason"], "capacity")

    def test_unlabelled_series(self):
        parsed = parse_metrics("some_metric 42.0\n")
        self.assertEqual(parsed["some_metric"][0].value, 42.0)
        self.assertEqual(parsed["some_metric"][0].labels, {})

    def test_ignores_blank_and_malformed_lines(self):
        parsed = parse_metrics("\n\ngarbage line here\nok_metric 1.0\n")
        self.assertIn("ok_metric", parsed)
        self.assertNotIn("garbage", parsed)

    def test_ignores_non_finite_values(self):
        parsed = parse_metrics("a_metric NaN\nb_metric 1.0\n")
        self.assertNotIn("a_metric", parsed)
        self.assertIn("b_metric", parsed)


if __name__ == "__main__":
    unittest.main()
