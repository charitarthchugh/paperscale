"""Tests for the vLLM /metrics scraper."""

import pathlib
import unittest

from paperscale.vllm_stats import Snapshot, metrics_url, parse_metrics, snapshot_from  # noqa: F401

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


class SnapshotTest(unittest.TestCase):
    def setUp(self):
        self.snap = snapshot_from(parse_metrics(FIXTURE.read_text()))

    def test_counters_summed_across_engines(self):
        self.assertEqual(self.snap.generation_tokens, 300.0)  # 200 + 100
        self.assertEqual(self.snap.prompt_tokens, 1500.0)  # 1000 + 500
        self.assertEqual(self.snap.cache_hits, 1200.0)  # 900 + 300
        self.assertEqual(self.snap.cache_queries, 1500.0)  # 1000 + 500

    def test_request_gauges_summed(self):
        self.assertEqual(self.snap.running, 8.0)  # 5 + 3
        self.assertEqual(self.snap.waiting, 2.0)  # 2 + 0

    def test_kv_usage_averaged_not_summed(self):
        # A fraction summed across engines would read 1.0 and look like a full cache.
        self.assertAlmostEqual(self.snap.kv_usage, 0.5)  # mean(0.4, 0.6)

    def test_absent_metric_is_none_not_zero(self):
        snap = snapshot_from(parse_metrics('vllm:num_requests_running{engine="0"} 1.0\n'))
        self.assertIsNone(snap.generation_tokens)
        self.assertEqual(snap.running, 1.0)

    def test_falls_back_to_alternate_names(self):
        text = 'vllm:prompt_tokens_total{engine="0"} 100.0\nvllm:prompt_tokens_cached_total{engine="0"} 60.0\nvllm:gpu_cache_usage_perc{engine="0"} 0.25\n'
        snap = snapshot_from(parse_metrics(text))
        self.assertEqual(snap.cache_hits, 60.0)  # prefix_cache_hits_total absent
        self.assertEqual(snap.cache_queries, 100.0)  # prefix_cache_queries_total absent
        self.assertEqual(snap.kv_usage, 0.25)  # kv_cache_usage_perc absent


class MetricsUrlTest(unittest.TestCase):
    def test_strips_v1_suffix(self):
        self.assertEqual(metrics_url("http://localhost:8000/v1"), "http://localhost:8000/metrics")

    def test_handles_trailing_slash(self):
        self.assertEqual(metrics_url("http://localhost:8000/v1/"), "http://localhost:8000/metrics")

    def test_bare_base_url(self):
        self.assertEqual(metrics_url("http://gigaspark:8000"), "http://gigaspark:8000/metrics")


if __name__ == "__main__":
    unittest.main()
