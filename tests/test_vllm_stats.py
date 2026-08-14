"""Tests for the vLLM /metrics scraper."""

import pathlib
import unittest
from unittest import mock

from paperscale.vllm_stats import Rates, Snapshot, VLLMStats, VLLMStatsPoller, format_rate, metrics_url, parse_metrics, push_vllm_stats, snapshot_from  # noqa: F401

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


class _FakeClock:
    """Deterministic monotonic clock. Advance with `tick`."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float) -> None:
        self.now += seconds


class VLLMStatsTest(unittest.TestCase):
    def setUp(self):
        self.clock = _FakeClock()
        self.stats = VLLMStats(window=60.0, clock=self.clock)

    def test_single_sample_has_no_rate(self):
        # One point cannot make a rate; must be `-`, not 0.
        self.stats.add(Snapshot(generation_tokens=100.0))
        self.assertIsNone(self.stats.rates().gen_tps)

    def test_windowed_rate_from_two_samples(self):
        self.stats.add(Snapshot(generation_tokens=100.0))
        self.clock.tick(10.0)
        self.stats.add(Snapshot(generation_tokens=600.0))
        self.assertAlmostEqual(self.stats.rates().gen_tps, 50.0)  # 500 tokens / 10 s

    def test_window_trims_old_samples(self):
        self.stats.add(Snapshot(generation_tokens=0.0))
        self.clock.tick(120.0)  # older than the 60 s window
        self.stats.add(Snapshot(generation_tokens=1200.0))
        self.clock.tick(10.0)
        self.stats.add(Snapshot(generation_tokens=1700.0))
        # Window holds only the last two samples: 500 tokens over 10 s.
        self.assertAlmostEqual(self.stats.rates().gen_tps, 50.0)

    def test_avg_spans_since_first_sample(self):
        self.stats.add(Snapshot(generation_tokens=0.0))
        self.clock.tick(100.0)
        self.stats.add(Snapshot(generation_tokens=1000.0))
        self.assertAlmostEqual(self.stats.rates().gen_tps_avg, 10.0)  # 1000 / 100 s

    def test_kv_hit_is_a_windowed_ratio(self):
        self.stats.add(Snapshot(cache_hits=0.0, cache_queries=0.0))
        self.clock.tick(10.0)
        self.stats.add(Snapshot(cache_hits=80.0, cache_queries=100.0))
        self.assertAlmostEqual(self.stats.rates().kv_hit, 0.8)

    def test_counter_reset_yields_none_not_negative(self):
        # A vLLM restart sends counters backwards. A negative tok/s is nonsense.
        self.stats.add(Snapshot(generation_tokens=1000.0))
        self.clock.tick(10.0)
        self.stats.add(Snapshot(generation_tokens=5.0))
        rates = self.stats.rates()
        self.assertIsNone(rates.gen_tps)
        self.assertIsNone(rates.gen_tps_avg)

    def test_recovers_after_counter_reset(self):
        self.stats.add(Snapshot(generation_tokens=1000.0))
        self.clock.tick(10.0)
        self.stats.add(Snapshot(generation_tokens=5.0))  # reset, window cleared
        self.clock.tick(10.0)
        self.stats.add(Snapshot(generation_tokens=105.0))
        self.assertAlmostEqual(self.stats.rates().gen_tps, 10.0)

    def test_zero_elapsed_time_yields_none(self):
        self.stats.add(Snapshot(generation_tokens=100.0))
        self.stats.add(Snapshot(generation_tokens=200.0))  # same instant
        self.assertIsNone(self.stats.rates().gen_tps)

    def test_zero_queries_in_window_yields_none(self):
        self.stats.add(Snapshot(cache_hits=10.0, cache_queries=50.0))
        self.clock.tick(10.0)
        self.stats.add(Snapshot(cache_hits=10.0, cache_queries=50.0))  # idle server
        self.assertIsNone(self.stats.rates().kv_hit)

    def test_gauges_come_from_the_latest_sample(self):
        self.stats.add(Snapshot(running=1.0, waiting=9.0, kv_usage=0.1))
        self.clock.tick(5.0)
        self.stats.add(Snapshot(running=4.0, waiting=0.0, kv_usage=0.7))
        rates = self.stats.rates()
        self.assertEqual(rates.running, 4.0)
        self.assertEqual(rates.waiting, 0.0)
        self.assertAlmostEqual(rates.kv_usage, 0.7)

    def test_empty_stats_are_all_none(self):
        rates = self.stats.rates()
        self.assertIsNone(rates.gen_tps)
        self.assertIsNone(rates.running)

    def test_missing_counter_does_not_break_others(self):
        self.stats.add(Snapshot(generation_tokens=None, prompt_tokens=0.0))
        self.clock.tick(10.0)
        self.stats.add(Snapshot(generation_tokens=None, prompt_tokens=100.0))
        rates = self.stats.rates()
        self.assertIsNone(rates.gen_tps)
        self.assertAlmostEqual(rates.prompt_tps, 10.0)

    def test_none_masked_regression_across_window_is_still_a_reset(self):
        # A transient partial scrape (generation_tokens missing) sits between
        # the pre-restart sample and the post-restart sample. `_is_reset` must
        # not be fooled just because the immediately-preceding sample can't
        # prove the regression itself.
        self.stats.add(Snapshot(generation_tokens=1000.0, prompt_tokens=0.0))
        self.clock.tick(10.0)
        self.stats.add(Snapshot(generation_tokens=None, prompt_tokens=50.0))
        self.clock.tick(10.0)
        self.stats.add(Snapshot(generation_tokens=5.0, prompt_tokens=100.0))
        rates = self.stats.rates()
        self.assertIsNone(rates.gen_tps)
        self.assertIsNone(rates.gen_tps_avg)
        for value in (rates.gen_tps, rates.gen_tps_avg, rates.prompt_tps, rates.prompt_tps_avg, rates.kv_hit, rates.kv_hit_avg):
            if value is not None:
                self.assertGreaterEqual(value, 0.0)


class VLLMStatsPollerTest(unittest.TestCase):
    def setUp(self):
        self.stats = VLLMStats(clock=_FakeClock())

    def test_tick_feeds_stats(self):
        poller = VLLMStatsPoller("http://x/metrics", self.stats, fetch=lambda url: FIXTURE.read_text())
        self.assertTrue(poller._tick())
        self.assertTrue(poller.available)
        self.assertEqual(self.stats.rates().running, 8.0)

    def test_connection_error_marks_unavailable_without_raising(self):
        def boom(url):
            raise OSError("connection refused")

        poller = VLLMStatsPoller("http://x/metrics", self.stats, fetch=boom)
        self.assertFalse(poller._tick())  # must not raise
        self.assertFalse(poller.available)

    def test_http_error_marks_unavailable(self):
        # A backend with no /metrics (qianfan, surya) 404s. Not fatal.
        def not_found(url):
            raise RuntimeError("404")

        poller = VLLMStatsPoller("http://x/metrics", self.stats, fetch=not_found)
        self.assertFalse(poller._tick())
        self.assertFalse(poller.available)

    def test_malformed_body_marks_unavailable(self):
        poller = VLLMStatsPoller("http://x/metrics", self.stats, fetch=lambda url: "<html>not prometheus</html>")
        self.assertFalse(poller._tick())
        self.assertFalse(poller.available)

    def test_warns_once_then_debug(self):
        def boom(url):
            raise OSError("nope")

        poller = VLLMStatsPoller("http://x/metrics", self.stats, fetch=boom)
        with mock.patch("paperscale.vllm_stats.logger") as log:
            poller._tick()
            poller._tick()
            poller._tick()
        self.assertEqual(log.warning.call_count, 1)
        self.assertEqual(log.debug.call_count, 2)

    def test_recovers_after_failure(self):
        calls = {"n": 0}

        def flaky(url):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("nope")
            return FIXTURE.read_text()

        poller = VLLMStatsPoller("http://x/metrics", self.stats, fetch=flaky)
        self.assertFalse(poller._tick())
        self.assertTrue(poller._tick())
        self.assertTrue(poller.available)

    def test_stop_is_safe_before_start(self):
        poller = VLLMStatsPoller("http://x/metrics", self.stats, fetch=lambda url: "")
        poller.stop()  # must not raise


class _RecordingRep:
    def __init__(self):
        self.stats = {}

    def set_stat(self, name, value, *, group="run"):
        self.stats.setdefault(group, {})[name] = value


class FormatRateTest(unittest.TestCase):
    def test_none_renders_as_dash_not_zero(self):
        self.assertEqual(format_rate(None), "-")

    def test_zero_is_a_real_measurement(self):
        self.assertEqual(format_rate(0.0), "0")

    def test_thousands_are_abbreviated(self):
        self.assertEqual(format_rate(6100.0), "6.1k")


class PushVllmStatsTest(unittest.TestCase):
    def test_unavailable_poller_shows_status_only(self):
        rep = _RecordingRep()
        poller = VLLMStatsPoller("http://x/metrics", self.stats_obj(), fetch=lambda url: "")
        poller.available = False
        push_vllm_stats(rep, self.stats_obj(), poller)
        self.assertEqual(rep.stats["vllm"], {"status": "unavailable"})

    def test_populates_all_rows_when_available(self):
        clock = _FakeClock()
        stats = VLLMStats(clock=clock)
        stats.add(Snapshot(generation_tokens=0.0, prompt_tokens=0.0, cache_hits=0.0, cache_queries=0.0, running=8.0, waiting=2.0))
        clock.tick(10.0)
        stats.add(Snapshot(generation_tokens=4120.0, prompt_tokens=100.0, cache_hits=93.0, cache_queries=100.0, running=8.0, waiting=2.0))
        rep = _RecordingRep()
        push_vllm_stats(rep, stats, None)
        self.assertIn("412 tok/s", rep.stats["vllm"]["gen"])
        self.assertEqual(rep.stats["vllm"]["kv hit"], "93%")

    def test_no_stats_is_a_noop(self):
        rep = _RecordingRep()
        push_vllm_stats(rep, None, None)
        self.assertEqual(rep.stats, {})

    def stats_obj(self):
        return VLLMStats(clock=_FakeClock())


if __name__ == "__main__":
    unittest.main()
