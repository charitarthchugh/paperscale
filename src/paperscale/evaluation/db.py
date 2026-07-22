"""SQLite persistence + leaderboard for reference-free OCR evaluation.

One table per metric; one ``pplx_<model>`` table per model (perplexity columns
vary per scorer and are queried per-model). Every ``write_*`` is idempotent:
it deletes the rows it is about to (re)write for the affected model(s) first,
so re-running an evaluation replaces rather than duplicates.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Iterable

# Static per-metric schema. pplx tables are created on demand (see _pplx_table).
_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    model TEXT PRIMARY KEY,
    input_path TEXT,
    pplx_scorer_model TEXT
);
CREATE TABLE IF NOT EXISTS correction_rate (
    model TEXT, doc TEXT, page INTEGER, correction_rate REAL, uncorrectable_rate REAL
);
CREATE TABLE IF NOT EXISTS garbage_fraction (
    model TEXT, doc TEXT, page INTEGER, score REAL
);
CREATE TABLE IF NOT EXISTS peer_agreement (
    model TEXT, peer TEXT, doc TEXT, page INTEGER, bow_f1 REAL, one_minus_ned REAL
);
CREATE TABLE IF NOT EXISTS textlayer_agreement (
    model TEXT, doc TEXT, page INTEGER, bow_f1 REAL, one_minus_ned REAL
);
CREATE TABLE IF NOT EXISTS reject_rate (
    model TEXT, doc TEXT, fallback_pages INTEGER, total_pages INTEGER
);
"""

_PPLX_COLS = (
    "n_tokens_raw",
    "sum_logprob_raw",
    "ppl_raw",
    "n_tokens_corrected",
    "sum_logprob_corrected",
    "ppl_corrected",
)


def _pplx_table(model: str) -> str:
    return "pplx_" + re.sub(r"\W+", "_", model)


def _distinct_models(rows: list, idx: int = 0) -> set:
    return {r[idx] for r in rows}


class EvalDB:
    def __init__(self, path: str | Path) -> None:
        self.conn = sqlite3.connect(str(path))
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # --- writers ---------------------------------------------------------

    def register_run(
        self, model: str, input_path: str, pplx_scorer_model: str | None = None
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO runs (model, input_path, pplx_scorer_model) VALUES (?, ?, ?)",
            (model, input_path, pplx_scorer_model),
        )
        self.conn.commit()

    def _replace(self, table: str, rows: Iterable, cols: tuple[str, ...]) -> None:
        rows = list(rows)
        for model in _distinct_models(rows):
            self.conn.execute(f"DELETE FROM {table} WHERE model = ?", (model,))
        placeholders = ", ".join("?" * len(cols))
        self.conn.executemany(
            f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})", rows
        )
        self.conn.commit()

    def write_correction_rate(self, rows) -> None:
        self._replace(
            "correction_rate", rows, ("model", "doc", "page", "correction_rate", "uncorrectable_rate")
        )

    def write_garbage_fraction(self, rows) -> None:
        self._replace("garbage_fraction", rows, ("model", "doc", "page", "score"))

    def write_peer_agreement(self, rows) -> None:
        self._replace(
            "peer_agreement",
            rows,
            ("model", "peer", "doc", "page", "bow_f1", "one_minus_ned"),
        )

    def write_textlayer_agreement(self, rows) -> None:
        self._replace(
            "textlayer_agreement",
            rows,
            ("model", "doc", "page", "bow_f1", "one_minus_ned"),
        )

    def write_reject_rate(self, rows) -> None:
        self._replace(
            "reject_rate", rows, ("model", "doc", "fallback_pages", "total_pages")
        )

    def write_pplx(self, model: str, rows) -> None:
        table = _pplx_table(model)
        cols = ("doc", "page") + _PPLX_COLS
        coldefs = "doc TEXT, page INTEGER, " + ", ".join(f"{c} REAL" for c in _PPLX_COLS)
        # n_tokens columns are integers, but REAL stores them faithfully via sqlite affinity.
        self.conn.execute(f"DROP TABLE IF EXISTS {table}")
        self.conn.execute(f"CREATE TABLE {table} ({coldefs})")
        placeholders = ", ".join("?" * len(cols))
        self.conn.executemany(
            f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})",
            list(rows),
        )
        self.conn.commit()

    # --- leaderboard -----------------------------------------------------

    def _pplx_tables(self) -> list[str]:
        cur = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'pplx_%'"
        )
        return [r[0] for r in cur.fetchall()]

    def _models(self) -> list[str]:
        """All models seen across any table (union), sorted for determinism."""
        models: set = set()
        for table in ("runs", "correction_rate", "garbage_fraction", "peer_agreement",
                      "textlayer_agreement", "reject_rate"):
            cur = self.conn.execute(f"SELECT DISTINCT model FROM {table}")
            models.update(r[0] for r in cur.fetchall())
        return sorted(models)

    def _per_doc_means(self, table: str, value_expr: str) -> dict[str, dict[str, float]]:
        """{model: {doc: mean(value_expr) over that model+doc's rows}}."""
        cur = self.conn.execute(
            f"SELECT model, doc, AVG({value_expr}) FROM {table} GROUP BY model, doc"
        )
        out: dict[str, dict[str, float]] = {}
        for model, doc, mean in cur.fetchall():
            out.setdefault(model, {})[doc] = mean
        return out

    def _overall_mean(self, table: str, value_expr: str) -> dict[str, float]:
        cur = self.conn.execute(
            f"SELECT model, AVG({value_expr}) FROM {table} GROUP BY model"
        )
        return {model: mean for model, mean in cur.fetchall()}

    @staticmethod
    def _win_rate(per_doc: dict[str, dict[str, float]], higher_better: bool) -> dict[str, str]:
        """Fraction of common docs where a model is best. 'n/a' if <2 models / no common docs."""
        models = list(per_doc)
        if len(models) < 2:
            return {m: "n/a" for m in models}
        common = set.intersection(*(set(per_doc[m]) for m in models)) if models else set()
        if not common:
            return {m: "n/a" for m in models}
        wins = {m: 0 for m in models}
        for doc in common:
            vals = {m: per_doc[m][doc] for m in models}
            best = max(vals.values()) if higher_better else min(vals.values())
            for m, v in vals.items():
                if v == best:
                    wins[m] += 1
        n = len(common)
        return {m: f"{wins[m] / n:.2f}" for m in models}

    def leaderboard(self) -> str:
        models = self._models()
        if not models:
            return "(no models)"

        # Each spec: (col-label, {model: mean-str}, {model: win-rate-str})
        columns: list[tuple[str, dict[str, str], dict[str, str] | None]] = []

        def add_scalar(label: str, table: str, expr: str, higher_better: bool) -> None:
            overall = self._overall_mean(table, expr)
            per_doc = self._per_doc_means(table, expr)
            means = {m: (f"{overall[m]:.3f}" if m in overall else "-") for m in models}
            wr = self._win_rate(per_doc, higher_better)
            wins = {m: wr.get(m, "n/a") for m in models}
            columns.append((label, means, wins))

        # correction_rate: how much the spell checker changed the text (lower better,
        # win-rate on it); uncorrectable_rate: garbage it couldn't fix (mean only).
        add_scalar("corr_rate", "correction_rate", "correction_rate", higher_better=False)
        uncorr = self._overall_mean("correction_rate", "uncorrectable_rate")
        columns.append(("uncorr_rate", {m: (f"{uncorr[m]:.3f}" if m in uncorr else "-") for m in models}, None))
        add_scalar("garbage", "garbage_fraction", "score", higher_better=False)

        # peer_agreement: two value columns, win-rate on bow_f1.
        for label, expr in (("peer_f1", "bow_f1"), ("peer_ned", "one_minus_ned")):
            overall = self._overall_mean("peer_agreement", expr)
            means = {m: (f"{overall[m]:.3f}" if m in overall else "-") for m in models}
            if label == "peer_f1":
                wr = self._win_rate(self._per_doc_means("peer_agreement", expr), True)
                columns.append((label, means, {m: wr.get(m, "n/a") for m in models}))
            else:
                columns.append((label, means, None))

        for label, expr in (("tl_f1", "bow_f1"), ("tl_ned", "one_minus_ned")):
            overall = self._overall_mean("textlayer_agreement", expr)
            means = {m: (f"{overall[m]:.3f}" if m in overall else "-") for m in models}
            if label == "tl_f1":
                wr = self._win_rate(self._per_doc_means("textlayer_agreement", expr), True)
                columns.append((label, means, {m: wr.get(m, "n/a") for m in models}))
            else:
                columns.append((label, means, None))

        # reject_rate: per-doc value is already fallback/total (one row per doc);
        # lower is better.
        add_scalar("reject_rate", "reject_rate",
                   "CAST(fallback_pages AS REAL) / total_pages", higher_better=False)

        # pplx: mean ppl_raw / ppl_corrected across each model's own table.
        pplx_tables = self._pplx_tables()
        if pplx_tables:
            raw: dict[str, str] = {}
            corr: dict[str, str] = {}
            for m in models:
                t = _pplx_table(m)
                if t in pplx_tables:
                    row = self.conn.execute(
                        f"SELECT AVG(ppl_raw), AVG(ppl_corrected) FROM {t}"
                    ).fetchone()
                    raw[m] = f"{row[0]:.3f}" if row[0] is not None else "-"
                    corr[m] = f"{row[1]:.3f}" if row[1] is not None else "-"
                else:
                    raw[m] = corr[m] = "-"
            columns.append(("ppl_raw", raw, None))
            columns.append(("ppl_corr", corr, None))

        return self._render(models, columns)

    @staticmethod
    def _render(models, columns) -> str:
        # Build header + rows. Each metric column becomes "mean" and, when it has
        # a win-rate, a "<label>_win" column.
        headers = ["model"]
        cells_per_model: dict[str, list[str]] = {m: [m] for m in models}
        for label, means, wins in columns:
            headers.append(label)
            for m in models:
                cells_per_model[m].append(means[m])
            if wins is not None:
                headers.append(label + "_win")
                for m in models:
                    cells_per_model[m].append(wins[m])

        rows = [cells_per_model[m] for m in models]
        widths = [
            max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i])
            for i in range(len(headers))
        ]
        def fmt(cells):
            return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

        lines = [fmt(headers), fmt(["-" * w for w in widths])]
        lines += [fmt(r) for r in rows]
        return "\n".join(lines)

    def close(self) -> None:
        self.conn.close()
