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
CREATE TABLE IF NOT EXISTS eval_doc (
    model TEXT, doc TEXT, phase TEXT, text_sha1 TEXT,
    UNIQUE(model, doc, phase)
);
"""

# Resume phases -> (table, columns). The phase name is what `eval_doc` records and
# what the CLI passes to done_docs(); keeping the mapping here is what lets one
# generic per-doc writer serve every metric.
_PHASE_TABLES: dict[str, tuple[str, tuple[str, ...]]] = {
    "corrections": ("correction_rate", ("model", "doc", "page", "correction_rate", "uncorrectable_rate")),
    "garbage": ("garbage_fraction", ("model", "doc", "page", "score")),
    "textlayer": ("textlayer_agreement", ("model", "doc", "page", "bow_f1", "one_minus_ned")),
}

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
        # Staleness guard: pplx rows scored by a different scorer model are not
        # comparable -- drop them rather than resume on top of them.
        row = self.conn.execute(
            "SELECT pplx_scorer_model FROM runs WHERE model = ?", (model,)
        ).fetchone()
        stored = row[0] if row else None
        if pplx_scorer_model is None:
            pplx_scorer_model = stored  # not scoring pplx this run; keep the record
        elif stored is not None and stored != pplx_scorer_model:
            self.clear_pplx(model)
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

    # --- resume bookkeeping ----------------------------------------------
    #
    # Which (model, doc, phase) triples are complete. Row presence in a metric
    # table cannot answer this: textlayer legitimately writes zero rows for a
    # blank-layer or fallback doc after already paying for its pdftotext calls.
    # The checksum is over the doc's page texts, so re-OCR'ing a run under the
    # same label invalidates exactly the docs whose output changed.

    def mark_doc_done(self, model: str, doc: str, phase: str, text_sha1: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO eval_doc (model, doc, phase, text_sha1) VALUES (?, ?, ?, ?)",
            (model, doc, phase, text_sha1),
        )
        self.conn.commit()

    def write_doc(self, phase: str, model: str, doc: str, rows, text_sha1: str) -> None:
        """Replace one doc's rows for ``phase`` and mark it done -- atomically.

        The row write and the done-mark share one transaction on purpose: a doc
        marked done with no rows behind it would make the next run skip work that
        never happened. ``rows`` may be empty (a doc whose pages were all skipped).
        Commits per doc, so an interrupted phase keeps every doc that finished.
        """
        table, cols = _PHASE_TABLES[phase]
        placeholders = ", ".join("?" * len(cols))
        with self.conn:  # commit on success, roll back on any exception
            self.conn.execute(f"DELETE FROM {table} WHERE model = ? AND doc = ?", (model, doc))
            self.conn.executemany(
                f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})", list(rows)
            )
            self.conn.execute(
                "INSERT OR REPLACE INTO eval_doc (model, doc, phase, text_sha1) VALUES (?, ?, ?, ?)",
                (model, doc, phase, text_sha1),
            )

    def done_docs(self, model: str, phase: str, checksums: dict[str, str]) -> set[str]:
        """Docs of ``model`` whose ``phase`` is complete AND whose text is unchanged.

        ``checksums`` maps doc -> checksum for the run as just loaded; a doc absent
        from it (deleted PDF) is never reported done.
        """
        cur = self.conn.execute(
            "SELECT doc, text_sha1 FROM eval_doc WHERE model = ? AND phase = ?",
            (model, phase),
        )
        return {doc for doc, sha in cur.fetchall() if checksums.get(doc) == sha}

    # Metric tables keyed by a plain `model` column. peer_agreement is handled
    # separately everywhere below: it also has a `peer` column, and a row is stale
    # if EITHER side of the pair is rescored.
    _MODEL_TABLES = ("correction_rate", "garbage_fraction", "textlayer_agreement",
                     "reject_rate", "eval_doc")

    def clear_model(self, model: str) -> None:
        """Forget everything cached for one label (backs ``--no-resume``)."""
        with self.conn:
            for table in self._MODEL_TABLES:
                self.conn.execute(f"DELETE FROM {table} WHERE model = ?", (model,))
            self.conn.execute(
                "DELETE FROM peer_agreement WHERE model = ? OR peer = ?", (model, model)
            )
        self.clear_pplx(model)

    def prune_missing_docs(self, model: str, keep: set[str]) -> None:
        """Drop rows for docs no longer present in this model's run.

        Without this a PDF you deleted from the workspace keeps contributing to the
        leaderboard's per-doc means forever.
        """
        placeholders = ", ".join("?" * len(keep))
        keep_list = list(keep)
        # "NOT IN ()" is invalid SQL, so an empty keep-set means delete everything.
        cond = f"doc NOT IN ({placeholders})" if keep else "1"
        with self.conn:
            for table in self._MODEL_TABLES:
                self.conn.execute(f"DELETE FROM {table} WHERE model = ? AND {cond}", (model, *keep_list))
            self.conn.execute(
                f"DELETE FROM peer_agreement WHERE (model = ? OR peer = ?) AND {cond}",
                (model, model, *keep_list),
            )

    def mark_docs_done(self, rows) -> None:
        """Batch form of `mark_doc_done` -- ``[(model, doc, phase, text_sha1), ...]``
        in one transaction. The peer phase re-marks every doc on every run, so a
        commit per doc would cost one fsync per doc even on a pure no-op."""
        rows = list(rows)
        if not rows:
            return
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO eval_doc (model, doc, phase, text_sha1) VALUES (?, ?, ?, ?)",
                rows,
            )

    def clear_peer_docs(self, pairs) -> None:
        """Drop every peer row touching ``model`` on ``doc``, for each ``(model, doc)``.

        Both directions go: a pair scored against the old text is stale whether this
        model is the row's ``model`` or its ``peer``. One transaction for the batch --
        this runs over every doc on every run, so per-doc commits would cost one
        fsync per doc even when nothing changed.
        """
        pairs = list(pairs)
        if not pairs:
            return
        with self.conn:
            self.conn.executemany(
                "DELETE FROM peer_agreement WHERE doc = ? AND (model = ? OR peer = ?)",
                [(doc, model, model) for model, doc in pairs],
            )

    def marked_docs(self, model: str, phase: str) -> set[str]:
        """Docs with any checksum record for ``phase``, matching or not.

        Unlike `done_docs` this ignores the checksum value; it exists to spot rows
        written before eval_doc did (see the pplx adoption in the CLI).
        """
        cur = self.conn.execute(
            "SELECT doc FROM eval_doc WHERE model = ? AND phase = ?", (model, phase)
        )
        return {r[0] for r in cur.fetchall()}

    def append_peer_agreement(self, rows) -> None:
        """Insert peer rows WITHOUT deleting existing ones.

        Resume computes only the missing pairs, so `_replace`'s delete-by-model
        would destroy exactly the pairs being reused.
        """
        rows = list(rows)
        if not rows:
            return
        with self.conn:
            self.conn.executemany(
                "INSERT INTO peer_agreement (model, peer, doc, page, bow_f1, one_minus_ned)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )

    def stored_peer_pairs(self) -> dict[tuple[str, int], set[tuple[str, str]]]:
        """``{(doc, page): {(model, peer), ...}}`` for every peer row on disk.

        Peer agreement resumes off these rows rather than off eval_doc: it always
        writes a row for what it scores, so row presence is a truthful record.
        """
        out: dict[tuple[str, int], set[tuple[str, str]]] = {}
        cur = self.conn.execute("SELECT doc, page, model, peer FROM peer_agreement")
        for doc, page, model, peer in cur.fetchall():
            out.setdefault((doc, page), set()).add((model, peer))
        return out

    def _ensure_pplx_table(self, model: str) -> str:
        table = _pplx_table(model)
        coldefs = "doc TEXT, page INTEGER, " + ", ".join(f"{c} REAL" for c in _PPLX_COLS)
        # n_tokens columns are integers, but REAL stores them faithfully via sqlite affinity.
        self.conn.execute(
            f"CREATE TABLE IF NOT EXISTS {table} ({coldefs}, UNIQUE(doc, page))"
        )
        return table

    def write_pplx_doc(self, model: str, rows) -> None:
        """Streaming write: one call (and one commit) per completed doc, so a
        crashed long run resumes from what's already scored."""
        table = self._ensure_pplx_table(model)
        cols = ("doc", "page") + _PPLX_COLS
        placeholders = ", ".join("?" * len(cols))
        self.conn.executemany(
            f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) VALUES ({placeholders})",
            list(rows),
        )
        self.conn.commit()

    def pplx_done_docs(self, model: str) -> set[str]:
        """Docs already scored for this model (resume support)."""
        table = self._ensure_pplx_table(model)
        cur = self.conn.execute(f"SELECT DISTINCT doc FROM {table}")
        return {r[0] for r in cur.fetchall()}

    def clear_pplx(self, model: str) -> None:
        self.conn.execute(f"DROP TABLE IF EXISTS {_pplx_table(model)}")
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

    @staticmethod
    def _doc_mean(per_doc: dict[str, dict[str, float]]) -> dict[str, float]:
        """Mean over a model's per-doc means -- each doc weighted equally, matching
        the win-rate weighting (a 100-page doc must not dominate a 1-page doc)."""
        return {m: sum(d.values()) / len(d) for m, d in per_doc.items() if d}

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
            per_doc = self._per_doc_means(table, expr)
            overall = self._doc_mean(per_doc)
            means = {m: (f"{overall[m]:.3f}" if m in overall else "-") for m in models}
            wr = self._win_rate(per_doc, higher_better)
            wins = {m: wr.get(m, "n/a") for m in models}
            columns.append((label, means, wins))

        # correction_rate: how much the spell checker changed the text (lower better,
        # win-rate on it); uncorrectable_rate: garbage it couldn't fix (mean only).
        add_scalar("corr_rate", "correction_rate", "correction_rate", higher_better=False)
        uncorr = self._doc_mean(self._per_doc_means("correction_rate", "uncorrectable_rate"))
        columns.append(("uncorr_rate", {m: (f"{uncorr[m]:.3f}" if m in uncorr else "-") for m in models}, None))
        add_scalar("garbage", "garbage_fraction", "score", higher_better=False)

        # peer_agreement: two value columns, win-rate on bow_f1.
        for label, expr in (("peer_f1", "bow_f1"), ("peer_ned", "one_minus_ned")):
            per_doc = self._per_doc_means("peer_agreement", expr)
            overall = self._doc_mean(per_doc)
            means = {m: (f"{overall[m]:.3f}" if m in overall else "-") for m in models}
            if label == "peer_f1":
                wr = self._win_rate(per_doc, True)
                columns.append((label, means, {m: wr.get(m, "n/a") for m in models}))
            else:
                columns.append((label, means, None))

        for label, expr in (("tl_f1", "bow_f1"), ("tl_ned", "one_minus_ned")):
            per_doc = self._per_doc_means("textlayer_agreement", expr)
            overall = self._doc_mean(per_doc)
            means = {m: (f"{overall[m]:.3f}" if m in overall else "-") for m in models}
            if label == "tl_f1":
                wr = self._win_rate(per_doc, True)
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
