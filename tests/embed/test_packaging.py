"""Test obligation 39 -- `paperscale` for OCR still imports without the embed extra.

No single module agent could write this: the promise is about the whole package's
import graph, and it only breaks when two modules are combined. It is checked three
ways, because the cheap check is the misleading one.

The obvious test -- import the OCR entry points and assert `numpy` is absent from
`sys.modules` -- is **wrong here**, and the reason is worth writing down so nobody
"fixes" this file back to it. `pypdfium2` is a *core* OCR dependency and it does
``try: import numpy / except ImportError: numpy = None`` at module scope, so
`paperscale.pipeline` legitimately pulls numpy into `sys.modules` on any machine
where numpy happens to be installed -- while still working perfectly on one where it
is not. A `sys.modules` assertion therefore fails on a correct install and would be
"fixed" by deleting the very thing it is meant to protect.

So the availability promise is tested by *simulating the uninstall*: a meta-path
finder that raises `ImportError` for the three distribution names, which is what the
interpreter does when a package genuinely is not there. That exercises the real code
path, including `pypdfium2`'s except-branch.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import unittest

# The three distributions that live in the `embed` extra and must never be required
# by an OCR-only install.
HEAVY = ("numpy", "lancedb", "pyarrow")

# Installed into a child interpreter to make the three distributions genuinely
# unimportable. Raising from `find_spec` is what a missing package looks like, so the
# except-branch in `pypdfium2` (and anywhere else) runs for real.
_BLOCKER = """
import sys
BLOCKED = {"numpy", "lancedb", "pyarrow"}


class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError("No module named %r (simulated OCR-only install)" % name)
        return None


sys.meta_path.insert(0, _Blocker())
"""

_EMBED_MODULES = (
    "adapters",
    "budget",
    "chunking",
    "client",
    "lance_sink",
    "names",
    "npz_sink",
    "panel",
    "records",
    "resume",
    "run",
    "vectors",
)


def _run_without_the_extra(body: str) -> subprocess.CompletedProcess[str]:
    """Run `body` in a child interpreter where numpy/lancedb/pyarrow do not exist."""
    return subprocess.run(
        [sys.executable, "-c", _BLOCKER + textwrap.dedent(body)],
        capture_output=True,
        text=True,
    )


class OcrOnlyInstallTest(unittest.TestCase):
    """Obligation 39 -- the OCR command line works with the extra uninstalled."""

    def test_the_ocr_entry_points_import(self):
        done = _run_without_the_extra("""
            import paperscale.cli
            import paperscale.models
            import paperscale.pipeline
            print("ok", sorted(m for m in BLOCKED if m in sys.modules))
            """)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout.split("\n")[0], "ok []")

    def test_importing_the_embed_package_needs_nothing_heavy(self):
        # `paperscale.cli` reaches `paperscale.embed` for `embed --help`, so the
        # package's `__init__` has to stay importable without the extra too.
        done = _run_without_the_extra("""
            import paperscale.embed
            print("ok")
            """)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout.strip(), "ok")

    def test_embed_help_works_without_the_extra(self):
        """The operator-visible half: a real usage message, never an ImportError.

        This is why `cli._handle_embed` defers its import into the function body --
        argparse has to be able to describe the subcommand on a machine that cannot
        run it.
        """
        done = _run_without_the_extra("""
            from paperscale.cli import main
            try:
                main(["embed", "--help"])
            except SystemExit as exc:
                print("exit", exc.code)
            """)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("--embed-model", done.stdout)
        self.assertIn("exit 0", done.stdout)
