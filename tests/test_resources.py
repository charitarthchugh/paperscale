"""Resource-governor tests for VLM OCR safety invariants.

Trace: plan acceptance tests 9 and 11 plus fixed acquisition order invariant.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paperscale.resources import ResourceGovernor, ResourceKind, ResourceOrderError


class ResourceGovernorTests(unittest.TestCase):
    def test_acquisition_order_violation_raises(self) -> None:
        governor = ResourceGovernor()

        with governor.acquire(ResourceKind.PROVIDER):
            with self.assertRaises(ResourceOrderError):
                with governor.acquire(ResourceKind.FILE_DESCRIPTOR):
                    pass

    def test_file_opens_go_through_governor_token(self) -> None:
        observations: list[bool] = []

        def recording_opener(path: Path, mode: str, **kwargs):
            observations.append(governor.is_held(ResourceKind.FILE_DESCRIPTOR))
            return Path(path).open(mode, **kwargs)

        governor = ResourceGovernor(file_opener=recording_opener)

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "page.md"
            with governor.open_file(target, "w", encoding="utf-8") as handle:
                handle.write("# Page\n")

        self.assertEqual(observations, [True])

    def test_release_must_follow_reverse_order(self) -> None:
        governor = ResourceGovernor()
        outer = governor.acquire(ResourceKind.SCHEDULER)
        inner = governor.acquire(ResourceKind.RENDER)

        outer.__enter__()
        inner.__enter__()
        with self.assertRaises(ResourceOrderError):
            outer.__exit__(None, None, None)

        inner.__exit__(None, None, None)
        outer.__exit__(None, None, None)


if __name__ == "__main__":
    unittest.main()
