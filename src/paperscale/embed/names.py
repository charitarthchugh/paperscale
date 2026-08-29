"""Document identity: the .npz filename, the LanceDB primary key, and the Resume key.

One string does all three jobs (design 7), so a weak derivation fails in three places at
one time: the wrong file on disk, the wrong row replaced, and a Resume that either skips a
Document it never embedded or re-embeds the whole corpus.

The name is a **pure function of one Record's** ``Source-File``. It does not depend on
iteration order, on the Sink layout, or on which other Documents share the Invocation --
which is exactly what lets a later Invocation derive the same name and recognise its own
outputs.

Two things here deliberately differ from ``pipeline.get_markdown_path``, the export's
answer to the same problem:

* **The source extension is appended, never replaced.** ``case.pdf`` -> ``case.pdf``, so
  the Sink writes ``case.pdf.npz``. Replacing it is issue #32: ``case.pdf`` and
  ``case.tiff`` collapse onto one output and the second silently overwrites the first.
  #22 wanted the two functions consistent; consistency with a bug is not worth keeping.
* **Normalization is ``os.path.normpath``**, not the export's
  ``[p for p in parts if p and p != ".."]``. That sanitizer *drops* ``..`` instead of
  resolving it, so ``/a/b/../c.pdf`` and ``/a/b/c.pdf`` derive the same name, and it keeps
  ``.``, so ``/a/./b.pdf`` derives ``a/./b.pdf`` while the file it names lands at
  ``a/b.pdf``. Resume's correctness rests entirely on name stability, so ``embed`` cannot
  inherit either behaviour.

``realpath`` is **not** used. The PDFs may be gone by embed time, and resolving symlinks
would change *which Documents are considered the same* rather than normalize *how one is
spelled* -- a different, larger decision than this function is entitled to make.
"""

from __future__ import annotations

import hashlib
import os

# The manifest (``paperscale-embed.json``) and the failures file
# (``paperscale-embed-failures.txt``) sit at ``<out>/`` in both layouts, so the prefix is
# reserved rather than defended per-file: a Document named ``paperscale-embed`` writes its
# sidecar straight over the manifest.
RESERVED_PREFIX = "paperscale-embed"

# ext4 caps one path component at 255 *bytes*, and long legal filenames do reach it -- in
# a non-ASCII corpus well before 255 characters, which is why this is measured on the
# encoded form. Past the cap there is no usable filename, so the digest becomes the name.
MAX_COMPONENT_BYTES = 255

# The ``.npz`` Sink writes ``<name>.npz`` *and* ``<name>.json``, so the component that
# becomes the filename has to fit the cap with the longer of those two suffixes already
# on it. Checking the bare component against 255 would pass a 255-byte name straight into
# an ENAMETOOLONG at write time -- exactly the failure this fallback exists to prevent.
# Only the last component is affected: the earlier ones become directories and get
# nothing appended, so they keep the full 255.
MAX_SUFFIX_BYTES = len(".json")


class NameCollisionError(RuntimeError):
    """Two Source-File values in one Run want the same Document name (or a reserved one)."""


def source_digest(source_file: str) -> str:
    """Digest the raw ``Source-File`` **string** -- never the Document text.

    Every Record already carries ``id = sha1(document_text)`` (``pipeline.py:654``). It is
    the tempting choice and it is the wrong one: that digest changes on every re-OCR, and
    with no content-change detection (standing decision 7) adopting it as identity turns
    "Resume silently keeps stale vectors" into "Resume silently duplicates the whole
    corpus" -- the opposite failure, equally invisible. The path string survives a re-OCR.

    64 bits is far past what a corpus of this size needs; 16 hex characters stay usable as
    a filename, which is the digest's second job (see :func:`document_name`). **Both Sinks
    must call this one function**, or they disagree about one Document's identity, which
    is precisely the failure the digest exists to prevent.
    """
    return hashlib.sha256(source_file.encode()).hexdigest()[:16]


def _sanitized_parts(relative_path: str) -> list[str]:
    """Normalize, drop the root, then drop what ``normpath`` cannot.

    ``normpath`` resolves *interior* ``..`` and ``.`` but leaves a *leading* ``..`` alone
    (``../a`` stays ``../a``), so the traversal guard still runs after it rather than
    being replaced by it. A bare ``.`` survives too, for any path that normalizes to the
    current directory; ``.`` is not a Document name, so it is dropped here and the caller
    falls back to the digest.
    """
    normalized = os.path.normpath(relative_path).lstrip("/")
    return [p for p in normalized.split("/") if p and p not in ("..", ".")]


def document_name(source_file: str) -> str:
    """Derive the Document name from a Record's ``metadata["Source-File"]``.

    The name is the sanitized relative path *including* the source extension --
    ``/media/cc/data/law/doc9419897.pdf`` -> ``media/cc/data/law/doc9419897.pdf``. The
    ``.npz`` Sink appends ``.npz`` and ``.json`` to it; the LanceDB Sink stores it
    verbatim in ``document_name``. One derivation, two Sinks.

    Falls back to :func:`source_digest` as the *whole* name whenever no usable path
    exists: ``Source-File`` missing or empty, a path that sanitizes to nothing (``/``,
    ``..``, ``.``), or a component past :data:`MAX_COMPONENT_BYTES` (less
    :data:`MAX_SUFFIX_BYTES` for the last one, which is the component a Sink suffixes).

    Note what this function does **not** do: it never consults the layout, the Run label,
    or the other Documents. Anything order-dependent here would move under Resume's feet
    between Invocations (design 7.3).
    """
    if not source_file:
        # "Missing" and "empty" are one case (design 7.3 job 2), and a Record with no
        # Source-File key reaches here as None from a plain .get(), so both derive the
        # same digest name -- two such Records then collide loudly at startup instead of
        # one silently overwriting the other.
        return source_digest(source_file or "")

    if "::" in source_file:
        # Tarball form, ``archive.tar.gz::internal/path.pdf``. The archive basename minus
        # ``.tar``/``.tar.gz`` becomes a directory and the internal path continues beneath
        # it, reusing the markdown export's handling. Note that ``x.tar.gz::doc.pdf`` and
        # ``x.tar::doc.pdf`` therefore collapse onto one name -- deliberately left as a
        # startup collision (see :func:`check_collisions`) rather than papered over, since
        # any escape from the collapse would have to encode the archive's extension into
        # the name and no other class of source pays that cost.
        tarball_path, internal_path = source_file.split("::", 1)
        basename = os.path.splitext(os.path.basename(tarball_path))[0]
        if basename.endswith(".tar"):
            basename = basename[:-4]
        # The internal path arrives from inside an archive, so its ``..`` is the one case
        # here that is plausibly hostile rather than merely sloppy; sanitizing the joined
        # path covers it with the same guard the plain branch uses.
        # Joined by hand: ``os.path.join`` discards ``basename`` outright when the member is
        # recorded with a leading ``/``, so ``corpus.tar.gz::/internal/doc.pdf`` came out as
        # ``/internal/doc.pdf``. That drops the archive namespace this branch exists to add,
        # and two archives holding the same internal path then collide on one name.
        relative_path = f"{basename}/{internal_path.lstrip('/')}"
    else:
        relative_path = source_file

    parts = _sanitized_parts(relative_path)
    if not parts:
        return source_digest(source_file)
    *directories, filename = parts
    if any(len(p.encode()) > MAX_COMPONENT_BYTES for p in directories):
        return source_digest(source_file)
    if len(filename.encode()) > MAX_COMPONENT_BYTES - MAX_SUFFIX_BYTES:
        return source_digest(source_file)
    return "/".join(parts)


def _is_reserved(name: str) -> bool:
    """Would this Document's outputs land on the reserved prefix at the output root?

    Only the first component can: the manifest and the failures file sit at ``<out>/`` in
    both layouts, so ``a/paperscale-embed`` is safe while a Document named
    ``paperscale-embed`` writes ``<out>/paperscale-embed.json`` over the manifest.

    The check ignores the layout on purpose. The name is layout-independent, and a
    ``labelled`` Invocation (where ``<out>/<label>/`` would have shielded the name) can
    drop to a single ``--run`` later; discovering the collision then, against outputs
    Resume already keys on, is worse than refusing it now.
    """
    return name.split("/", 1)[0].startswith(RESERVED_PREFIX)


def check_collisions(run_label: str, pairs: list[tuple[str, str]]) -> None:
    """Stop the Invocation before any GPU work if one Run's names are not usable.

    ``pairs`` is ``[(document_name, source_file)]`` for **one** Run. The scoping is not an
    optimization: both Sinks key on ``(run_label, document_name)`` and the ``labelled``
    layout gives each Run its own subtree, so two Runs cannot collide with each other.

    Rules 2 and 3 of the derivation already prevent the two classes a derivation *can*
    prevent (``..`` components, and extension replacement). The residue is fatal here
    rather than tiebroken -- ``/a/case.pdf`` beside ``a/case.pdf``, ``x.tar.gz::doc.pdf``
    beside ``x.tar::doc.pdf``, two Records with an empty ``Source-File``, and any name
    landing on :data:`RESERVED_PREFIX`. A tiebreak would have to be stable across
    Invocations, because Resume derives its state from the outputs, and neither available
    scheme is: suffixing the loser depends on iteration order, and suffixing every member
    of a colliding set silently reverts when the set changes. Both are silent costs; this
    is loud, costs seconds, and names exactly what to fix.

    Not in tension with "one bad PDF must never end a Run" (#30): that rule governs
    embedding failures mid-Invocation. This runs before the first request is sent.

    The message lists **every** colliding group with **both** raw ``Source-File`` values,
    quoted -- a collision is frequently two strings that differ only in leading slashes or
    whitespace, which an unquoted listing hides.
    """
    sources_by_name: dict[str, list[str]] = {}
    for name, source_file in pairs:
        sources_by_name.setdefault(name, []).append(source_file)

    groups: list[str] = []
    bad_names = 0
    for name in sorted(sources_by_name):
        sources = sources_by_name[name]
        reasons = []
        if len(sources) > 1:
            reasons.append(f"{len(sources)} Source-File values derive it")
        if _is_reserved(name):
            reasons.append(f"it is reserved -- {RESERVED_PREFIX}.json (the manifest) and {RESERVED_PREFIX}-failures.txt live at the output root")
        if not reasons:
            continue
        bad_names += 1
        groups.append(f"  {name!r}: " + "; ".join(reasons))
        groups.extend(f"      {s!r}" for s in sources)

    if not groups:
        return

    raise NameCollisionError(
        f"run {run_label!r}: {bad_names} unusable Document name(s) out of {len(pairs)} Documents; nothing has been embedded.\n"
        + "\n".join(groups)
        + "\n  Rename the colliding sources, or re-OCR them from one consistent root, so that each derives a distinct name."
    )
