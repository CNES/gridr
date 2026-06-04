"""
Reflow Markdown bullet text to a target width (default 100 cols), respecting
nested list indentation and preserving inline code spans, headings and italic
metadata lines.

Algorithm:
- Pass through lines that are headings, blank, italic-only metadata, or
  fenced code blocks unchanged.
- For each bullet (lines starting with `-`, `*`, or `  -` etc.), collect the
  full logical paragraph (the bullet line + any continuation lines indented
  past the bullet marker) into a single buffer.
- Re-wrap the buffer using textwrap, with subsequent_indent set to the
  proper continuation indent so the bullet glyph aligns visually.
- Word-boundary wrap: never break inside backtick spans.
"""

from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path

WIDTH = 100

# Match a bullet: indent + (-|*|number.) + space
BULLET_RE = re.compile(r"^(?P<indent>\s*)(?P<marker>[-*+]|\d+\.)\s+(?P<rest>.*)$")
HEADING_RE = re.compile(r"^#{1,6}\s")
FENCE_RE = re.compile(r"^```")
ITALIC_META_RE = re.compile(r"^\*[A-Za-z][^*]*\*\s*$")  # *Module: foo*


def is_passthrough(line: str) -> bool:
    """Lines that must be emitted verbatim regardless of width."""
    if not line.strip():
        return True
    if HEADING_RE.match(line):
        return True
    if ITALIC_META_RE.match(line):
        return True
    return False


def split_preserving_code(text: str) -> list[str]:
    """Split text on whitespace, but treat backtick-delimited spans as atoms."""
    parts: list[str] = []
    buf = []
    in_code = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "`":
            buf.append(ch)
            in_code = not in_code
            i += 1
            continue
        if ch.isspace() and not in_code:
            if buf:
                parts.append("".join(buf))
                buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    if buf:
        parts.append("".join(buf))
    return parts


def wrap_paragraph(text: str, initial_indent: str, subsequent_indent: str,
                   width: int) -> str:
    """Wrap `text` to `width`, never breaking inside backtick spans."""
    tokens = split_preserving_code(text)
    if not tokens:
        return initial_indent.rstrip()

    lines: list[str] = []
    current = initial_indent
    first_token_on_line = True

    for tok in tokens:
        candidate_len = len(current) + (0 if first_token_on_line else 1) + len(tok)
        if first_token_on_line or candidate_len <= width:
            if first_token_on_line:
                current = current + tok
                first_token_on_line = False
            else:
                current = current + " " + tok
        else:
            lines.append(current.rstrip())
            current = subsequent_indent + tok
            first_token_on_line = False
    lines.append(current.rstrip())
    return "\n".join(lines)


def reflow(text: str, width: int = WIDTH) -> str:
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    in_fence = False

    while i < len(lines):
        line = lines[i]

        # Code fences pass through
        if FENCE_RE.match(line):
            in_fence = not in_fence
            out.append(line)
            i += 1
            continue
        if in_fence:
            out.append(line)
            i += 1
            continue

        # Indented code blocks (4+ spaces after a blank line, no bullet marker)
        # We treat lines with exactly 4-space indent and no bullet as code only
        # if they don't look like a bullet continuation. To keep it simple, we
        # let the bullet detection handle these; non-bullet 4-space lines stay
        # as passthrough.
        if is_passthrough(line):
            out.append(line)
            i += 1
            continue

        m = BULLET_RE.match(line)
        if not m:
            # Non-bullet prose paragraph; collect until blank/heading/bullet,
            # then wrap.
            buf = [line.rstrip()]
            j = i + 1
            while j < len(lines):
                nxt = lines[j]
                if not nxt.strip():
                    break
                if HEADING_RE.match(nxt) or FENCE_RE.match(nxt):
                    break
                if BULLET_RE.match(nxt):
                    break
                if ITALIC_META_RE.match(nxt):
                    break
                buf.append(nxt.strip())
                j += 1
            paragraph = " ".join(b.strip() for b in buf)
            out.append(wrap_paragraph(paragraph, "", "", width))
            i = j
            continue

        # Bullet line: collect continuations
        indent = m.group("indent")
        marker = m.group("marker")
        rest = m.group("rest")
        bullet_prefix = f"{indent}{marker} "
        cont_indent = " " * len(bullet_prefix)

        buf = [rest]
        j = i + 1
        while j < len(lines):
            nxt = lines[j]
            if not nxt.strip():
                break
            # A new bullet at any indent level ends this bullet's continuation
            mn = BULLET_RE.match(nxt)
            if mn:
                # Sub-bullet (more indented than current) ends this bullet too
                break
            if HEADING_RE.match(nxt) or FENCE_RE.match(nxt):
                break
            # Continuation must be indented at least to cont_indent depth
            stripped = nxt.lstrip()
            if not nxt.startswith(cont_indent[:min(len(cont_indent), len(nxt) - len(stripped))]):
                # Loose continuation (less indented) — still treat it as part
                # of the bullet text rather than a new paragraph.
                pass
            buf.append(stripped)
            j += 1

        merged = " ".join(b.strip() for b in buf if b.strip())
        out.append(wrap_paragraph(merged, bullet_prefix, cont_indent, width))
        i = j
        continue

    # Preserve trailing newline if input had one
    suffix = "\n" if text.endswith("\n") else ""
    return "\n".join(out) + suffix


def main():
    if len(sys.argv) != 3:
        print("Usage: reflow_md.py <input.md> <output.md>", file=sys.stderr)
        sys.exit(1)
    src = Path(sys.argv[1]).read_text(encoding="utf-8")
    dst = reflow(src, WIDTH)
    Path(sys.argv[2]).write_text(dst, encoding="utf-8")
    # Report max line length for sanity
    max_len = max((len(l) for l in dst.splitlines()), default=0)
    print(f"Wrote {sys.argv[2]}: {len(dst.splitlines())} lines, max width {max_len}")


if __name__ == "__main__":
    main()