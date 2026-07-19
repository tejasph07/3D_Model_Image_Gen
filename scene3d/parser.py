"""Parser for the 3-token scene DSL.

Grammar (one command per line):

    verb,arg1,arg2

  * Case-insensitive verbs; entity names are case-sensitive identifiers.
  * `entity.part` dotted references select a body part or object anchor.
  * Numeric args allowed where the verb expects them (scale, turn, hover).
  * `a:b` packs two values into one arg (at,human,1.5:2.0 / between,a,b:c).
  * '#' starts a comment; blank lines ignored.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Ref:
    """A parsed argument: entity name with optional .part, or a literal."""
    raw: str
    entity: str | None = None
    part: str | None = None
    number: float | None = None
    pair: tuple | None = None

    def __repr__(self):
        return f"Ref({self.raw})"


@dataclass
class Command:
    verb: str
    a: Ref
    b: Ref
    line_no: int
    raw: str = ""


class ParseError(ValueError):
    pass


_NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _parse_ref(tok: str, line_no: int) -> Ref:
    tok = tok.strip()
    r = Ref(raw=tok)
    if _NUM_RE.match(tok):
        r.number = float(tok)
        return r
    if ":" in tok:
        parts = tok.split(":")
        vals = []
        for p in parts:
            p = p.strip()
            vals.append(float(p) if _NUM_RE.match(p) else p)
        r.pair = tuple(vals)
        return r
    if "." in tok:
        ent, part = tok.split(".", 1)
        if not _IDENT_RE.match(ent) or not _IDENT_RE.match(part):
            raise ParseError(f"line {line_no}: bad reference '{tok}'")
        r.entity, r.part = ent, part.lower()
        return r
    if not _IDENT_RE.match(tok):
        raise ParseError(f"line {line_no}: bad token '{tok}'")
    r.entity = tok
    return r


def parse(text: str) -> list[Command]:
    cmds = []
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            raise ParseError(
                f"line {i}: expected 'verb,arg1,arg2' (3 comma-separated tokens), "
                f"got {len(parts)}: '{line}'")
        verb = parts[0].lower()
        if not _IDENT_RE.match(verb):
            raise ParseError(f"line {i}: bad verb '{parts[0]}'")
        cmds.append(Command(verb=verb, a=_parse_ref(parts[1], i),
                            b=_parse_ref(parts[2], i), line_no=i, raw=line))
    return cmds
