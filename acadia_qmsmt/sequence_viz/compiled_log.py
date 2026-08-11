"""
Decode the DMA command stream straight out of an archived ``compiled.log``.

No execution and no acadia version dependence -- this reads what actually ran.
Per ``Acadia.command_dma`` the command word is ``(address << 16) | (length - 1)``
for arbitrary commands and ``length - 1`` for every other type.

Useful as a cross-check on :func:`~.folder.trace_folder`: if the re-trace and the
archive disagree, the re-trace is wrong.
"""
import re
from collections import Counter
from pathlib import Path

CMD_RE = re.compile(
    r"; Command DMA for (?P<ch>[AD][DA]C\d+), type (?P<type>\d): (?P<word>[0-9A-F]{8})\s*$")
TRIGGER_RE = re.compile(r"; Trigger DMAs\s*$")

KIND = {0: "ARB_CONT", 1: "ARB", 2: "CONST_CONT", 3: "DWELL"}


def parse(path):
    """Return ``(commands, n_triggers)``.

    Each command is ``(channel, kind, address, length_cycles)``; ``address`` is
    ``None`` for non-arbitrary commands. Lines whose command word did not resolve
    to a literal are skipped -- in practice these are branch-target ``Symbol``
    reprs that quote a later command's comment, not commands of their own.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "compiled.log"

    commands, triggers = [], 0
    for line in path.read_text().splitlines():
        match = CMD_RE.search(line)
        if match:
            word = int(match["word"], 16)
            kind = int(match["type"])
            address = (word >> 16) if kind == 1 else None
            length = ((word & 0xFFFF) if kind == 1 else word) + 1
            commands.append((match["ch"], KIND.get(kind, kind), address, length))
        elif TRIGGER_RE.search(line):
            triggers += 1
    return commands, triggers


def compare(trace, folder):
    """Cross-check a :class:`SequenceTrace` against the folder's ``compiled.log``.

    Compares the multiset of ``(channel, kind, length)`` and the block/trigger
    count. Returns a dict; ``match`` is True when the re-trace reproduces the
    archived program exactly.
    """
    archived, triggers = parse(folder)
    archived_counts = Counter((ch, kind, length) for ch, kind, _, length in archived)

    retrace_counts = Counter()
    symbolic = 0
    for c in trace.static_commands:
        if c.symbolic:
            symbolic += 1
            continue
        retrace_counts[(c.channel, c.kind, c.length)] += 1

    only_archive = archived_counts - retrace_counts
    only_retrace = retrace_counts - archived_counts
    return {
        "match": not only_archive and not only_retrace and triggers == len(trace.blocks),
        "blocks": len(trace.blocks),
        "triggers": triggers,
        "commands_retrace": sum(retrace_counts.values()),
        "commands_archive": sum(archived_counts.values()),
        "symbolic_retrace": symbolic,
        "only_in_archive": dict(only_archive),
        "only_in_retrace": dict(only_retrace),
    }
