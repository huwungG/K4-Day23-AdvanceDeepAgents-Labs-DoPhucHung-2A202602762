"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"

_REF_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
_REF_LINE = re.compile(r"^\s*\[(\d+)\]")
_URL_RE = re.compile(r"https?://\S+")
_GROUP_RE = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()")
_CODE_RE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)


def _strip_code(text):
    """Remove fenced code blocks and inline code spans (citations inside code don't count)."""
    return _CODE_RE.sub("", text)


def _expand_group(group):
    """Expand '1', '1, 2', '1-3' (hyphen or en dash) into a list of ints."""
    numbers = []
    for part in re.split(r"\s*,\s*", group.strip()):
        if not part:
            continue
        span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
        if span:
            a, b = int(span.group(1)), int(span.group(2))
            if a <= b and b - a <= 200:
                numbers.extend(range(a, b + 1))
            else:
                numbers.extend([a, b])
        elif re.fullmatch(r"\d+", part):
            numbers.append(int(part))
    return numbers


def _cited_in_body(body):
    """Set of citation numbers in the body, expanding groups, ignoring code and markdown links."""
    cited = set()
    for match in _GROUP_RE.finditer(_strip_code(body)):
        for n in _expand_group(match.group(1)):
            cited.add(n)
    return cited


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK).

    PSEUDO-CODE:
      problems = []
      if sources is empty: return ["no sources in sources.json"]
      for each source entry:
          n must be an int                       -> problem if not
          url must start with http:// or https://-> problem if not
          the same url must not appear twice     -> problem if duplicated
      split report_text at the heading "## References":
          body = text before it; if the heading is missing -> problem
      cited = set of numbers found as [n] in the BODY only (not in the reference list; use a regex)
      every number in `cited` must exist in sources -> problem "[n] cited but missing from sources.json"
      every source number must be in `cited`        -> problem "source [n] never cited"
      the lines of the References section that start with "[n]" (regex) are the reference lines:
          every source needs exactly ONE reference line (none missing, no number twice, no number that is not a source)
          each reference line holds exactly ONE http(s) URL and it must equal that source's url
          (a line bundling several sources under one number is a problem)
      return problems
    """
    problems = []

    # Rule 1: sources must be a non-empty list.
    if not isinstance(sources, list) or len(sources) == 0:
        return ["no sources in sources.json"]

    # Rule 2: per-source checks.
    by_n = {}
    seen_urls = {}
    valid_numbers = set()
    for entry in sources:
        if not isinstance(entry, dict):
            problems.append(f"source entry is not an object: {entry!r}")
            continue
        n = entry.get("n")
        url = entry.get("url")
        # n must be an int (bool is not accepted).
        if not isinstance(n, int) or isinstance(n, bool):
            problems.append(f"source n={n!r} is not an integer")
            continue
        by_n[n] = entry
        valid_numbers.add(n)
        # url must start with http:// or https://.
        if not isinstance(url, str) or not (url.startswith("http://") or url.startswith("https://")):
            problems.append(f"source [{n}] has a bad url: {url!r}")
            url_ok = False
        else:
            url_ok = True
        # duplicate urls.
        if url_ok:
            if url in seen_urls:
                problems.append(f"duplicate url {url} in sources [{seen_urls[url]}] and [{n}]")
            else:
                seen_urls[url] = n

    # Rule 3: split at ## References heading.
    matches = list(_REF_HEADING.finditer(report_text))
    if not matches:
        problems.append("missing ## References heading")
        body = report_text
        ref_section = ""
    else:
        body = report_text[: matches[-1].start()]
        ref_section = report_text[matches[-1].end():]

    # Rule 4: citations in the body only.
    cited = _cited_in_body(body)
    for n in sorted(cited):
        if n not in valid_numbers:
            problems.append(f"[{n}] cited but missing from sources.json")
    for n in sorted(valid_numbers):
        if n not in cited:
            problems.append(f"source [{n}] never cited")

    # Rule 5 + 6: reference lines.
    if matches:
        ref_lines = []
        for line in ref_section.splitlines():
            m = _REF_LINE.match(line)
            if m:
                ref_lines.append((int(m.group(1)), line.strip()))
        # Count occurrences of each number.
        counts = {}
        for num, _line in ref_lines:
            counts[num] = counts.get(num, 0) + 1
        for n in sorted(valid_numbers):
            c = counts.get(n, 0)
            if c == 0:
                problems.append(f"source [{n}] missing from References")
            elif c > 1:
                problems.append(f"source [{n}] appears {c} times in References (need exactly one line)")
        for num in sorted(counts):
            if num not in valid_numbers:
                problems.append(f"References has [{num}] which is not in sources.json")
        # Each reference line must hold exactly one URL equal to that source's url.
        by_line = {}
        for num, line in ref_lines:
            by_line.setdefault(num, []).append(line)
        for num, lines in by_line.items():
            if num not in by_n:
                continue
            for line in lines:
                urls = _URL_RE.findall(line)
                # Strip trailing punctuation that is not part of the URL.
                cleaned = [u.rstrip(").,;]\"'") for u in urls]
                if len(cleaned) != 1:
                    problems.append(f"References [{num}] must contain exactly one URL (found {len(cleaned)})")
                elif cleaned[0] != by_n[num].get("url"):
                    problems.append(
                        f"References [{num}] URL {cleaned[0]} does not match sources.json {by_n[num].get('url')}"
                    )

    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
