"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent
from model import make_model
from sandbox import download, open_sandbox, upload

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    slug = re.sub(r"\W+", "-", str(topic or "").strip().lower(), flags=re.UNICODE)
    slug = slug.strip("-")[:60].strip("-")
    return slug or "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    topic = str(topic or "").strip()
    return (f"Write a survey report about: {topic}\n\n"
            f"Follow your system instructions exactly: plan with write_todos, split into >= 4 sub-questions, "
            f"delegate to parallel `researcher` subagents covering at least 3 source families, verify their notes, "
            f"merge into {SOURCES_PATH}, write the report body to {REPORT_PATH} per REPORT_TEMPLATE.md "
            f"(TL;DR, Background, 3-6 thematic synthesis sections, Trends and open problems; inline [n] citations; "
            f"NO ## References section — the finalizer generates it), then run the finalizer "
            f"(python3 {FINALIZER_PATH}) and the validator (python3 {VALIDATOR_PATH}) via `execute` until OK, "
            f"and have `citation-checker` spot-check claims. Be specific (titles, years, numbers from the notes), "
            f"synthesize by theme, and cite Hugging Face papers alongside arXiv and web sources.")


def _iter_tool_calls(messages):
    for msg in messages:
        calls = None
        if isinstance(msg, dict):
            calls = msg.get("tool_calls") or []
            for call in calls:
                if isinstance(call, dict):
                    yield call.get("name", "")
                else:
                    yield str(getattr(call, "name", "") or "")
        else:
            calls = getattr(msg, "tool_calls", None) or []
            for call in calls:
                if isinstance(call, dict):
                    yield call.get("name", "")
                else:
                    name = getattr(call, "name", "")
                    if not name and isinstance(call, (list, tuple)) and len(call) >= 1:
                        name = call[0]
                    yield str(name or "")


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}.

    PSEUDO-CODE: walk the lead's messages; for every message with tool_calls count call["name"] (subagent_calls = the
    count of "task"); add the input/output token counts from each message's usage_metadata when present.
    (Lead messages only: subagent tokens are not included, so this undercounts the real cost.)
    elapsed_s rounded to 0.1.
    """
    counter = Counter()
    subagent_calls = 0
    for name in _iter_tool_calls(messages):
        if not name:
            continue
        counter[name] += 1
        if name == "task":
            subagent_calls += 1
    in_tokens = out_tokens = 0
    for msg in messages:
        meta = None
        if isinstance(msg, dict):
            meta = msg.get("usage_metadata") or msg.get("usage") or None
        else:
            meta = getattr(msg, "usage_metadata", None)
        if isinstance(meta, dict):
            try:
                in_tokens += int(meta.get("input_tokens", 0) or 0)
                out_tokens += int(meta.get("output_tokens", 0) or 0)
            except (TypeError, ValueError):
                pass
    return {"model": model_name,
            "elapsed_s": round(float(elapsed), 1),
            "subagent_calls": subagent_calls,
            "tool_calls": dict(counter),
            "tokens": {"input": in_tokens, "output": out_tokens}}


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path.

    PSEUDO-CODE:
      files = download(backend, [REPORT_PATH, SOURCES_PATH])
      if the report is missing/empty or sources.json is missing/invalid JSON: raise RuntimeError and WRITE NOTHING
          (a failed run must never leave an empty or half-written report behind)
      write <slug>.sources.json, <slug>.meta.json (topic + summarize(...) + n_sources + source_families: the sorted
      distinct "source" values of sources.json) and <slug>.md
    """
    slug = slugify(topic)
    reports_dir = Path(reports_dir)
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    raw_report = files.get(REPORT_PATH)
    raw_sources = files.get(SOURCES_PATH)
    # Validate BEFORE writing anything.
    if raw_report is None:
        raise RuntimeError("report.md is missing in the sandbox")
    report_text = raw_report.decode("utf-8", errors="replace") if isinstance(raw_report, (bytes, bytearray)) else str(raw_report)
    if not report_text.strip():
        raise RuntimeError("report.md is empty")
    if raw_sources is None:
        raise RuntimeError("sources.json is missing in the sandbox")
    sources_text = raw_sources.decode("utf-8", errors="replace") if isinstance(raw_sources, (bytes, bytearray)) else str(raw_sources)
    try:
        sources = json.loads(sources_text)
    except ValueError as exc:
        raise RuntimeError(f"sources.json is not valid JSON: {exc}") from exc
    if not isinstance(sources, list) or not sources:
        raise RuntimeError("sources.json is empty or not a list")
    # All checks passed: now write.
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"{slug}.sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (reports_dir / f"{slug}.md").write_text(report_text if report_text.endswith("\n") else report_text + "\n", encoding="utf-8")
    families = sorted({str(s.get("source", "")).strip() for s in sources
                       if isinstance(s, dict) and str(s.get("source", "")).strip()})
    summary = summarize(messages, elapsed, model_name)
    meta = {"topic": topic, **summary, "n_sources": len(sources), "source_families": families}
    (reports_dir / f"{slug}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return reports_dir / f"{slug}.md"


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic).

    PSEUDO-CODE:
      empty topic -> print usage to stderr, return 2
      model = make_model(); start = time.monotonic()
      with open_sandbox() as backend:                # the sandbox is always cleaned up, even on errors
          backend.execute("mkdir -p <WORKDIR>/research/notes <WORKDIR>/report")
          upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(), FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()})
          agent = build_lead_agent(backend, model)
          result = agent.invoke({"messages": [{"role": "user", "content": build_prompt(topic)}]},
                                config={"recursion_limit": 1000})
          save_outputs(...); on RuntimeError print "FAILED: ..." to stderr and return 1
      print where the report was saved; return 0
    """
    topic = str(topic or "").strip()
    if not topic:
        print('Usage: python research.py "<topic>"', file=sys.stderr)
        return 2
    try:
        model = make_model()
    except Exception as exc:
        print(f"FAILED: cannot create model: {exc}", file=sys.stderr)
        return 1
    try:
        model_name = getattr(model, "model", None) or getattr(model, "model_name", None) or os.getenv("LAB_MODEL", "")
    except Exception:
        model_name = os.getenv("LAB_MODEL", "")
    start = time.monotonic()
    try:
        with open_sandbox() as backend:  # always stopped/removed on exit, even on errors
            backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
            upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                             FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()})
            agent = build_lead_agent(backend, model)
            result = agent.invoke({"messages": [{"role": "user", "content": build_prompt(topic)}]},
                                  config={"recursion_limit": 1000})
            messages = result.get("messages", []) if isinstance(result, dict) else getattr(result, "messages", [])
            elapsed = time.monotonic() - start
            try:
                out = save_outputs(backend, topic, messages, elapsed, str(model_name))
            except RuntimeError as exc:
                print(f"FAILED: {exc}", file=sys.stderr)
                return 1
        print(f"Report saved to {out}")
        return 0
    except Exception as exc:  # noqa: BLE001 - a failed run exits non-zero and writes nothing
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
