#!/usr/bin/env python3
"""Local-first memory retrieval and briefing utilities.

This module keeps Markdown files and eval JSON artifacts as source-of-truth,
then builds a derived SQLite index for fast search and safe snippet reads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_INDEX_REL_PATH = Path("memory/evals/retrieval_index.sqlite")
DEFAULT_SESSION_LIMIT = 10
CHUNK_LINES = 22
CHUNK_OVERLAP = 6


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    path: str
    source: str
    start_line: int
    end_line: int
    text: str


@dataclass(frozen=True)
class SearchHit:
    path: str
    source: str
    start_line: int
    end_line: int
    score: float
    snippet: str

    @property
    def citation(self) -> str:
        if self.start_line == self.end_line:
            return f"{self.path}#L{self.start_line}"
        return f"{self.path}#L{self.start_line}-L{self.end_line}"


def _repo_root(path: Path | None = None) -> Path:
    if path is not None:
        return path
    return Path(__file__).resolve().parents[1]


def _safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_whitespace(text: str) -> str:
    return " ".join(str(text).split())


def _chunk_markdown(path: str, source: str, body: str) -> list[ChunkRecord]:
    lines = body.splitlines()
    chunks: list[ChunkRecord] = []
    if not lines:
        return chunks

    step = max(1, CHUNK_LINES - CHUNK_OVERLAP)
    cursor = 0
    while cursor < len(lines):
        end = min(len(lines), cursor + CHUNK_LINES)
        window = lines[cursor:end]
        text = "\n".join(window).strip()
        if text:
            key = f"{path}:{cursor + 1}:{end}:{_hash_text(text)[:10]}"
            chunks.append(
                ChunkRecord(
                    chunk_id=_hash_text(key),
                    path=path,
                    source=source,
                    start_line=cursor + 1,
                    end_line=end,
                    text=text,
                )
            )
        if end >= len(lines):
            break
        cursor += step
    return chunks


def _line_count(text: str) -> int:
    if text == "":
        return 1
    return text.count("\n") + 1


def _safe_get(mapping: dict[str, Any], *keys: str, default: Any = "") -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value is None else value


def _iter_recent_markdown_paths(directory: Path, limit: int) -> list[Path]:
    if not directory.exists():
        return []
    paths = sorted(directory.glob("*.md"), key=lambda p: p.name, reverse=True)
    return paths[: max(0, int(limit))]


def _memory_markdown_paths(root: Path, session_limit: int = DEFAULT_SESSION_LIMIT) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []

    stable_files = [
        ("canon", root / "memory/canon.md"),
        ("mood", root / "memory/moods.md"),
        ("instrument", root / "memory/instruments.md"),
        ("instrument", root / "memory/instrument_ranges.md"),
        ("governance", root / "memory/governance/active.md"),
    ]
    for source, path in stable_files:
        if path.exists() and path.is_file():
            paths.append((source, path))

    fundamentals = sorted((root / "memory/fundamentals").glob("*.md"))
    for path in fundamentals:
        if path.is_file():
            paths.append(("fundamental", path))

    instruments = sorted((root / "memory/instruments").glob("*.md"))
    for path in instruments:
        if path.is_file():
            paths.append(("instrument", path))

    sessions = _iter_recent_markdown_paths(root / "memory/sessions", session_limit)
    for path in sessions:
        if path.is_file():
            paths.append(("session", path))

    work_journal = _iter_recent_markdown_paths(root / "memory/work_journal", session_limit)
    for path in work_journal:
        if path.is_file():
            paths.append(("journal", path))

    unique: dict[str, tuple[str, Path]] = {}
    for source, path in paths:
        rel = str(path.relative_to(root)).replace(os.sep, "/")
        unique[rel] = (source, path)
    return list(unique.values())


def _governance_active_guidance(root: Path, max_items: int = 3) -> list[str]:
    path = root / "memory/governance/active.md"
    if not path.exists() or not path.is_file():
        return []
    lines = _safe_read_text(path).splitlines()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("- [gov:"):
            continue
        out.append(stripped[2:].strip())
        if len(out) >= max(1, int(max_items)):
            break
    return out


def _format_eval_summary(artifact: dict[str, Any]) -> str:
    run_id = str(artifact.get("run_id", ""))
    timestamp = str(artifact.get("timestamp_utc", ""))
    status = str(artifact.get("status", ""))

    composition = artifact.get("composition", {}) if isinstance(artifact.get("composition"), dict) else {}
    reflection = artifact.get("reflection", {}) if isinstance(artifact.get("reflection"), dict) else {}
    fingerprints = artifact.get("fingerprints", {}) if isinstance(artifact.get("fingerprints"), dict) else {}

    mood = str(composition.get("mood", ""))
    key_name = str(composition.get("key_name", ""))
    tempo = composition.get("tempo_bpm", "")
    signature = str(composition.get("signature", ""))
    meter_bpm = str(fingerprints.get("meter_bpm", ""))

    novelty = reflection.get("novelty_score")
    similarity = reflection.get("similarity_to_reference")
    flags = reflection.get("repetition_flags", [])
    prompts = reflection.get("prompts", [])

    lines: list[str] = [
        f"run_id: {run_id}",
        f"timestamp_utc: {timestamp}",
        f"status: {status}",
        f"mood: {mood}",
        f"key_name: {key_name}",
        f"tempo_bpm: {tempo}",
        f"signature: {signature}",
        f"meter_bpm: {meter_bpm}",
        f"novelty_score: {novelty}",
        f"similarity_to_reference: {similarity}",
    ]

    if isinstance(flags, Sequence) and not isinstance(flags, (str, bytes)):
        for flag in flags[:5]:
            lines.append(f"repetition_flag: {flag}")

    merit = reflection.get("merit_rubric", {}) if isinstance(reflection.get("merit_rubric"), dict) else {}
    for key in sorted(merit.keys()):
        lines.append(f"merit_{key}: {merit.get(key)}")

    identity = reflection.get("instrument_identity", {}) if isinstance(reflection.get("instrument_identity"), dict) else {}
    if identity:
        lines.append(f"instrument_identity_status: {identity.get('status', '')}")
        identity_flags = identity.get("flags", [])
        if isinstance(identity_flags, Sequence) and not isinstance(identity_flags, (str, bytes)):
            for flag in identity_flags[:5]:
                lines.append(f"identity_flag: {flag}")

    if isinstance(prompts, Sequence) and not isinstance(prompts, (str, bytes)):
        for prompt in prompts[:3]:
            lines.append(f"reflection_prompt: {_normalize_whitespace(prompt)}")

    return "\n".join(lines)


def _eval_artifact_paths(root: Path, limit: int = 120) -> list[Path]:
    index_path = root / "memory/evals/composition_index.json"
    artifacts: list[Path] = []
    resolved_root = root.resolve()

    if index_path.exists():
        try:
            payload = json.loads(_safe_read_text(index_path))
            entries = payload.get("entries", []) if isinstance(payload, dict) else []
            if isinstance(entries, list):
                for entry in entries[:limit]:
                    if not isinstance(entry, dict):
                        continue
                    rel = entry.get("artifact_path")
                    if isinstance(rel, str) and rel.strip():
                        candidate = root / rel
                        candidate_resolved = candidate.resolve()
                        try:
                            candidate_resolved.relative_to(resolved_root)
                        except ValueError:
                            continue
                        if candidate_resolved.exists() and candidate_resolved.is_file():
                            artifacts.append(candidate_resolved)
        except json.JSONDecodeError:
            pass

    if not artifacts:
        candidates = sorted(
            (root / "memory/evals/compositions").glob("**/*.json"),
            key=lambda p: p.name,
            reverse=True,
        )
        artifacts.extend(candidates[:limit])

    unique: dict[str, Path] = {}
    for path in artifacts:
        try:
            rel = str(path.resolve().relative_to(resolved_root)).replace(os.sep, "/")
        except ValueError:
            continue
        unique[rel] = path
    return list(unique.values())


class RetrievalIndex:
    """Derived memory/eval search index."""

    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        index_rel_path: Path = DEFAULT_INDEX_REL_PATH,
        session_limit: int = DEFAULT_SESSION_LIMIT,
    ) -> None:
        self.repo_root = _repo_root(repo_root)
        self.index_path = self.repo_root / index_rel_path
        self.session_limit = max(1, int(session_limit))

    def _connect(self) -> sqlite3.Connection:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.index_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    @staticmethod
    def _fts_available(conn: sqlite3.Connection) -> bool:
        try:
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(chunk_id UNINDEXED, text, path UNINDEXED, source UNINDEXED)"
            )
            return True
        except sqlite3.OperationalError:
            return False

    def _ensure_schema(self, conn: sqlite3.Connection) -> bool:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
              chunk_id TEXT PRIMARY KEY,
              path TEXT NOT NULL,
              source TEXT NOT NULL,
              start_line INTEGER NOT NULL,
              end_line INTEGER NOT NULL,
              text TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS index_meta (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            )
            """
        )
        return self._fts_available(conn)

    def _clear(self, conn: sqlite3.Connection, *, fts_enabled: bool) -> None:
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM index_meta")
        if fts_enabled:
            conn.execute("DELETE FROM chunks_fts")

    def _index_markdown(self, conn: sqlite3.Connection, *, fts_enabled: bool) -> int:
        total = 0
        for source, path in _memory_markdown_paths(self.repo_root, self.session_limit):
            rel = str(path.relative_to(self.repo_root)).replace(os.sep, "/")
            body = _safe_read_text(path)
            for chunk in _chunk_markdown(rel, source, body):
                self._insert_chunk(conn, chunk, fts_enabled=fts_enabled)
                total += 1
        return total

    def _index_eval_artifacts(self, conn: sqlite3.Connection, *, fts_enabled: bool) -> int:
        total = 0
        resolved_root = self.repo_root.resolve()
        for path in _eval_artifact_paths(self.repo_root):
            rel = str(path.resolve().relative_to(resolved_root)).replace(os.sep, "/")
            try:
                artifact = json.loads(_safe_read_text(path))
            except json.JSONDecodeError:
                continue
            if not isinstance(artifact, dict):
                continue
            text = _format_eval_summary(artifact)
            chunk = ChunkRecord(
                chunk_id=_hash_text(f"{rel}:1:{_hash_text(text)[:12]}"),
                path=rel,
                source="eval",
                start_line=1,
                end_line=max(1, _line_count(text)),
                text=text,
            )
            self._insert_chunk(conn, chunk, fts_enabled=fts_enabled)
            total += 1
        return total

    @staticmethod
    def _insert_chunk(conn: sqlite3.Connection, chunk: ChunkRecord, *, fts_enabled: bool) -> None:
        conn.execute(
            """
            INSERT OR REPLACE INTO chunks (chunk_id, path, source, start_line, end_line, text)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                chunk.chunk_id,
                chunk.path,
                chunk.source,
                chunk.start_line,
                chunk.end_line,
                chunk.text,
            ),
        )
        if fts_enabled:
            conn.execute(
                "INSERT INTO chunks_fts (chunk_id, text, path, source) VALUES (?, ?, ?, ?)",
                (chunk.chunk_id, chunk.text, chunk.path, chunk.source),
            )

    def rebuild(self) -> dict[str, int | str | bool]:
        conn = self._connect()
        try:
            fts_enabled = self._ensure_schema(conn)
            self._clear(conn, fts_enabled=fts_enabled)

            markdown_chunks = self._index_markdown(conn, fts_enabled=fts_enabled)
            eval_chunks = self._index_eval_artifacts(conn, fts_enabled=fts_enabled)

            conn.execute(
                "INSERT OR REPLACE INTO index_meta (key, value) VALUES (?, ?)",
                ("fts_enabled", "true" if fts_enabled else "false"),
            )
            conn.execute(
                "INSERT OR REPLACE INTO index_meta (key, value) VALUES (?, ?)",
                ("markdown_chunks", str(markdown_chunks)),
            )
            conn.execute(
                "INSERT OR REPLACE INTO index_meta (key, value) VALUES (?, ?)",
                ("eval_chunks", str(eval_chunks)),
            )
            conn.commit()

            return {
                "index_path": str(self.index_path),
                "fts_enabled": fts_enabled,
                "markdown_chunks": markdown_chunks,
                "eval_chunks": eval_chunks,
                "total_chunks": markdown_chunks + eval_chunks,
            }
        finally:
            conn.close()

    def status(self) -> dict[str, Any]:
        conn = self._connect()
        try:
            fts_enabled = self._ensure_schema(conn)
            total_chunks = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
            by_source = {
                row[0]: int(row[1])
                for row in conn.execute("SELECT source, COUNT(*) FROM chunks GROUP BY source").fetchall()
            }
            return {
                "index_path": str(self.index_path),
                "exists": self.index_path.exists(),
                "fts_enabled": fts_enabled,
                "total_chunks": total_chunks,
                "chunks_by_source": by_source,
            }
        finally:
            conn.close()

    def _search_fts(self, conn: sqlite3.Connection, query: str, max_results: int) -> list[SearchHit]:
        tokens = re.findall(r"[A-Za-z0-9_]+", query)
        fts_query = " OR ".join(f'\"{token}\"' for token in tokens) if tokens else query.strip()
        if not fts_query:
            return []
        rows = conn.execute(
            """
            SELECT c.path, c.source, c.start_line, c.end_line, c.text,
                   bm25(chunks_fts) AS rank
            FROM chunks_fts
            JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
            WHERE chunks_fts MATCH ?
            ORDER BY rank ASC
            LIMIT ?
            """,
            (fts_query, max_results),
        ).fetchall()

        hits: list[SearchHit] = []
        for row in rows:
            rank = float(row["rank"]) if row["rank"] is not None else 9999.0
            score = round(1.0 / (1.0 + max(0.0, rank)), 6)
            hits.append(
                SearchHit(
                    path=str(row["path"]),
                    source=str(row["source"]),
                    start_line=int(row["start_line"]),
                    end_line=int(row["end_line"]),
                    score=score,
                    snippet=str(row["text"]),
                )
            )
        return hits

    @staticmethod
    def _fallback_score(text: str, terms: Sequence[str]) -> float:
        lowered = text.lower()
        matches = sum(lowered.count(term) for term in terms)
        if matches <= 0:
            return 0.0
        return round(float(matches) / float(max(1, len(text.split()))), 6)

    def _search_fallback(self, conn: sqlite3.Connection, query: str, max_results: int) -> list[SearchHit]:
        terms = [term.lower() for term in query.split() if term.strip()]
        if not terms:
            return []

        where = " OR ".join(["LOWER(text) LIKE ?" for _ in terms])
        params = [f"%{term}%" for term in terms]
        rows = conn.execute(
            f"SELECT path, source, start_line, end_line, text FROM chunks WHERE {where} LIMIT ?",
            (*params, max_results * 4),
        ).fetchall()

        scored: list[SearchHit] = []
        for row in rows:
            text = str(row["text"])
            score = self._fallback_score(text, terms)
            if score <= 0:
                continue
            scored.append(
                SearchHit(
                    path=str(row["path"]),
                    source=str(row["source"]),
                    start_line=int(row["start_line"]),
                    end_line=int(row["end_line"]),
                    score=score,
                    snippet=text,
                )
            )
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:max_results]

    def search(self, *, query: str, max_results: int = 6, min_score: float = 0.0) -> list[SearchHit]:
        cleaned = query.strip()
        if not cleaned:
            return []

        conn = self._connect()
        try:
            fts_enabled = self._ensure_schema(conn)
            total_chunks = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
            if total_chunks <= 0:
                self.rebuild()
                conn.close()
                conn = self._connect()
                fts_enabled = self._ensure_schema(conn)

            if fts_enabled:
                hits = self._search_fts(conn, cleaned, max_results)
            else:
                hits = self._search_fallback(conn, cleaned, max_results)

            return [hit for hit in hits if float(hit.score) >= float(min_score)]
        finally:
            conn.close()

    def read_window(self, *, rel_path: str, start_line: int | None = None, lines: int | None = None) -> dict[str, Any]:
        normalized = str(rel_path).strip().replace("\\", "/")
        if not normalized:
            raise ValueError("path is required")

        abs_path = (self.repo_root / normalized).resolve()
        memory_root = (self.repo_root / "memory").resolve()
        try:
            abs_path.relative_to(memory_root)
        except ValueError as exc:
            raise ValueError("path must stay under memory/") from exc

        if not abs_path.exists() or not abs_path.is_file():
            raise ValueError("path does not exist")

        if abs_path.suffix.lower() not in {".md", ".json", ".toml"}:
            raise ValueError("path extension is not supported")

        content = _safe_read_text(abs_path)
        if start_line is None and lines is None:
            return {
                "path": normalized,
                "from": 1,
                "lines": _line_count(content),
                "text": content,
            }

        safe_start = max(1, int(start_line or 1))
        safe_lines = max(1, int(lines or _line_count(content)))
        split_lines = content.splitlines()
        window = split_lines[safe_start - 1 : safe_start - 1 + safe_lines]
        return {
            "path": normalized,
            "from": safe_start,
            "lines": safe_lines,
            "text": "\n".join(window),
        }

    def brief(
        self,
        *,
        meter: str | None = None,
        bpm: float | None = None,
        mood: str | None = None,
        key_name: str | None = None,
        focus: str | None = None,
        max_results: int = 6,
    ) -> str:
        terms = [
            "composition",
            "arrangement",
            "reflection",
            "novelty",
            "repetition",
        ]
        if meter:
            terms.append(str(meter))
        if bpm is not None:
            terms.append(str(int(round(float(bpm)))))
        if mood:
            terms.append(str(mood))
        if key_name:
            terms.append(str(key_name))
        if focus:
            terms.append(str(focus))

        query = " ".join(term for term in terms if term)
        hits = self.search(query=query, max_results=max(1, int(max_results)))
        governance_lines = _governance_active_guidance(self.repo_root, max_items=3)

        if not hits and not governance_lines:
            return "Memory brief: no indexed context found for this request."

        lines: list[str] = [
            "Memory brief:",
            f"- context query: {query}",
        ]
        for item in governance_lines:
            lines.append(f"- governance :: {item}")
        for idx, hit in enumerate(hits, start=1):
            one_liner = _normalize_whitespace(hit.snippet).strip()
            lines.append(
                f"- [{idx}] ({hit.source}) {hit.citation} :: {one_liner[:220]}"
            )
        return "\n".join(lines)


def build_context_brief(
    *,
    meter: str | None = None,
    bpm: float | None = None,
    mood: str | None = None,
    key_name: str | None = None,
    focus: str | None = None,
    max_results: int = 6,
    repo_root: Path | None = None,
) -> str:
    index = RetrievalIndex(repo_root=repo_root)
    return index.brief(
        meter=meter,
        bpm=bpm,
        mood=mood,
        key_name=key_name,
        focus=focus,
        max_results=max_results,
    )


def _print_search(hits: Sequence[SearchHit]) -> None:
    if not hits:
        print("No results.")
        return
    for idx, hit in enumerate(hits, start=1):
        print(f"[{idx}] score={hit.score:.6f} source={hit.source}")
        print(f"path={hit.citation}")
        print(hit.snippet)
        if idx != len(hits):
            print()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="Show retrieval index status")
    status.add_argument("--repo-root", default=None, help="Optional repository root")

    index_cmd = sub.add_parser("index", help="Rebuild retrieval index")
    index_cmd.add_argument("--repo-root", default=None, help="Optional repository root")

    search = sub.add_parser("search", help="Search indexed memory/eval chunks")
    search.add_argument("--repo-root", default=None, help="Optional repository root")
    search.add_argument("--query", required=True, help="Search query")
    search.add_argument("--max-results", type=int, default=6, help="Max hits")
    search.add_argument("--min-score", type=float, default=0.0, help="Score floor")

    get = sub.add_parser("get", help="Read allowed memory file snippet")
    get.add_argument("--repo-root", default=None, help="Optional repository root")
    get.add_argument("--path", required=True, help="Repo-relative path under memory/")
    get.add_argument("--from", dest="start_line", type=int, default=None, help="Start line (1-based)")
    get.add_argument("--lines", type=int, default=None, help="Line count")

    brief = sub.add_parser("brief", help="Build a run context brief")
    brief.add_argument("--repo-root", default=None, help="Optional repository root")
    brief.add_argument("--meter", default=None, help="Meter string (for example 5/4)")
    brief.add_argument("--bpm", type=float, default=None, help="Tempo in BPM")
    brief.add_argument("--mood", default=None, help="Mood label")
    brief.add_argument("--key-name", default=None, help="Key label")
    brief.add_argument("--focus", default=None, help="Focus label")
    brief.add_argument("--max-results", type=int, default=6, help="Max hits")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = None if args.repo_root in (None, "") else Path(args.repo_root)
    index = RetrievalIndex(repo_root=root)

    if args.command == "status":
        payload = index.status()
        print(f"index_path={payload['index_path']}")
        print(f"exists={str(payload['exists']).lower()}")
        print(f"fts_enabled={str(payload['fts_enabled']).lower()}")
        print(f"total_chunks={payload['total_chunks']}")
        by_source = payload.get("chunks_by_source", {})
        for key in sorted(by_source.keys()):
            print(f"chunks_{key}={by_source[key]}")
        return 0

    if args.command == "index":
        payload = index.rebuild()
        print(f"index_path={payload['index_path']}")
        print(f"fts_enabled={str(payload['fts_enabled']).lower()}")
        print(f"markdown_chunks={payload['markdown_chunks']}")
        print(f"eval_chunks={payload['eval_chunks']}")
        print(f"total_chunks={payload['total_chunks']}")
        return 0

    if args.command == "search":
        hits = index.search(
            query=str(args.query),
            max_results=max(1, int(args.max_results)),
            min_score=float(args.min_score),
        )
        _print_search(hits)
        return 0

    if args.command == "get":
        payload = index.read_window(
            rel_path=str(args.path),
            start_line=args.start_line,
            lines=args.lines,
        )
        print(f"path={payload['path']}")
        print(f"from={payload['from']}")
        print(f"lines={payload['lines']}")
        print(payload["text"])
        return 0

    if args.command == "brief":
        print(
            index.brief(
                meter=args.meter,
                bpm=args.bpm,
                mood=args.mood,
                key_name=args.key_name,
                focus=args.focus,
                max_results=max(1, int(args.max_results)),
            )
        )
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
