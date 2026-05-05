"""Skill discovery and loading — adapted from skillsbench/libs/terminus_agent/agents/terminus_2/skill_docs.py.

Key adaptation: uses local filesystem (Path.read_text / Path.iterdir) instead of
Harbor's BaseEnvironment.exec() for container-based file access.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SkillMetadata:
    name: str
    description: str
    frontmatter: str
    location: str


class SkillDocLoader:
    """Discovers and loads SKILL.md files from local filesystem directories."""

    def __init__(
        self,
        max_total_chars: int = 16_000,
        max_skill_chars: int = 4_000,
    ) -> None:
        self._max_total_chars = max_total_chars
        self._max_skill_chars = max_skill_chars
        self._last_metadata: list[SkillMetadata] = []

    def build_index(self, roots: Iterable[Path]) -> str:
        metadata = self._collect_metadata(roots)
        self._last_metadata = metadata
        if not metadata:
            return "No skills available."

        lines: list[str] = ["Available skills:"]
        remaining = self._max_total_chars
        for entry in metadata:
            line = f"- {entry.name}: {entry.description or 'No description provided.'}"
            if remaining <= 0:
                lines.append("(Additional skills omitted for length.)")
                break
            if len(line) > remaining:
                line = line[:remaining] + " (Truncated)"
            lines.append(line)
            remaining -= len(line)
        return "\n".join(lines).strip()

    def load_skill(self, name: str, roots: Iterable[Path]) -> str | None:
        skill_dir = self._find_skill_dir(name, roots)
        if skill_dir is None:
            return None
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            return None
        content = skill_md.read_text()
        if len(content) > self._max_skill_chars:
            return content[: self._max_skill_chars] + "\n(Truncated)"
        return content

    def load_references(self, name: str, roots: Iterable[Path]) -> list[tuple[str, str]]:
        skill_dir = self._find_skill_dir(name, roots)
        if skill_dir is None:
            return []
        ref_dir = skill_dir / "references"
        if not ref_dir.exists():
            return []
        refs: list[tuple[str, str]] = []
        for f in sorted(ref_dir.iterdir()):
            if f.suffix == ".md" and f.is_file():
                content = f.read_text()
                if len(content) > self._max_skill_chars:
                    content = content[: self._max_skill_chars] + "\n(Truncated)"
                refs.append((f.name, content))
        return refs

    def get_metadata(self) -> list[SkillMetadata]:
        return list(self._last_metadata)

    # ── private ──

    def _collect_metadata(self, roots: Iterable[Path]) -> list[SkillMetadata]:
        seen: set[str] = set()
        metadata: list[SkillMetadata] = []
        for root in roots:
            if not root.exists():
                continue
            for child in sorted(root.iterdir()):
                if not child.is_dir():
                    continue
                skill_name = child.name
                if skill_name in seen:
                    continue
                skill_md = child / "SKILL.md"
                if not skill_md.exists():
                    continue
                text = skill_md.read_text()
                frontmatter = self._parse_frontmatter(text)
                description = frontmatter.get("description", "").strip()
                frontmatter_block = self._extract_frontmatter_block(text)
                metadata.append(SkillMetadata(
                    name=skill_name,
                    description=description,
                    frontmatter=frontmatter_block,
                    location=str(child),
                ))
                seen.add(skill_name)
        return metadata

    def _find_skill_dir(self, name: str, roots: Iterable[Path]) -> Path | None:
        for root in roots:
            candidate = root / name
            if (candidate / "SKILL.md").exists():
                return candidate
        return None

    @staticmethod
    def _parse_frontmatter(text: str) -> dict[str, str]:
        lines = [l for l in text.splitlines() if l.strip()]
        if not lines or lines[0].strip() != "---":
            return {}
        end_index = None
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                end_index = i
                break
        if end_index is None:
            return {}
        metadata: dict[str, str] = {}
        current_key = None
        current_value_lines: list[str] = []
        for line in lines[1:end_index]:
            if ":" in line and not line.startswith(" "):
                if current_key:
                    metadata[current_key] = " ".join(current_value_lines).strip().strip("\"'")
                key, value = line.split(":", 1)
                current_key = key.strip()
                val = value.strip().strip("\"'")
                current_value_lines = [val] if val and val != ">" else []
            elif current_key and line.startswith(" "):
                current_value_lines.append(line.strip())
        if current_key:
            metadata[current_key] = " ".join(current_value_lines).strip().strip("\"'")
        return metadata

    @staticmethod
    def _extract_frontmatter_block(text: str) -> str:
        lines = text.splitlines()
        if not lines or lines[0].strip() != "---":
            return ""
        end_index = None
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                end_index = i
                break
        if end_index is None:
            return ""
        return "\n".join(lines[: end_index + 1]).strip()
