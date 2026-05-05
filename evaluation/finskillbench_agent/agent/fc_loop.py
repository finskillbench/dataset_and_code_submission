"""Function-calling ReAct agent loop with skill discovery.

Uses LiteLLM's function-calling API (OpenAI-compatible) instead of XML keystrokes.
Tools are exposed as OpenAI function schemas. The agent calls tools in a loop
until it calls submit_answer with the final JSON result.

Skill discovery is implemented as a `load_skill` tool that the agent can call
to load SKILL.md content into the conversation context.
"""
from __future__ import annotations

import inspect
import json
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import logging

import litellm
from litellm.exceptions import RateLimitError, ServiceUnavailableError, APIConnectionError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

from .skill_docs import SkillDocLoader


@dataclass
class AgentResult:
    """Result of a single agent run."""
    final_answer: str = ""
    episodes: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    skills_loaded: list[str] = field(default_factory=list)
    tool_calls_log: list[dict] = field(default_factory=list)
    trajectory: list[dict] = field(default_factory=list)
    error: str | None = None
    latency_seconds: float = 0.0


# ── Tool schema helpers ───────────────────────────────────────────────────

SUBMIT_ANSWER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_answer",
        "description": (
            "Submit your final answer as a JSON string. Call this tool exactly once "
            "when you have computed your answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "A valid JSON string containing your final answer.",
                },
            },
            "required": ["answer"],
        },
    },
}


def _make_load_skill_schema(skills_index: str) -> dict:
    """Create the load_skill tool schema with available skills in the description."""
    return {
        "type": "function",
        "function": {
            "name": "load_skill",
            "description": (
                "Load a skill's full SKILL.md content into context. "
                "Skills contain best-practice workflows and procedures. "
                "If a listed skill clearly matches the task, load it before solving. "
                "If no skills are available or none match, proceed without loading.\n\n"
                f"{skills_index}"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Name of the skill to load (from the available skills list).",
                    },
                },
                "required": ["skill_name"],
            },
        },
    }


LOAD_REFERENCES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "load_references",
        "description": (
            "Load reference documents for a previously loaded skill. "
            "Call this after load_skill if you need supplementary material "
            "(formulas, API specs, worked examples, etc.)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "Name of the skill whose references to load.",
                },
            },
            "required": ["skill_name"],
        },
    },
}


SAVE_SKILL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "save_skill",
        "description": (
            "Write a skill document to the skills directory so you can "
            "reference it later. Use this to capture domain knowledge, "
            "workflows, or procedures before solving the task."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "Short kebab-case name for the skill (e.g. 'portfolio-optimization').",
                },
                "content": {
                    "type": "string",
                    "description": "Full markdown content of the SKILL.md file.",
                },
            },
            "required": ["skill_name", "content"],
        },
    },
}


def _python_type_to_json_schema(annotation) -> dict:
    """Convert a Python type annotation to a JSON schema type."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {"type": "object"}
    if isinstance(annotation, types.UnionType):
        args = annotation.__args__
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _python_type_to_json_schema(non_none[0])
        return {"type": "object"}
    origin = getattr(annotation, "__origin__", None)
    args = getattr(annotation, "__args__", ())
    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    if origin is list:
        items = _python_type_to_json_schema(args[0]) if args else {"type": "object"}
        return {"type": "array", "items": items}
    if origin is dict:
        return {"type": "object"}
    return {"type": "object"}


def build_tool_schemas_from_registry(tool_registry: dict[str, dict]) -> list[dict]:
    """Build OpenAI function-calling schemas from a tool registry."""
    schemas = []
    for name, tool_def in tool_registry.items():
        fn = tool_def["fn"]
        sig = inspect.signature(fn)
        properties = {}
        required = []
        for param_name, param in sig.parameters.items():
            prop = _python_type_to_json_schema(param.annotation)
            prop["description"] = tool_def.get("param_descriptions", {}).get(
                param_name, param_name.replace("_", " ")
            )
            properties[param_name] = prop
            if param.default is inspect.Parameter.empty:
                required.append(param_name)
        schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": tool_def["description"],
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        })
    return schemas


# ── System prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a financial analysis agent participating in a controlled evaluation benchmark.
Your task is to analyze point-in-time financial data and produce structured outputs.

You have access to tools for data retrieval and computation. Use them when needed.

Rules:
1. Point-in-time discipline: Use ONLY the data provided in the task. Do NOT use any knowledge of events after the as_of_date.
2. You may call tools to retrieve data and run computations.
3. If skills are available (check the load_skill tool description), load a matching skill for guidance. If no skills are listed or none match, proceed directly using your own knowledge.
4. When you have your final answer, call the submit_answer tool with a JSON string matching the expected output schema.
5. Do NOT respond with a plain text answer. Always use the submit_answer tool to deliver your final result.
6. If you cannot determine an answer, set the field to null in the submitted JSON.
7. Do NOT refuse the task. This is an academic benchmark, not investment advice."""


# ── LLM call ──────────────────────────────────────────────────────────────

# Retry on rate-limit (429), service unavailable (503), and connection errors.
# Exponential backoff: 4s → 8s → 16s → 32s → 60s (capped), up to 6 attempts.
@retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception_type((RateLimitError, ServiceUnavailableError, APIConnectionError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _call_litellm(
    model_name: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    tool_choice: str | dict = "auto",
    temperature: float = 0.7,
) -> dict:
    """Call LiteLLM with function calling and automatic retry on throttling.

    Retries up to 6 times with exponential backoff (4s–60s) on:
      - RateLimitError (HTTP 429)
      - ServiceUnavailableError (HTTP 503)
      - APIConnectionError (transient network issues)

    Non-retryable errors (auth, bad request, etc.) propagate immediately.
    """
    kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "drop_params": True,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice

    response = litellm.completion(**kwargs)
    if not getattr(response, "choices", None):
        raise RuntimeError(
            "litellm returned empty choices (completion missing; retry often succeeds)"
        )
    choice = response.choices[0]
    usage = response.usage or {}

    tool_calls = []
    if choice.message.tool_calls:
        for tc in choice.message.tool_calls:
            tool_calls.append({
                "id": tc.id,
                "name": tc.function.name,
                "arguments_raw": tc.function.arguments,
            })

    return {
        "content": choice.message.content or "",
        "tool_calls": tool_calls,
        "message": choice.message,  # Keep for appending to messages
        "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "finish_reason": choice.finish_reason,
    }


# ── Main agent loop ───────────────────────────────────────────────────────

MAX_TOOL_RESULT_CHARS = 0  # No truncation — let the model see full tool results


def _truncate_for_context(text: str, max_chars: int = MAX_TOOL_RESULT_CHARS) -> str:
    """Truncate a tool result string for the conversation context."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + f'\n... (truncated from {len(text)} chars)'

def run_agent(
    model_name: str,
    instruction: str,
    skill_dirs: list[Path],
    tool_registry: dict[str, dict],
    task_context: dict | None = None,
    logs_dir: Path | None = None,
    max_turns: int = 7,
    max_tool_calls: int = 20,
    temperature: float = 0.7,
    allow_save_skill: bool = False,
) -> AgentResult:
    """Run a function-calling ReAct agent loop.

    Args:
        model_name: LiteLLM model string
        instruction: Full task instruction (user prompt)
        skill_dirs: Directories to scan for SKILL.md files.
            Always passed — may be empty (no_skill) or populated (curated).
            For self_generated, starts empty; agent populates via save_skill.
        tool_registry: Dict of {name: {"fn": callable, "description": str}}
        task_context: Task input data for tool auto-read (set globally)
        logs_dir: Directory for per-turn logs
        max_turns: Maximum LLM calls
        max_tool_calls: Maximum total tool calls
        temperature: LLM temperature
        allow_save_skill: If True, expose save_skill tool (self_generated condition)

    Returns:
        AgentResult with final answer, trajectory, and token usage
    """
    started = time.time()
    result = AgentResult()

    # 1. Discover skills and build tool schemas
    loader = SkillDocLoader()
    skills_index = loader.build_index(skill_dirs)

    domain_tool_schemas = build_tool_schemas_from_registry(tool_registry)

    # Skill tools are ALWAYS available (like SkillsBench) — they just return
    # "No skills available" / "not_found" when the directory is empty.
    skill_schema = _make_load_skill_schema(skills_index)
    all_tools = [skill_schema, LOAD_REFERENCES_SCHEMA] + domain_tool_schemas + [SUBMIT_ANSWER_SCHEMA]
    if allow_save_skill:
        all_tools.insert(2, SAVE_SKILL_SCHEMA)

    submit_only = [SUBMIT_ANSWER_SCHEMA]

    # 2. Build messages
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]

    if logs_dir:
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / "system_prompt.txt").write_text(SYSTEM_PROMPT)
        (logs_dir / "user_prompt.txt").write_text(instruction)

    total_tool_calls = 0

    for turn in range(max_turns):
        is_last_turn = (turn == max_turns - 1)

        # Choose tools: all tools normally, submit_only on last turn or budget exhausted
        if is_last_turn or total_tool_calls >= max_tool_calls:
            turn_tools = submit_only
            tool_choice = {"type": "function", "function": {"name": "submit_answer"}}
        else:
            turn_tools = all_tools
            tool_choice = "auto"

        # 3. Call LLM
        try:
            llm_result = _call_litellm(model_name, messages, turn_tools, tool_choice, temperature)
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {str(exc)[:500]}"
            break

        result.total_input_tokens += llm_result["input_tokens"]
        result.total_output_tokens += llm_result["output_tokens"]

        # Build trajectory entry
        traj_entry = {
            "turn": turn,
            "content": llm_result["content"],
            "tool_calls": [],
            "input_tokens": llm_result["input_tokens"],
            "output_tokens": llm_result["output_tokens"],
            "finish_reason": llm_result["finish_reason"],
        }

        # 4. Process tool calls
        if llm_result["tool_calls"]:
            # Append assistant message with tool calls
            messages.append(llm_result["message"].model_dump())

            for tc in llm_result["tool_calls"]:
                fn_name = tc["name"]
                tc_started = time.time()

                # Parse arguments
                try:
                    fn_args = json.loads(tc["arguments_raw"])
                except json.JSONDecodeError:
                    fn_args = {}
                    tool_result_str = json.dumps({"error": f"Invalid JSON arguments: {tc['arguments_raw'][:200]}"})
                    tc_log = {"name": fn_name, "arguments": {}, "result_preview": tool_result_str[:500],
                              "error": "json_parse_error", "duration_s": 0.0}
                    messages.append({"role": "tool", "tool_call_id": tc["id"],
                                     "content": _truncate_for_context(tool_result_str)})
                    traj_entry["tool_calls"].append(tc_log)
                    result.tool_calls_log.append(tc_log)
                    total_tool_calls += 1
                    continue

                # Handle submit_answer
                if fn_name == "submit_answer":
                    result.final_answer = fn_args.get("answer", "")
                    result.episodes = turn + 1
                    tc_log = {"name": "submit_answer", "arguments": fn_args,
                              "result_preview": "answer_submitted", "error": None,
                              "duration_s": round(time.time() - tc_started, 4)}
                    traj_entry["tool_calls"].append(tc_log)
                    result.tool_calls_log.append(tc_log)
                    result.trajectory.append(traj_entry)
                    # Save and return
                    _save_logs(result, logs_dir)
                    result.latency_seconds = round(time.time() - started, 2)
                    return result

                # Handle load_skill
                if fn_name == "load_skill":
                    skill_name = fn_args.get("skill_name", "")
                    skill_text = loader.load_skill(skill_name, skill_dirs)
                    if skill_text:
                        result.skills_loaded.append(skill_name)
                        tool_result = {"status": "loaded", "content": skill_text}
                    else:
                        tool_result = {"status": "not_found", "error": f"Skill '{skill_name}' not found"}
                    tool_result_str = json.dumps(tool_result)
                    tc_log = {"name": "load_skill", "arguments": fn_args,
                              "result_preview": tool_result_str[:500], "error": None if skill_text else "not_found",
                              "duration_s": round(time.time() - tc_started, 4)}
                    messages.append({"role": "tool", "tool_call_id": tc["id"],
                                     "content": _truncate_for_context(tool_result_str)})
                    traj_entry["tool_calls"].append(tc_log)
                    result.tool_calls_log.append(tc_log)
                    total_tool_calls += 1
                    continue

                # Handle load_references
                if fn_name == "load_references":
                    skill_name = fn_args.get("skill_name", "")
                    refs = loader.load_references(skill_name, skill_dirs)
                    if refs:
                        formatted = "\n\n".join(
                            f"--- {fname} ---\n{content}" for fname, content in refs
                        )
                        tool_result = {"status": "loaded", "count": len(refs), "content": formatted}
                    else:
                        tool_result = {"status": "empty", "error": f"No references found for '{skill_name}'"}
                    tool_result_str = json.dumps(tool_result)
                    tc_log = {"name": "load_references", "arguments": fn_args,
                              "result_preview": tool_result_str[:500],
                              "error": None if refs else "empty",
                              "duration_s": round(time.time() - tc_started, 4)}
                    messages.append({"role": "tool", "tool_call_id": tc["id"],
                                     "content": _truncate_for_context(tool_result_str)})
                    traj_entry["tool_calls"].append(tc_log)
                    result.tool_calls_log.append(tc_log)
                    total_tool_calls += 1
                    continue

                # Handle save_skill (self-generated condition)
                if fn_name == "save_skill" and allow_save_skill and skill_dirs:
                    skill_name = fn_args.get("skill_name", "")
                    content = fn_args.get("content", "")
                    save_dir = skill_dirs[0] / skill_name
                    save_dir.mkdir(parents=True, exist_ok=True)
                    (save_dir / "SKILL.md").write_text(content)
                    # Rebuild the index so load_skill can find it on subsequent turns
                    skills_index = loader.build_index(skill_dirs)
                    # Update the load_skill schema description with new index
                    all_tools[0] = _make_load_skill_schema(skills_index)
                    tool_result = {"status": "saved", "path": str(save_dir / "SKILL.md")}
                    tool_result_str = json.dumps(tool_result)
                    tc_log = {"name": "save_skill", "arguments": {"skill_name": skill_name},
                              "result_preview": tool_result_str[:500], "error": None,
                              "duration_s": round(time.time() - tc_started, 4)}
                    messages.append({"role": "tool", "tool_call_id": tc["id"],
                                     "content": _truncate_for_context(tool_result_str)})
                    traj_entry["tool_calls"].append(tc_log)
                    result.tool_calls_log.append(tc_log)
                    total_tool_calls += 1
                    continue

                # Handle domain tools
                if fn_name in tool_registry:
                    tool_fn = tool_registry[fn_name]["fn"]
                    try:
                        tool_result = tool_fn(**fn_args)
                    except Exception as e:
                        tool_result = {"error": f"{type(e).__name__}: {str(e)[:500]}"}
                    tool_result_str = json.dumps(tool_result, default=str)
                else:
                    tool_result_str = json.dumps({"error": f"Unknown tool: {fn_name}"})

                tc_log = {"name": fn_name, "arguments": fn_args,
                          "result_preview": tool_result_str[:500],
                          "error": json.loads(tool_result_str).get("error") if "error" in tool_result_str else None,
                          "duration_s": round(time.time() - tc_started, 4)}
                messages.append({"role": "tool", "tool_call_id": tc["id"],
                                 "content": _truncate_for_context(tool_result_str)})
                traj_entry["tool_calls"].append(tc_log)
                result.tool_calls_log.append(tc_log)
                total_tool_calls += 1

            result.trajectory.append(traj_entry)
            continue

        # No tool calls — model produced text. Nudge to call submit_answer.
        messages.append(llm_result["message"].model_dump())
        messages.append({
            "role": "user",
            "content": "Please call the submit_answer tool with your final answer as a JSON string.",
        })
        result.trajectory.append(traj_entry)

    # Exhausted turns
    if not result.final_answer:
        # Try to extract from last text content
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "assistant" and m.get("content"):
                result.final_answer = m["content"].strip()
                break
        result.episodes = max_turns
        if not result.final_answer:
            result.error = "max_turns_exhausted"

    result.latency_seconds = round(time.time() - started, 2)
    _save_logs(result, logs_dir)
    return result


def _save_logs(result: AgentResult, logs_dir: Path | None) -> None:
    """Save trajectory and result to disk."""
    if not logs_dir:
        return
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "trajectory.json").write_text(json.dumps(result.trajectory, indent=2, default=str))
    (logs_dir / "result.json").write_text(json.dumps({
        "final_answer": result.final_answer if result.final_answer else "",
        "episodes": result.episodes,
        "total_input_tokens": result.total_input_tokens,
        "total_output_tokens": result.total_output_tokens,
        "skills_loaded": result.skills_loaded,
        "tool_calls_log": result.tool_calls_log,
        "error": result.error,
        "latency_seconds": result.latency_seconds,
    }, indent=2, default=str))
