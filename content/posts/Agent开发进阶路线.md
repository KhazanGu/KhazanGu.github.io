+++
date = 2026-09-14T23:31:00+08:00
draft = false
title = "Agent 开发进阶路线"
tags = ["Agent", "LLM", "工程实践"]
categories = ["笔记"]
ShowToc = true
slug = "agent-dev-roadmap"
summary = "零 Agent 经验、已有基础编程能力。独立交付一个范围受控、可评估、可观测、可回滚的 Agent，并具备启动内部试点的能力。"
+++


> 目标不是在几个月内“精通所有框架”，而是独立交付一个**范围受控、可评估、可观测、可回滚**的 Agent，并具备启动内部试点的能力。
> 真正完成生产试点，还需要持续落实阶段 6 的工程与运营门槛；高风险行业、多租户平台和严格 SLA 场景则需要更长期的经验。

## 前置认知：先看清本质

Agent 的核心循环可以简化为：

```text
模型提出工具调用
→ 宿主程序校验参数、权限和风险
→ 宿主执行工具
→ 将结构化结果回灌给模型
→ 模型继续推理
→ 满足终止条件后输出结果
```

模型只负责**提出意图**，真正的执行权始终在宿主程序。框架、多 Agent、记忆、RAG 和 MCP，都是围绕这个循环增加状态管理、工具接入、流程控制或可观测性。

新手最容易走的弯路，是一上来就学习框架，却没有亲手实现过循环。这样很容易停留在“会改示例、不会定位问题”的层次。框架和 API 会变化，但以下四条主线相对稳定：

1. **控制循环**：什么时候调用工具、重试或停止。
2. **上下文与状态**：每次模型调用应该看到什么。
3. **评估与可观测性**：怎样证明一次改动真的更好。
4. **权限与安全**：模型可以建议什么，系统允许执行什么。

评估和安全不是最后才补的功能。从第一个练习开始，就要有最小测试集、模型轮次与工具调用预算，以及明确的权限边界；后续阶段再逐步增强。

---

## 阶段 0：地基

### 要掌握

- Python 3.11+：异常处理、类型注解、`dataclass`；理解 `async/await`，但第一个循环可以先用同步代码
- 运行时数据与模型校验（如 Pydantic）；若需要直接校验任意 JSON Schema，再了解 `jsonschema` 等实现
- HTTP、JSON 与 JSON Schema
- 环境变量、密钥管理、超时和基础重试
- 一次**不经过 Agent 框架**的 LLM API 调用
- Tool calling / function calling 的请求和响应结构

阶段 1 会先手写少量运行时校验，看清到底需要拦什么，再用 Pydantic 或等价方案替换。重点搞清楚四件事：

1. 工具描述怎样影响模型选择；
2. 参数 Schema 怎样约束输入；
3. 模型返回的工具调用结构是什么；
4. 为什么工具参数即使来自模型，也必须由宿主再次校验。

### 阶段交付物

- 一个最小 API 调用脚本
- 一个精确锁定依赖版本的项目文件
- 一份说明密钥配置、运行命令和错误处理方式的 README

### 通过标准

不要求背下某家 SDK 的方法名。你应该能够不依赖框架完成一次工具调用，并准确解释：请求里有什么、模型返回了什么、宿主执行了什么、错误如何回到下一轮。同时应能从全新虚拟环境按锁版文件复现安装，并只依赖 README 配置密钥、运行脚本和触发一条预期错误。

---

## 阶段 1：手写一个安全边界明确的 Agent

下面的代码刻意使用兼容 Chat Completions 的消息结构，让循环保持可见。该接口仍受支持，但并非所有当前模型都兼容；`OPENAI_MODEL` 必须选择支持 Chat Completions 与工具调用的模型。新项目还应对照服务商文档评估 Responses API。实际运行前请确认接口能力，并在项目中锁定验证过的 SDK 版本。核心学习目标是循环本身，而不是记忆 API 名称。

### 最小实现

```python
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

# SDK 默认从 OPENAI_API_KEY 读取密钥；模型必须兼容 Chat Completions 工具调用。
MODEL = os.environ.get("OPENAI_MODEL")
if not MODEL:
    raise RuntimeError("请先设置 OPENAI_MODEL")

WORKSPACE = Path(os.environ.get("AGENT_WORKSPACE", ".")).resolve()
if not WORKSPACE.is_dir():
    raise RuntimeError("AGENT_WORKSPACE 必须指向已存在的目录")
if not os.access(WORKSPACE, os.R_OK | os.W_OK | os.X_OK):
    raise RuntimeError("AGENT_WORKSPACE 必须可读、可写且可进入")

MAX_TOOL_OUTPUT_CHARS = 8_000
MAX_DIR_ENTRIES = 20
MAX_PATH_CHARS = 1_024
MAX_REPORT_CHARS = 20_000
MAX_TOOL_ARGUMENT_CHARS = MAX_REPORT_CHARS * 6 + 2_000
MAX_CALL_ID_CHARS = 256
MAX_TOOL_NAME_CHARS = 128
MAX_MODEL_TURNS_LIMIT = 50
MAX_TOOL_CALLS_LIMIT = 500

if MAX_TOOL_OUTPUT_CHARS < 256:
    raise RuntimeError("MAX_TOOL_OUTPUT_CHARS 过小，无法容纳结构化错误")

client = OpenAI(timeout=30.0, max_retries=2)

SYSTEM_PROMPT = """你是一个能使用工具完成文件任务的助手。
只能使用提供的工具，不得假设工具已经成功执行。
所有路径必须相对于工作区；工具失败后先分析原因，再决定是否更换参数重试。
list_dir 分页返回：next_offset 非 null 说明还有条目，必须用该值继续翻页，
直到 next_offset 为 null，才可以对整个目录下结论。
任何结果带 truncated=true 时，不得把不完整结果当作完整结果。
不要用完全相同的参数重复失败调用。完成任务后直接给出结果摘要。
"""


def resolve_workspace_path(path: str) -> Path:
    """拒绝绝对路径和解析后越界路径；仅适用于不可被并发篡改的练习目录。"""
    if not isinstance(path, str) or not path.strip():
        raise ValueError("路径必须是非空字符串")
    if len(path) > MAX_PATH_CHARS:
        raise ValueError("路径过长")

    relative_path = Path(path)
    if relative_path.is_absolute():
        raise ValueError("只允许工作区相对路径")

    target = (WORKSPACE / relative_path).resolve()
    if target != WORKSPACE and WORKSPACE not in target.parents:
        raise ValueError("路径超出工作区")
    return target


def has_symlink_component(path: str) -> bool:
    """拒绝路径中现存的符号链接组件；仍不能消除检查与使用之间的竞态。"""
    current = WORKSPACE
    for part in Path(path).parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def entry_type(path: Path) -> str:
    if path.is_symlink():
        return "symlink"
    if path.is_dir():
        return "directory"
    if path.is_file():
        return "file"
    return "other"


def directory_page(
    entries: list[dict[str, str]], offset: int, total: int
) -> dict[str, Any]:
    end = offset + len(entries)
    has_more = end < total
    return {
        "ok": True,
        "entries": entries,
        "offset": offset,
        "returned": len(entries),
        "total": total,
        "next_offset": end if has_more else None,
        "truncated": has_more,
    }


# --- 1. 工具实现 ---
def list_dir(path: str, offset: int) -> dict[str, Any]:
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        return {"ok": False, "error": "offset 必须是非负整数"}

    try:
        target = resolve_workspace_path(path)
        if has_symlink_component(path):
            return {"ok": False, "error": "不允许列出包含符号链接的路径"}
        if not target.is_dir():
            return {"ok": False, "error": "目标不是目录"}

        # 先全量排序再切窗口：分页窗口必须由名称决定，不能由文件系统枚举顺序决定。
        # 练习期间还应冻结目录；若页间发生增删改，仅靠 offset 仍无法得到一致快照。
        names = sorted(child.name for child in target.iterdir())
        if offset > len(names):
            return {"ok": False, "error": "offset 超出目录范围；请使用返回的 next_offset"}

        entries: list[dict[str, str]] = []
        for name in names[offset : offset + MAX_DIR_ENTRIES]:
            candidate_entries = entries + [
                {"name": name, "type": entry_type(target / name)}
            ]
            candidate = directory_page(candidate_entries, offset, len(names))
            encoded = json.dumps(candidate, ensure_ascii=True, allow_nan=False)
            if len(encoded) > MAX_TOOL_OUTPUT_CHARS:
                break
            entries = candidate_entries

        if offset < len(names) and not entries:
            return {
                "ok": False,
                "error": "单个目录项无法在工具输出预算内编码",
            }
        return directory_page(entries, offset, len(names))
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    except OSError:
        return {"ok": False, "error": "目录访问失败"}


def stat_file(path: str) -> dict[str, Any]:
    try:
        target = resolve_workspace_path(path)
        if has_symlink_component(path):
            return {"ok": False, "error": "不允许统计包含符号链接的路径"}
        if not target.is_file():
            return {"ok": False, "error": "目标不是普通文件"}
        return {"ok": True, "path": path, "size_bytes": target.stat().st_size}
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    except OSError:
        return {"ok": False, "error": "文件访问失败"}


def write_report(content: str) -> dict[str, Any]:
    if not isinstance(content, str) or not content.strip():
        return {"ok": False, "error": "报告内容必须是非空字符串"}
    if len(content) > MAX_REPORT_CHARS:
        return {"ok": False, "error": f"写入内容超过 {MAX_REPORT_CHARS} 字符"}

    target = WORKSPACE / "report.md"
    try:
        with target.open("x", encoding="utf-8") as report:
            report.write(content)
        return {"ok": True, "path": "report.md", "chars_written": len(content)}
    except FileExistsError:
        return {"ok": False, "error": "report.md 已存在；本练习禁止覆盖"}
    except OSError:
        return {"ok": False, "error": "报告写入失败"}


Tool = Callable[..., dict[str, Any]]
TOOL_REGISTRY: dict[str, Tool] = {
    "list_dir": list_dir,
    "stat_file": stat_file,
    "write_report": write_report,
}


def validate_path_argument(path: Any) -> str | None:
    if not isinstance(path, str) or not path.strip() or len(path) > MAX_PATH_CHARS:
        return f"path 必须是长度不超过 {MAX_PATH_CHARS} 的非空字符串"
    return None


def validate_tool_arguments(tool_name: str, arguments: dict[str, Any]) -> str | None:
    """在宿主侧执行最小运行时校验，而不是只依赖模型侧 Schema。"""
    if tool_name == "list_dir":
        if set(arguments) != {"path", "offset"}:
            return "参数必须且只能包含 path 和 offset"
        if path_error := validate_path_argument(arguments["path"]):
            return path_error
        offset = arguments["offset"]
        # JSON 的 true 在 Python 里是 int 子类，必须显式排除 bool。
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            return "offset 必须是非负整数"
        return None

    if tool_name == "stat_file":
        if set(arguments) != {"path"}:
            return "参数必须且只能包含 path"
        return validate_path_argument(arguments["path"])

    if tool_name == "write_report":
        if set(arguments) != {"content"}:
            return "参数必须且只能包含 content"
        content = arguments["content"]
        if not isinstance(content, str) or not content.strip():
            return "content 必须是非空字符串"
        if len(content) > MAX_REPORT_CHARS:
            return f"content 超过 {MAX_REPORT_CHARS} 字符"
        return None

    return "未知工具"


# --- 2. 工具 Schema：模型看到的是描述，宿主依然负责最终校验 ---
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": (
                "按名称排序分页列出工作区相对目录中的条目，每页最多 20 项；"
                "若编码预算不足可能更少。next_offset 非 null 时需继续调用。"
            ),
            # strict 模式要求所有属性都出现在 required 中，因此 offset 必须显式传入。
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "不超过 1024 字符的相对目录，如 . 或 data",
                    },
                    "offset": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "首页传 0，后续传上一次返回的 next_offset",
                    },
                },
                "required": ["path", "offset"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stat_file",
            "description": "读取工作区内普通文件的字节大小，不读取文件内容。",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "不超过 1024 字符的文件相对路径",
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_report",
            "description": (
                "在工作区根目录新建 report.md；目标已存在时拒绝覆盖；"
                "内容必须非空且不超过 20000 字符。"
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "完整 Markdown 内容"},
                },
                "required": ["content"],
                "additionalProperties": False,
            },
        },
    },
]


def encode_tool_result(result: dict[str, Any]) -> str:
    """保持工具结果为合法 JSON，并保证编码后长度不超过固定预算。"""
    try:
        payload = json.dumps(result, ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError):
        payload = json.dumps(
            {"ok": False, "error": "工具结果无法编码为标准 JSON"},
            ensure_ascii=True,
            allow_nan=False,
        )

    if len(payload) <= MAX_TOOL_OUTPUT_CHARS:
        return payload

    # 通用编码器不能用 preview 破坏分页控制字段。可分页工具必须在工具内部缩页；
    # 其他超大结果返回显式错误，由调用方改用分页或摘要接口。
    return json.dumps(
        {
            "ok": False,
            "truncated": True,
            "error": "工具结果超过输出预算；请改用分页或摘要接口",
        },
        ensure_ascii=True,
        allow_nan=False,
    )


def reject_non_json_constant(value: str) -> Any:
    raise ValueError(f"非法 JSON 常量：{value}")


# --- 3. Agent 循环 ---
def run_agent(
    user_input: str,
    max_model_turns: int = 15,
    max_tool_calls: int = 40,
) -> str:
    if not isinstance(user_input, str) or not user_input.strip():
        return "任务必须是非空字符串。"
    if (
        isinstance(max_model_turns, bool)
        or not isinstance(max_model_turns, int)
        or not 1 <= max_model_turns <= MAX_MODEL_TURNS_LIMIT
    ):
        return f"max_model_turns 必须是 1～{MAX_MODEL_TURNS_LIMIT} 的整数。"
    if (
        isinstance(max_tool_calls, bool)
        or not isinstance(max_tool_calls, int)
        or not 1 <= max_tool_calls <= MAX_TOOL_CALLS_LIMIT
    ):
        return f"max_tool_calls 必须是 1～{MAX_TOOL_CALLS_LIMIT} 的整数。"

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]
    tool_calls_used = 0

    for model_turn in range(max_model_turns):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOL_SCHEMAS,
            )
        except Exception as exc:
            return f"模型调用失败（{type(exc).__name__}）。"

        if not response.choices:
            return "模型没有返回候选结果。"

        choice = response.choices[0]
        if choice.finish_reason not in {"stop", "tool_calls"}:
            return f"模型响应未正常完成：{choice.finish_reason}"

        message = choice.message
        content = getattr(message, "content", None)
        if content is not None and not isinstance(content, str):
            return "模型返回了非法的 assistant content。"

        raw_tool_calls = getattr(message, "tool_calls", None)
        if raw_tool_calls is None:
            tool_calls: list[Any] = []
        elif isinstance(raw_tool_calls, (list, tuple)):
            tool_calls = list(raw_tool_calls)
        else:
            return "模型返回了非法的 tool_calls 列表。"

        if (choice.finish_reason == "tool_calls") != bool(tool_calls):
            return "模型 finish_reason 与 tool_calls 不一致，已停止且未执行工具。"

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": content or "",
        }
        prepared_calls: list[tuple[str, str, str]] = []

        if tool_calls:
            seen_call_ids: set[str] = set()
            normalized_calls: list[dict[str, Any]] = []

            for call in tool_calls:
                call_id = getattr(call, "id", None)
                call_type = getattr(call, "type", None)
                function_call = getattr(call, "function", None)
                tool_name = getattr(function_call, "name", None)
                arguments_json = getattr(function_call, "arguments", None)

                if (
                    not isinstance(call_id, str)
                    or not call_id.strip()
                    or len(call_id) > MAX_CALL_ID_CHARS
                    or call_id in seen_call_ids
                ):
                    return "工具调用 ID 必须非空、唯一且长度受限；本批未执行。"
                if call_type != "function":
                    return "工具调用 type 必须为 function；本批未执行。"
                if (
                    not isinstance(tool_name, str)
                    or not tool_name.strip()
                    or len(tool_name) > MAX_TOOL_NAME_CHARS
                ):
                    return "工具名称必须是长度受限的非空字符串；本批未执行。"
                if (
                    not isinstance(arguments_json, str)
                    or len(arguments_json) > MAX_TOOL_ARGUMENT_CHARS
                ):
                    return "工具参数必须是长度受限的 JSON 字符串；本批未执行。"

                seen_call_ids.add(call_id)
                prepared_calls.append((call_id, tool_name, arguments_json))
                normalized_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": arguments_json,
                        },
                    }
                )

            call_names = [tool_name for _, tool_name, _ in prepared_calls]
            if "write_report" in call_names and len(prepared_calls) != 1:
                return "write_report 必须单独成批调用；本批未执行。"
            if tool_calls_used + len(prepared_calls) > max_tool_calls:
                return f"将超过工具调用上限 {max_tool_calls}，已停止且未执行本批工具。"
            # 最后一个模型轮次必须留给最终答案，不能执行结果无人消费的副作用。
            if model_turn == max_model_turns - 1:
                return f"达到模型轮次上限 {max_model_turns}，本批工具未执行。"

            assistant_message["tool_calls"] = normalized_calls

        messages.append(assistant_message)

        if not prepared_calls:
            return content or "任务结束，但模型没有返回文本结果。"

        for call_id, tool_name, arguments_json in prepared_calls:
            tool_calls_used += 1
            function = TOOL_REGISTRY.get(tool_name)

            try:
                arguments = json.loads(
                    arguments_json,
                    parse_constant=reject_non_json_constant,
                )
            except (ValueError, RecursionError):
                result = {"ok": False, "error": "工具参数不是合法 JSON"}
            else:
                if not isinstance(arguments, dict):
                    result = {"ok": False, "error": "工具参数必须是 JSON 对象"}
                elif function is None:
                    result = {"ok": False, "error": f"未知工具：{tool_name}"}
                elif validation_error := validate_tool_arguments(tool_name, arguments):
                    result = {"ok": False, "error": validation_error}
                else:
                    try:
                        result = function(**arguments)
                        if not isinstance(result, dict):
                            result = {"ok": False, "error": "工具返回类型错误"}
                    except Exception as exc:
                        result = {
                            "ok": False,
                            "error": f"工具执行失败（{type(exc).__name__}）",
                        }

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": encode_tool_result(result),
                }
            )

    return f"达到模型轮次上限 {max_model_turns}，任务仍未完成。"


if __name__ == "__main__":
    task = input("请输入任务：").strip()
    if not task:
        raise SystemExit("任务不能为空")
    print(run_agent(task))
```

这仍然只是教学实现，几处取舍需要说清楚。

`max_model_turns` 和 `max_tool_calls` 限制的是模型轮次与工具调用数量，不是硬性的墙钟时间；真实外部工具还必须有独立超时和取消机制。循环会把最后一个模型轮次保留给最终回答：若模型在最后一轮仍要求调用工具，本批不会执行，避免写入已经发生却没有后续模型轮次消费结果。

`Path.resolve()` 与路径中现存符号链接的检查，只能降低静态、本地练习目录中的越界风险，不能替代操作系统沙箱，也无法消除检查与使用之间的竞态。`open("x")` 可以防止覆盖已有报告，但进程崩溃或磁盘故障仍可能留下半成品；生产写入应采用同文件系统临时文件、必要的刷盘和原子发布策略。

`list_dir` 先对目录名做全量排序，再按条目数和**编码后的输出预算**共同切页。这样不会再由通用截断器丢失 `next_offset`、`total` 等分页控制字段。不过，名称排序只能保证目录内容不变时的复现性；若页间发生增删改，offset 分页仍会漂移。因此评估时必须冻结 fixture，生产场景则应使用稳定快照或游标索引。

`encode_tool_result` 使用 `allow_nan=False`，无法编码或超过预算时返回短小的结构化错误，不再把整个 JSON 切成一个 `preview`。可分页工具应像 `list_dir` 一样在工具内部缩页；不可分页的大结果应改成摘要、对象存储引用或专门的分页接口。

模型响应同样是不可信输入。宿主先核对 `finish_reason`、调用列表、调用 ID、`type`、工具名和参数字符串，再执行任何工具；包含 `write_report` 的批次必须只有这一个调用，避免同批前序失败后仍然落盘。`assistant_message` 只把 `None` 规范化为空字符串，模型若同时返回非空文本与工具调用则保留原文。部分兼容层要求 `content` 为 `null`，接入前仍需确认。

Schema 里的 `strict: True` 依赖模型支持严格函数调用。示例把 OpenAI 严格模式当前支持的数值下界写进 Schema；字符串长度与“非空白”等供应商子集未必支持的约束继续由描述和宿主校验双重承担。使用其他模型或微调模型时，应先核对其 JSON Schema 子集，不能为了表面完整加入服务端不支持的关键字。

`validate_tool_arguments` 是刻意手写的，目的是看清一个校验层到底要拦什么：多余字段、缺失字段、类型错误、取值范围，以及 `offset` 这类必须显式排除 `bool` 的陷阱。看懂之后，把它换成 Pydantic 模型是一次很好的练习——行数会少很多，但你已经知道它在替你做什么。

为了让循环一眼可读，示例在导入时初始化模型客户端并固定工作区。测试和服务化时应改为显式注入客户端、工作区与预算配置，避免模块全局状态让用例彼此污染。持久化、人工审批、完整 tracing、token 估算和并发调度留待后续阶段。任意 Shell 更不应该在没有沙箱、超时、命令策略和用户确认的情况下直接加入。

### 练手项目

只给 Agent 三个工具：`list_dir`、`stat_file`、`write_report`。让它完成：

> 统计当前目录下最大的三个普通文件，并将文件名、字节数和排序依据写入 `report.md`。排序规则固定为“字节数降序、相同时文件名升序”。

这个任务会强制 Agent 经历“发现文件 → 查询元数据 → 聚合 → 写入”的多步过程，而不是用一个 Shell 命令绕过工具设计。

准备一个内容冻结的临时目录，放入 21～30 个普通文件并混入目录、符号链接和 FIFO，即可触发至少两页，同时仍能落在默认 40 次工具调用预算内。若逐个统计 `N` 个普通文件，保守预算公式是“列目录页数 + `N` 次 `stat_file` + 1 次 `write_report`”；同一轮可提出多个彼此独立的只读调用，但调用数仍逐个计入。每次评估都重新创建 fixture，运行结束后销毁，不能让上次产生的 `report.md` 或页间文件变化污染下一次。

### 从第一天开始建立最小评估

准备固定 10 个用例，为每个用例预先写清输入、oracle（精确期望输出或期望拒绝行为）、分母和判定规则，并标注它属于功能项还是安全项。每个用例至少在独立、重置后的临时工作区重复 5 次。用例应覆盖：空目录、只有两个文件、同大小文件的排序、文件名含空格、目录项恰好等于单页上限、目录项超过单页上限（需翻页）、路径越界、非法 `offset`、工具返回错误、畸形工具调用信封、最后模型轮次请求写入，以及 `report.md` 已存在等。

分页未翻完时，正确行为是继续用 `next_offset` 取回后续页，或明确说明结论只覆盖了部分条目；不能拿第一页就声称找到了全目录最大文件。报告已存在时，正确行为是拒绝覆盖并说明原因。最后模型轮次才提出写入时，正确行为是停止且不创建文件。

这里要区分两类保障。`next_offset` 和 `truncated` 只是把事实告知模型，是否正确翻页属于**模型行为评估**；而路径越界、参数非法、拒绝覆盖由宿主代码强制，属于**安全边界**。如果业务绝不允许基于不完整清单写报告，就必须在宿主状态机里记录“是否已翻完”并据此阻止写入，不能只依赖 Prompt——这正是阶段 6 中“高风险操作在执行层前置”的雏形。

示例中的 `messages` 只保存在内存，不等于已经具备可观测性。练习时应补一份最小调用日志，至少记录模型轮次、工具名、参数摘要、成功状态和耗时；不要把密钥、完整文件内容、未经脱敏的原始异常或报告正文写入日志。每次修改 Prompt 或 Schema 后重新运行并记录：

- 最终结果是否正确；
- 是否选择了正确工具；
- 是否重复调用相同失败参数；
- 是否在模型轮次与工具调用预算内结束；
- 是否发生越权读取或写入。

### 通过标准

- 每个功能用例在 5 次独立运行中至少通过 4 次，不能用总体平均掩盖某个稳定失败的场景；所有安全项必须 5 次全部正确拒绝且无副作用；
- 超过单页上限的冻结目录能被完整翻页，且 fixture 不变时同一 `offset` 重复调用返回相同条目；
- 同大小文件按既定次级排序稳定输出，报告已存在或最后轮次预算耗尽时不会落盘；
- 能从一次可恢复的工具错误中继续完成任务；
- 达到模型轮次或工具调用上限时能够停止，并说明触发了哪一项预算；
- 能根据执行轨迹解释失败发生在模型决策、响应信封校验、宿主参数校验、工具执行还是结果回灌。

---

## 阶段 2：上下文与状态工程

上下文工程不是“尽量多塞信息”，而是设计每次模型调用看到什么、按什么顺序看到、什么被压缩、什么按需取回、什么不应进入窗口。长上下文能力很有价值，但无关、重复或低信号内容仍可能降低质量并增加成本。

### 要掌握

- **工具输出裁剪与分页**：日志、网页和大文件不能原样回灌
- **历史压缩（compaction）**：保留目标、约束、关键决策、未完成事项和证据引用
- **外部化状态**：长期状态放数据库、文件或检查点，不依赖对话历史充当数据库
- **工作记忆与长期记忆分层**：分别定义写入、检索、更新和遗忘策略
- **结构化中间结果**：让工具和子任务返回可合并的数据，而不是长篇自然语言
- **检索（RAG）**：把它视为按需取回信息的一种手段，而不是 Agent 的全部
- **状态恢复**：进程重启后能够从已确认的检查点继续，而不是重新猜测之前做过什么

### 练手项目

建立 10、50、200 个文件三档、内容分布一致的 UTF-8 Markdown fixture。每个文件有一个一级标题和若干行 `TODO:`；新增一个本地工具，只返回单个文件的 `{path, title, todo_count}`，不要把原文放进模型上下文。Agent 最终输出以下确定性结构，并按 `path` 升序排列 `files`：

```json
{
  "files": [{"path": "notes/a.md", "title": "A", "todo_count": 2}],
  "total_files": 1,
  "total_todos": 2
}
```

每处理固定数量的文件就原子更新检查点，检查点至少记录 fixture 版本、已确认路径、结构化结果和 Schema 版本。处理到 50% 时主动终止进程，再从检查点恢复：已确认文件不得重复计数，恢复结果必须与不中断运行逐字段一致。一次运行期间冻结 fixture；若输入版本变化，应拒绝沿用旧检查点或显式重新开始。

再做两组小实验，避免“要掌握”停留在名词：

1. **记忆生命周期**：写入三条带来源和过期策略的事实，更新一条、删除一条，验证旧值和已删除值不再被检索；涉及用户数据时同时记录同意与删除路径。
2. **RAG**：准备固定问题—证据集，预先定义 `recall@k`、有证据答案正确率和“证据不足时拒答”的阈值；分别评估检索与生成，不能只看最终文风。

这里不要笼统追求“总 token 随文件数亚线性增长”。如果必须准确检查全部文件，读取和本地计算至少是线性的。合理目标是：

- 峰值模型上下文保持有界；
- 原始文件内容不在线性累积到对话历史中；
- 增加文件数量时，模型侧输入主要增长在高信号聚合结果上；
- 对结果正确率、输入/输出 token、耗时和峰值上下文分别记录。

### 通过标准

在 10、50、200 三档数据上用相同口径测量，证明峰值模型上下文没有随原始总字节数按比例或近线性增长；画出一次代表性模型调用的上下文组成，解释每一部分为什么存在、从哪里取回、何时失效。主动中断测试必须逐字段复现不中断结果且无重复计数；记忆的更新/删除和 RAG 的检索、回答、拒答指标都达到实验前写下的阈值。

---

## 阶段 3：再选择框架，而不是追逐框架

2026 年的 Agent 工具链仍在快速变化，不宜断言“行业已经收敛为三个框架”。更稳定的比较方式是看心智模型。下面是代表性方案，不是完整排名：

| 方案 | 主要心智模型 | 更适合的场景 | 注意点 |
|---|---|---|---|
| OpenAI Agents SDK | Agent、handoff、guardrail、session、tracing | 轻量循环、简单委派、OpenAI 生态 | 不要把 handoff 当成所有多 Agent 问题的答案 |
| Claude Agent SDK | 类 Claude Code 的工具循环、上下文与权限控制 | 编码、研究和需要内置文件/命令工具的 Agent | 与 Claude 能力和权限模型结合较深 |
| LangGraph | 显式状态图与持久化执行 | 分支、重试、审批、断点续跑、长流程 | 控制力高，但状态设计和调试成本也高 |
| Microsoft Agent Framework | Agent 与显式 workflow | Python/.NET、企业集成、由 AutoGen 或 Semantic Kernel 迁移 | 关注版本与迁移指南 |
| Google ADK | Agent 层级、工作流与工具生态 | Google Cloud 生态或需要多种工作流 Agent | 区分 SDK 能力与托管平台能力 |
| CrewAI | 角色、任务、Crew / Flow | 角色清晰、声明式协作流程 | 角色叙事不能替代状态、权限和评估设计 |

### 建议路径

1. 先选一个与你的模型供应商和部署环境匹配的轻量 SDK；
2. 用它重写阶段 2 的项目；
3. 比较时固定模型快照、采样参数、任务 Prompt、工具契约、测试数据、SDK/API 版本、缓存状态、重试与并发策略，并提前定义“模型轮次”和“工具调用数”的统计口径；
4. 无法保持一致的框架默认行为必须单独记录，不能把差异全部归因于框架优劣；
5. 只有真正需要显式分支、持久化、人工审批或复杂补偿逻辑时，再引入图或 workflow。

不要为了简历同时浅学多个框架。

### 时效性提示

- OpenAI 已明确用 Agents SDK 取代实验性质的 Swarm，新项目不应再以 Swarm 为主线。
- Microsoft Agent Framework 是 AutoGen 和 Semantic Kernel 用户的重要迁移方向。与其笼统写“AutoGen 已进入维护模式”，更稳妥的做法是查看微软当前支持策略和迁移文档。

### 通过标准

用框架重写阶段 2 的项目，在上述控制变量和统一统计口径下对比成功率、模型轮次、工具调用数、token、延迟和可恢复性；对无法控制的差异单独披露。你还应能说明框架替你管理了什么、引入了什么约束、哪些状态仍由业务代码负责。

---

## 阶段 4：MCP——标准化工具与上下文接入

MCP（Model Context Protocol）是连接 AI 应用、工具和数据源的开放协议，已经进入多个模型、IDE、SDK 和云平台生态，但不同产品支持的规范版本、能力子集和生产成熟度并不相同。相比争论它是否是“事实标准”，更重要的是理解它解决了什么边界问题，并在集成前核对双方能力。

### 先理解三个角色

- **Host**：承载用户体验和 Agent 的应用，例如 Kiro；
- **Client**：Host 内负责连接某个 MCP Server 的协议客户端；
- **Server**：暴露工具、资源或提示模板的独立服务。

同时区分：

- **Tools**：可被模型请求调用的动作；
- **Resources**：可读取的上下文数据；
- **Prompts**：可复用的交互模板。

### 学法

自己写一个只暴露 2～3 个低风险工具的 MCP Server，并接入 Kiro 或另一个第三方客户端。至少实践：

- **`server/discover`**：规范要求服务端必须实现这个 RPC，用来声明支持的协议版本、能力和身份。客户端可以在任何其他请求之前调用它来选定版本；
- **逐请求能力协商**：不再有 `initialize` 握手，协议版本与客户端能力随每个请求通过 `_meta` 传递，版本不匹配返回 `UnsupportedProtocolVersionError`；
- **工具定义与参数 Schema**：`tools/list` 的返回还需带 `ttlMs` 和 `cacheScope` 缓存字段，并建议以确定性顺序返回，便于客户端缓存和提示缓存命中；
- **本地 stdio**：子进程生命周期、环境变量凭据和操作系统权限边界；
- **远程 Streamable HTTP**：每个 POST 必须带 `MCP-Protocol-Version`；`Mcp-Method` 用于所有请求，`Mcp-Name` 仅用于 `tools/call`、`resources/read`、`prompts/get`。这些头部必须与消息体对应字段一致；协议级会话和 `Mcp-Session-Id` 已移除，跨调用状态改用服务端签发、通过普通参数传递的 handle；
- **多轮往返（MRTR）**：需要补充信息时，服务端返回 `resultType: "input_required"` 与 `inputRequests`，客户端带 `inputResponses` 并使用新的 JSON-RPC 请求 ID 重试原方法；
- **超时、取消和结构化错误**；
- **按请求日志级别**：`logging/setLevel` 已移除，改为在 `_meta` 的 `io.modelcontextprotocol/logLevel` 中传递；
- **调用轨迹**：规范给出 `traceparent` / `tracestate` / `baggage` 在 `_meta` 中的 OpenTelemetry 传播约定，可以把 MCP 调用并入阶段 5 的 trace；
- **不可信数据边界**：Server 返回内容要继续当作不可信数据；客户端也必须把工具 annotations 视为不可信，除非来自受信服务端。

上面这些是 2026-07-28 规范的要求，与更早版本差异较大。动手前请以你选定的目标版本规范为准，并确认所用 SDK 已支持该版本。

### MCP 与 A2A

可以用一个不严格但实用的比喻理解：

- MCP 通常解决 Host / Agent 如何连接工具与上下文；
- A2A（Agent2Agent）通常解决彼此独立、可能跨供应商的 Agent 如何发现能力、交换任务和协作。

它们可以互补，但这不是绝对的“纵向/横向”协议边界。只有在确实需要跨系统 Agent 协作时，才需要深入 A2A。

### 版本提醒

MCP 在 **2026-07-28** 规范中进行了一次重大修订，核心是无状态化：移除协议级会话与 `Mcp-Session-Id`、移除 `initialize` / `notifications/initialized` 握手、改为逐请求携带协议版本与能力、新增必须实现的 `server/discover`，并用 `subscriptions/listen` 取代原来的 HTTP GET 端点与 `resources/subscribe`。SSE 流的可恢复性（`Last-Event-ID`）也被移除，断流后客户端必须以新的请求 ID 重发。

该版本还把 Roots、Sampling 和 Logging 标为弃用：兼容期内仍可用，但新实现不应新增依赖；目录或文件可通过工具参数、资源 URI 或服务配置传递，模型调用可直接集成供应商 API，日志优先使用 stderr（stdio）或 OpenTelemetry。

这意味着依赖会话粘连、初始化握手或流恢复的旧教程，部署方法可能整体失效。学习时应直接核对目标版本的规范、SDK 和迁移说明，不要只看发布日期较早的教程。

### 通过标准

你的 MCP Server 能被至少一个第三方 Host 发现和调用，`server/discover` 能正确返回支持的协议版本与能力，对不支持的版本返回明确错误，列表结果包含并验证 `ttlMs` / `cacheScope`。选择 stdio 时，应验证子进程退出、环境变量凭据、非法参数、取消和超时；选择 Streamable HTTP 时，还必须覆盖缺失或不匹配的标准请求头、未认证、未授权和网络错误。

两种路径都至少有一条自动化集成用例验证完整调用链，并完成一次 `input_required` → `inputResponses` 重试；trace 测试应确认 `_meta` 上下文贯通，安全测试应确认未经信任的 annotations 不会直接提升权限。若所用 SDK 尚未实现目标版本中的某项能力，应把该项标成已知缺口，而不是伪造通过结果。

---

## 阶段 5：系统化评估与可观测性

阶段 1 已经建立最小回归集；这一阶段把它升级为可用于发布决策的评估系统。Agent 的失败可能发生在任何一层，所以不能只检查最终文字是否“看起来不错”。

### 三个评估层级

- **Run**：单次运行的最终结果是否满足任务和安全约束；
- **Trace**：工具选择、参数、顺序、重试和中间状态是否合理；
- **Thread**：跨轮目标、记忆、权限和状态是否保持一致。

### 建议指标

| 维度 | 示例指标 |
|---|---|
| 任务质量 | 成功率、字段正确率、人工验收通过率 |
| 路径质量 | 正确工具率、冗余调用数、不可恢复错误率 |
| 安全 | 越权执行数、危险操作拦截率、注入攻击成功率 |
| 可靠性 | 重试后恢复率、超时率、断点恢复成功率 |
| 效率 | 输入/输出 token、工具耗时、端到端延迟、单次成本 |

### 动手做

1. 扩充到 20～30 条代表性用例，覆盖正常流程、边界、工具失败、恶意输入和状态恢复；
2. 对存在随机性的关键用例重复运行 3～5 次，记录波动，而不是只跑一次；
3. 固定并记录模型版本、Prompt、工具 Schema、测试数据和代码提交；
4. 接入 LangSmith、Langfuse、Phoenix 或基于 OpenTelemetry 的 tracing；如果系统里有 MCP Server，按规范的 `_meta` 传播约定把它的调用并入同一条 trace，避免出现观测盲区；
5. 将确定性断言、模型评分和人工复核分开，不让模型评分替代所有事实检查；
6. 在运行评估前写下数值或布尔发布门槛，每次 Prompt、模型或工具变更都运行回归并输出逐项判定。

### 通过标准

能提供一份版本化报告，至少展示一次受控改动前后的指标、波动范围、预先定义的发布阈值、每项通过/失败和最终发布判断；累计三次以上改动后，再比较趋势。出现失败时能从 trace 定位到具体层级，不能只给“总体看起来更好”的结论。

---

## 阶段 6：生产化

### 成本与延迟

- 简单步骤使用更小模型，复杂决策再升级模型；
- 对稳定输入和确定性工具结果做安全缓存；
- 只并行执行彼此独立且没有写冲突的工具；
- 设置 token、步骤、时间和费用预算，并支持取消。

### 可靠性与状态

- 对网络故障做有上限的退避重试；
- 写操作使用幂等键、事务或补偿逻辑；
- 将关键状态持久化到可恢复的检查点；
- 为模型不可用、工具不可用和结果不确定设计降级路径。

### Human-in-the-loop

- 高风险操作在执行层前置审批，而不是只依赖 Prompt 要求模型“先询问”；
- 审批内容要包含实际参数、影响范围和过期时间；
- 审批状态必须可审计，并能跨进程重启恢复。

### 安全

- 最小权限、工作区隔离、网络出口控制和密钥分域；
- 文件、网页、邮件、检索结果和工具返回一律视为不可信数据；
- 外部内容不能直接提升权限或改变系统策略；
- Prompt injection 防护必须落在权限校验、数据隔离、工具策略和审批机制上，不能只依赖另一段 Prompt；
- 保存可追溯、完整性受保护的操作审计，同时避免将密钥和敏感正文写入日志；如果业务要求“不可抵赖”，还需额外设计身份绑定、防篡改、可信时间与签名机制。

### 发布与运营

- 定义成功率、延迟、成本和安全事件等 SLO；
- 使用灰度发布、功能开关、速率限制、告警和一键停用；
- 建立失败样本回流、人工接管、回滚和事故复盘机制；
- 只有单 Agent 的工具与状态边界已经清楚，并且数据证明存在协作收益时，再引入多 Agent 或 A2A。

### 通过标准

在预生产环境完成故障注入、权限测试、注入攻击测试和断点恢复；至少演练一次幂等写重试、模型或工具降级、预算取消、功能开关停用和版本回滚，并保存结果与负责人。上线试点还必须具备量化 SLO、审批矩阵、监控告警、回滚步骤和明确的 go/no-go 判定；仅有方案文档、不做演练不算通过。

---

## 五个最常见的误区

1. **过早追求多 Agent**：单 Agent 配合清晰工具通常能覆盖大量任务，多 Agent 会显著增加状态和调试复杂度。
2. **把框架 API 当成知识本身**：稳定能力来自循环、状态、权限和评估设计。
3. **把对话历史当数据库**：历史会被截断、压缩或污染，关键状态必须外部化。
4. **等功能做完才写 eval**：没有基线，就无法判断重构、换模型或改 Prompt 是否真的更好。
5. **把安全留到上线前**：如果第一阶段就给了任意 Shell，后补 Prompt 无法建立可靠权限边界。

---

## 时间线参考

按每周投入约 10 小时计算：

| 阶段 | 周期 | 累计参考 |
|---|---:|---:|
| 0 地基 | 1～2 周 | 1～2 周 |
| 1 安全的手写 Agent | 1～2 周 | 2～4 周 |
| 2 上下文与状态工程 | 4 周 | 6～8 周 |
| 3 框架 | 2～4 周 | 8～12 周 |
| 4 MCP | 1～2 周 | 9～14 周 |
| 5 系统化评估与可观测性 | 2～3 周 | 11～17 周 |
| 6 生产化 | 持续 | — |

约 3～4 个月可以达到“独立构建受控范围 Agent，并具备启动内部试点的能力”。真正完成生产试点还需要继续落实阶段 6 的 SLO、权限、灰度、监控和回滚门槛；其时间取决于业务风险、合规要求、流量规模和持续运营经验，不能只由学习时长保证。

---

## 进度清单

- [ ] 阶段 0：完成一次无框架工具调用，并解释完整请求—执行—回灌链路
- [ ] 阶段 0：在全新环境按锁版文件安装，并仅依赖 README 完成配置、运行和错误排查
- [ ] 阶段 0：密钥不进入代码仓库
- [ ] 阶段 1：从空项目实现有模型轮次和总工具调用预算的多轮 Agent
- [ ] 阶段 1：路径越界测试 100% 被拒绝，工具错误可恢复
- [ ] 阶段 1：冻结目录能完整翻页，分页控制字段不被输出截断破坏
- [ ] 阶段 1：建立逐用例 oracle、独立 fixture 和最小工具调用日志
- [ ] 阶段 1：把手写参数校验替换为 Pydantic 或等价模型并保持测试通过
- [ ] 阶段 2：实现工具输出裁剪、分页、历史压缩和版本化检查点
- [ ] 阶段 2：在多档规模上证明峰值上下文有界，关键状态可跨重启恢复
- [ ] 阶段 2：验证记忆更新/删除与 RAG 检索—生成分层指标
- [ ] 阶段 3：用一个框架重写项目，并在固定控制变量和统一统计口径下完成前后对比
- [ ] 阶段 4：MCP Server 实现 `server/discover`、逐请求版本协商和列表缓存字段
- [ ] 阶段 4：按所选传输覆盖凭据、授权、请求头、MRTR、取消、超时和错误用例
- [ ] 阶段 4：贯通 trace，并把未经信任的工具 annotations 保持在权限边界之外
- [ ] 阶段 5：20+ 条测试用例、重复运行统计和 tracing 接入
- [ ] 阶段 5：每次变更报告都写明门槛、逐项判定和最终发布结论
- [ ] 阶段 6：完成权限矩阵、人工审批、Prompt injection 测试和受保护的审计日志
- [ ] 阶段 6：定义量化 SLO，并演练幂等、降级、预算取消、停用和回滚

---

## 参考来源

以下优先列出官方规范、项目文档和基金会公告；框架与协议状态变化较快，使用时应再次核对发布日期和版本。

### Agent API 与框架

- [OpenAI Function Calling 指南](https://developers.openai.com/api/docs/guides/function-calling/)
- [OpenAI Agents SDK 指南](https://developers.openai.com/api/docs/guides/agents)
- [OpenAI Swarm README：由 Agents SDK 取代](https://github.com/openai/swarm/blob/main/README.md)
- [Claude Agent SDK 概览](https://code.claude.com/docs/en/agent-sdk/overview)
- [LangGraph 官方文档](https://docs.langchain.com/oss/python/langgraph/overview)
- [CrewAI 官方文档](https://docs.crewai.com/)
- [Google Agent Development Kit 文档](https://adk.dev/)
- [AutoGen 到 Microsoft Agent Framework 迁移指南](https://learn.microsoft.com/en-us/agent-framework/migration-guide/from-autogen/)

### 协议与治理

- [MCP 当前规范入口](https://modelcontextprotocol.io/specification/)
- [MCP 2026-07-28 版本变更记录](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
- [MCP 2026-07-28 授权规范](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization)
- [MCP 2026-07-28 Streamable HTTP 规范](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [Linux Foundation：Agentic AI Foundation 与 MCP](https://www.linuxfoundation.org/press/linux-foundation-announces-the-formation-of-the-agentic-ai-foundation)
- [Linux Foundation：Agent2Agent 项目](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents)
- [MCP 2026-07-28 版本协商与 `server/discover`](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning)
- [MCP 2026-07-28 工具规范](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)

### 上下文、评估与安全

- [Microsoft：AI Agents 上下文工程教程](https://github.com/microsoft/ai-agents-for-beginners/blob/main/12-context-engineering/README.md)
- [LangChain：Agent Evals](https://www.langchain.com/resources/agent-evals)
- [OWASP：Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)

本文只对外部资料做摘要、比较和路线编排，不复制长段原文；相关内容均经转述整理。
