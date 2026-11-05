import ast
import ctypes
import html.parser
import math
import operator
import os
import platform
from datetime import datetime
from urllib.error import URLError
from urllib.parse import parse_qs, quote_plus, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from mcp.server.fastmcp import FastMCP

from app.core.config import settings
from app.db.session import SessionLocal
from app.files import (
    create_excel_workbook as create_excel_file,
    create_managed_text_file,
    create_word_document as create_word_file,
    list_managed_files,
    read_managed_text_file,
    read_excel_workbook as read_excel_file,
    read_word_document as read_word_file,
)
from app.models import TaskNote


mcp = FastMCP("AgentOS")

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MAX_EXPRESSION_LENGTH = 256
_MAX_NUMBER_MAGNITUDE = 1e100


def _evaluate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        value = node.value
    elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponents must be between -100 and 100.")
        value = _BINARY_OPERATORS[type(node.op)](left, right)
    elif isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        value = _UNARY_OPERATORS[type(node.op)](_evaluate(node.operand))
    else:
        raise ValueError("Use only numbers, parentheses, and basic arithmetic operators.")

    if not math.isfinite(value) or abs(value) > _MAX_NUMBER_MAGNITUDE:
        raise ValueError("The calculation result is outside the supported range.")
    return value


@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression without executing code."""
    if not expression.strip() or len(expression) > _MAX_EXPRESSION_LENGTH:
        raise ValueError(f"Expression must contain 1 to {_MAX_EXPRESSION_LENGTH} characters.")
    try:
        parsed = ast.parse(expression, mode="eval")
        result = _evaluate(parsed.body)
    except (SyntaxError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError("Could not calculate that expression.") from exc
    return str(result)


@mcp.tool()
def get_current_datetime() -> str:
    """Get the current local date and time, including the UTC offset."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


@mcp.tool()
def create_task_note(title: str, description: str) -> str:
    """Save a task note in the AgentOS database."""
    title = title.strip()
    description = description.strip()
    if not title or len(title) > 200:
        raise ValueError("Title must contain 1 to 200 characters.")
    if len(description) > 10_000:
        raise ValueError("Description must contain no more than 10000 characters.")
    with SessionLocal() as db:
        note = TaskNote(title=title, description=description)
        db.add(note)
        db.commit()
        db.refresh(note)
        return f"Saved note #{note.id}: {note.title}"


@mcp.tool()
def list_agent_files() -> list[dict[str, str]]:
    """List supported files in the dedicated AgentOS files folder."""
    return list_managed_files(settings.agent_files_dir)


@mcp.tool()
def read_agent_file(filename: str) -> str:
    """Read a UTF-8 text file from the dedicated AgentOS files folder."""
    return read_managed_text_file(settings.agent_files_dir, filename)


@mcp.tool()
def write_agent_file(filename: str, content: str) -> str:
    """Create a new text or Python file in the AgentOS files folder without overwriting files."""
    return create_managed_text_file(settings.agent_files_dir, filename, content)


@mcp.tool()
def check_python_syntax(source: str) -> str:
    """Check Python source syntax without running the code."""
    if not isinstance(source, str) or len(source.encode("utf-8")) > 100_000:
        raise ValueError("Python source must be no larger than 100 KB.")
    try:
        ast.parse(source)
    except SyntaxError as exc:
        return f"Syntax error on line {exc.lineno}: {exc.msg}"
    return "Python syntax is valid."


@mcp.tool()
def get_system_info() -> dict[str, str | int | float]:
    """Get basic CPU, memory, and Windows version details without exposing user files."""
    info: dict[str, str | int] = {
        "operating_system": platform.system(),
        "windows_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "logical_cpu_count": os.cpu_count() or 0,
    }
    if os.name == "nt":
        class _MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        memory = _MemoryStatus()
        memory.dwLength = ctypes.sizeof(_MemoryStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            raise OSError("Windows could not report physical memory information.")
        info["total_memory_gb"] = round(memory.ullTotalPhys / (1024 ** 3), 2)
        info["available_memory_gb"] = round(memory.ullAvailPhys / (1024 ** 3), 2)
    return info


class _SearchResultsParser(html.parser.HTMLParser):
    _VOID_TAGS = {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    }

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._current: dict[str, str] | None = None
        self._capture: str | None = None
        self._capture_tags: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = (attributes.get("class") or "").split()
        if "result__a" in classes:
            self._current = {"url": self._result_url(attributes.get("href") or ""), "title": ""}
            self._capture, self._capture_tags = "title", [tag]
        elif self._current is not None and "result__snippet" in classes:
            self._capture, self._capture_tags = "snippet", [tag]
        elif self._capture is not None:
            if tag not in self._VOID_TAGS:
                self._capture_tags.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if self._capture is None or tag in self._VOID_TAGS or tag not in self._capture_tags:
            return
        while self._capture_tags:
            closed_tag = self._capture_tags.pop()
            if closed_tag == tag:
                break
        if not self._capture_tags:
            if self._current is not None and self._capture is not None:
                self._current[self._capture] = self._current.get(self._capture, "").strip()
                if self._capture == "title":
                    self.results.append(self._current)
            self._capture = None

    def handle_data(self, data: str) -> None:
        if self._current is not None and self._capture is not None:
            self._current[self._capture] = (
                self._current.get(self._capture, "") + data
            )

    @staticmethod
    def _result_url(href: str) -> str:
        parsed = urlsplit(href)
        if parsed.path.endswith("/l/"):
            target = parse_qs(parsed.query).get("uddg", [""])[0]
            if target:
                return target
        if href.startswith("//"):
            return urlunsplit(("https", parsed.netloc, parsed.path, parsed.query, ""))
        return href


@mcp.tool()
def web_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Search the public web with DuckDuckGo and return titles, links, and snippets."""
    query = query.strip()
    if not query or len(query) > 500:
        raise ValueError("Search query must contain 1 to 500 characters.")
    if not 1 <= max_results <= 5:
        raise ValueError("max_results must be between 1 and 5.")
    request = Request(
        f"https://html.duckduckgo.com/html/?q={quote_plus(query)}",
        headers={"User-Agent": "AgentOS/1.0 (local assistant web search)"},
    )
    try:
        with urlopen(request, timeout=10) as response:
            if response.status != 200:
                raise RuntimeError("Web search provider returned an unavailable response.")
            document = response.read(1_000_001)
    except (TimeoutError, URLError) as exc:
        raise RuntimeError("Web search is temporarily unavailable.") from exc
    if len(document) > 1_000_000:
        raise ValueError("Search response exceeded the supported size.")
    parser = _SearchResultsParser()
    parser.feed(document.decode("utf-8", errors="replace"))
    results = [
        {
            "title": item.get("title", "").strip(),
            "url": item.get("url", "").strip(),
            "snippet": item.get("snippet", "").strip(),
        }
        for item in parser.results
        if item.get("url") and item.get("title")
    ]
    if not results:
        raise RuntimeError("Web search returned no results or the provider blocked the request.")
    return results[:max_results]


@mcp.tool()
def read_word_document(filename: str) -> str:
    """Read a bounded text preview from a Word file in the AgentOS files folder."""
    return read_word_file(settings.agent_files_dir, filename)


@mcp.tool()
def read_excel_workbook(filename: str) -> str:
    """Read a bounded preview of an Excel workbook in the AgentOS files folder."""
    return read_excel_file(settings.agent_files_dir, filename)


@mcp.tool()
def create_word_document(filename: str, title: str, content: str) -> str:
    """Create a new Word document in the AgentOS files folder without overwriting files."""
    return create_word_file(settings.agent_files_dir, filename, title, content)


@mcp.tool()
def create_excel_workbook(
    filename: str,
    sheet_name: str,
    headers: list[str],
    rows: list[list[str]],
) -> str:
    """Create a new Excel workbook in the AgentOS files folder without overwriting files."""
    return create_excel_file(settings.agent_files_dir, filename, sheet_name, headers, rows)


if __name__ == "__main__":
    mcp.run(transport="stdio")
