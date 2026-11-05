import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

from app.agent import run_agent
from app.core.config import settings
from app.mcp_server import _SearchResultsParser, web_search


BACKEND_DIR = Path(__file__).resolve().parents[1]


class AgentOSMCPTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_agent_discovers_and_calls_mcp_tools(self):
        tool_call = MagicMock()
        tool_call.id = "call_calculate"
        tool_call.function = SimpleNamespace(
            name="calculate",
            arguments='{"expression": "2 + 4"}',
        )
        tool_call.model_dump.return_value = {
            "id": tool_call.id,
            "type": "function",
            "function": {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
            },
        }
        responses = [
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content=None, tool_calls=[tool_call]
            ))]),
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content="The answer is 6.", tool_calls=None
            ))]),
        ]
        completion = AsyncMock(side_effect=responses)
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=completion))
        )
        fake_db = MagicMock()
        fake_db.execute.return_value.scalars.return_value.all.return_value = []

        with patch("app.agent.AsyncGroq", return_value=fake_client), patch.object(
            settings, "groq_api_key", "test-api-key"
        ):
            answer = await run_agent(1, fake_db)

        self.assertEqual(answer, "The answer is 6.")
        available_tools = completion.await_args_list[0].kwargs["tools"]
        self.assertIn("calculate", {
            tool["function"]["name"] for tool in available_tools
        })
        tool_response = completion.await_args_list[1].kwargs["messages"][-1]
        self.assertEqual(tool_response["content"], "6")

    async def test_mcp_tools_are_available_to_agentos_chat(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            parameters = StdioServerParameters(
                command=sys.executable,
                args=["-m", "app.mcp_server"],
                cwd=str(BACKEND_DIR),
                env={
                    "DATABASE_URL": settings.database_url,
                    "AGENT_FILES_DIR": temporary_directory,
                },
            )
            async with stdio_client(parameters) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tools = (await session.list_tools()).tools
                    tool_names = {tool.name for tool in tools}
                    self.assertTrue({
                        "calculate",
                        "get_current_datetime",
                        "create_task_note",
                        "list_agent_files",
                        "read_agent_file",
                        "write_agent_file",
                        "check_python_syntax",
                        "get_system_info",
                        "web_search",
                        "read_word_document",
                        "read_excel_workbook",
                        "create_word_document",
                        "create_excel_workbook",
                    }.issubset(tool_names))

                    calculation = await session.call_tool(
                        "calculate", {"expression": "(12 + 3) * 2"}
                    )
                    self.assertEqual(
                        [item.text for item in calculation.content if isinstance(item, TextContent)],
                        ["30"],
                    )

                    created = await session.call_tool(
                        "create_word_document",
                        {
                            "filename": "Meeting Notes.docx",
                            "title": "Meeting Notes",
                            "content": "Review the launch plan",
                        },
                    )
                    self.assertFalse(created.isError)

                    created_python = await session.call_tool(
                        "write_agent_file",
                        {
                            "filename": "hello_tool.py",
                            "content": 'def hello():\n    return "hello"\n',
                        },
                    )
                    self.assertFalse(created_python.isError)
                    python_listing = await session.call_tool("list_agent_files", {})
                    self.assertIn("hello_tool.py", str(python_listing))
                    python_source = await session.call_tool(
                        "read_agent_file", {"filename": "hello_tool.py"}
                    )
                    self.assertIn('return "hello"', str(python_source))

                    syntax_check = await session.call_tool(
                        "check_python_syntax",
                        {"source": "def valid():\n    return 42\n"},
                    )
                    self.assertIn("syntax is valid", str(syntax_check))
                    invalid_python = await session.call_tool(
                        "write_agent_file",
                        {"filename": "broken.py", "content": "def broken(:\n"},
                    )
                    self.assertTrue(invalid_python.isError)

                    listing = await session.call_tool("list_agent_files", {})
                    self.assertFalse(listing.isError)
                    self.assertIn("Meeting Notes.docx", str(listing))

                    system_info = await session.call_tool("get_system_info", {})
                    self.assertFalse(system_info.isError)
                    self.assertIn("logical_cpu_count", str(system_info))

                    unsafe_calculation = await session.call_tool(
                        "calculate", {"expression": "__import__('os').getcwd()"}
                    )
                    self.assertTrue(unsafe_calculation.isError)

    def test_web_search_parser_extracts_result_metadata(self):
        parser = _SearchResultsParser()
        parser.feed(
            '<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com">'
            "Example <b>Result</b></a>"
            '<a class="result__snippet">A useful snippet</a>'
        )
        self.assertEqual(
            parser.results,
            [{
                "url": "https://example.com",
                "title": "Example Result",
                "snippet": "A useful snippet",
            }],
        )

    def test_web_search_returns_results_and_reports_provider_challenges(self):
        response = MagicMock()
        response.status = 200
        response.read.return_value = (
            b'<a class="result__a" href="https://example.com">Example Result</a>'
            b'<a class="result__snippet">A useful snippet</a>'
        )
        response.__enter__.return_value = response
        with patch("app.mcp_server.urlopen", return_value=response):
            results = web_search("example query", 2)
        self.assertEqual(results[0]["url"], "https://example.com")

        blocked_response = MagicMock()
        blocked_response.status = 202
        blocked_response.read.return_value = b"challenge page"
        blocked_response.__enter__.return_value = blocked_response
        with patch("app.mcp_server.urlopen", return_value=blocked_response):
            with self.assertRaisesRegex(RuntimeError, "unavailable response"):
                web_search("example query")


if __name__ == "__main__":
    unittest.main()
