import asyncio
import os
import json
from typing import Optional, Dict, List, Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        
        # .env 파일에서 API_KEY를 가져와 OpenAI 클라이언트 초기화
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")
            
        self.openai = OpenAI(api_key=api_key)

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        server_params = StdioServerParameters(
            command="python",
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using OpenAI API and available tools"""
        # 시스템 메시지 추가
        messages = [
            {
                "role": "system", 
                "content": "당신은 날씨 정보를 제공하는 도우미입니다. 사용자의 질문에 답하기 위해 제공된 도구를 활용하세요."
            },
            {
                "role": "user",
                "content": query
            }
        ]

        # MCP에서 사용 가능한 도구 목록 가져오기
        response = await self.session.list_tools()
        
        # OpenAI 형식에 맞게 도구 변환
        available_tools = []
        for tool in response.tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            }
            available_tools.append(openai_tool)

        # OpenAI API 호출
        response = self.openai.chat.completions.create(
            model="gpt-4o",  # 또는 다른 적절한 모델
            messages=messages,
            tools=available_tools,
            tool_choice="auto"
        )

        # 응답 처리
        final_text = []
        response_message = response.choices[0].message

        # 텍스트 응답이 있으면 추가
        if response_message.content:
            final_text.append(response_message.content)

        # 도구 호출이 있는 경우 처리
        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                # 도구 호출 로그 추가
                final_text.append(f"[도구 호출: {function_name}, 인자: {function_args}]")
                
                # MCP 서버에 도구 호출 요청
                result = await self.session.call_tool(function_name, function_args)
                
                # 도구 호출 결과를 메시지에 추가
                messages.append(response_message.model_dump())
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result.content
                })
                
                # 최종 응답 생성
                final_response = self.openai.chat.completions.create(
                    model="gpt-4o",  # 또는 다른 적절한 모델
                    messages=messages
                )
                
                final_text.append(final_response.choices[0].message.content)

        return "\n".join(final_text)


    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())