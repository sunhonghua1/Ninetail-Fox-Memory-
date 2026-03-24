import asyncio
import os
import json
from typing import Optional, List, Dict, Any
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)
from pydantic import AnyUrl

# Import our V4 Logic
from openclaw_memory_v4 import EnhancedMemoryCore

# Initialize Memory Core
# Note: storage_path should be configurable, defaulting to memory.json
STORAGE_PATH = os.environ.get("MEMORY_STORAGE_PATH", "memory_v4.json")
memory = EnhancedMemoryCore(storage_path=STORAGE_PATH)

app = Server("local-memory-mcp")

@app.list_resources()
async def list_resources() -> list[Resource]:
    """List available static resources (e.g., system prompts)."""
    return [
        Resource(
            uri=AnyUrl("memory://config"),
            name="Memory Stats",
            description="Status of the local memory system",
            mimeType="application/json",
        )
    ]

@app.read_resource()
async def read_resource(uri: AnyUrl) -> str:
    """Read a specific resource."""
    if str(uri) == "memory://config":
        stats = memory.get_memory_stats()
        return json.dumps(stats, indent=2)
    raise ValueError(f"Unknown resource: {uri}")

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available memory tools."""
    return [
        Tool(
            name="search_memory",
            description="Search local long-term and short-term memory using hybrid vector search.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "user_id": {"type": "string", "description": "Unique user ID (default: 'default')"},
                    "scope": {"type": "string", "description": "Optional search scope (e.g., 'project', 'personal')"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_user_profile",
            description="Retrieve structured user traits and current dynamic context (Supermemory mode).",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "Unique user ID"},
                },
                "required": ["user_id"],
            },
        ),
        Tool(
            name="add_memory",
            description="Add a new item to the local memory log for future retrieval.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Content to store"},
                    "scope": {"type": "string", "description": "Optional category"},
                    "user_id": {"type": "string", "description": "User ID associated with this memory"},
                },
                "required": ["content"],
            },
        ),
        Tool(
            name="extract_facts",
            description="Automatically extract facts from a conversation block and save to user profile.",
            inputSchema={
                "type": "object",
                "properties": {
                    "conversation": {"type": "string", "description": "Raw chat history to analyze"},
                    "user_id": {"type": "string", "description": "User ID to update"},
                },
                "required": ["conversation", "user_id"],
            },
        ),
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    if name == "search_memory":
        query = arguments["query"]
        user_id = arguments.get("user_id", "default")
        scope = arguments.get("scope")
        
        # Use our V4 hybrid recall
        results = memory.smart_recall(query, scope=scope)
        profile_context = memory.profile_manager.get_context_string(user_id)
        
        formatted_results = "\n".join([f"- {r['content']}" for r in results])
        output = f"PROFILE CONTEXT:\n{profile_context}\n\nSEARCH RESULTS:\n{formatted_results}"
        return [TextContent(type="text", text=output)]

    elif name == "get_user_profile":
        user_id = arguments["user_id"]
        profile = memory.profile_manager.get_context_string(user_id)
        return [TextContent(type="text", text=profile or "No profile information found for this user.")]

    elif name == "add_memory":
        content = arguments["content"]
        scope = arguments.get("scope", "default")
        memory.add_memory(content, scope=scope)
        return [TextContent(type="text", text=f"Memory successfully stored in scope '{scope}'.")]

    elif name == "extract_facts":
        conversation = arguments["conversation"]
        user_id = arguments["user_id"]
        
        # Format as prompt for the extractor (llm_call needs to be configured in real use)
        # For now, we reuse the profile_manager directly if the user provides direct facts
        # Or we can implement the logic to call the extractor if an API key is available
        facts = memory.extractor.extract_facts([{"role": "user", "content": conversation}])
        
        extracted = []
        for f in facts:
            memory.profile_manager.add_fact(user_id, f['fact'], f['type'], f.get('ttl_days'))
            extracted.append(f['fact'])
            
        return [TextContent(type="text", text=f"Extracted and saved facts: {', '.join(extracted)}" if extracted else "No significant facts found.")]

    raise ValueError(f"Unknown tool: {name}")

async def main():
    """Run the server using standard input/output."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
