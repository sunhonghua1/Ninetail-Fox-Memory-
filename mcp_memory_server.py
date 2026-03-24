#!/usr/bin/env python3
"""
OpenClaw Memory V4 — MCP Server
A production-ready Model Context Protocol server for local AI memory.

Provides tools for:
  - Hybrid semantic + keyword memory search
  - Structured user profile management (Supermemory mode)
  - Autonomous fact extraction from conversations
  - Memory lifecycle management (add, delete, cleanup)

Environment Variables:
  MEMORY_STORAGE_PATH  — Path to JSON memory file (default: ./memory_v4.json)
  PROFILES_DB_PATH     — Path to SQLite profiles DB (default: ./profiles.sqlite)
  LLM_API_KEY          — API key for fact extraction LLM (required for extract_facts)
  LLM_BASE_URL         — OpenAI-compatible endpoint (default: DashScope)
  LLM_MODEL            — Model name (default: qwen-plus)
  LOG_LEVEL            — Logging level: DEBUG/INFO/WARNING (default: INFO)

Usage:
  python3 mcp_memory_server.py
"""

import asyncio
import os
import sys
import json
import logging
from typing import List, Dict, Any
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
)
from pydantic import AnyUrl

# ===== Logging Setup =====
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,  # MCP uses stdout for protocol, logs go to stderr
)
logger = logging.getLogger("openclaw-memory")

# ===== Add project root to Python path =====
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# ===== Import V4 Core =====
from user_profile_manager import UserProfileManager
from fact_extractor import FactExtractor

# Try to import the full search engine; fallback to profile-only mode
try:
    from openclaw_memory_v4 import EnhancedMemoryCore
    STORAGE_PATH = os.environ.get("MEMORY_STORAGE_PATH", os.path.join(PROJECT_DIR, "memory_v4.json"))
    memory = EnhancedMemoryCore(storage_path=STORAGE_PATH)
    HAS_SEARCH_ENGINE = True
    logger.info(f"Full search engine loaded. Storage: {STORAGE_PATH}")
except Exception as e:
    memory = None
    HAS_SEARCH_ENGINE = False
    logger.warning(f"Search engine unavailable ({e}). Running in profile-only mode.")

# ===== Initialize Profile Manager =====
PROFILES_DB = os.environ.get("PROFILES_DB_PATH", os.path.join(PROJECT_DIR, "profiles.sqlite"))
profile_manager = UserProfileManager(PROFILES_DB)

# Cleanup expired facts on startup
cleaned = profile_manager.cleanup_expired()
if cleaned:
    logger.info(f"Startup cleanup: removed {cleaned} expired facts.")

# ===== Initialize Fact Extractor =====
extractor = FactExtractor()  # Auto-uses openai_compatible_call

# ===== MCP Server =====
app = Server("openclaw-memory-v4")


@app.list_resources()
async def list_resources() -> list[Resource]:
    """List available resources."""
    resources = [
        Resource(
            uri=AnyUrl("memory://status"),
            name="Memory System Status",
            description="Overview of the memory system: storage stats, profile counts, and health.",
            mimeType="application/json",
        )
    ]
    return resources


@app.read_resource()
async def read_resource(uri: AnyUrl) -> str:
    """Read a resource."""
    if str(uri) == "memory://status":
        status = {
            "search_engine": "active" if HAS_SEARCH_ENGINE else "unavailable",
            "profiles_db": PROFILES_DB,
            "users": profile_manager.list_all_users(),
            "llm_configured": bool(os.environ.get("LLM_API_KEY")),
        }
        if HAS_SEARCH_ENGINE:
            try:
                status["memory_stats"] = memory.get_memory_stats()
            except Exception:
                status["memory_stats"] = "error"
        return json.dumps(status, indent=2, ensure_ascii=False)
    raise ValueError(f"Unknown resource: {uri}")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List all available MCP tools."""
    tools = [
        Tool(
            name="search_memory",
            description="Search local long-term and short-term memory using hybrid vector+keyword search. Returns relevant memories combined with user profile context.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "user_id": {"type": "string", "description": "User ID for profile context (default: 'default')."},
                    "scope": {"type": "string", "description": "Memory scope filter (e.g., 'project', 'personal')."},
                    "max_results": {"type": "integer", "description": "Max results to return (default: 5)."},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="add_memory",
            description="Store a new piece of information in long-term memory for future retrieval.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The content to memorize."},
                    "scope": {"type": "string", "description": "Category (e.g., 'project', 'personal'). Default: 'default'."},
                },
                "required": ["content"],
            },
        ),
        Tool(
            name="get_user_profile",
            description="Retrieve structured user traits (STATIC) and current dynamic context for a specific user. Returns formatted text ready for system prompt injection.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "The unique user identifier."},
                },
                "required": ["user_id"],
            },
        ),
        Tool(
            name="add_user_fact",
            description="Manually add a fact to a user's profile. Use type 'STATIC' for permanent traits (name, preferences) or 'DYNAMIC' for temporary states (busy this week).",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "The user to update."},
                    "fact": {"type": "string", "description": "The fact to store."},
                    "fact_type": {"type": "string", "enum": ["STATIC", "DYNAMIC"], "description": "STATIC or DYNAMIC."},
                    "ttl_days": {"type": "integer", "description": "Days until expiry (for DYNAMIC facts)."},
                },
                "required": ["user_id", "fact"],
            },
        ),
        Tool(
            name="delete_user_fact",
            description="Delete a specific fact from a user's profile by its ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "fact_id": {"type": "integer", "description": "The fact ID to delete."},
                },
                "required": ["fact_id"],
            },
        ),
        Tool(
            name="list_users",
            description="List all users who have stored profile facts, with their fact counts.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="extract_facts",
            description="Use an LLM to automatically extract user facts from a block of conversation text. Requires LLM_API_KEY to be configured.",
            inputSchema={
                "type": "object",
                "properties": {
                    "conversation": {"type": "string", "description": "Raw conversation text to analyze."},
                    "user_id": {"type": "string", "description": "User ID to save extracted facts to."},
                },
                "required": ["conversation", "user_id"],
            },
        ),
    ]
    return tools


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Route and execute MCP tool calls."""
    try:
        if name == "search_memory":
            return await _search_memory(arguments)
        elif name == "add_memory":
            return await _add_memory(arguments)
        elif name == "get_user_profile":
            return await _get_user_profile(arguments)
        elif name == "add_user_fact":
            return await _add_user_fact(arguments)
        elif name == "delete_user_fact":
            return await _delete_user_fact(arguments)
        elif name == "list_users":
            return await _list_users(arguments)
        elif name == "extract_facts":
            return await _extract_facts(arguments)
        else:
            return [TextContent(type="text", text=f"❌ Unknown tool: {name}")]
    except Exception as e:
        logger.error(f"Tool '{name}' failed: {e}", exc_info=True)
        return [TextContent(type="text", text=f"❌ Error in {name}: {str(e)}")]


# ===== Tool Implementations =====

async def _search_memory(args: dict) -> list[TextContent]:
    query = args["query"]
    user_id = args.get("user_id", "default")
    scope = args.get("scope")
    max_results = args.get("max_results", 5)

    parts = []

    # Profile context
    profile_str = profile_manager.get_context_string(user_id)
    if profile_str:
        parts.append(profile_str)

    # Vector search
    if HAS_SEARCH_ENGINE:
        results = memory.smart_recall(query, max_results=max_results, scope=scope)
        if results:
            parts.append("\n=== Relevant Memories ===")
            for r in results:
                score = r.get('score', 0)
                parts.append(f"- [{score:.2f}] {r['content']}")
        else:
            parts.append("\n(No matching memories found)")
    else:
        parts.append("\n⚠️ Search engine unavailable. Showing profile only.")

    output = "\n".join(parts) if parts else "No memories or profile data found."
    logger.info(f"search_memory: query='{query[:50]}', results={len(parts)}")
    return [TextContent(type="text", text=output)]


async def _add_memory(args: dict) -> list[TextContent]:
    content = args["content"]
    scope = args.get("scope", "default")

    if not HAS_SEARCH_ENGINE:
        return [TextContent(type="text", text="❌ Search engine unavailable. Cannot add memory.")]

    memory.add_memory(content, scope=scope)
    logger.info(f"add_memory: scope='{scope}', content='{content[:50]}'")
    return [TextContent(type="text", text=f"✅ Memory stored in scope '{scope}'.")]


async def _get_user_profile(args: dict) -> list[TextContent]:
    user_id = args["user_id"]
    profiles = profile_manager.get_profiles(user_id)
    context_str = profile_manager.get_context_string(user_id)

    if not context_str:
        return [TextContent(type="text", text=f"No profile data found for user '{user_id}'.")]

    summary = f"Static: {len(profiles['static_facts'])} | Dynamic: {len(profiles['dynamic_contexts'])}"
    output = f"{context_str}\n\n--- {summary} ---"
    return [TextContent(type="text", text=output)]


async def _add_user_fact(args: dict) -> list[TextContent]:
    user_id = args["user_id"]
    fact = args["fact"]
    fact_type = args.get("fact_type", "STATIC")
    ttl_days = args.get("ttl_days")

    profile_manager.add_fact(user_id, fact, fact_type, ttl_days)
    ttl_info = f" (expires in {ttl_days} days)" if ttl_days else ""
    return [TextContent(type="text", text=f"✅ {fact_type} fact added for '{user_id}'{ttl_info}: {fact}")]


async def _delete_user_fact(args: dict) -> list[TextContent]:
    fact_id = args["fact_id"]
    success = profile_manager.delete_fact(fact_id)
    if success:
        return [TextContent(type="text", text=f"✅ Fact #{fact_id} deleted.")]
    return [TextContent(type="text", text=f"⚠️ Fact #{fact_id} not found or already deleted.")]


async def _list_users(args: dict) -> list[TextContent]:
    users = profile_manager.list_all_users()
    if not users:
        return [TextContent(type="text", text="No users with profile data found.")]

    lines = ["User Profiles Overview:"]
    for u in users:
        lines.append(f"- {u['user_id']}: {u['total_facts']} facts "
                     f"({u['static_count']} static, {u['dynamic_count']} dynamic) "
                     f"| Last updated: {u['last_updated']}")
    return [TextContent(type="text", text="\n".join(lines))]


async def _extract_facts(args: dict) -> list[TextContent]:
    conversation = args["conversation"]
    user_id = args["user_id"]

    if not os.environ.get("LLM_API_KEY"):
        return [TextContent(type="text", text="❌ LLM_API_KEY not configured. Cannot extract facts. "
                           "Set the LLM_API_KEY environment variable to enable this feature.")]

    messages = [{"role": "user", "content": conversation}]
    facts = extractor.extract_facts(messages)

    if not facts:
        return [TextContent(type="text", text="No significant facts found in the conversation.")]

    saved = []
    for f in facts:
        profile_manager.add_fact(user_id, f['fact'], f.get('type', 'STATIC'), f.get('ttl_days'))
        ttl = f" (TTL: {f['ttl_days']}d)" if f.get('ttl_days') else ""
        saved.append(f"- [{f.get('type', 'STATIC')}] {f['fact']}{ttl}")

    output = f"✅ Extracted and saved {len(saved)} facts for '{user_id}':\n" + "\n".join(saved)
    return [TextContent(type="text", text=output)]


# ===== Entry Point =====

async def main():
    """Run the MCP server via STDIO."""
    logger.info("Starting OpenClaw Memory V4 MCP Server...")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
