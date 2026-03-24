"""
Autonomous Fact Extractor (Supermemory Consolidator)

Uses an LLM to extract structured user facts from raw conversation logs.
Identifies if a fact is STATIC (long-term trait) or DYNAMIC (temporary state).

Supports any OpenAI-compatible API endpoint via environment variables:
  - LLM_API_KEY: Your API key for the LLM provider
  - LLM_BASE_URL: Base URL (default: https://dashscope.aliyuncs.com/compatible-mode/v1)
  - LLM_MODEL: Model name (default: qwen-plus)
"""

import json
import re
import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("openclaw-memory")

# Try to use httpx (installed with mcp), fallback to urllib
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    import urllib.request
    HAS_HTTPX = False


def openai_compatible_call(prompt: str, system_prompt: str) -> str:
    """
    Call any OpenAI-compatible LLM endpoint.
    Configured via environment variables: LLM_API_KEY, LLM_BASE_URL, LLM_MODEL.
    """
    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model = os.environ.get("LLM_MODEL", "qwen-plus")

    if not api_key:
        logger.warning("LLM_API_KEY not set. Fact extraction will return empty results.")
        return "[]"

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 1024,
    }

    try:
        if HAS_HTTPX:
            resp = httpx.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        else:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))

        return data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"LLM API call failed: {e}")
        return "[]"


class FactExtractor:
    """
    Autonomous Fact Extractor (Supermemory Consolidator Port)

    Uses an LLM to extract structured user facts from raw conversation logs.
    Identifies if a fact is STATIC (long-term trait) or DYNAMIC (temporary state).
    """

    SYSTEM_PROMPT = """
You are the "Memory Consolidator" for an AI agent. 
Your task is to analyze the recent conversation history and extract important facts about the user.

FACT TYPES:
1. STATIC: Long-term traits, preferences, personal info (e.g., name, job, birthday, technology stacks).
2. DYNAMIC: Temporary states, current tasks, immediate plans (e.g., "busy this week", "traveling to Tokyo tomorrow", "currently working on a React project").

OUTPUT FORMAT (JSON ONLY):
[
  {"fact": "User is a senior Python developer", "type": "STATIC"},
  {"fact": "User is busy with a project launch this week", "type": "DYNAMIC", "ttl_days": 7},
  {"fact": "User prefers dark mode in UI", "type": "STATIC"}
]

Only extract NEW and SIGNIFICANT information. If no new facts are found, return an empty list [].
    """

    def __init__(self, llm_provider_callback=None):
        """
        Args:
            llm_provider_callback: A function/method that takes (prompt, system_prompt) 
                                   and returns a string response from LLM.
                                   If None, uses the built-in openai_compatible_call.
        """
        self.llm_call = llm_provider_callback or openai_compatible_call

    def extract_facts(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Analyze messages and return list of facts.

        Args:
            messages: List of {"role": "user/assistant", "content": "..."}
        """
        if not messages:
            return []

        context = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        user_prompt = f"Please extract facts from the following recent conversation:\n\n{context}"

        try:
            response_text = self.llm_call(user_prompt, self.SYSTEM_PROMPT)
            facts = self._parse_json(response_text)
            logger.info(f"Extracted {len(facts)} facts from {len(messages)} messages.")
            return facts
        except Exception as e:
            logger.error(f"Fact extraction failed: {e}")
            return []

    def _parse_json(self, text: str) -> List[Dict[str, Any]]:
        """Parse JSON from LLM response, handling markdown code blocks and extra text."""
        # Try markdown code block first
        code_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', text, re.DOTALL)
        if code_match:
            try:
                return json.loads(code_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try raw JSON array
        json_match = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Fallback: try the entire text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse LLM response as JSON: {text[:200]}")
            return []
