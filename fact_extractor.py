import json
import re
from typing import List, Dict, Any, Optional

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

    def __init__(self, llm_provider_callback):
        """
        Args:
            llm_provider_callback: A function/method that takes (prompt, system_prompt) 
                                   and returns a string response from LLM.
        """
        self.llm_call = llm_provider_callback

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
            return self._parse_json(response_text)
        except Exception as e:
            print(f"⚠️ Fact extraction failed: {e}")
            return []

    def _parse_json(self, text: str) -> List[Dict[str, Any]]:
        # Find JSON block in case LLM added garbage
        json_match = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except:
                pass
        
        # Fallback: simple split if it looks like a list
        try:
            return json.loads(text)
        except:
            return []
