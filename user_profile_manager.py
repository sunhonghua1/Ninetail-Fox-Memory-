import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

class UserProfileManager:
    """
    User Profile Manager (Local Supermemory Mode)
    
    Manages structured user facts (STATIC and DYNAMIC) in a local SQLite database.
    Inspired by Supermemory's user context and mem9's persistent infrastructure,
    but implemented as a lightweight, zero-dependency local solution.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database with user_profiles table."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table with TTL support
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                fact_type TEXT CHECK(fact_type IN ('STATIC', 'DYNAMIC')) NOT NULL,
                content TEXT NOT NULL,
                expires_at DATETIME,
                is_active INTEGER DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Index for fast retrieval
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_profiles_user_id ON user_profiles(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_profiles_type ON user_profiles(fact_type)')
        
        conn.commit()
        conn.close()

    def add_fact(self, user_id: str, content: str, fact_type: str = 'STATIC', ttl_days: Optional[int] = None):
        """
        Add or update a user fact.
        
        Args:
            user_id: Unique identifier for the user or group.
            content: The extracted fact string.
            fact_type: 'STATIC' (long-term) or 'DYNAMIC' (temporary).
            ttl_days: Days until the fact expires (optional, used for DYNAMIC).
        """
        expires_at = None
        if fact_type == 'DYNAMIC' and ttl_days:
            expires_at = (datetime.now() + timedelta(days=ttl_days)).isoformat()
        elif ttl_days: # Even static facts could have a very long TTL if desired
            expires_at = (datetime.now() + timedelta(days=ttl_days)).isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Simple implementation: always append for now (deduplication can be handled by extractor or manual cleanup)
        cursor.execute('''
            INSERT INTO user_profiles (user_id, fact_type, content, expires_at)
            VALUES (?, ?, ?, ?)
        ''', (user_id, fact_type, content, expires_at))
        
        conn.commit()
        conn.close()
        return True

    def get_profiles(self, user_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve active profile facts for a user.
        Filters out expired dynamic facts.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        
        cursor.execute('''
            SELECT * FROM user_profiles 
            WHERE user_id = ? 
            AND is_active = 1
            AND (expires_at IS NULL OR expires_at > ?)
        ''', (user_id, now))
        
        rows = cursor.fetchall()
        conn.close()

        static_facts = []
        dynamic_contexts = []
        
        for row in rows:
            fact = {
                "id": row["id"],
                "content": row["content"],
                "fact_type": row["fact_type"],
                "created_at": row["created_at"],
                "expires_at": row["expires_at"]
            }
            if row["fact_type"] == 'STATIC':
                static_facts.append(fact)
            else:
                dynamic_contexts.append(fact)
                
        return {
            "static_facts": static_facts,
            "dynamic_contexts": dynamic_contexts
        }

    def delete_fact(self, fact_id: int):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('UPDATE user_profiles SET is_active = 0 WHERE id = ?', (fact_id,))
        conn.commit()
        conn.close()

    def get_context_string(self, user_id: str) -> str:
        """Returns a formatted string for inclusion in LLM prompt."""
        profiles = self.get_profiles(user_id)
        parts = []
        
        if profiles["static_facts"]:
            parts.append("【User Static Traits】")
            for f in profiles["static_facts"]:
                parts.append(f"- {f['content']}")
        
        if profiles["dynamic_contexts"]:
            parts.append("【Current User Status/Context】")
            for f in profiles["dynamic_contexts"]:
                parts.append(f"- {f['content']}")
                
        return "\n".join(parts) if parts else ""
