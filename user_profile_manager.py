"""
User Profile Manager (Local Supermemory Mode)

Manages structured user facts (STATIC and DYNAMIC) in a local SQLite database.
Inspired by Supermemory's user context and mem9's persistent infrastructure,
but implemented as a lightweight, zero-dependency local solution.
"""

import sqlite3
import json
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger("openclaw-memory")


class UserProfileManager:
    """
    SQLite-backed User Profile storage with TTL support.
    
    Fact Types:
        STATIC: Long-term traits (name, preferences, skills)
        DYNAMIC: Temporary states (busy this week, traveling tomorrow)
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database with user_profiles table."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

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

        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_profiles_user_id ON user_profiles(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_profiles_type ON user_profiles(fact_type)')

        conn.commit()
        conn.close()
        logger.info(f"UserProfileManager initialized at {self.db_path}")

    def add_fact(self, user_id: str, content: str, fact_type: str = 'STATIC', ttl_days: Optional[int] = None) -> bool:
        """
        Add a user fact.

        Args:
            user_id: Unique identifier for the user or group.
            content: The extracted fact string.
            fact_type: 'STATIC' (long-term) or 'DYNAMIC' (temporary).
            ttl_days: Days until the fact expires (optional).
        """
        expires_at = None
        if ttl_days:
            expires_at = (datetime.now() + timedelta(days=ttl_days)).isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check for duplicates before inserting
        cursor.execute('''
            SELECT id FROM user_profiles
            WHERE user_id = ? AND content = ? AND is_active = 1
        ''', (user_id, content))
        
        if cursor.fetchone():
            # Update the timestamp instead of duplicating
            cursor.execute('''
                UPDATE user_profiles SET updated_at = CURRENT_TIMESTAMP, expires_at = ?
                WHERE user_id = ? AND content = ? AND is_active = 1
            ''', (expires_at, user_id, content))
            logger.debug(f"Updated existing fact for {user_id}: {content[:50]}")
        else:
            cursor.execute('''
                INSERT INTO user_profiles (user_id, fact_type, content, expires_at)
                VALUES (?, ?, ?, ?)
            ''', (user_id, fact_type, content, expires_at))
            logger.info(f"Added {fact_type} fact for {user_id}: {content[:50]}")

        conn.commit()
        conn.close()
        return True

    def get_profiles(self, user_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve active profile facts for a user.
        Automatically filters out expired dynamic facts.
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
            ORDER BY updated_at DESC
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

    def delete_fact(self, fact_id: int) -> bool:
        """Soft-delete a fact by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('UPDATE user_profiles SET is_active = 0 WHERE id = ?', (fact_id,))
        affected = cursor.rowcount
        conn.commit()
        conn.close()
        logger.info(f"Deleted fact #{fact_id} (affected: {affected})")
        return affected > 0

    def list_all_users(self) -> List[Dict[str, Any]]:
        """List all users with their fact counts."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT user_id, 
                   COUNT(*) as total_facts,
                   SUM(CASE WHEN fact_type = 'STATIC' THEN 1 ELSE 0 END) as static_count,
                   SUM(CASE WHEN fact_type = 'DYNAMIC' THEN 1 ELSE 0 END) as dynamic_count,
                   MAX(updated_at) as last_updated
            FROM user_profiles 
            WHERE is_active = 1
            GROUP BY user_id
            ORDER BY last_updated DESC
        ''')
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "user_id": row[0],
                "total_facts": row[1],
                "static_count": row[2],
                "dynamic_count": row[3],
                "last_updated": row[4]
            }
            for row in rows
        ]

    def cleanup_expired(self) -> int:
        """Remove expired facts from the database. Returns count of cleaned records."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        cursor.execute('''
            UPDATE user_profiles SET is_active = 0 
            WHERE expires_at IS NOT NULL AND expires_at <= ? AND is_active = 1
        ''', (now,))
        cleaned = cursor.rowcount
        conn.commit()
        conn.close()
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} expired facts.")
        return cleaned

    def get_context_string(self, user_id: str) -> str:
        """Returns a formatted string for inclusion in LLM prompt."""
        profiles = self.get_profiles(user_id)
        parts = []

        if profiles["static_facts"]:
            parts.append("【User Traits】")
            for f in profiles["static_facts"]:
                parts.append(f"- {f['content']}")

        if profiles["dynamic_contexts"]:
            parts.append("【Current Context】")
            for f in profiles["dynamic_contexts"]:
                parts.append(f"- {f['content']}")

        return "\n".join(parts) if parts else ""
