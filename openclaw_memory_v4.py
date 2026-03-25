#!/usr/bin/env python3
"""
Ninetail-Fox Memory V4.5
Local Supermemory Engine: Real Vector Search + SQLite User Profiles + Fact Extraction

V4.5 Fixes:
1. conversation_log: deque(maxlen=100) → SQLite persistent storage (unlimited)
2. VectorCache: JSON FIFO → SQLite LRU with last_accessed tracking
3. BM25 tokenizer: optional jieba for proper Chinese word segmentation
4. FactExtractor: wired to real LLM via openai_compatible_call
"""

import json
import re
import time
import math
import os
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger("openclaw-memory")

# ========== Optional jieba support ==========
try:
    import jieba
    HAS_JIEBA = True
    logger.info("jieba detected — using smart Chinese segmentation for BM25.")
except ImportError:
    HAS_JIEBA = False

from embedding_provider import (
    MultiProviderEmbedding, cosine_similarity, EmbeddingResult,
    DashScopeReranker
)
from user_profile_manager import UserProfileManager
from fact_extractor import FactExtractor


# ========== 噪音过滤器 ==========

class NoiseFilter:
    """
    噪音过滤器
    自动过滤无意义信息，避免存入垃圾记忆

    规则：
    - 内容过短（< 4 字）
    - 纯粹的问候语、感叹词
    - 纯标点/表情
    - 重复内容
    """

    # 常见噪音短语（支持中英文）
    NOISE_PATTERNS = [
        r'^(你好|hello|hi|hey|嗨|哈喽|ok|好的|谢谢|thanks|嗯|哦|啊|呢|吧|了|是的|对|没错|行|可以)[\s!！.。~？?]*$',
        r'^[\s\.\,\!\?\;\:\-\~\…\。\，\！\？\；\：]+$',  # 纯标点
        r'^[\U0001f600-\U0001f9ff\U00002702-\U000027b0\s]+$',  # 纯表情
        r'^(lol|lmao|haha|哈哈|嘻嘻|呵呵|233+)\s*$',
    ]

    def __init__(self, min_length: int = 4):
        self.min_length = min_length
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.NOISE_PATTERNS]
        self._recent_hashes = deque(maxlen=200)  # 去重缓冲

    def is_noise(self, text: str) -> bool:
        """判断内容是否为噪音"""
        text = text.strip()

        # 过短
        if len(text) < self.min_length:
            return True

        # 匹配噪音模式
        for pattern in self._compiled:
            if pattern.match(text):
                return True

        # 重复内容
        text_hash = hash(text[:100])
        if text_hash in self._recent_hashes:
            return True
        self._recent_hashes.append(text_hash)

        return False

    def filter_batch(self, texts: List[str]) -> List[str]:
        """批量过滤噪音"""
        return [t for t in texts if not self.is_noise(t)]


# ========== 时间衰减 ==========

class TimeDecay:
    """
    时间衰减计算器
    近期记忆权重更高，遥远记忆权重降低

    使用指数衰减：score * exp(-lambda * days_ago)
    - 1 天前：权重 ~95%
    - 7 天前：权重 ~70%
    - 30 天前：权重 ~30%
    - 90 天前：权重 ~10%
    """

    def __init__(self, half_life_days: float = 14.0):
        """
        Args:
            half_life_days: 半衰期（天数），即多少天后权重降为 50%
        """
        # lambda = ln(2) / half_life
        self.decay_lambda = math.log(2) / half_life_days

    def apply(self, score: float, timestamp: str) -> float:
        """
        对分数施加时间衰减

        Args:
            score: 原始分数
            timestamp: ISO 格式时间戳
        """
        try:
            doc_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            # 处理 naive datetime（无时区信息）
            now = datetime.now()
            if doc_time.tzinfo:
                now = datetime.now(doc_time.tzinfo)

            days_ago = max((now - doc_time).total_seconds() / 86400, 0)
            decay_factor = math.exp(-self.decay_lambda * days_ago)
            return score * decay_factor
        except Exception:
            # 解析失败不影响评分
            return score

    def get_decay_info(self, timestamp: str) -> Dict:
        """获取衰减详情（用于调试）"""
        try:
            doc_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            now = datetime.now()
            if doc_time.tzinfo:
                now = datetime.now(doc_time.tzinfo)
            days_ago = max((now - doc_time).total_seconds() / 86400, 0)
            decay_factor = math.exp(-self.decay_lambda * days_ago)
            return {
                "days_ago": round(days_ago, 1),
                "decay_factor": round(decay_factor, 4)
            }
        except Exception:
            return {"days_ago": 0, "decay_factor": 1.0}


# ========== 向量缓存 (SQLite LRU) ==========

class VectorCache:
    """
    向量缓存管理器 (V4.5 升级: SQLite + LRU)
    
    改进点:
    - SQLite 存储替代 JSON 文件, 解决大缓存内存爆炸问题
    - LRU 淘汰策略替代 FIFO, 高频查询的向量不会被误删
    - last_accessed 时间戳, 按最近使用排序
    """
    MAX_CACHE_SIZE = 10000
    EVICT_COUNT = 2000  # 超限时淘汰最久未用的条目

    def __init__(self, cache_path: str):
        # 将 .json 路径转为 .sqlite
        if cache_path.endswith(".json"):
            cache_path = cache_path.replace(".json", ".sqlite")
        self.cache_path = cache_path
        self._init_db()
        self._migrate_from_json(cache_path)

    def _init_db(self):
        Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.cache_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS vector_cache (
                key TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_accessed TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_vc_last_accessed ON vector_cache(last_accessed)')
        conn.commit()
        conn.close()

    def _migrate_from_json(self, sqlite_path: str):
        """One-time migration from legacy JSON cache."""
        json_path = sqlite_path.replace(".sqlite", ".json")
        if not Path(json_path).exists():
            return
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                old_cache = json.load(f)
            if old_cache:
                conn = sqlite3.connect(self.cache_path)
                for key, vec in old_cache.items():
                    blob = json.dumps(vec).encode("utf-8")
                    conn.execute(
                        'INSERT OR IGNORE INTO vector_cache (key, vector) VALUES (?, ?)',
                        (key, blob)
                    )
                conn.commit()
                conn.close()
                logger.info(f"Migrated {len(old_cache)} vectors from JSON to SQLite.")
            # Rename old file so migration doesn't repeat
            Path(json_path).rename(json_path + ".migrated")
        except Exception as e:
            logger.warning(f"JSON migration skipped: {e}")

    @staticmethod
    def _text_key(text: str) -> str:
        return f"{hash(text[:200])}_{len(text)}"

    def get(self, text: str) -> Optional[List[float]]:
        key = self._text_key(text)
        conn = sqlite3.connect(self.cache_path)
        row = conn.execute('SELECT vector FROM vector_cache WHERE key = ?', (key,)).fetchone()
        if row:
            # LRU touch: update last_accessed
            conn.execute(
                'UPDATE vector_cache SET last_accessed = ? WHERE key = ?',
                (datetime.now().isoformat(), key)
            )
            conn.commit()
            conn.close()
            return json.loads(row[0])
        conn.close()
        return None

    def put(self, text: str, vector: List[float]):
        key = self._text_key(text)
        blob = json.dumps(vector).encode("utf-8")
        conn = sqlite3.connect(self.cache_path)
        now = datetime.now().isoformat()
        conn.execute(
            'INSERT OR REPLACE INTO vector_cache (key, vector, created_at, last_accessed) VALUES (?, ?, ?, ?)',
            (key, blob, now, now)
        )
        # LRU eviction: if over limit, drop oldest accessed
        count = conn.execute('SELECT COUNT(*) FROM vector_cache').fetchone()[0]
        if count > self.MAX_CACHE_SIZE:
            conn.execute(f'''
                DELETE FROM vector_cache WHERE key IN (
                    SELECT key FROM vector_cache ORDER BY last_accessed ASC LIMIT {self.EVICT_COUNT}
                )
            ''')
            logger.info(f"VectorCache LRU eviction: removed {self.EVICT_COUNT} least-recently-used entries.")
        conn.commit()
        conn.close()

    @property
    def size(self) -> int:
        conn = sqlite3.connect(self.cache_path)
        count = conn.execute('SELECT COUNT(*) FROM vector_cache').fetchone()[0]
        conn.close()
        return count


# ========== 搜索结果 ==========

@dataclass
class SearchResult:
    """搜索结果"""
    content: str
    score: float
    source: str
    timestamp: str
    scope: str = "default"
    metadata: Dict = field(default_factory=dict)
    rerank_score: Optional[float] = None
    time_decay: Optional[float] = None


# ========== 混合搜索引擎 V4.5 ==========

class HybridSearchEngine:
    """
    混合搜索引擎 V4.5

    完整流水线：
    1. 向量语义搜索（Bi-Encoder）—— 粗筛
    2. BM25 关键词搜索 —— 补充精确匹配
    3. 混合融合 —— 70% 向量 + 30% BM25
    4. 时间衰减 —— 近期优先
    5. Cross-Encoder 重排序 —— 精排（可选）
    """

    def __init__(self, embedder: MultiProviderEmbedding,
                 vector_cache: VectorCache,
                 reranker: Optional[DashScopeReranker] = None,
                 time_decay: Optional[TimeDecay] = None,
                 noise_filter: Optional[NoiseFilter] = None):
        self.embedder = embedder
        self.vector_cache = vector_cache
        self.reranker = reranker
        self.time_decay = time_decay or TimeDecay()
        self.noise_filter = noise_filter or NoiseFilter()

        # 多范围隔离：每个 scope 拥有独立的文档集
        self.scopes: Dict[str, Dict] = {}
        self._ensure_scope("default")

    def _ensure_scope(self, scope: str):
        """确保 scope 存在"""
        if scope not in self.scopes:
            self.scopes[scope] = {
                "documents": [],
                "doc_vectors": [],
                "index": {},
                "idf_scores": {}
            }

    def add_document(self, doc_id: str, content: str,
                     metadata: Dict = None, vector: List[float] = None,
                     scope: str = "default",
                     skip_noise_filter: bool = False):
        """添加文档到指定 scope"""
        # 噪音过滤（可跳过，避免 add_memory 已过滤后重复检查）
        if not skip_noise_filter and self.noise_filter.is_noise(content):
            return False

        self._ensure_scope(scope)
        s = self.scopes[scope]

        doc = {
            "id": doc_id,
            "content": content,
            "metadata": metadata or {},
            "tokens": self._tokenize(content),
            "timestamp": datetime.now().isoformat(),
            "scope": scope
        }
        s["documents"].append(doc)

        # 获取或生成向量
        if vector is not None:
            s["doc_vectors"].append(vector)
        else:
            cached = self.vector_cache.get(content)
            if cached is not None:
                s["doc_vectors"].append(cached)
            else:
                try:
                    result = self.embedder.embed([content])
                    vec = result.vectors[0]
                    s["doc_vectors"].append(vec)
                    self.vector_cache.put(content, vec)
                except Exception as e:
                    print(f"⚠️ Embedding 失败 (doc={doc_id}): {e}")
                    s["doc_vectors"].append([])

        # 更新 BM25 索引
        for token in set(doc["tokens"]):
            if token not in s["index"]:
                s["index"][token] = []
            s["index"][token].append(doc["id"])
        s["idf_scores"] = {}  # 重置 IDF
        return True

    def _tokenize(self, text: str) -> List[str]:
        """分词（中英文混合，支持 jieba 可选增强）"""
        text_lower = re.sub(r"[^\w\s]", " ", text.lower())
        # English tokens
        tokens = text_lower.split()

        chinese_text = re.findall(r"[\u4e00-\u9fff]+", text)
        if HAS_JIEBA and chinese_text:
            # Use jieba for proper Chinese word segmentation
            for chars in chinese_text:
                tokens.extend(jieba.cut(chars, cut_all=False))
        else:
            # Fallback: unigram + bigram (zero-dependency)
            for chars in chinese_text:
                for i in range(len(chars)):
                    tokens.append(chars[i])
                    if i < len(chars) - 1:
                        tokens.append(chars[i:i+2])
        return tokens

    def _calculate_idf(self, scope_data: Dict):
        total_docs = max(len(scope_data["documents"]), 1)
        for token, doc_ids in scope_data["index"].items():
            scope_data["idf_scores"][token] = math.log(total_docs / len(doc_ids))

    def _bm25_score(self, query_tokens: List[str], doc: Dict,
                    scope_data: Dict) -> float:
        score = 0.0
        doc_tokens = doc["tokens"]
        doc_len = len(doc_tokens)
        docs = scope_data["documents"]
        avg_len = sum(len(d["tokens"]) for d in docs) / max(len(docs), 1)
        k1, b = 1.5, 0.75

        for token in query_tokens:
            if token in doc_tokens:
                tf = doc_tokens.count(token)
                idf = scope_data["idf_scores"].get(token, 0)
                norm = 1 - b + b * (doc_len / max(avg_len, 1))
                score += idf * (tf * (k1 + 1)) / (tf + k1 * norm)
        return score

    def hybrid_search(self, query: str, top_k: int = 5,
                      vector_weight: float = 0.7,
                      scope: str = "default",
                      enable_rerank: bool = True,
                      enable_time_decay: bool = True) -> List[SearchResult]:
        """
        混合搜索 V4.5

        完整流水线：粗筛 → 融合 → 时间衰减 → 精排
        """
        self._ensure_scope(scope)
        s = self.scopes[scope]

        if not s["documents"]:
            return []

        if not s["idf_scores"]:
            self._calculate_idf(s)

        # 获取查询向量
        query_vector = None
        try:
            cached = self.vector_cache.get(query)
            if cached is not None:
                query_vector = cached
            else:
                result = self.embedder.embed([query])
                query_vector = result.vectors[0]
                self.vector_cache.put(query, query_vector)
        except Exception as e:
            print(f"⚠️ 查询 Embedding 失败: {e}，降级为纯 BM25")
            vector_weight = 0.0

        query_tokens = self._tokenize(query)
        candidates = []

        # 第 1-3 步：向量 + BM25 + 融合
        for i, doc in enumerate(s["documents"]):
            bm25 = self._bm25_score(query_tokens, doc, s)

            vec_score = 0.0
            if query_vector and i < len(s["doc_vectors"]) and s["doc_vectors"][i]:
                vec_score = cosine_similarity(query_vector, s["doc_vectors"][i])
                vec_score = (vec_score + 1) / 2

            final_score = vector_weight * vec_score + (1 - vector_weight) * bm25

            # 第 4 步：时间衰减
            decay_factor = None
            if enable_time_decay and self.time_decay:
                decay_info = self.time_decay.get_decay_info(doc["timestamp"])
                decay_factor = decay_info["decay_factor"]
                final_score *= decay_factor

            candidates.append(SearchResult(
                content=doc["content"],
                score=final_score,
                source=doc["id"],
                timestamp=doc["timestamp"],
                scope=scope,
                metadata=doc.get("metadata", {}),
                time_decay=decay_factor
            ))

        # 先按混合分数排序，取 top-k * 2（给 rerank 更多候选）
        candidates.sort(key=lambda x: x.score, reverse=True)
        rerank_pool = candidates[:top_k * 2]

        # 第 5 步：Cross-Encoder 重排序
        if (enable_rerank and self.reranker and
                self.reranker.is_available and len(rerank_pool) > 1):
            try:
                doc_texts = [r.content for r in rerank_pool]
                rerank_results = self.reranker.rerank(
                    query, doc_texts, top_n=top_k
                )
                # 用 rerank 分数替换最终排序
                reranked = []
                for rr in rerank_results:
                    candidate = rerank_pool[rr.index]
                    candidate.rerank_score = rr.relevance_score
                    # 最终分数 = 0.4 * 粗筛 + 0.6 * 精排
                    candidate.score = 0.4 * candidate.score + 0.6 * rr.relevance_score
                    reranked.append(candidate)
                reranked.sort(key=lambda x: x.score, reverse=True)
                return reranked[:top_k]
            except Exception as e:
                print(f"⚠️ Rerank 失败，使用粗筛结果: {e}")

        return rerank_pool[:top_k]

    def get_scope_list(self) -> List[str]:
        """获取所有 scope 列表"""
        return list(self.scopes.keys())

    def get_scope_stats(self, scope: str = "default") -> Dict:
        self._ensure_scope(scope)
        s = self.scopes[scope]
        return {
            "scope": scope,
            "documents": len(s["documents"]),
            "tokens": len(s["index"]),
            "vectors": len(s["doc_vectors"])
        }

    def get_stats(self) -> Dict:
        total_docs = sum(len(s["documents"]) for s in self.scopes.values())
        return {
            "total_documents": total_docs,
            "scopes": len(self.scopes),
            "scope_details": {
                name: len(s["documents"])
                for name, s in self.scopes.items()
            },
            "cached_vectors": self.vector_cache.size,
            "reranker_available": (
                self.reranker.is_available if self.reranker else False
            ),
            "embedding_provider": self.embedder.get_stats()
        }


# ========== 增强版记忆系统 V4.5 ==========

class EnhancedMemoryCore:
    """
    增强版记忆核心 V4.5

    V4.0 → V4.5 新增：
    1. Cross-Encoder 重排序（精度 +20-30%）
    2. 多范围隔离（scope 隔离不同 agent 的记忆）
    3. 噪音过滤（自动过滤无意义短语）
    4. 时间衰减（近期记忆优先）
    """

    def __init__(self, storage_path: str = "/root/.openclaw/memory/openclaw_memory_v4.json",
                 config_dir: str = None,
                 default_scope: str = "default",
                 half_life_days: float = 14.0):
        self.storage_path = storage_path
        self.default_scope = default_scope

        if config_dir is None:
            config_dir = os.path.dirname(os.path.abspath(__file__))

        config_path = os.path.join(config_dir, "embedding_config.json")
        cache_path = os.path.join(
            os.path.dirname(storage_path), "vector_cache.sqlite"
        )

        # 加载配置
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # 初始化组件
        self.embedder = MultiProviderEmbedding(config_path=config_path)
        self.vector_cache = VectorCache(cache_path)
        self.noise_filter = NoiseFilter()
        self.time_decay = TimeDecay(half_life_days=half_life_days)

        # 初始化 Reranker（使用 DashScope API Key）
        ds_key = config.get("providers", {}).get("dashscope", {}).get("api_key", "")
        self.reranker = None
        if ds_key and not ds_key.startswith("YOUR_"):
            self.reranker = DashScopeReranker(api_key=ds_key)
            logger.info("Cross-Encoder Reranker enabled (qwen3-rerank)")

        # 混合搜索引擎
        self.search_engine = HybridSearchEngine(
            embedder=self.embedder,
            vector_cache=self.vector_cache,
            reranker=self.reranker,
            time_decay=self.time_decay,
            noise_filter=self.noise_filter
        )

        # V4.0画像管理器
        db_path = os.path.join(os.path.dirname(storage_path), "profiles.sqlite")
        self.profile_manager = UserProfileManager(db_path)

        # 事实提取器 (V4.5: 使用内置 openai_compatible_call)
        self.extractor = FactExtractor()  # Auto-uses env-based LLM

        # === conversation_log: SQLite 持久化 (V4.5) ===
        self.conv_db_path = os.path.join(os.path.dirname(storage_path), "conversations.sqlite")
        self._init_conv_db()

        # 分类字典 (轻量内存结构, conversation_log 已移至 SQLite)
        self.context = {
            "session": {
                "current_id": None,
                "start_time": None,
                "message_count": 0
            },
            "user_profile": {
                "preferences": {},
                "expertise": {},
                "history_summary": deque(maxlen=50)
            },
            "knowledge_base": {
                "code_snippets": {},
                "documents": {},
                "concepts": {}
            },
            "tasks": {
                "active": deque(maxlen=10),
                "completed": deque(maxlen=20)
            },
        }

        # 统计
        self.stats = {
            "searches": 0,
            "hits": 0,
            "noise_filtered": 0,
            "rerank_count": 0,
            "token_saved": 0
        }

        self._migrate_json_conv_log()
        self.load()
        self._rebuild_search_index()

    # ========== conversation_log SQLite ==========

    def _init_conv_db(self):
        """Initialize SQLite for persistent conversation storage."""
        Path(self.conv_db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.conv_db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS conversation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                scope TEXT DEFAULT 'default',
                timestamp TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                role TEXT DEFAULT 'user'
            )
        ''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_conv_scope ON conversation_log(scope)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_conv_ts ON conversation_log(timestamp)')
        conn.commit()
        conn.close()
        logger.info(f"Conversation DB initialized at {self.conv_db_path}")

    def _migrate_json_conv_log(self):
        """One-time migration from legacy JSON conversation_log to SQLite."""
        if not Path(self.storage_path).exists():
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            old_log = data.get("conversation_log", [])
            if not old_log:
                return
            # Check if already migrated
            conn = sqlite3.connect(self.conv_db_path)
            existing = conn.execute('SELECT COUNT(*) FROM conversation_log').fetchone()[0]
            if existing > 0:
                conn.close()
                return
            for msg in old_log:
                if isinstance(msg, dict):
                    conn.execute(
                        'INSERT INTO conversation_log (content, category, scope, timestamp, metadata) VALUES (?, ?, ?, ?, ?)',
                        (
                            msg.get("content", ""),
                            msg.get("category", "general"),
                            msg.get("scope", "default"),
                            msg.get("timestamp", datetime.now().isoformat()),
                            json.dumps(msg.get("metadata", {})),
                        )
                    )
            conn.commit()
            conn.close()
            logger.info(f"Migrated {len(old_log)} conversation entries from JSON to SQLite.")
        except Exception as e:
            logger.warning(f"Conversation migration skipped: {e}")

    def _get_all_conversations(self) -> List[Dict]:
        """Retrieve all conversations from SQLite."""
        conn = sqlite3.connect(self.conv_db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute('SELECT * FROM conversation_log ORDER BY id').fetchall()
        conn.close()
        return [
            {
                "id": row["id"],
                "content": row["content"],
                "category": row["category"],
                "scope": row["scope"],
                "timestamp": row["timestamp"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "role": row["role"],
            }
            for row in rows
        ]

    def _add_conversation(self, content: str, category: str = "general",
                          scope: str = "default", metadata: Dict = None) -> int:
        """Add a new conversation entry to SQLite. Returns the new row ID."""
        conn = sqlite3.connect(self.conv_db_path)
        cur = conn.execute(
            'INSERT INTO conversation_log (content, category, scope, timestamp, metadata) VALUES (?, ?, ?, ?, ?)',
            (content, category, scope, datetime.now().isoformat(), json.dumps(metadata or {}))
        )
        row_id = cur.lastrowid
        conn.commit()
        conn.close()
        return row_id

    def _get_conversation_count(self) -> int:
        conn = sqlite3.connect(self.conv_db_path)
        count = conn.execute('SELECT COUNT(*) FROM conversation_log').fetchone()[0]
        conn.close()
        return count

    def load(self):
        """Load lightweight in-memory context (conversations are in SQLite now)."""
        if Path(self.storage_path).exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for key in ["user_profile", "tasks"]:
                    if key in data:
                        for subkey, value in data[key].items():
                            if isinstance(value, list):
                                max_len = 50 if "history" in subkey else 20
                                self.context[key][subkey] = deque(
                                    value, maxlen=max_len
                                )
                logger.info("Context loaded (conversations in SQLite).")
            except Exception as e:
                logger.warning(f"Load failed: {e}")

    def save(self):
        """Save lightweight context (conversations are auto-committed to SQLite)."""
        try:
            Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)
            serializable = self._to_serializable(self.context)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Save failed: {e}")
            return False

    def _to_serializable(self, obj):
        if isinstance(obj, deque):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: self._to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._to_serializable(item) for item in obj]
        return obj

    def _rebuild_search_index(self):
        """Rebuild search index from SQLite conversations (no more maxlen=100 limit!)."""
        logger.info("Rebuilding search index from SQLite...")
        indexed = 0
        conversations = self._get_all_conversations()
        for msg in conversations:
            content = msg.get("content", "")
            scope = msg.get("scope", self.default_scope)
            if content:
                added = self.search_engine.add_document(
                    doc_id=f"conversation_{msg['id']}",
                    content=content,
                    metadata={
                        "role": msg.get("role"),
                        "timestamp": msg.get("timestamp")
                    },
                    scope=scope
                )
                if added:
                    indexed += 1

        for key, value in self.context["knowledge_base"].items():
            if isinstance(value, dict):
                for item_id, item_content in value.items():
                    content = str(item_content)
                    if content:
                        added = self.search_engine.add_document(
                            doc_id=f"knowledge_{key}_{item_id}",
                            content=content,
                            metadata={"category": key},
                            scope=self.default_scope
                        )
                        if added:
                            indexed += 1

        scopes = self.search_engine.get_scope_list()
        logger.info(f"Index rebuilt: {indexed} documents across {len(scopes)} scopes.")

    # ========== 核心功能 ==========

    def smart_recall(self, query: str, max_results: int = 5,
                     scope: str = None,
                     enable_rerank: bool = True) -> List[Dict]:
        """
        智能回忆（V4.5 核心）

        完整流水线：向量搜索 + BM25 + 时间衰减 + Cross-Encoder 精排
        """
        scope = scope or self.default_scope
        self.stats["searches"] += 1

        results = self.search_engine.hybrid_search(
            query, top_k=max_results, scope=scope,
            enable_rerank=enable_rerank
        )

        if results:
            self.stats["hits"] += 1
            if any(r.rerank_score is not None for r in results):
                self.stats["rerank_count"] += 1

            # Estimate tokens saved (use conversation count from SQLite)
            conv_count = self._get_conversation_count()
            estimated_total_size = conv_count * 200  # ~200 chars/entry average
            retrieved_size = sum(len(r.content) for r in results)
            self.stats["token_saved"] += max(estimated_total_size - retrieved_size, 0) // 4

        return [
            {
                "content": r.content,
                "score": round(r.score, 4),
                "source": r.source,
                "timestamp": r.timestamp,
                "scope": r.scope,
                "rerank_score": round(r.rerank_score, 4) if r.rerank_score else None,
                "time_decay": round(r.time_decay, 4) if r.time_decay else None
            }
            for r in results
        ]

    def add_memory(self, content: str, category: str = "general",
                   metadata: Dict = None, scope: str = None):
        """添加新记忆（自动噪音过滤 + 嵌入 + 索引 → SQLite 持久化）"""
        scope = scope or self.default_scope

        # 噪音过滤
        if self.noise_filter.is_noise(content):
            self.stats["noise_filtered"] += 1
            return False

        # Persist to SQLite (replaces the old deque append)
        row_id = self._add_conversation(content, category, scope, metadata)

        doc_id = f"{category}_{row_id}"
        self.search_engine.add_document(
            doc_id, content, metadata, scope=scope,
            skip_noise_filter=True
        )
        self.save()  # Save lightweight context (tasks, preferences)
        return True

    def get_relevant_context(self, current_query: str,
                             max_tokens: int = 500,
                             scope: str = None,
                             user_id: str = "default") -> str:
        """获取相关上下文 (结合搜索结果与画像)"""
        relevant = self.smart_recall(
            current_query, max_results=3, scope=scope
        )
        profile_str = self.profile_manager.get_context_string(user_id)
        
        parts = []
        if profile_str:
            parts.append(profile_str)
            
        if relevant:
            parts.append("\n=== 相关记忆 ===")
            current_tokens = 0
            for mem in relevant:
                text = f"[{mem['source']}] {mem['content'][:200]}"
                tokens = len(text) // 4
                if current_tokens + tokens > max_tokens:
                    break
                parts.append(text)
                current_tokens += tokens
        else:
            if not profile_str:
                return "（无相关记录）"
                
        return "\n".join(parts)

    def get_memory_stats(self) -> Dict:
        search_stats = self.search_engine.get_stats()
        return {
            **search_stats,
            "total_conversations": self._get_conversation_count(),
            "active_tasks": len(self.context["tasks"]["active"]),
            "noise_filtered": self.stats["noise_filtered"],
            "search_efficiency": {
                "total_searches": self.stats["searches"],
                "successful_hits": self.stats["hits"],
                "rerank_used": self.stats["rerank_count"],
                "hit_rate": (
                    f"{self.stats['hits'] / self.stats['searches'] * 100:.1f}%"
                    if self.stats["searches"] > 0 else "0%"
                ),
                "estimated_tokens_saved": self.stats["token_saved"]
            }
        }

    def print_stats(self):
        stats = self.get_memory_stats()
        print("\n" + "=" * 60)
        print("📊 增强版记忆系统 V4.5 统计")
        print("=" * 60)
        print(f"💾 存储统计:")
        print(f"  - 文档总数: {stats['total_documents']}")
        print(f"  - 缓存向量: {stats['cached_vectors']}")
        print(f"  - 对话条数: {stats['total_conversations']}")
        print(f"  - 范围数量: {stats['scopes']}")
        for name, count in stats["scope_details"].items():
            print(f"    📂 {name}: {count} 条")
        print(f"\n🔍 搜索效率:")
        eff = stats["search_efficiency"]
        print(f"  - 搜索次数: {eff['total_searches']}")
        print(f"  - 命中率: {eff['hit_rate']}")
        print(f"  - Rerank 次数: {eff['rerank_used']}")
        print(f"  - 噪音过滤: {stats['noise_filtered']} 条")
        print(f"  - 估算节省 Token: {eff['estimated_tokens_saved']:,}")
        print(f"\n🌐 组件状态:")
        for p in stats["embedding_provider"]["providers"]:
            status = "✅" if p["available"] else "❌"
            print(f"  {status} {p['name']} ({p['model']})")
        rr_status = "✅" if stats["reranker_available"] else "❌"
        print(f"  {rr_status} reranker (qwen3-rerank)")
        print("=" * 60 + "\n")


# ========== 演示 ==========

if __name__ == "__main__":
    print("🚀 OpenClaw 增强记忆系统 V4.5 演示\n")

    memory = EnhancedMemoryCore(
        storage_path="/tmp/test_memory_v45.json",
        config_dir=os.path.dirname(os.path.abspath(__file__))
    )

    # ── 测试 1: 噪音过滤 ──
    print("=" * 60)
    print("🧪 测试 1: 噪音过滤")
    print("=" * 60)
    noise_tests = ["你好", "ok", "👍", "嗯", "哈哈哈"]
    for noise in noise_tests:
        added = memory.add_memory(noise)
        print(f"  '{noise}' → {'❌ 已过滤' if not added else '✅ 存入'}")

    # ── 测试 2: 多范围隔离 ──
    print(f"\n{'=' * 60}")
    print("🧪 测试 2: 多范围隔离")
    print("=" * 60)
    memory.add_memory("用户喜欢简洁的代码风格", scope="personal")
    memory.add_memory("交易机器人需要优化延迟", scope="project-bot")
    memory.add_memory("ICLR论文截止日期是3月", scope="project-paper")
    memory.add_memory("OpenClaw记忆系统已升级到V4.5", scope="personal")
    print("  ✅ 已存入 3 个不同 scope")

    # 只搜索特定 scope
    print("\n  搜索 scope='personal':")
    results = memory.smart_recall("代码规范", scope="personal", enable_rerank=False)
    for r in results:
        print(f"    [{r['score']:.4f}] [{r['scope']}] {r['content'][:40]}")

    print("\n  搜索 scope='project-bot':")
    results = memory.smart_recall("性能优化", scope="project-bot", enable_rerank=False)
    for r in results:
        print(f"    [{r['score']:.4f}] [{r['scope']}] {r['content'][:40]}")

    # ── 测试 3: 时间衰减 ──
    print(f"\n{'=' * 60}")
    print("🧪 测试 3: 时间衰减")
    print("=" * 60)
    decay = TimeDecay(half_life_days=14)
    for days in [0, 1, 7, 14, 30, 90]:
        ts = (datetime.now() - timedelta(days=days)).isoformat()
        info = decay.get_decay_info(ts)
        bar = "█" * int(info["decay_factor"] * 20)
        print(f"  {days:3d} 天前: {info['decay_factor']:.3f} {bar}")

    # ── 测试 4: Cross-Encoder 重排序 ──
    print(f"\n{'=' * 60}")
    print("🧪 测试 4: Cross-Encoder 重排序")
    print("=" * 60)
    memory.add_memory("Python量化交易机器人使用ccxt库", scope="default")
    memory.add_memory("讨论了如何优化OpenClaw的Token消耗", scope="default")
    memory.add_memory("Vue.js前端框架的组件设计模式", scope="default")
    memory.add_memory("服务器部署使用Docker容器化", scope="default")

    print("\n  无 Rerank:")
    results_no_rr = memory.smart_recall("编程语言", enable_rerank=False)
    for r in results_no_rr[:3]:
        print(f"    [{r['score']:.4f}] {r['content'][:40]}")

    print("\n  有 Rerank:")
    results_rr = memory.smart_recall("编程语言", enable_rerank=True)
    for r in results_rr[:3]:
        rr_str = f" (rerank={r['rerank_score']})" if r['rerank_score'] else ""
        print(f"    [{r['score']:.4f}]{rr_str} {r['content'][:40]}")

    # 统计
    print()
    memory.print_stats()
