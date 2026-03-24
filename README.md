<div align="center">
  <img src="https://img.shields.io/badge/OpenClaw-Memory_V4-FF4B4B?style=for-the-badge&logo=ai" alt="OpenClaw Memory V4" />
  <h1>🧠 OpenClaw Memory V4 (Local Supermemory)</h1>
  <p><strong>The Universal, Privacy-First MCP Memory Plugin for AI Agents</strong></p>

  [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
  [![MCP Support](https://img.shields.io/badge/Protocol-MCP_Stable-blue.svg)](https://modelcontextprotocol.io/)
  [![Local First](https://img.shields.io/badge/Cloud-Zero_API_Required-orange.svg)]()
</div>

> **One Memory Engine to Rule Them All.** 
> V4 flawlessly fuses the semantic precision of **OpenClaw V3**, the persistent sandbox architecture of **mem9**, and the autonomous profiling intelligence of **Supermemory**—all running 100% locally on your machine via the standard **Model Context Protocol (MCP)**.

---

## 🌟 Why OpenClaw Memory V4?

Most AI memory solutions force you to choose between vendor lock-in, expensive cloud APIs, or dumb keyword search. **V4 changes the game:**

*   **🔌 Universal MCP Support**: Use your local memory not just in OpenClaw, but in **Cursor**, **Claude Desktop**, **Windsurf**, or any MCP-compatible client. Your memory follows *you*, not the framework.
*   **🛡️ 100% Local (Zero Cloud API)**: Your data stays on your hard drive (SQLite + JSON). No data is sent to third-party memory servers. It's your personal, private vault.
*   **🧠 "Supermemory" Profiling**: Automatically extracts `STATIC` traits (e.g., "User is a Python dev") and `DYNAMIC` states (e.g., "Busy this week") with self-expiring TTLs.
*   **🎯 Hybrid Search V4.5**: True Vector Semantic Search (70%) + BM25 Keyword Search (30%) + Cross-Encoder Reranking. Finds exactly what you mean, not just what you said.

---

## ⚔️ The Ultimate Feature Fusion

| Feature | `mem9` | `Supermemory` | **OpenClaw V4** |
|:---|:---:|:---:|:---:|
| **Zero Cloud API (Local SQLite)**| ❌ | ❌ | ✅ **Yes** |
| **Universal MCP Plugin** | ❌ | ❌ | ✅ **Yes** |
| **Auto-Expiring Context (TTL)**| ❌ | ✅ | ✅ **Yes** |
| **Structured User Profiling** | ❌ | ✅ | ✅ **STATIC & DYNAMIC** |
| **Hybrid Search (Vector+BM25)**| ✅ | ✅ | ✅ **Self-Hosted** |
| **Open Source** | ✅ | Partial | ✅ **MIT License** |

---

## 🚀 Quick Start (MCP Plugin)

Turn your local machine into a persistent brain for any AI assistant in under 1 minute.

### 1. Installation

```bash
git clone https://github.com/sunhonghua1/openclaw-memory-v3.git
cd openclaw-memory-v3
pip install -e .
```

### 2. Connect to Claude Desktop / Cursor

Open your MCP config file (e.g., `claude_desktop_config.json`) and add:

```json
{
  "mcpServers": {
    "local-supermemory": {
      "command": "python3",
      "args": ["<path-to-repo>/mcp_memory_server.py"],
      "env": {
        "MEMORY_STORAGE_PATH": "./memory_v4.json",
        "PROFILES_DB_PATH": "./profiles.sqlite"
        // Optional: Add local LLM API for Auto-Fact Extraction
        // "LLM_API_KEY": "your-key",
        // "LLM_BASE_URL": "http://localhost:11434/v1" 
      }
    }
  }
}
```

---

## 🛠️ MCP Tools Reference

Once connected, your AI assistant gains access to these powerful native capabilities:

| 🔧 Tool Name | 📝 Description |
|---|---|
| `search_memory` | Hybrid search combining semantic meaning, exact keywords, and your active user profile. |
| `get_user_profile` | Injects your permanent traits and current context into the LLM prompt. |
| `add_user_fact` | Teach the AI a `STATIC` (permanent) or `DYNAMIC` (temporary) fact about you. |
| `extract_facts` | *(Requires LLM)* The AI autonomously reads the chat and distills new facts into SQLite. |
| `list_users` | View all user profiles and their active fact counts. |

---

## 🏗️ Architecture Visualization

```mermaid
graph TD
    classDef client fill:#4b9de3,stroke:#333,stroke-width:2px,color:#fff;
    classDef core fill:#ff4b4b,stroke:#333,stroke-width:2px,color:#fff;
    classDef db fill:#2eaf7d,stroke:#333,stroke-width:2px,color:#fff;

    ClientA([Cursor IDE]):::client
    ClientB([Claude Desktop]):::client
    ClientC([OpenClaw Agent]):::client

    ClientA & ClientB & ClientC -->|MCP Protocol| MCP[mcp_memory_server.py]:::core
    
    MCP --> Search[Hybrid Vector Engine]:::core
    MCP --> Profiles[(SQLite Profiles)]:::db
    MCP --> Ex[Autonomous Extractor]:::core
    
    Search -->|70% Vector| VectorDB[(Local Vector Cache)]:::db
    Search -->|30% BM25| KeywordDB[(In-Memory BM25)]:::db
    
    Ex -.->|Optional Local/API LLM| LLM{LLM Inference}
    LLM -.->|Distilled Facts| Profiles
```

---

<div align="center">
  <i>Engineered for Data Sovereignty by <a href="https://github.com/sunhonghua1">sunhonghua</a></i><br>
  <i>Powered by the <b>Foxbot Engine</b> & <b>OpenClaw Community</b></i>
</div>
