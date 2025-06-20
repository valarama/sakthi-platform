# Sakthi Platform

The **MCP Language (Sakthi)** is a specialized framework for designing and executing **Model Context Protocols (MCP)** in Natural Language Processing (NLP). It provides a structured approach to managing context-aware workflows, semantic parsing, and integrating with large language models (LLMs) for decision-making and language transformation.

---

## 🚀 Features

- **MCP Core Engine**  
  - Define and execute MCP workflows using Sakthi's structured syntax.
  - Context-aware metadata handling.
  - Multi-format input support (JSON, SQL, natural language).

- **NLP Modules**  
  - Intent recognition and entity extraction.
  - Contextual sentiment analysis and semantic similarity scoring.
  - Modular design for easy integration with existing systems.

- **GenAI Integration**  
  - Supports LLM-based workflows with custom prompt engineering.
  - DeepSeek integration for local LLM inference.

- **Web Interface**  
  - Built with Next.js for seamless interaction with MCP workflows.
  - Real-time visualization of processed data.

- **Deployment Ready**  
  - Dockerized services and Kubernetes configurations for scalable deployment.

---

## 📂 Project Structure

```plaintext
sakthi-platform/
├── README.md                    ← Main project documentation
├── .env                         ← Environment configuration
├── .gitignore                   ← Git ignore file
├── requirements.txt             ← Python dependencies
├── docker-compose.yml           ← Docker configuration
├── start-dev.sh                 ← Quick start script
├── deploy.sh                    ← Deployment script
│
├── backend/                     ← FastAPI Backend
│   ├── __init__.py
│   ├── main.py                  ← Main FastAPI app
│   └── api/
│       └── __init__.py
│
├── sakthi-language/             ← Core Sakthi Engine
│   ├── __init__.py
│   └── core.py                  ← Sakthi language processor
│
├── document-processor/          ← Document handling
│   ├── __init__.py
│   ├── processor.py             ← Multi-format processor
│   └── Dockerfile
│
├── genai-modeling-agent/        ← AI Agents
│   ├── __init__.py
│   ├── agent_system.py          ← AutoGen + LangGraph
│   └── Dockerfile
│
├── sakthi-llm-integration/      ← LLM Integration
│   ├── __init__.py
│   └── llm_provider.py          ← DeepSeek LLM integration
│
├── web-interface/               ← Next.js Frontend
│   ├── package.json
│   ├── next.config.js
│   ├── tailwind.config.js
│   ├── Dockerfile
│   ├── pages/
│   │   └── index.js
│   ├── components/
│   │   └── Dashboard.jsx
│   └── styles/
│       └── globals.css
│
├── deployment/                  ← Deployment configs
│   ├── nginx.conf
│   ├── init-db.sql
│   └── kubernetes/
│       └── sakthi-platform.yaml
│
├── config/                      ← Configuration files
│   ├── logging.conf
│   └── settings.json
│
├── docs/                        ← Documentation
│   ├── architecture.md
│   ├── api-guide.md
│   └── user-manual.md
│
├── tests/                       ← Test files
│   ├── __init__.py
│   └── test_basic.py
│
├── logs/                        ← Log files (auto-created)
├── uploads/                     ← File uploads (auto-created)
├── storage/                     ← Storage (auto-created)
└── chromadb/                    ← Vector DB (auto-created)

# 📋 Copy Artifact Contents to Files

Use the following mapping to populate your project files with the artifact content:

| **Artifact Name**                     | **File Location**                         |
|---------------------------------------|-------------------------------------------|
| Sakthi Language - Core Implementation | `sakthi-language/core.py`                 |
| Document Processing Layer             | `document-processor/processor.py`         |
| GenAI Modeling Agent                  | `genai-modeling-agent/agent_system.py`    |
| DeepSeek LLM Integration              | `sakthi-llm-integration/llm_provider.py`  |
| FastAPI Backend                       | `backend/main.py`                         |
| Next.js Web Interface                 | `web-interface/components/Dashboard.jsx`  |
| Environment Configuration             | `.env`                                    |
| Deployment Configuration              | `docker-compose.yml`                      |
| Deployment Script                     | `deploy.sh`                               |


