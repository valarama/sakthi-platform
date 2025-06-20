# 🌟 **_Sakthi Platform_**

The **Sakthi Platform** is an **enterprise-grade, AI-powered system** designed to transform natural language inputs into actionable outputs. Powered by the **MCP Language (Sakthi)**, **DeepSeek LLM**, **ChromaDB**, and **LangGraph**, it delivers **scalable, context-aware solutions** for schema transformation, document processing, and workflow orchestration. With real-time monitoring, dynamic rule processing, and multi-format outputs, it’s built for modern enterprise needs.

---

## 🚀 **_Key Features_**

- 🌐 **Natural Language Interface**: Process tasks with plain English, e.g.:
  - "Convert Oracle HR schema to BigQuery."
  - "Extract revenue data from this PDF."
  - "Monitor competitor pricing daily."
- 🤖 **AI-Powered Processing**: Uses **DeepSeek LLM** (e.g., DeepSeek-Coder-6.7B, Codestral-22B) for intent recognition, SQL generation, and data transformation.
- 🧠 **Context-Aware Workflows**: Leverages **ChromaDB** for Retrieval-Augmented Generation (RAG) to deliver smarter results.
- ⚙️ **Dynamic Rule Processing**: Applies rules from `rules.csv` for SQL conditions and validations.
- 📦 **Batch Processing**: Handles up to 1000 target fields with `EnhancedTargetProcessor`.
- 🏢 **Enterprise-Grade Deployment**: Dockerized, Kubernetes-ready, with Nginx, WebSocket updates, and LangGraph monitoring.
- 📄 **Multi-format Outputs**: Generates JSON, SQL, CSV, or API-ready formats.

---

## 🔄 **_How It Works_**

1. **Input**: Users submit queries (e.g., "Convert schema") or upload documents (PDF, XLSX, CSV).
2. **Processing**:
   - **Sakthi Language Parser** identifies intent via DeepSeek LLM.
   - **ChromaDB** retrieves historical context using RAG.
   - **LangGraph** orchestrates workflows and monitors progress.
   - **EnhancedTargetProcessor** processes target fields in batches.
3. **Output**: Delivers structured data (JSON, SQL, CSV) via APIs or saved to `output/`.
4. **Storage**: Stores results and context in **ChromaDB** for future use.

---

## 📂 **_Project Structure_**

```plaintext
sakthi-platform/
├── sakthi-language/             # Core Sakthi Engine
│   ├── core.py                  # Language processor
│   └── __init__.py
├── document-processor/          # Multi-format document handler
│   ├── processor.py             # Processes PDF, XLSX, CSV
│   ├── Dockerfile
│   └── __init__.py
├── genai-modeling-agent/        # AI agents with AutoGen + LangGraph
│   ├── agent_system.py          # Manages LLM workflows
│   ├── Dockerfile
│   └── __init__.py
├── backend/                     # FastAPI backend
│   ├── main.py                  # API endpoints
│   ├── api/
│   │   └── __init__.py
│   ├── requirements.txt
│   └── Dockerfile
├── web-interface/               # Next.js frontend
│   ├── pages/
│   ├── components/
│   │   └── Dashboard.jsx        # Interactive dashboard
│   ├── package.json
│   └── Dockerfile
├── deployment/                  # Deployment configurations
│   ├── docker-compose.yml       # Docker setup
│   ├── kubernetes/
│   │   └── sakthi-platform.yaml # Kubernetes manifests
│   └── nginx.conf               # Nginx proxy configuration
├── config/                      # Configuration files
│   ├── prompt_template.json     # Dynamic LLM prompt templates
│   └── .env                     # Environment variables
├── docs/                        # Documentation
├── logs/                        # Log storage
├── uploads/                     # File uploads (PDF, XLSX, CSV)
├── output/                      # Generated outputs (JSON, SQL, CSV)
├── storage/                     # Data storage
└── chromadb/                    # Vector database for RAG

## 📋 Artifact-to-File Mapping

| Artifact Name                           | File Location                                  |
|-----------------------------------------|------------------------------------------------|
| Sakthi Language - Core Implementation   | sakthi-language/core.py                        |
| Document Processing Layer               | document-processor/processor.py                |
| GenAI Modeling Agent                    | genai-modeling-agent/agent_system.py           |
| DeepSeek LLM Integration                | sakthi-llm-integration/llm_provider.py         |
| FastAPI Backend                         | backend/main.py                                |
| Next.js Web Interface                   | web-interface/components/Dashboard.jsx         |
| Environment Configuration               | config/.env                                    |
| Prompt Template                         | config/prompt_template.json                    |
| Deployment Configuration                | deployment/docker-compose.yml                  |
| Deployment Script                       | deployment/deploy.sh                           |
| SQL Validation Script                   | scripts/sql_validator.py                       |

# 🌟 **Sakthi Platform Deployment Guide**

## 🚀 Deployment Instructions

### 🌍 Local Development Setup

1. Clone the repository:
  
   git clone https://github.com/your-org/sakthi-platform.git
   cd sakthi-platform
   ```

2. Set up the Python environment:

   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r backend/requirements.txt
   ```

3. Install frontend dependencies:
   ```bash
   cd web-interface
   npm install
   ```

4. Run services locally:
   - Backend:
     ```bash
     cd backend
     python main.py
     ```
   - Frontend:
     ```bash
     cd web-interface
     npm run dev
     ```

5. Start LLM servers (e.g., DeepSeek-Coder-6.7B, Codestral-22B):
   ```bash
   bash deployment/launch_enhanced_llm_servers.sh
   ```

---

### 🐳 Docker Deployment

1. Build and start services:
 
   docker-compose up --build
   ```

2. Access services:
   - 🌐 Web Interface: [http://localhost:3000](http://localhost:3000)
   - 📖 API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
   - 🤖 LLM Endpoints:
     - [http://localhost:11434/api/generate](http://localhost:11434/api/generate)
     - [http://localhost:11435/v1/chat/completions](http://localhost:11435/v1/chat/completions)

---

### ☸️ Kubernetes Deployment

1. Prepare the Kubernetes cluster:
   ```bash
   kubectl create namespace sakthi-platform
   kubectl apply -f deployment/kubernetes/sakthi-platform.yaml
   ```

2. Monitor the deployment:
   ```bash
   kubectl get pods -n sakthi-platform
   ```

---

### 🤖 LLM Server Setup

1. Launch LLM servers for models like DeepSeek-Coder-6.7B and Codestral-22B:

   # Example: launch_enhanced_llm_servers.sh
   #!/bin/bash
   source /home/appadmin/virenv/bin/activate
   llama_cpp.server --model /mnt/modelslist/deepseek-coder-6.7b-instruct --port 11434 --n_gpu_layers 0
   llama_cpp.server --model /mnt/modelslist/codestral-22b --port 11435 --n_gpu_layers 0
   ```

2. Verify endpoints:

   curl http://localhost:11434/api/generate -d '{"prompt": "Test"}'
   curl http://localhost:11435/v1/chat/completions -d '{"messages": [{"role": "user", "content": "Test"}]}'
   ```

---

## 🎯 Use Case Examples

### Schema Migration 📊
- **Input**: "Convert Oracle HR schema to BigQuery."
- **Process**: DeepSeek LLM generates BigQuery DDL; EnhancedTargetProcessor maps fields in batches.
- **Output**: BigQuery DDL, ETL scripts, and validation reports in `output/`.

### Document Analysis 📄
- **Input**: Upload `financial_report.pdf` or `Account_Silver_Table_Column_List.xlsx`.
- **Intent**: "Extract quarterly revenue data by region."
- **Process**: `document-processor/processor.py` handles XLSX/CSV; LangGraph monitors extraction.
- **Output**: JSON, CSV, and API endpoints in `output/`.

### Web Data Integration 🌐
- **Input**: URL of competitor pricing page.
- **Intent**: "Monitor pricing changes daily."
- **Process**: DeepSeek LLM with SERPAPI for scraping; ChromaDB for context storage.
- **Output**: Automated trend analysis in `output/trends.json`.

---

## 🛠️ Configuration Guide

1. Create a configuration file (`config/.env`):
   ```plaintext
   OPENAI_API_KEY=your_openai_key
   SERPAPI_KEY=your_serpapi_key
   DATABASE_URL=postgresql://username:password@localhost/sakthi_db
   REDIS_URL=redis://localhost:6379
   CHROMADB_HOST=localhost
   CHROMADB_PORT=8001
   DEEPSEEK_MODEL_PATH=/mnt/modelslist/deepseek-coder-6.7b-instruct
   ```

2. Dynamic prompt template (`config/prompt_template.json`):

   {
     "template": "Generate SQL for {{rule_type}}: {{rule_value}}",
     "rules": "{{rules_csv_content}}"
   }
   ```

---

## 📈 SQL Validation

1. Validate generated SQL against rules.csv:
   ```bash
   pytest scripts/sql_validator.py --asyncio
   ```

2. Ensure SQL adheres to dynamic rules (e.g., email formats, numeric ranges) using `pytest-asyncio`, LangGraph, and DeepSeek LLM.

---

## 🔧 Advanced Setup

### Batch Processing 📦
Configure `EnhancedTargetProcessor` in `genai-modeling-agent/agent_system.py` for 1000-field batches:
```python
processor = EnhancedTargetProcessor(batch_size=1000)
processor.process_target_excel("uploads/Account_Silver_Table_Column_List.xlsx")
```

### Model Mounting 💾
Mount `/home/appadmin/modelslist` from one server (e.g., `10.100.15.67`) to another (e.g., `10.100.15.66`) using NFS:

mount 10.100.15.67:/home/appadmin/modelslist /mnt/modelslist
```

---

## 📊 Roadmap

- 🌟 Advanced semantic similarity scoring for RAG.
- 🤝 Enhanced multi-agent schema mapping with LangGraph.
- 📈 Real-time analytics dashboard in Next.js.
- 🆕 Support for additional models (e.g., Mistral_7B, Phi_2).

---

# 🌟 Sakthi Platform

The **Sakthi Platform** is a cutting-edge AI-driven data processing solution designed for schema transformations, natural language processing, and workflow orchestration.

---

## 🏛️ Architecture Overview

The following diagram illustrates the Sakthi Platform’s architecture, showcasing the flow between user interfaces, core services, AI agents, and storage systems.

```mermaid
graph TD
    %% ========== LAYER DEFINITIONS ==========
    subgraph "User Interface Layer"
        UI[("Web Interface<br/>(Next.js)")]
        API_GW[["API Gateway"]]
    end

    subgraph "Sakthi Core Services"
        ENGINE[["Sakthi Engine"]]
        PARSER[["Parser"]]
        INTENT[["Intent Analyzer"]]
        OUTPUT[["Output Generator"]]
    end

    subgraph "Document Processing"
        DOC_PROC[["Document Processor"]]
        PDF[("PDF Handler")]
        CSV[("CSV Handler")]
        DOCX[("DOCX Handler")]
        JSON[("JSON Handler")]
        WEB[["Web Scraper"]]
    end

    subgraph "AI Agent Framework"
        ORCHESTRATOR[["LangGraph<br/>Orchestrator"]]
        EXTRACTOR[("Extractor Agent<br/>(AutoGen)")]
        MAPPER[("Mapper Agent<br/>(AutoGen)")]
        VERIFIER[("Verifier Agent<br/>(AutoGen)")]
    end

    subgraph "MCPP Generation"
        MCPP[["Schema Generator"]]
        PATTERN[("Pattern Recognition")]
        MAPPING[("Mapping Engine")]
        VERSION[("Version Control")]
    end

    subgraph "External Services"
        SERPAPI[("SerpAPI")]
        OPENAI[("OpenAI/LLM")]
        METADATA[("Metadata APIs")]
        ATLAN[("Atlan")]
        COLLIBRA[("Collibra")]
    end

    subgraph "Storage Layer"
        STORAGE[["Storage Manager"]]
        DB_CONN[["Database<br/>Connectors"]]
        CLOUD[("Cloud Storage")]
        CACHE[("Redis Cache")]
    end

    subgraph "Target Systems"
        RDBMS[("Relational DBs<br/>(Oracle, PostgreSQL)")]
        DW[("Data Warehouses<br/>(BigQuery, Snowflake)")]
        NOSQL[("NoSQL<br/>(MongoDB)")]
        OBJ_STORE[("Object Storage<br/>(AWS S3)")]
    end

    %% ========== PRIMARY DATA FLOW ==========
    UI -->|User Requests| API_GW
    API_GW -->|Process| ENGINE
    
    %% Document Processing
    UI -->|Uploads| DOC_PROC
    DOC_PROC --> PDF & CSV & DOCX & JSON & WEB
    WEB -->|Search| SERPAPI
    PDF & CSV & DOCX & JSON -->|Extracted Data| ENGINE
    
    %% Sakthi Processing
    ENGINE -->|Parse| PARSER
    PARSER -->|Analyze| INTENT
    INTENT -->|Generate| OUTPUT
    
    %% AI Agent Workflow
    ENGINE -->|Orchestrate| ORCHESTRATOR
    ORCHESTRATOR --> EXTRACTOR & MAPPER & VERIFIER
    
    %% MCPP Generation
    MAPPER -->|Generate Schema| MCPP
    MCPP --> PATTERN & MAPPING & VERSION
    
    %% External Integrations
    EXTRACTOR & MAPPER & VERIFIER -->|LLM Calls| OPENAI
    VERIFIER -->|Metadata| METADATA
    METADATA --> ATLAN & COLLIBRA
    
    %% Storage & Output
    OUTPUT -->|Store| STORAGE
    STORAGE --> DB_CONN & CLOUD & CACHE
    OUTPUT -->|Write| DB_CONN
    
    %% Target Connections
    DB_CONN --> RDBMS & DW & NOSQL
    CLOUD --> OBJ_STORE

    %% ========== LEGEND ==========
    LEGEND[("Legend:<br/>
    🔵 User Interface | 🟣 Core Services<br/>
    🟢 Document Processing | 🟡 AI Agents<br/>
    🟠 Schema Generation | 🔴 External Services<br/>
    🟣 Storage | 🔗 Connectors | 🟢 Target Systems")]


## 🛡️ License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

Start transforming your data with the **Sakthi Platform** today! 🚀

