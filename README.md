---
title: "ShastraDocs"
emoji: "📚"
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
license: mit
tags: [rag, document-analysis, llm, enterprise, ai]
---

<div align="center">

# 📚 ShastraDocs v2
## Enterprise RAG System for Document Analysis

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-ready-green.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue.svg)

**🚀 Production-ready API • 📄 8+ Document Formats • 🤖 Multi-LLM Support • ⚡ Advanced Retrieval**

[**Try the API**](#-quick-start) | [**Full Docs**](https://github.com/Team-DevBytes/ShastraDocs2) | [**GitHub**](https://github.com/Team-DevBytes/ShastraDocs2)

</div>

---

## 🚀 Overview

ShastraDocs v2 is a production-ready, modular RAG system designed for comprehensive document analysis and intelligent question answering. Built with enterprise requirements in mind, it supports 8+ document formats, features intelligent multi-provider LLM management, and provides advanced retrieval techniques with comprehensive monitoring capabilities.

### ✨ Key Highlights

- **🎯 Multi-Format Support**: PDF, DOCX, PPTX, XLSX, Images, Text, CSV, and URLs
- **⚡ Intelligent Processing**: Automatic format detection with specialized handlers.
- **🔄 Multi-Provider LLM**: Smart rotation between Groq, Gemini, and OpenAI with rate limit handling.
- **🔍 Advanced Retrieval**: Hybrid search with BM25 + semantic search and cross-encoder reranking.
- **📂 Retrieval Across Multiple Files**: Query across a collection of documents in a persistent session, with responses synthesized from all relevant sources.
- **🖥️ UI Integration**: A complete web interface for user login, persisting sessions, document uploads, and interactive chat with smart referencing and document previews powered by the **Apryse Web SDK**.
- **📊 Production Features**: Comprehensive logging, monitoring, and health checks.
- **🐳 Docker Ready**: Containerized deployment with HuggingFace Spaces optimization.
- **💰 Cost Effective**: Process 200+ questions at $0 cost using free tier rotation.

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ShastraDocs v2                           │
├─────────────────────────────────────────────────────────────────┤
│    User Interface (Login, Sessions, Uploads, Chat, File Preview)│
├─────────────────────────────────────────────────────────────────┤
│  FastAPI REST API (Authentication, Endpoints, Health Checks)    │
├─────────────────────────────────────────────────────────────────┤
│     Multi-Provider LLM Handler (Groq, Gemini, OpenAI)           │
├─────────────────────────────────────────────────────────────────┤
│        Advanced RAG Processor (Query Expansion, Reranking)      │
├─────────────────────────────────────────────────────────────────┤
│   Document Preprocessing (8+ Formats, OCR, Table Extraction)    │
├─────────────────────────────────────────────────────────────────┤
│    Vector Storage & Search (Qdrant, Hybrid Search, Caching)     │
├─────────────────────────────────────────────────────────────────┤
│  Comprehensive Logging & Monitoring (Request Tracking, Stats)   │
└─────────────────────────────────────────────────────────────────┘
```

## 📦 Project Structure

```
shastradocs-v2/
├── 📁 api/                          # FastAPI REST API
│   ├── __init__.py
│   └── api.py                       # Main API endpoints, UI logic, and authentication
├── 📁 templates/                   # HTML templates for the UI
│   ├── index.html                   # Main single-page application
│   └── service-worker.js            # PWA service worker
├── 📁 static/                      # Static assets for the UI
│   └── lib/apryse/                  # Apryse WebViewer SDK extraction location
├── 📁 config/                       # Centralized configuration
│   ├── __init__.py
│   └── config.py                    # Auto-detecting multi-provider configs
├── 📁 LLM/                         # Multi-provider LLM management
│   ├── __init__.py
│   ├── llm_handler.py               # Unified multi-provider handler
│   ├── one_shotter.py               # Enhanced QA with web scraping
│   ├── image_answerer.py            # Specialized image analysis
│   ├── tabular_answer.py            # Structured data handler
│   └── lite_llm.py                  # Lightweight handler
├── 📁 RAG/                         # Advanced retrieval system
│   ├── __init__.py
│   ├── advanced_rag_processor.py    # Main RAG orchestrator
│   └── rag_modules/                 # Modular RAG components
│       ├── query_expansion.py       # Query decomposition
│       ├── embedding_manager.py     # Semantic embeddings
│       ├── search_manager.py        # Hybrid search engine
│       ├── reranking_manager.py     # Cross-encoder reranking
│       ├── context_manager.py       # Context assembly
│       └── answer_generator.py      # LLM answer generation
├── 📁 shared/                      # Shared utility modules
│   ├── __init__.py
│   └── model_manager.py             # Ensures one embedding model instance is created
├── 📁 preprocessing/               # Document processing pipeline
│   ├── __init__.py
│   ├── preprocessing.py             # Main entry point and CLI
│   └── preprocessing_modules/       # Specialized extractors
│       ├── modular_preprocessor.py  # Main orchestrator
│       ├── file_downloader.py       # Universal file downloading
│       ├── pdf_extractor.py         # Advanced PDF processing
│       ├── docx_extractor.py        # Word document handling
│       ├── pptx_extractor.py        # PowerPoint processing
│       ├── xlsx_extractor.py        # Excel with OCR support
│       ├── image_extractor.py       # Image and table extraction
│       ├── text_chunker.py          # Smart text chunking
│       ├── embedding_manager.py     # Batch embedding generation
│       ├── vector_storage.py        # Qdrant integration
│       └── metadata_manager.py      # Document metadata
├── 📁 logger/                      # Advanced logging system
│   ├── __init__.py
│   └── logger.py                    # In-memory logging with analytics
├── 📄 app.py                       # Application entry point
├── 📄 startup.sh                   # Production startup script
├── 📄 Dockerfile                   # Container configuration
├── 📄 requirements.txt             # Python dependencies
├── 📄 LICENSE                      # MIT License
└── 📄 README.md                    # This file
```

## 🎯 Core Features

### 🔧 Multi-Provider LLM Management

**Smart Rate Limit Handling**
- Automatic rotation between Groq, Gemini, and OpenAI
- 60-second cooldown management per provider
- Intelligent fallback with zero downtime
- Real-time provider health monitoring

**Multi-Instance Support**
- Up to 10 API keys per provider
- Custom model assignment per instance
- Priority-based routing (Groq → Gemini → OpenAI)
- Cost-effective free tier optimization

### 📋 Document Processing Pipeline

**Supported Formats**
| Format | Extensions | Special Features |
|--------|------------|-----------------|
| PDF | .pdf | CID font mapping, table extraction, parallel processing |
| Word | .docx | Text boxes, tables, gridSpan handling |
| PowerPoint | .pptx | OCR Space API for images, notes extraction |
| Excel | .xlsx | Cell processing, embedded image OCR |
| Images | .png, .jpg, .jpeg | Table detection, OCR text extraction |
| Text | .txt, .csv | Direct processing, structured data handling |
| URLs | http/https | Google Docs conversion, web scraping |

**Advanced Processing**
- **Smart Chunking**: Sentence-boundary aware with configurable overlap
- **OCR Integration**: OCR Space API and Tesseract support
- **Table Extraction**: Automatic detection and markdown formatting
- **Caching System**: Document-level caching to avoid reprocessing
- **Parallel Processing**: Multi-threaded operations for efficiency

### 🔍 Advanced RAG System

**Query Processing**
- **Query Expansion**: Automatic decomposition into focused sub-queries
- **Multi-aspect Analysis**: Process/document/contact identification
- **Budget Management**: Intelligent retrieval budget distribution

**Hybrid Search Engine**
- **Semantic Search**: Dense vector similarity (SentenceTransformers)
- **Keyword Search**: BM25 for exact term matching
- **Score Fusion**: Reciprocal Rank Fusion with weighted combination
- **Reranking**: Cross-encoder models for relevance refinement

**Context-Aware Generation**
- **Multi-Document Context**: Synthesizes answers from multiple documents in a single query.
- **Enhanced Prompting**: Specialized prompts for policy documents.
- **Source Referencing**: Generates answers with citations pointing to the source document and page.
- **Error Handling**: Graceful handling of edge cases.

### 🌐 Production-Ready API

**REST Endpoints**
- `POST /signup`, `POST /login` - User authentication
- `GET /my_sessions/{user_id}` - Manage user sessions
- `POST /upload` - Upload documents to a session
- `POST /query` - Ask questions within a session's context
- `POST /hackrx/run` - Core multi-document processing and Q&A
- `GET /health` - System health monitoring
- `POST /preprocess` - Batch document preprocessing (admin)
- `GET /logs` - Request logs export with filtering (admin)
- `GET /collections` - List processed documents (admin)

**Security Features**
- User email/password authentication for UI.
- Bearer token authentication for core API endpoints.
- Admin token for administrative functions.
- Request validation using Pydantic models.
- CORS and security headers configuration.

### 📊 Comprehensive Monitoring

**Request Tracking**
- Unique request ID generation
- Pipeline stage timing breakdown
- Per-question performance metrics
- Success/failure tracking

**Performance Analytics**
- Real-time processing statistics
- Provider usage distribution
- Memory and resource monitoring
- Export capabilities with filtering

**Health Monitoring**
- System component status
- Provider availability tracking
- Database connection health
- Resource usage monitoring

## ⚙️ Quick Setup

### Prerequisites

- Python 3.10+
- Docker (optional)
- At least one LLM provider API key (Groq/Gemini/OpenAI)
- OCR Space API key (for PowerPoint images)

### 🚀 Local Development Setup

1.  **Clone Repository**
    ```bash
    git clone https://github.com/Rahul-Samedavar/ShastraDocs2.git
    cd ShastraDocs2
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment**
    Create a `.env` file with your API keys:
    ```bash
    # === LLM PROVIDERS ===
    # Groq (Primary provider - fastest)
    GROQ_API_KEY_1=your_first_groq_key
    DEFAULT_GROQ_MODEL=llama3-70b-8192
    
    # You can add more API keys for each provider by incrementing the number
    # GEMINI_API_KEY_1=...
    # OPENAI_API_KEY_1=...

    # === SERVICES ===
    OCR_SPACE_API_KEY=your_ocr_space_key
    BEARER_TOKEN=your_secure_api_token
    ```

4.  **Apryse SDK Setup**
    - [Download](https://docs.apryse.com/downloads) the Apryse WebViewer SDK.
    - Extract the contents into the `static/lib/apryse/` directory. Your structure should look like `static/lib/apryse/core/...`, `static/lib/apryse/ui/...` etc.

5.  **Run Application**
    This will start the FastAPI server and serve the user interface.
    ```bash
    python app.py
    ```
    Access the UI at `http://127.0.0.1:7860`.

### 🐳 Docker Deployment

1.  **Build Image**
    ```bash
    docker build -t shastradocs-v2 .
    ```

2.  **Run Container**
    ```bash
    docker run -p 7860:7860 --env-file .env shastradocs-v2
    ```

### ☁️ HuggingFace Spaces Deployment

The application is optimized for HuggingFace Spaces:

1.  Upload project files to your Space.
2.  Set environment variables in Space settings.
3.  The `startup.sh` script handles database initialization.
4.  Access via your Space URL.

## 📖 Usage Examples

### Python Client (Multi-Document)

```python
import httpx
import asyncio

async def analyze_documents():
    # Use the port from your config, default is 7860
    url = "http://localhost:7860/hackrx/run"
    headers = {"Authorization": "Bearer your_secure_api_token"}
    
    data = {
        "documents": [
            "https://example.com/policy_a.pdf",
            "https://example.com/policy_b.docx"
        ],
        "questions": [
            "Compare the claim submission process from both documents.",
            "What is the contact for help in policy_a.pdf?"
        ]
    }
    
    async with httpx.AsyncClient(timeout=600) as client:
        response = await client.post(url, json=data, headers=headers)
        result = response.json()
        
        print("📄 Document Analysis Results:")
        for i, answer in enumerate(result["answers"]):
            print(f"\nQ{i+1}: {data['questions'][i]}")
            print(f"A{i+1}: {answer}")

asyncio.run(analyze_documents())
```

### cURL Examples

```bash
# Process multiple documents with questions
curl -X POST "http://localhost:7860/hackrx/run" \
  -H "Authorization: Bearer your_secure_api_token" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      "https://example.com/policy.pdf",
      "https://example.com/faq.docx"
    ],
    "questions": [
      "Summarize the key policy highlights from the PDF.",
      "How do I submit a claim according to the FAQ?"
    ]
  }'

# Check system health
curl -X GET "http://localhost:7860/health"

# Get request logs (admin)
curl -X GET "http://localhost:7860/logs?minutes=60&limit=50" \
  -H "Authorization: Bearer your_admin_token"

# Preprocess a document (admin)
curl -X POST "http://localhost:7860/preprocess" \
  -H "Authorization: Bearer your_admin_token" \
  -F "document_url=https://example.com/document.pdf" \
  -F "force=false"
```

### CLI Usage

```bash
# Process single document
python -m preprocessing.preprocessing --url "https://example.com/document.pdf"

# Process multiple documents from a file
python -m preprocessing.preprocessing --urls-file urls.txt

# List processed documents
python -m preprocessing.preprocessing --list

# Show statistics
python -m preprocessing.preprocessing --stats
```

## 🎛️ Configuration Guide

### Environment Variables

**Required Variables**
```bash
# At least one LLM provider
GROQ_API_KEY_1=your_key        # OR
GEMINI_API_KEY_1=your_key      # OR  
OPENAI_API_KEY_1=your_key

# Authentication
BEARER_TOKEN=your_secure_token

# OCR for PowerPoint processing
OCR_SPACE_API_KEY=your_ocr_key
```

**Optional Variables**
```bash
# Additional LLM keys (up to 10 per provider)
GROQ_API_KEY_2=backup_key
GEMINI_API_KEY_2=backup_key

# Custom models per provider
DEFAULT_GROQ_MODEL=llama3-70b-8192

# API configuration
API_HOST=0.0.0.0
API_PORT=7860
API_RELOAD=false

# RAG configuration
TOP_K=9
CHUNK_SIZE=1600
ENABLE_RERANKING=true
```

### Processing Modes

The system automatically selects optimal processing modes:

**1. Advanced RAG Processing**
- Complex documents requiring full pipeline
- Vector database storage and hybrid search
- Best for policy documents, manuals

**2. OneShot Processing**
- Simple text documents
- Direct LLM processing without vector search
- Faster for short documents

**3. Tabular Analysis**
- Excel, CSV files with structured data
- Specialized data analysis prompts
- Optimized for numerical data

**4. Image Processing**
- Visual content with OCR
- Table detection in images
- Automatic cleanup after processing

## 📊 Performance Metrics

### Processing Speed
- **Simple Queries**: 1-2 seconds
- **Complex Multi-aspect**: 2-4 seconds
- **Document Preprocessing**: 2-5 pages/second (PDF)
- **Embedding Generation**: 100-500 chunks/second

### Cost Optimization
- **Free Tier Usage**: 200+ questions at $0 cost
- **Provider Rotation**: Automatic cost-effective routing  
- **Rate Limit Avoidance**: Prevents unnecessary paid calls
- **Intelligent Caching**: Reduces redundant processing

### Resource Usage
- **Memory**: 500MB-1GB (model dependent)
- **CPU**: Moderate during processing, minimal idle

## 🛠️ Troubleshooting

### Common Issues

**1. No LLM Providers Available**
```python
# Check provider status
from LLM.llm_handler import llm_handler
status = llm_handler.get_provider_status()
print(f"Available providers: {len(status)}")

# Reset cooldowns if needed
llm_handler.reset_cooldowns()
```

**2. Document Processing Failures**
```bash
# Check document accessibility
curl -I "https://your-document-url.pdf"

# Force reprocessing (admin token required)
curl -X POST "http://localhost:7860/preprocess" \
  -H "Authorization: Bearer admin_token" \
  -F "document_url=your_url" -F "force=true"
```

**3. OCR Space API Issues**
```bash
# Verify OCR API key
export OCR_SPACE_API_KEY="your_key"

# Test OCR endpoint
curl -X POST "https://api.ocr.space/parse/image" \
  -F "apikey=your_key" \
  -F "url=https://example.com/image.jpg"
```

**4. Memory Issues**
```python
# Reduce batch sizes in config.py
BATCH_SIZE = 16
CHUNK_SIZE = 1200
```

### Debug Mode

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Check system health
from api.api import app
# Health check includes detailed component status
```

### Health Monitoring

```bash
# System health check
curl http://localhost:7860/health

# Detailed logs export (admin token required)
curl -H "Authorization: Bearer admin_token" \
  "http://localhost:7860/logs?minutes=60" > debug_logs.json
```

## 🚀 Production Deployment

### Docker Production Setup

```dockerfile
# Multi-stage build for optimization
FROM python:3.10-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.10-slim
COPY --from=builder /root/.local /root/.local
COPY . /app
WORKDIR /app

# Environment setup
ENV PATH=/root/.local/bin:$PATH
ENV HF_HOME=/app/.cache/huggingface
EXPOSE 7860

CMD ["bash", "startup.sh"]
```

### Environment-Specific Configuration

**Development**
```bash
API_RELOAD=true
API_HOST=127.0.0.1
LOG_LEVEL=DEBUG
```

**Staging**
```bash
API_RELOAD=false
API_HOST=0.0.0.0
LOG_LEVEL=INFO
```

**Production**
```bash
API_RELOAD=false
API_HOST=0.0.0.0
LOG_LEVEL=WARNING
# Multiple API keys for redundancy
GROQ_API_KEY_1=prod_key_1
GROQ_API_KEY_2=prod_key_2
```

### Monitoring Setup

```bash
# Health check endpoint for load balancers
curl -f http://localhost:7860/health || exit 1

# Prometheus metrics (custom implementation)
curl http://localhost:7860/metrics

# Log aggregation (admin token required)
curl -H "Authorization: Bearer admin_token" \
  "http://localhost:7860/logs" | jq '.metadata'
```

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Development Setup
1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Follow modular architecture patterns
4. Maintain async/await compatibility
5. Add comprehensive error handling
6. Include type hints and documentation

### Code Standards
- **Python**: Follow PEP 8 style guidelines
- **Documentation**: Update README for new features
- **Testing**: Add tests for new components
- **Error Handling**: Implement graceful error recovery

### Pull Request Process
1. Update documentation
2. Add tests for new functionality
3. Ensure all tests pass
4. Update CHANGELOG.md
5. Submit PR with detailed description

## 🔒 Security Considerations

### Authentication
- **User Authentication**: Email and password-based system for the UI.
- **Bearer Tokens**: Secure API access with rotation support for core endpoints.
- **Admin Endpoints**: Separate authentication for sensitive operations.
- **Input Validation**: Comprehensive request sanitization.

### Data Security
- **Database Storage**: User and session metadata are stored in a SQLite database.
- **Document Caching**: Documents are stored on the server filesystem for processing.
- **Automatic Cleanup**: Temporary files removed after processing.
- **Secure Headers**: CORS and security headers configured.

### Rate Limiting
- **Request Throttling**: Built-in concurrency limits
- **Provider Management**: Smart rate limit handling
- **Graceful Degradation**: Continues operation during issues

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Copyright (c) 2025 Rahul Samedavar and Sambhaji Patil**

## 🙏 Acknowledgments

- **HuggingFace**: For model hosting and Spaces platform
- **Qdrant**: For vector database capabilities  
- **FastAPI**: For modern API framework
- **SentenceTransformers**: For embedding models
- **Community Contributors**: For feedback and improvements

---

<div align="center">

**ShastraDocs v2** - *Enterprise-grade RAG system for intelligent document analysis*

[🌟 Star on GitHub](https://github.com/Team-DevBytes/ShastraDocs2) 

</div>
