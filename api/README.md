
# ShastraDocs API Package

A production-ready FastAPI REST API for the ShastraDocs document analysis system. This package provides secure, authenticated endpoints for document processing, question answering, and system management, along with a full set of endpoints to power its web-based user interface.

## 🚀 Overview

The API package serves as the main interface for the ShastraDocs system, offering:
- **User & Session Management**: Endpoints for user signup, login, and management of persistent document sessions.
- **Multi-Document Processing**: Upload and analyze multiple documents (8+ formats) within a session.
- **Interactive Question Answering**: A chat-style endpoint for querying across all documents in a session, with structured, referenced answers.
- **System Management**: Admin endpoints for monitoring and maintenance.
- **Enhanced Logging**: Detailed request tracking and performance analytics.

## 📦 Package Structure

```
api/
├── __init__.py          # Package initialization
└── api.py              # Main FastAPI application with all endpoints
```

## 🎯 Core Features

### 🔐 Security & Authentication
- **User Authentication**: Email/password-based system for the UI.
- **Bearer Token Authentication**: Secure core API access with configurable tokens.
- **Admin Endpoints**: Separate authentication for administrative functions.
- **Request Validation**: Comprehensive input validation using Pydantic models.

### ⚡ Intelligent Document Processing
- **Multi-Document Context**: Process and query across collections of documents in a single request.
- **Optimized Flow**: Checks for pre-processed documents to avoid redundant work.
- **Parallel Processing**: Concurrent question answering with configurable limits.
- **Fallback Handling**: Graceful degradation for unsupported formats.

### 📊 Advanced Processing Modes
- **Standard RAG**: Full pipeline for complex documents.
- **OneShot Processing**: Fast processing for simple text documents.
- **Tabular Analysis**: Specialized handling for structured data.
- **Image Analysis**: OCR and visual question answering.

### 🔍 Monitoring & Observability
- **Real-time Logging**: Detailed request tracking with unique IDs.
- **Performance Metrics**: Pipeline timing breakdown and statistics.
- **Health Monitoring**: System status and component health checks.
- **Export Capabilities**: JSON log export with filtering options.

## 📋 API Endpoints

### User & Session Management

- `POST /signup`: Create a new user account.
- `POST /login`: Log in a user and retrieve their user ID.
- `GET /my_sessions/{user_id}`: List all sessions for a specific user.
- `POST /new_session`: Create a new, empty session for a user.
- `DELETE /session/{session_id}`: Delete a session and its associated messages.
- `POST /clone`: Create a copy of an existing session.

### Document & Chat Endpoints

#### `POST /upload` - Upload Documents to Session
Upload one or more files, or provide a URL, to add documents to a specific session.
- **Request**: `multipart/form-data` with `session_id` (str), `files` (List[UploadFile]), or `url` (str).
- **Response**: `{ "message": "File(s) uploaded", "doc_ids": ["1", "2"] }`

#### `POST /query` - Ask a Question in a Session
Submit a question to get an answer synthesized from all documents within the specified session.
**Request:**
```json
{
  "session_id": "your_session_id",
  "question": "What is the policy on remote work?"
}
```
**Response (Structured with References):**
```json
{
  "answer_parts": [
    {
      "text": "The policy states that remote work is available to all full-time employees. ",
      "references": []
    },
    {
      "text": "Approval from a direct manager is required before starting a remote work arrangement.",
      "references": [
        {
          "doc_id": "3",
          "page_num": 4,
          "text_snippet": "All remote work arrangements must be approved by the employee's direct manager..."
        }
      ]
    }
  ],
  "metadata": { "confidence_score": 0.92, "language": "en" }
}
```
#### `GET /get_doc/{session_id}/{doc_id}` - Retrieve a Document
Serves a document file for viewing (e.g., in Apryse WebViewer).

### Core Processing Endpoint

#### `POST /hackrx/run` - Multi-Document Processing & QA
Process one or more documents and answer questions in a single, stateless request.

**Request:**
```json
{
  "documents": [
    "https://example.com/policy_a.pdf",
    "https://example.com/policy_b.docx"
  ],
  "questions": [
    "Compare the claim submission process from both documents."
  ]
}
```

**Response:**
```json
{
  "answers": [
    "In policy A, the claim process involves filling out Form-X and submitting it online. In policy B, claims must be initiated via a phone call to the claims department..."
  ]
}
```

### Administrative & Health Endpoints (Admin Token Required)

- `GET /health`: Simple health check endpoint.
- `POST /preprocess`: Pre-process a document for faster future queries.
- `GET /collections`: List information about all processed document collections.
- `GET /collections/stats`: Get statistics about the document database.
- `GET /logs`: Export detailed API request logs.
- `GET /logs/summary`: Get aggregated performance metrics.

## 🔧 Configuration

### Environment Variables

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=7860
API_RELOAD=True

# Authentication
BEARER_TOKEN=your_secure_api_token
ADMIN_TOKEN=your_secure_admin_token

# LLM Provider Keys (auto-detects multiple keys)
GROQ_API_KEY_1=your_groq_key_1
GEMINI_API_KEY_1=your_gemini_key_1

# OCR Service
OCR_SPACE_API_KEY=your_ocr_space_key
```

## 🚀 Usage Examples

### Python Client

```python
import httpx
import asyncio

async def process_documents():
    url = "http://localhost:7860/hackrx/run"
    headers = {"Authorization": "Bearer your_token"}
    
    data = {
        "documents": [
            "https://example.com/policy.pdf",
            "https://example.com/faq.docx"
        ],
        "questions": [
            "Summarize the policy coverage.",
            "How do I file a claim according to the FAQ?"
        ]
    }
    
    async with httpx.AsyncClient(timeout=600) as client:
        response = await client.post(url, json=data, headers=headers)
        result = response.json()
        
        for i, answer in enumerate(result["answers"]):
            print(f"Q{i+1}: {data['questions'][i]}")
            print(f"A{i+1}: {answer}\n")

asyncio.run(process_documents())
```

### cURL Examples

```bash
# Process multiple documents
curl -X POST "http://localhost:7860/hackrx/run" \
  -H "Authorization: Bearer your_token" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": ["https://example.com/doc1.pdf", "https://example.com/doc2.pdf"],
    "questions": ["Compare the conclusions of both documents."]
  }'

# Ask a question within a UI session
curl -X POST "http://localhost:7860/query" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "your_session_id_from_ui",
    "question": "What is this document about?"
  }'

# Upload a file to a session
curl -X POST "http://localhost:7860/upload" \
  -F "session_id=your_session_id_from_ui" \
  -F "files=@/path/to/your/document.pdf"
```

## 🛠️ Development

### Running the API

```bash
# Development mode with auto-reload from the project root
python app.py

# Production mode with uvicorn
uvicorn api.api:app --host 0.0.0.0 --port 7860 --workers 4
```

### Testing

```python
import pytest
from fastapi.testclient import TestClient
from api.api import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_process_document_unauthorized():
    data = {
        "documents": ["https://example.com/test.pdf"],
        "questions": ["What is this about?"]
    }
    response = client.post("/hackrx/run", json=data)
    assert response.status_code == 403 # Depends on if Depends is caught by test client
```

### Custom Error Handling

The API includes comprehensive error handling for both user and system issues.
```json
// Example error responses
{ "detail": "Invalid authentication token" }
{ "detail": "Failed to process document: Unsupported file format" }
{ "error": "No documents in this session" }
```

## 🔒 Security Considerations

### Authentication
- **User Passwords**: Stored directly in the database. For production, hashing should be implemented.
- **Bearer Token**: Core endpoints require a valid bearer token.
- **Admin Token**: Administrative functions use a separate token.

### Data Security
- **Document Caching**: Uploaded documents are stored on the server's filesystem. Ensure appropriate permissions.
- **Database**: Session and user data are stored in a SQLite database.
- **Secure Headers**: CORS and security headers are configured.

---

**ShastraDocs API Package** - Production-ready REST API for advanced, multi-document analysis and interactive question answering.