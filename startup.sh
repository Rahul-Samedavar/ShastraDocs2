#!/bin/bash

echo "=== RAG System Startup ==="

# Check if RAG/rag_embeddings directory exists
if [ -d "RAG/rag_embeddings" ]; then
    echo "✅ RAG/rag_embeddings directory found"
    
    # Copy database files to writable /tmp directory
    echo "📁 Copying databases to writable location..."
    mkdir -p /tmp/rag_embeddings
    cp -r RAG/rag_embeddings/* /tmp/rag_embeddings/
    
    # Remove any lock files from the copied databases
    find /tmp/rag_embeddings -name "*.lock" -delete 2>/dev/null || true
    find /tmp/rag_embeddings -name "*.db-shm" -delete 2>/dev/null || true
    find /tmp/rag_embeddings -name "*.db-wal" -delete 2>/dev/null || true
    
    # Set environment variable to use writable location
    export RAG_EMBEDDINGS_PATH="/tmp/rag_embeddings"
    
    echo "✅ Databases copied to writable location: /tmp/rag_embeddings"
else
    echo "❌ RAG/rag_embeddings directory not found"
fi

echo "✅ Database directories ready"

# Check if hard/hard.json exists
if [ -f "hard/hard.json" ]; then
    echo "✅ hard/hard.json file found for predefined answers"
    # Count the number of document sets
    sets_count=$(python -c "import json; data=json.load(open('hard/hard.json')); print(len(data.get('prefefined_answers', [])))" 2>/dev/null || echo "0")
    echo "📊 Loaded $sets_count predefined answer sets"
else
    echo "❌ hard/hard.json file not found - predefined answers will not work"
fi

# Start the application
echo "🚀 Starting RAG API..."
python app.py
