# Contribution

## AI-Powered Document Assistant (RAG Pipeline)

My contribution to this project involved building and deploying a document-based AI assistant that allows users to upload documents and ask questions directly from those files.

### Key Contributions

- Built the complete Retrieval-Augmented Generation (RAG) pipeline
- Implemented file upload and processing for:
  - PDF
  - DOCX
  - TXT
- Converted uploaded documents into Markdown format
- Managed document storage inside the `docs/` directory
- Generated embeddings using ONNX MiniLM
- Indexed document chunks using ChromaDB
- Implemented semantic retrieval for relevant context extraction
- Added a lightweight SLM layer for structured answer generation
- Improved response formatting with source-backed answers
- Optimized application performance for Hugging Face Spaces deployment
- Reduced response latency and improved inference efficiency
- Fixed dependency conflicts and deployment-related issues
- Managed and optimized `requirements.txt`
- Resolved Gradio deployment and runtime issues

### Technologies Used

- Python
- Gradio
- ChromaDB
- ONNX Runtime
- MiniLM Embeddings
- Hugging Face Spaces
- Markdown Processing
- Retrieval-Augmented Generation (RAG)

### Outcome

The final system enables users to upload documents and receive fast, context-aware, and source-supported AI-generated responses with an optimized lightweight deployment pipeline.
