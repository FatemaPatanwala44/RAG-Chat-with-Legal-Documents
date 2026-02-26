📘 Chat with Legal Documents (RAG System)
🎓 Project Information

Course: Generative AI
Institution: Medicaps University – Datagami Skill Based Course
Academic Year: 2025–2026
Project Title: Chat with Legal Documents using RAG

🚀 Project Overview

This project is a Retrieval-Augmented Generation (RAG) based AI chatbot that allows users to upload legal or professional PDF documents and ask questions in natural language.

Instead of manually searching long documents, the system:

Reads the document

Understands the content

Retrieves relevant sections

Generates accurate answers based only on the uploaded file

This ensures context-based, reliable answers directly grounded in the document.

❓ Problem Statement

Most legal and professional documents are available in PDF format, but:

Searching information manually is time-consuming.

Keyword search is inefficient.

Users need to read entire documents to find small details.

This project solves that problem by building an AI assistant that understands documents and answers questions contextually.

🎯 Objectives

Build a smart chatbot to read and understand PDFs

Implement Retrieval-Augmented Generation (RAG)

Convert document text into semantic embeddings

Enable fast and accurate document-based search

Reduce manual effort in legal document analysis

🏗️ System Architecture

The system follows this workflow:

📌 Step-by-Step Flow

User uploads a PDF

Text is extracted from the PDF

Text is split into smaller chunks

Each chunk is converted into embeddings (vector form)

Embeddings are stored in a vector database (FAISS)

User asks a question

System converts the question into embedding

Similar chunks are retrieved

LLM generates a context-based answer

📊 Database Architecture (Page 4 Diagram Explanation)

The ER diagram (Page 4) shows a structured backend system with:

User → stores authentication and profile data

Document → stores uploaded PDF metadata

DocumentChunk → stores text chunks

Embedding → stores vector embeddings

ChatSession → tracks user conversations

ChatMessage → stores chat history

Citation → links answers to document chunks

This design ensures:

Scalability

Tracking of user sessions

Storing embeddings efficiently

Source citation support

🔄 Activity Workflow (Page 5 Diagram Explanation)

The Activity Diagram shows:

📥 Document Processing Flow

User login

PDF upload

File validation

Text extraction (OCR if needed)

Text cleaning

Chunking

Embedding generation

Store embeddings

Mark document as ready

💬 Question-Answer Flow

User asks question

Convert query into embedding

Similarity search (Top-K chunks)

Prepare prompt

Call LLM API

Generate answer with citations

Store session and messages

🧠 Technologies Used
Component	Technology
Programming Language	Python
Framework	Streamlit
Document Loader	PyPDF
Text Splitting	LangChain Text Splitter
Embeddings	Sentence Transformers
Vector Database	FAISS
LLM	Gemini API
IDE	VS Code
🖥️ Features

📂 Upload PDF documents

🧩 Smart text chunking

🔍 Semantic similarity search

⚡ Fast retrieval using FAISS

🤖 AI-generated answers

🧾 Answer citations

💬 Chat-style interface

🌐 Offline capability support (Ollama/Llama3 mentioned)

📸 Output

The screenshots (Pages 7–9) show:

Clean UI: “Legal RAG Assistant”

Successful document upload

Chunking details

Chat interface

Auto-detection of previously processed documents

Context-based answers

🏆 Results

Successfully implemented a document-based AI chatbot

Accurate question-answering from uploaded PDFs

End-to-end RAG pipeline working

Context-restricted response generation

🔮 Future Enhancements

Multi-document support

Chat memory system

Improved chat UI

Advanced source citations with page numbers

Cloud deployment

More accurate advanced LLM models

🧩 How RAG Works in This Project

RAG = Retrieval + Generation

Instead of letting AI guess answers:

The system first retrieves relevant text from the document

Then generates answer using that retrieved context

This reduces hallucinations and improves accuracy.
