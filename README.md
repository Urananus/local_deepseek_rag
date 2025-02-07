# Advanced RAG System

This project implements an advanced Retrieval-Augmented Generation (RAG) system for PDF documents using LangChain, PyPDF, and Gradio. Upload PDFs, ask questions, and receive answers with relevant sources.

## Features
- Extracts text from PDF documents.
- Splits text into manageable chunks.
- Creates embeddings and a vector store using Chroma.
- Provides a retrieval-based QA pipeline via LangChain.
- Interactive interface with Gradio.

## Requirements
- Python 3.8+
- Required packages:
  - langchain
  - langchain_community
  - langchain_ollama
  - pypdf
  - gradio

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>