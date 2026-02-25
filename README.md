# ⚖️ Legal Document Analyzer

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/yourusername/legal-doc-analyzer)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-black)](https://github.com/yourusername/legal-doc-analyzer)

A powerful legal document analysis tool combining **Stanford Material Contracts Corpus** (1M+ historical contracts) with **live SEC EDGAR filings**. Built with Dash and deployed on Hugging Face Spaces.

## ✨ Features

- 🔍 **Hybrid Search**: Search across 1M+ historical contracts + live SEC filings
- 📊 **Interactive Analytics**: Visualize contract types, clauses, and trends
- ⚡ **Real-time Updates**: Pull latest contracts from SEC EDGAR API
- 🏛️ **Stanford MCC Integration**: 1M+ contracts from 2000-2023
- 🔒 **Privacy First**: All processing happens locally

## 🛠️ Tech Stack

- **Frontend**: Dash + Plotly
- **Backend**: Python + LangChain
- **Vector DB**: ChromaDB with sentence-transformers
- **LLM**: Ollama (local, free)
- **Deployment**: Hugging Face Spaces

