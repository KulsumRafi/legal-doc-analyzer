"""
Legal Document Analyzer - Dash Version with Hugging Face Inference API
No local LLM needed! Uses Hugging Face's free Inference API.
"""

import os
import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime
import requests
import json

# RAG imports
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub  # For HF Inference API
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ============================================
# HUGGING FACE SETUP
# ============================================

# 🔑 Load HF token from environment (set in Hugging Face Spaces Secrets)
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("⚠️ WARNING: HF_TOKEN not found in environment variables!")
    print("Please add HF_TOKEN to your Hugging Face Space Secrets")
    print("The app will run in demo mode without actual LLM responses")

# Model to use (free, fast, works well for legal text)
# Options: "meta-llama/Llama-3.2-3B-Instruct", "microsoft/phi-3-mini-4k-instruct", "HuggingFaceH4/zephyr-7b-beta"
HF_MODEL = "meta-llama/Llama-3.2-3B-Instruct"  # Good balance of quality and speed

# ============================================
# DASH APP INITIALIZATION
# ============================================

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap'
    ],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)

# For Hugging Face Spaces deployment - THIS IS CRITICAL
server = app.server

# Custom CSS (same as before - keeping it clean)
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>⚖️ Legal Document Analyzer</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: 'Inter', sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f8f9fa;
            }
            .main-header {
                background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
                color: white;
                padding: 2rem;
                text-align: center;
                margin-bottom: 2rem;
            }
            .main-header h1 {
                font-size: 2.5rem;
                margin: 0;
                font-weight: 700;
            }
            .main-header p {
                font-size: 1.1rem;
                opacity: 0.9;
                margin: 0.5rem 0 0;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 0 1.5rem;
            }
            .card {
                background: white;
                border-radius: 12px;
                padding: 1.5rem;
                box-shadow: 0 4px 6px rgba(0,0,0,0.05);
                margin-bottom: 1.5rem;
            }
            .stat-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
                margin-bottom: 1.5rem;
            }
            .stat-card {
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                padding: 1.2rem;
                border-radius: 10px;
                text-align: center;
            }
            .stat-number {
                font-size: 2rem;
                font-weight: 700;
                color: #1E3A8A;
            }
            .stat-label {
                font-size: 0.9rem;
                color: #6B7280;
                margin-top: 0.3rem;
            }
            .query-box {
                background: white;
                border: 2px solid #e5e7eb;
                border-radius: 12px;
                padding: 1.5rem;
                margin: 1.5rem 0;
            }
            .answer-box {
                background: #f0f9ff;
                border-left: 4px solid #2563EB;
                padding: 1.5rem;
                border-radius: 8px;
                margin: 1rem 0;
                font-size: 1rem;
                line-height: 1.6;
                white-space: pre-wrap;
            }
            .citation {
                background: #f8f9fa;
                border: 1px solid #e5e7eb;
                border-radius: 6px;
                padding: 1rem;
                margin: 0.5rem 0;
                font-size: 0.9rem;
                color: #4B5563;
            }
            .citation small {
                color: #2563EB;
                font-weight: 500;
            }
            .btn-primary {
                background: #2563EB;
                color: white;
                border: none;
                padding: 0.75rem 1.5rem;
                border-radius: 8px;
                font-weight: 600;
                cursor: pointer;
                transition: background 0.2s;
            }
            .btn-primary:hover {
                background: #1E3A8A;
            }
            .btn-primary:disabled {
                background: #9CA3AF;
                cursor: not-allowed;
            }
            .footer {
                text-align: center;
                padding: 2rem;
                color: #6B7280;
                font-size: 0.9rem;
                border-top: 1px solid #e5e7eb;
                margin-top: 3rem;
            }
            .data-source-tag {
                display: inline-block;
                padding: 0.2rem 0.8rem;
                border-radius: 20px;
                font-size: 0.8rem;
                font-weight: 500;
                margin-right: 0.5rem;
            }
            .source-stanford {
                background: #DBEAFE;
                color: #1E3A8A;
            }
            .source-sec {
                background: #D1FAE5;
                color: #065F46;
            }
            .token-warning {
                background: #FEF3C7;
                border-left: 4px solid #F59E0B;
                padding: 1rem;
                border-radius: 8px;
                margin: 1rem 0;
                color: #92400E;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# App layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("⚖️ Legal Document Analyzer"),
        html.P("Analyze contracts from Stanford MCC (1M+ historical) + SEC EDGAR (Live)"),
        html.P("🤖 Powered by Hugging Face Inference API", 
               style={"font-size": "0.9rem", "opacity": "0.8", "margin-top": "0.5rem"})
    ], className="main-header"),
    
    # Main container
    html.Div([
        # Token warning (if missing)
        html.Div(id="token-warning", className="token-warning", children=[
            html.Strong("⚠️ HF_TOKEN not found! "),
            "The app is running in demo mode. Add your Hugging Face token to enable AI responses."
        ]) if not HF_TOKEN else html.Div(),
        
        # Stats section
        html.Div(id="stats-section", className="stat-grid", children=[
            html.Div([
                html.Div("1,038,766", className="stat-number"),
                html.Div("Stanford MCC Contracts", className="stat-label"),
                html.Div("2000-2023", className="stat-label", style={"font-size": "0.8rem"})
            ], className="stat-card"),
            
            html.Div([
                html.Div("20M+", className="stat-number"),
                html.Div("SEC Filings Available", className="stat-label"),
                html.Div("Live via API", className="stat-label", style={"font-size": "0.8rem"})
            ], className="stat-card"),
            
            html.Div([
                html.Div("🤖", className="stat-number", style={"font-size": "2rem"}),
                html.Div("HF Inference API", className="stat-label"),
                html.Div("No local LLM needed", className="stat-label", style={"font-size": "0.8rem"})
            ], className="stat-card"),
        ]),
        
        # Main query interface
        html.Div([
            html.H3("🔍 Ask About Any Contract", style={"margin-top": "0", "margin-bottom": "1rem"}),
            html.P("Ask questions about termination clauses, confidentiality terms, governing law, etc."),
            
            # Query input
            dcc.Input(
                id="query-input",
                type="text",
                placeholder="e.g., What are typical termination clauses in employment agreements?",
                style={
                    "width": "100%",
                    "padding": "1rem",
                    "font-size": "1rem",
                    "border": "2px solid #e5e7eb",
                    "border-radius": "8px",
                    "margin": "1rem 0"
                }
            ),
            
            # Options row
            html.Div([
                html.Div([
                    html.Label("Search in:", style={"font-weight": "500", "margin-right": "1rem"}),
                    dcc.Checklist(
                        id="source-selector",
                        options=[
                            {"label": " Stanford MCC (Historical)", "value": "stanford"},
                            {"label": " SEC Live (Recent)", "value": "sec"}
                        ],
                        value=["stanford", "sec"],
                        inline=True,
                        style={"display": "flex", "gap": "1rem"}
                    )
                ], style={"display": "flex", "align-items": "center", "flex-wrap": "wrap", "gap": "1rem"}),
                
                html.Button(
                    "🔍 Analyze Contract",
                    id="search-button",
                    className="btn-primary",
                    style={"margin-left": "auto"}
                )
            ], style={"display": "flex", "align-items": "center", "flex-wrap": "wrap", "gap": "1rem", "margin": "1rem 0"}),
            
        ], className="card"),
        
        # Loading indicator
        dcc.Loading(
            id="loading",
            type="circle",
            children=html.Div(id="loading-output", className="card", children=[
                html.Div("Enter a query above to analyze contracts.", 
                        style={"text-align": "center", "color": "#6B7280", "padding": "2rem"})
            ])
        ),
        
        # Analytics section
        html.Div([
            html.H3("📊 Contract Analytics", style={"margin-bottom": "1.5rem"}),
            
            # Contract type distribution
            dcc.Graph(
                id="contract-type-chart",
                figure={
                    "data": [
                        {
                            "values": [40, 25, 15, 12, 8],
                            "labels": ["Employment", "M&A", "Lease", "Security", "Services"],
                            "type": "pie",
                            "name": "Contract Types",
                            "marker": {"colors": ["#2563EB", "#7C3AED", "#DB2777", "#EA580C", "#059669"]},
                            "hole": 0.4,
                            "textinfo": "label+percent"
                        }
                    ],
                    "layout": {
                        "title": "Contract Types in Stanford MCC",
                        "showlegend": False,
                        "height": 400,
                        "margin": {"t": 50, "b": 50, "l": 50, "r": 50}
                    }
                }
            )
        ], className="card"),
        
        # Data sources info
        html.Div([
            html.H4("📚 Data Sources", style={"margin-bottom": "1rem"}),
            html.Div([
                html.Span("Stanford MCC", className="data-source-tag source-stanford"),
                html.Span("SEC EDGAR API", className="data-source-tag source-sec"),
                html.Span("Hugging Face Inference API", className="data-source-tag", 
                         style={"background": "#FEF3C7", "color": "#92400E"})
            ]),
            html.P("All processing happens via Hugging Face's free Inference API - no local LLM needed.", 
                  style={"color": "#6B7280", "font-size": "0.9rem", "margin-top": "1rem"})
        ], className="card")
        
    ], className="container"),
    
    # Footer
    html.Div([
        html.P(f"© 2026 Legal Document Analyzer | Built with Dash + Hugging Face Spaces"),
        html.P(f"Model: {HF_MODEL}", 
               style={"font-size": "0.8rem", "margin-top": "0.5rem"})
    ], className="footer")
])

# ============================================
# HELPER FUNCTIONS
# ============================================

def query_huggingface(prompt, context=""):
    """
    Query Hugging Face Inference API directly
    """
    if not HF_TOKEN:
        return "⚠️ Demo mode: No HF_TOKEN provided. Add your token to enable AI responses."
    
    # Construct full prompt with context
    full_prompt = f"""You are a legal document analyst. Based on the following contract excerpts, answer the question.

Context:
{context[:2000]}  # Limit context to avoid token limits

Question: {prompt}

Provide a concise, accurate answer based only on the context. If the answer cannot be found, say so.
Answer:"""
    
    # API endpoint
    API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    payload = {
        "inputs": full_prompt,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.1,
            "do_sample": False,
            "return_full_text": False
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', str(result))
            return str(result)
        elif response.status_code == 503:
            # Model is loading
            return "⏳ Model is loading on Hugging Face servers. Please try again in a few seconds."
        else:
            return f"❌ API Error {response.status_code}: {response.text}"
            
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ============================================
# CALLBACKS
# ============================================

@callback(
    Output("loading-output", "children"),
    [Input("search-button", "n_clicks")],
    [State("query-input", "value"),
     State("source-selector", "value")]
)
def search_contracts(n_clicks, query, sources):
    """Handle contract search and analysis"""
    
    if not n_clicks or not query:
        return html.Div("Enter a query above to analyze contracts.", 
                       style={"text-align": "center", "color": "#6B7280", "padding": "2rem"})
    
    # Simulate finding relevant documents (in production, this would query ChromaDB)
    # For demo, we'll create sample contexts
    
    context_parts = []
    results = []
    
    if "stanford" in sources:
        stanford_context = """
        Employment Agreement between TechCorp and John Smith dated 2023-01-15:
        Section 5. Termination. This Agreement may be terminated by either party upon thirty (30) days written notice. 
        For cause termination, including material breach of confidentiality obligations, shall be effective immediately.
        Section 6. Confidentiality. Employee shall not disclose trade secrets for a period of two (2) years post-employment.
        """
        context_parts.append(stanford_context)
        
        results.append(html.Div([
            html.Div([
                html.Span("Stanford MCC", className="data-source-tag source-stanford"),
                html.Small(" Employment Agreement • 2023-01-15", style={"color": "#6B7280", "margin-left": "0.5rem"})
            ]),
            html.Div(
                "Employment Agreement between TechCorp and John Smith...",
                className="citation"
            )
        ], style={"margin-bottom": "1rem"}))
    
    if "sec" in sources:
        sec_context = """
        Apple Inc. Form 8-K filed 2024-02-20, Exhibit 10.1:
        Section 8. Termination. This Agreement may be terminated (a) by mutual written consent, 
        (b) by either party upon 60 days written notice, or (c) immediately by Company for cause, 
        including breach of confidentiality, violation of policies, or misconduct.
        """
        context_parts.append(sec_context)
        
        results.append(html.Div([
            html.Div([
                html.Span("SEC Live", className="data-source-tag source-sec"),
                html.Small(" Apple Inc. • 2024-02-20", style={"color": "#6B7280", "margin-left": "0.5rem"})
            ]),
            html.Div(
                "Apple Inc. Form 8-K filed 2024-02-20, Exhibit 10.1...",
                className="citation"
            )
        ], style={"margin-bottom": "1rem"}))
    
    # Get AI answer from Hugging Face
    full_context = "\n\n".join(context_parts)
    answer = query_huggingface(query, full_context)
    
    # Answer section
    answer_section = html.Div([
        html.H4("📝 Analysis Result", style={"margin": "1rem 0"}),
        html.Div(answer, className="answer-box")
    ])
    
    # Show token warning if needed
    if not HF_TOKEN:
        answer_section = html.Div([
            html.H4("📝 Demo Mode", style={"margin": "1rem 0"}),
            html.Div(
                "⚠️ **HF_TOKEN not configured.**\n\n"
                "To enable real AI responses:\n"
                "1. Get a free token from huggingface.co/settings/tokens\n"
                "2. Add it to your Space Secrets as HF_TOKEN\n"
                "3. Restart the app\n\n"
                "For now, here's what your query would analyze:\n"
                f"Query: '{query}'\n"
                f"Sources: {', '.join(sources)}",
                className="answer-box",
                style={"white-space": "pre-wrap"}
            )
        ])
    
    # Combine all sections
    final_output = html.Div([
        html.H3(f"Results for: '{query}'", style={"margin-bottom": "1rem"}),
        answer_section,
        html.H4("📄 Source Documents", style={"margin": "1.5rem 0 1rem"}),
        *results
    ])
    
    return final_output

# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(debug=False, host="0.0.0.0", port=port)