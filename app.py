"""
Legal Document Analyzer - Dash Version
For Hugging Face Spaces deployment
Combines Stanford MCC + SEC API data
"""

import os
import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime
import base64
import io

# RAG imports
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

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

# Custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
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
        html.H1("Legal Document Analyzer"),
        html.P("Analyze contracts from Stanford MCC (1M+ historical) + SEC EDGAR (Live)")
    ], className="main-header"),
    
    # Main container
    html.Div([
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
                html.Div("🏛️ ⚡", className="stat-number", style={"font-size": "1.8rem"}),
                html.Div("Hybrid Search", className="stat-label"),
                html.Div("Historical + Live", className="stat-label", style={"font-size": "0.8rem"})
            ], className="stat-card"),
        ]),
        
        # Main query interface
        html.Div([
            html.H3("Ask About Any Contract", style={"margin-top": "0", "margin-bottom": "1rem"}),
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
                    "Analyze Contract",
                    id="search-button",
                    className="btn-primary",
                    style={"margin-left": "auto"}
                )
            ], style={"display": "flex", "align-items": "center", "flex-wrap": "wrap", "gap": "1rem", "margin": "1rem 0"}),
            
        ], className="card"),
        
        # Results section
        html.Div(id="results-section", className="card", children=[
            html.Div(id="loading-output", children=[
                html.Div("Enter a query above to analyze contracts.", 
                        style={"text-align": "center", "color": "#6B7280", "padding": "2rem"})
            ])
        ]),
        
        # Analytics section
        html.Div([
            html.H3("Contract Analytics", style={"margin-bottom": "1.5rem"}),
            
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
            html.H4("Data Sources", style={"margin-bottom": "1rem"}),
            html.Div([
                html.Span("Stanford MCC", className="data-source-tag source-stanford"),
                html.Span("SEC EDGAR API", className="data-source-tag source-sec"),
                html.Span("Local LLM (Ollama)", className="data-source-tag", 
                         style={"background": "#FEF3C7", "color": "#92400E"})
            ]),
            html.P("All processing happens locally - your queries are private.", 
                  style={"color": "#6B7280", "font-size": "0.9rem", "margin-top": "1rem"})
        ], className="card")
        
    ], className="container"),
    
    # Footer
    html.Div([
        html.P(f"© 2026 Legal Document Analyzer | Built with Dash + Hugging Face Spaces"),
        html.P("Data: Stanford Material Contracts Corpus + SEC EDGAR", 
               style={"font-size": "0.8rem", "margin-top": "0.5rem"})
    ], className="footer")
])

# Callback for search functionality
@callback(
    [Output("loading-output", "children"),
     Output("contract-type-chart", "figure")],
    [Input("search-button", "n_clicks")],
    [State("query-input", "value"),
     State("source-selector", "value")]
)
def search_contracts(n_clicks, query, sources):
    """Handle contract search and analysis"""
    
    if not n_clicks or not query:
        # Return default view
        default_fig = {
            "data": [
                {
                    "values": [40, 25, 15, 12, 8],
                    "labels": ["Employment", "M&A", "Lease", "Security", "Services"],
                    "type": "pie",
                    "hole": 0.4
                }
            ],
            "layout": {"title": "Contract Types in Stanford MCC", "height": 400}
        }
        return html.Div("Enter a query above to analyze contracts.", 
                       style={"text-align": "center", "color": "#6B7280", "padding": "2rem"}), default_fig
    
    # Simulate search results (in production, this would use your RAG pipeline)
    # For demo purposes, we'll show placeholder results
    
    results = []
    
    if "stanford" in sources:
        results.append(html.Div([
            html.Div([
                html.Span("Stanford MCC", className="data-source-tag source-stanford"),
                html.Small(" 2023-01-15", style={"color": "#6B7280", "margin-left": "0.5rem"})
            ]),
            html.P(f"Found relevant section about '{query}' in Employment Agreement"),
            html.Div(
                "This Employment Agreement may be terminated by either party upon 30 days written notice. For cause termination (including breach of confidentiality or misconduct) is effective immediately...",
                className="citation"
            )
        ], style={"margin-bottom": "1rem"}))
    
    if "sec" in sources:
        results.append(html.Div([
            html.Div([
                html.Span("SEC Live", className="data-source-tag source-sec"),
                html.Small(" AAPL • 2024-02-20", style={"color": "#6B7280", "margin-left": "0.5rem"})
            ]),
            html.P(f"Recent filing related to '{query}' from Apple Inc."),
            html.Div(
                "The Company may terminate this Agreement immediately upon written notice if the Supplier breaches any confidentiality obligation or fails to perform...",
                className="citation"
            )
        ], style={"margin-bottom": "1rem"}))
    
    # Answer section
    answer_section = html.Div([
        html.H4("📝 Analysis Result", style={"margin": "1rem 0"}),
        html.Div(
            f"Based on the contracts analyzed, {query.lower()} typically include provisions for mutual termination with 30-60 days notice, immediate termination for cause (breach, illegality), and survival of confidentiality obligations post-termination. Recent SEC filings show similar patterns with additional Sarbanes-Oxley compliance clauses.",
            className="answer-box"
        )
    ])
    
    # Combine all sections
    final_output = html.Div([
        html.H3(f"Results for: '{query}'", style={"margin-bottom": "1rem"}),
        answer_section,
        html.H4("📄 Source Documents", style={"margin": "1.5rem 0 1rem"}),
        *results
    ])
    
    # Update chart (in production, this would show actual distribution)
    updated_fig = {
        "data": [
            {
                "values": [42, 23, 14, 13, 8],
                "labels": ["Employment", "M&A", "Lease", "Security", "Services"],
                "type": "pie",
                "hole": 0.4
            }
        ],
        "layout": {"title": "Contract Types with Latest SEC Additions", "height": 400}
    }
    
    return final_output, updated_fig

# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(debug=False, host="0.0.0.0", port=port)