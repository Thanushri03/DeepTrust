# DeepTrust
DeepTrust is a Flask-based deepfake analysis tool that inspects uploaded images with Gemini backends and generates PDF evidence reports.
# DeepTrust 🔬

**End-to-end deepfake detection, verification, and reporting platform.**

## Architecture

```
┌─────────────────────────────────────────────────┐
│              DeepTrust Platform                 │
│                                                 │
│  ┌──────────────┐   ┌────────────────────────┐  │
│  │  Module 1    │   │      Module 2           │  │
│  │  Deepfake    │──▶│  Reverse Image Search  │  │
│  │  Detection   │   │  (SerpAPI / MCP)       │  │
│  │  (HuggingFace│   └────────────────────────┘  │
│  │   Transformer│            │                  │
│  └──────────────┘            ▼                  │
│         │          ┌────────────────────────┐   │
│         │          │      Module 3           │   │
│         └─────────▶│  Evidence Report (PDF) │   │
│                    │  + Cybercrime Portals  │   │
│                    └────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

## Modules

### 1. Deepfake Detection
Uses `prithivMLmods/Deep-Fake-Detector-v2-Model` from HuggingFace.

### 2. Reverse Image Search (MCP + SerpAPI)
- **MCP path**: Connects via Model Context Protocol to any MCP-compatible search server  
- **Fallback**: Direct SerpAPI Google Reverse Image Search  
- Set `SERPAPI_KEY` env var for live results

### 3. Cybercrime Portal Integration
Auto-generates a PDF evidence report and provides direct links to:
- India: cybercrime.gov.in
- USA: FBI IC3 (ic3.gov)
- UK: Action Fraud
- EU: Europol
- Global: INHOPE Network

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. CLI usage (standalone)
python deeptrust_backend.py path/to/image.jpg

# 3. API server (for web UI)
uvicorn server:app --reload --port 8000

# 4. Open the web UI
# Edit deeptrust_ui.html: set USE_REAL_BACKEND = true
# Open in browser
```

## Environment Variables

| Variable      | Purpose                          |
|---------------|----------------------------------|
| `SERPAPI_KEY` | SerpAPI key for reverse search   |
| `OPENAI_API_KEY` | OpenAI API key for AI agent reasoning |

## MCP Server (optional)

To connect a real MCP search server:
```python
platform = DeepTrust(
    mcp_command=["npx", "@anthropic-ai/mcp-server-web-search"],
    serpapi_key="your_key",
    openai_key="your_openai_key"
)
```

## Output

Each analysis produces:
- Console summary with verdict + matches
- PDF evidence report in `./reports/`
- JSON response (via API)
