"""
DeepTrust - End-to-End Deepfake Detection, Verification & Reporting Platform
=============================================================================
Modules:
  1. Deepfake Detection        - prithivMLmods/Deep-Fake-Detector-v2-Model
  2. Reverse Image Search      - SerpAPI / Google Lens / TinEye via MCP
  3. MCP Integration           - Model Context Protocol for external services
  4. Cybercrime Portal Report  - Auto-generates evidence PDF + portal links
"""

import os
import io
import json
import hashlib
import asyncio
import datetime
import textwrap
import urllib.parse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

# ── Deep-learning deps ──────────────────────────────────────────────────────
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ── HTTP / async ─────────────────────────────────────────────────────────────
import httpx                        # pip install httpx
from mcp import ClientSession, StdioServerParameters   # pip install mcp
from mcp.client.stdio import stdio_client

# ── AI Agent ────────────────────────────────────────────────────────────────
from openai import OpenAI            # pip install openai

# ── PDF generation ───────────────────────────────────────────────────────────
from reportlab.lib.pagesizes import A4          # pip install reportlab
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    prediction: str          # "Fake" | "Real"
    confidence: float        # 0-1
    label_scores: dict       # {label: score}
    image_hash: str
    timestamp: str

@dataclass
class ReverseSearchHit:
    url: str
    title: str
    source: str
    thumbnail: Optional[str] = None
    snippet: Optional[str] = None

@dataclass
class ReverseSearchResult:
    hits: list[ReverseSearchHit]
    total_found: int
    engine_used: str
    query_image_url: Optional[str] = None

@dataclass
class EvidenceReport:
    report_id: str
    generated_at: str
    image_path: str
    image_hash: str
    detection: DetectionResult
    reverse_search: ReverseSearchResult
    portal_links: list[dict]
    pdf_path: str


# ────────────────────────────────────────────────────────────────────────────
# MODULE 1 – DEEPFAKE DETECTION
# ────────────────────────────────────────────────────────────────────────────

class DeepfakeDetector:
    """Wraps the HuggingFace deepfake classifier."""

    MODEL_NAME = "prithivMLmods/Deep-Fake-Detector-v2-Model"

    def __init__(self):
        print("[DeepTrust] Loading detection model …")
        self.processor = AutoImageProcessor.from_pretrained(self.MODEL_NAME)
        self.model     = AutoModelForImageClassification.from_pretrained(self.MODEL_NAME)
        self.model.eval()
        print("[DeepTrust] Model ready.")

    def predict(self, image_path: str) -> DetectionResult:
        image  = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs  = torch.nn.functional.softmax(logits, dim=-1)[0]
        labels = self.model.config.id2label

        label_scores = {labels[i]: probs[i].item() for i in range(len(labels))}
        best_idx     = logits.argmax(-1).item()
        prediction   = labels[best_idx]
        confidence   = probs[best_idx].item()

        img_hash = hashlib.sha256(Path(image_path).read_bytes()).hexdigest()[:16]
        timestamp = datetime.datetime.now().isoformat()

        return DetectionResult(
            prediction   = prediction,
            confidence   = confidence,
            label_scores = label_scores,
            image_hash   = img_hash,
            timestamp    = timestamp,
        )


# ────────────────────────────────────────────────────────────────────────────
# MODULE 2 – MCP CLIENT (Model Context Protocol)
# ────────────────────────────────────────────────────────────────────────────

class MCPSearchClient:
    """
    Connects to an MCP server that exposes web-search / image-search tools.
    Falls back to a direct HTTP search when no MCP server is running.

    To run a real MCP search server:
        npx @anthropic-ai/mcp-server-web-search --api-key $SERPAPI_KEY
    """

    def __init__(self, server_command: Optional[list[str]] = None):
        self.server_command = server_command  # e.g. ["npx", "@anthropic-ai/mcp-server-web-search"]
        self._session: Optional[ClientSession] = None

    async def __aenter__(self):
        if self.server_command:
            self._params = StdioServerParameters(
                command=self.server_command[0],
                args=self.server_command[1:],
                env={"SERPAPI_KEY": os.environ.get("SERPAPI_KEY", "")},
            )
            self._cm = stdio_client(self._params)
            read, write = await self._cm.__aenter__()
            self._session = ClientSession(read, write)
            await self._session.__aenter__()
            await self._session.initialize()
        return self

    async def __aexit__(self, *args):
        if self._session:
            await self._session.__aexit__(*args)
            await self._cm.__aexit__(*args)

    async def reverse_image_search(
        self, image_path: str, serpapi_key: str = ""
    ) -> ReverseSearchResult:
        """
        Tries MCP tool first; falls back to SerpAPI Google Lens REST call.
        """
        # ── Try MCP tool ──────────────────────────────────────────────────
        if self._session:
            try:
                result = await self._session.call_tool(
                    "reverse_image_search",
                    {"image_path": image_path},
                )
                data = json.loads(result.content[0].text)
                hits = [ReverseSearchHit(**h) for h in data.get("hits", [])]
                return ReverseSearchResult(
                    hits=hits,
                    total_found=len(hits),
                    engine_used="MCP / Google Lens",
                )
            except Exception as e:
                print(f"[MCP] Tool call failed ({e}), falling back to HTTP …")

        # ── Fallback: SerpAPI Google Reverse Image Search ─────────────────
        return await self._serpapi_reverse_search(image_path, serpapi_key)

    async def _serpapi_reverse_search(
        self, image_path: str, api_key: str
    ) -> ReverseSearchResult:
        """
        Uploads image to SerpAPI via multipart POST (avoids URL-too-long error)
        and returns structured results.
        Requires SERPAPI_KEY env var or explicit api_key argument.
        """
        key = api_key or os.environ.get("SERPAPI_KEY", "")
        if not key:
            print("[ReverseSearch] No SERPAPI_KEY set – returning mock results.")
            return self._mock_results()

        img_bytes  = Path(image_path).read_bytes()
        img_name   = Path(image_path).name

        # ── Step 1: Upload image file to SerpAPI, get back a search URL ──
        upload_params = {
            "engine":  "google_reverse_image",
            "api_key": key,
            "output":  "json",
        }
        files = {"image_file": (img_name, img_bytes, "image/jpeg")}

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                # SerpAPI accepts multipart upload at the same endpoint
                resp = await client.post(
                    "https://serpapi.com/search",
                    params=upload_params,
                    files=files,
                )
                resp.raise_for_status()
                data = resp.json()

        except httpx.HTTPStatusError as e:
            print(f"[ReverseSearch] SerpAPI HTTP error {e.response.status_code}: {e.response.text[:200]}")
            return self._mock_results()
        except Exception as e:
            print(f"[ReverseSearch] Request failed: {e}")
            return self._mock_results()

        # ── Step 2: Parse results ─────────────────────────────────────────
        hits = []

        # Try multiple result keys SerpAPI may return
        result_items = (
            data.get("image_results") or
            data.get("inline_images") or
            data.get("organic_results") or
            []
        )

        for item in result_items[:10]:
            hits.append(ReverseSearchHit(
                url       = item.get("link", item.get("url", "")),
                title     = item.get("title", "Unknown"),
                source    = item.get("source", item.get("displayed_link", "")),
                thumbnail = item.get("thumbnail", ""),
                snippet   = item.get("snippet", ""),
            ))

        if not hits:
            # Check if SerpAPI returned an error message
            if "error" in data:
                print(f"[ReverseSearch] SerpAPI error: {data['error']}")
            else:
                print(f"[ReverseSearch] No results returned. Keys in response: {list(data.keys())}")

        return ReverseSearchResult(
            hits        = hits,
            total_found = len(hits),
            engine_used = "SerpAPI / Google Reverse Image",
        )

    @staticmethod
    def _mock_results() -> ReverseSearchResult:
        """Placeholder when no API key is available."""
        hits = [
            ReverseSearchHit(
                url     = "https://example-social.com/post/12345",
                title   = "Suspicious post featuring this image",
                source  = "example-social.com",
                snippet = "Image found on social media post (mock result – add SERPAPI_KEY)",
            )
        ]
        return ReverseSearchResult(
            hits        = hits,
            total_found = 1,
            engine_used = "Mock (no API key)",
        )


# ────────────────────────────────────────────────────────────────────────────
# MODULE 2.5 – AI AGENT (Reasoning & MCP Tool Calling)
# ────────────────────────────────────────────────────────────────────────────

class AIAgent:
    """
    Agentic AI that reasons about detection results and autonomously calls MCP tools.
    Uses OpenAI GPT-4 for reasoning and tool calling.
    """

    def __init__(self, openai_key: str, mcp_command: Optional[list[str]] = None):
        self.openai_key = openai_key
        self.mcp_command = mcp_command
        self.client = OpenAI(api_key=openai_key)

    async def analyze(self, detection: DetectionResult, image_path: str) -> ReverseSearchResult:
        """
        Takes detection result, reasons with AI, calls MCP tools as needed, returns search results.
        """
        async with MCPSearchClient(self.mcp_command) as mcp:
            if not mcp._session:
                # Fallback if no MCP
                return await mcp.reverse_image_search(image_path, "")

            # Get available MCP tools
            tools_response = await mcp._session.list_tools()
            mcp_tools = tools_response.tools

            # Convert MCP tools to OpenAI tool format
            openai_tools = []
            for tool in mcp_tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                })

            # AI Prompt
            system_prompt = (
                "You are an intelligent agent for deepfake detection and evidence gathering. "
                "Based on the detection result, decide if reverse image search is necessary. "
                "If the image is likely fake (high confidence), search online to find sources and context. "
                "If real or uncertain, you may skip the search. "
                "Use the available tools to gather information, then summarize the findings."
            )

            user_prompt = (
                f"Detection Result:\n"
                f"- Prediction: {detection.prediction}\n"
                f"- Confidence: {detection.confidence:.2%}\n"
                f"- Image Path: {image_path}\n\n"
                f"Decide if you need to call any tools. If calling reverse_image_search, provide the image_path."
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            # Call OpenAI with tools
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                tools=openai_tools,
                tool_choice="auto"
            )

            tool_calls = response.choices[0].message.tool_calls
            if tool_calls:
                for tool_call in tool_calls:
                    if tool_call.function.name == "reverse_image_search":
                        args = json.loads(tool_call.function.arguments)
                        # Call MCP tool
                        result = await mcp._session.call_tool(
                            tool_call.function.name,
                            args
                        )
                        # Parse result
                        data = json.loads(result.content[0].text)
                        hits = [ReverseSearchHit(**h) for h in data.get("hits", [])]
                        return ReverseSearchResult(
                            hits=hits,
                            total_found=len(hits),
                            engine_used="MCP via AI Agent"
                        )

            # No tool called or no results
            return ReverseSearchResult(
                hits=[],
                total_found=0,
                engine_used="AI Agent (no search performed)"
            )


# ────────────────────────────────────────────────────────────────────────────
# MODULE 3 – CYBERCRIME PORTAL LINKS
# ────────────────────────────────────────────────────────────────────────────

CYBERCRIME_PORTALS = [
    {
        "country": "India",
        "name":    "National Cyber Crime Reporting Portal",
        "url":     "https://cybercrime.gov.in",
        "note":    "Report under 'Report Cyber Crime > Other Cyber Crimes'",
    },
    {
        "country": "USA",
        "name":    "FBI Internet Crime Complaint Center (IC3)",
        "url":     "https://www.ic3.gov",
        "note":    "File complaint under 'Other' > 'Image / Video Manipulation'",
    },
    {
        "country": "UK",
        "name":    "Action Fraud",
        "url":     "https://www.actionfraud.police.uk",
        "note":    "Report under 'Fraud and cyber crime'",
    },
    {
        "country": "EU",
        "name":    "Europol Online Reporting",
        "url":     "https://www.europol.europa.eu/report-a-crime",
        "note":    "Use national member-state portal where possible",
    },
    {
        "country": "Global",
        "name":    "INHOPE (Hotline Network)",
        "url":     "https://www.inhope.org",
        "note":    "Coordinates takedowns across 50+ countries",
    },
]


# ────────────────────────────────────────────────────────────────────────────
# MODULE 4 – EVIDENCE REPORT GENERATOR (PDF)
# ────────────────────────────────────────────────────────────────────────────

class EvidenceReportGenerator:
    """Generates a professional PDF evidence report."""

    def generate(
        self,
        image_path:     str,
        detection:      DetectionResult,
        reverse_search: ReverseSearchResult,
        output_dir:     str = ".",
    ) -> EvidenceReport:

        report_id  = f"DT-{detection.image_hash.upper()}"
        timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_name   = f"DeepTrust_Report_{report_id}_{timestamp}.pdf"
        pdf_path   = str(Path(output_dir) / pdf_name)

        self._build_pdf(image_path, detection, reverse_search, pdf_path, report_id)

        return EvidenceReport(
            report_id      = report_id,
            generated_at   = detection.timestamp,
            image_path     = image_path,
            image_hash     = detection.image_hash,
            detection      = detection,
            reverse_search = reverse_search,
            portal_links   = CYBERCRIME_PORTALS,
            pdf_path       = pdf_path,
        )

    def _build_pdf(
        self,
        image_path:     str,
        detection:      DetectionResult,
        reverse_search: ReverseSearchResult,
        pdf_path:       str,
        report_id:      str,
    ):
        doc    = SimpleDocTemplate(pdf_path, pagesize=A4,
                                   leftMargin=2*cm, rightMargin=2*cm,
                                   topMargin=2*cm,  bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        story  = []

        # ── Heading style ──────────────────────────────────────────────────
        h1 = ParagraphStyle("H1", parent=styles["Heading1"],
                            textColor=colors.HexColor("#0d1117"),
                            fontSize=20, spaceAfter=4)
        h2 = ParagraphStyle("H2", parent=styles["Heading2"],
                            textColor=colors.HexColor("#0d6efd"),
                            fontSize=13, spaceAfter=4)
        body = ParagraphStyle("Body", parent=styles["Normal"],
                              fontSize=10, leading=14)
        mono = ParagraphStyle("Mono", parent=styles["Code"],
                              fontSize=8, leading=12,
                              backColor=colors.HexColor("#f6f8fa"),
                              borderPadding=4)

        # ── Banner ─────────────────────────────────────────────────────────
        banner_color = (colors.HexColor("#dc3545")
                        if detection.prediction.lower() == "fake"
                        else colors.HexColor("#198754"))
        verdict_text = ("⚠ DEEPFAKE DETECTED" if detection.prediction.lower() == "fake"
                        else "✔ AUTHENTIC IMAGE")

        banner_data = [[Paragraph(
            f'<font size="16" color="white"><b>DeepTrust Evidence Report &nbsp;|&nbsp; '
            f'{verdict_text}</b></font>', styles["Normal"]
        )]]
        banner = Table(banner_data, colWidths=[17*cm])
        banner.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), banner_color),
            ("TOPPADDING",    (0,0), (-1,-1), 12),
            ("BOTTOMPADDING", (0,0), (-1,-1), 12),
            ("LEFTPADDING",   (0,0), (-1,-1), 12),
        ]))
        story.append(banner)
        story.append(Spacer(1, 0.4*cm))

        # ── Meta table ─────────────────────────────────────────────────────
        meta = [
            ["Report ID",    report_id],
            ["Generated At", detection.timestamp],
            ["Image File",   Path(image_path).name],
            ["SHA-256 (16)", detection.image_hash],
        ]
        mt = Table(meta, colWidths=[4*cm, 13*cm])
        mt.setStyle(TableStyle([
            ("FONTNAME",    (0,0), (0,-1), "Helvetica-Bold"),
            ("FONTSIZE",    (0,0), (-1,-1), 9),
            ("ROWBACKGROUNDS", (0,0), (-1,-1),
             [colors.HexColor("#f0f4ff"), colors.white]),
            ("GRID",        (0,0), (-1,-1), 0.3, colors.lightgrey),
            ("TOPPADDING",  (0,0), (-1,-1), 4),
            ("BOTTOMPADDING",(0,0), (-1,-1), 4),
            ("LEFTPADDING", (0,0), (-1,-1), 6),
        ]))
        story.append(mt)
        story.append(Spacer(1, 0.5*cm))

        # ── Image + detection side-by-side ─────────────────────────────────
        story.append(Paragraph("1. Detection Analysis", h2))

        img_cell = ""
        try:
            pil = Image.open(image_path).convert("RGB")
            pil.thumbnail((200, 200))
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            buf.seek(0)
            img_cell = RLImage(buf, width=5*cm, height=5*cm)
        except Exception:
            img_cell = Paragraph("(image preview unavailable)", body)

        score_rows = [["Label", "Confidence"]] + [
            [lbl, f"{sc:.2%}"] for lbl, sc in sorted(
                detection.label_scores.items(), key=lambda x: -x[1]
            )
        ]
        st = Table(score_rows, colWidths=[5*cm, 3*cm])
        st.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,0),  colors.HexColor("#0d6efd")),
            ("TEXTCOLOR",   (0,0), (-1,0),  colors.white),
            ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",    (0,0), (-1,-1), 9),
            ("ROWBACKGROUNDS", (0,1), (-1,-1),
             [colors.HexColor("#f6f8fa"), colors.white]),
            ("GRID",        (0,0), (-1,-1), 0.3, colors.lightgrey),
            ("TOPPADDING",  (0,0), (-1,-1), 4),
            ("BOTTOMPADDING",(0,0), (-1,-1), 4),
            ("LEFTPADDING", (0,0), (-1,-1), 6),
        ]))

        side = Table([[img_cell, st]], colWidths=[5.5*cm, 8.5*cm])
        side.setStyle(TableStyle([
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("LEFTPADDING", (0,0), (-1,-1), 0),
        ]))
        story.append(side)
        story.append(Spacer(1, 0.4*cm))
        story.append(Paragraph(
            f"<b>Verdict:</b> {detection.prediction} &nbsp; "
            f"<b>Confidence:</b> {detection.confidence:.2%}", body
        ))
        story.append(Spacer(1, 0.5*cm))

        # ── Reverse Image Search ───────────────────────────────────────────
        story.append(HRFlowable(width="100%", color=colors.lightgrey))
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph("2. Reverse Image Search Results", h2))
        story.append(Paragraph(
            f"Engine: <b>{reverse_search.engine_used}</b> &nbsp;|&nbsp; "
            f"Matches Found: <b>{reverse_search.total_found}</b>", body
        ))
        story.append(Spacer(1, 0.3*cm))

        if reverse_search.hits:
            rs_data = [["#", "Title / Source", "URL"]]
            for i, hit in enumerate(reverse_search.hits[:8], 1):
                rs_data.append([
                    str(i),
                    Paragraph(f"<b>{hit.title[:60]}</b><br/><i>{hit.source}</i>", body),
                    Paragraph(f'<link href="{hit.url}">{hit.url[:55]}…</link>', body),
                ])
            rst = Table(rs_data, colWidths=[0.6*cm, 8*cm, 8.4*cm])
            rst.setStyle(TableStyle([
                ("BACKGROUND",  (0,0), (-1,0),  colors.HexColor("#6c757d")),
                ("TEXTCOLOR",   (0,0), (-1,0),  colors.white),
                ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
                ("FONTSIZE",    (0,0), (-1,-1), 8),
                ("ROWBACKGROUNDS", (0,1), (-1,-1),
                 [colors.HexColor("#f9f9f9"), colors.white]),
                ("GRID",        (0,0), (-1,-1), 0.3, colors.lightgrey),
                ("TOPPADDING",  (0,0), (-1,-1), 5),
                ("BOTTOMPADDING",(0,0), (-1,-1), 5),
                ("LEFTPADDING", (0,0), (-1,-1), 5),
                ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
            ]))
            story.append(rst)
        else:
            story.append(Paragraph("No matches found online.", body))

        story.append(Spacer(1, 0.5*cm))

        # ── Cybercrime Portals ─────────────────────────────────────────────
        story.append(HRFlowable(width="100%", color=colors.lightgrey))
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph("3. Official Cybercrime Complaint Portals", h2))
        story.append(Paragraph(
            "Submit this report along with the evidence PDF to the relevant portal below:", body
        ))
        story.append(Spacer(1, 0.3*cm))

        portal_data = [["Country", "Portal", "URL", "Notes"]]
        for p in CYBERCRIME_PORTALS:
            portal_data.append([
                p["country"],
                Paragraph(f"<b>{p['name']}</b>", body),
                Paragraph(f'<link href="{p["url"]}">{p["url"]}</link>', body),
                Paragraph(f'<i>{p["note"]}</i>', body),
            ])
        pt = Table(portal_data, colWidths=[2*cm, 4.5*cm, 5*cm, 5.5*cm])
        pt.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,0),  colors.HexColor("#343a40")),
            ("TEXTCOLOR",   (0,0), (-1,0),  colors.white),
            ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",    (0,0), (-1,-1), 8),
            ("ROWBACKGROUNDS", (0,1), (-1,-1),
             [colors.HexColor("#fff3cd"), colors.white]),
            ("GRID",        (0,0), (-1,-1), 0.3, colors.lightgrey),
            ("TOPPADDING",  (0,0), (-1,-1), 5),
            ("BOTTOMPADDING",(0,0), (-1,-1), 5),
            ("LEFTPADDING", (0,0), (-1,-1), 5),
            ("VALIGN",      (0,0), (-1,-1), "TOP"),
        ]))
        story.append(pt)
        story.append(Spacer(1, 0.5*cm))

        # ── Footer ─────────────────────────────────────────────────────────
        story.append(HRFlowable(width="100%", color=colors.lightgrey))
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph(
            "<i>This report was auto-generated by DeepTrust and is intended for use as "
            "supporting evidence in cybercrime complaints. For legal proceedings, "
            "please consult a certified digital forensics expert.</i>",
            ParagraphStyle("Footer", parent=styles["Normal"],
                           fontSize=8, textColor=colors.grey)
        ))

        doc.build(story)
        print(f"[Report] Saved → {pdf_path}")


# ────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR
# ────────────────────────────────────────────────────────────────────────────

class DeepTrust:
    """
    Main orchestrator that ties all modules together.

    Usage:
        platform = DeepTrust(serpapi_key="YOUR_KEY")
        report   = asyncio.run(platform.analyze("path/to/image.jpg"))
        print(report.pdf_path)
    """

    def __init__(
        self,
        serpapi_key:    str  = "",
        mcp_command:    Optional[list[str]] = None,
        output_dir:     str  = "reports",
        openai_key:     str  = "",
    ):
        self.serpapi_key = serpapi_key or os.environ.get("SERPAPI_KEY", "")
        self.mcp_command = mcp_command
        self.output_dir  = output_dir
        self.openai_key  = openai_key or os.environ.get("OPENAI_API_KEY", "")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        self.detector  = DeepfakeDetector()
        self.agent     = AIAgent(self.openai_key, self.mcp_command)
        self.reporter  = EvidenceReportGenerator()

    async def analyze(self, image_path: str) -> EvidenceReport:
        print(f"\n{'='*60}")
        print(f"  DeepTrust – Analyzing: {image_path}")
        print(f"{'='*60}")

        # Step 1 – Detect
        print("\n[1/3] Running deepfake detection …")
        detection = self.detector.predict(image_path)
        print(f"      Verdict    : {detection.prediction}")
        print(f"      Confidence : {detection.confidence:.2%}")

        # Step 2 – AI Agent: Reason and call MCP tools autonomously
        print("\n[2/3] AI Agent reasoning and tool calling …")
        reverse = await self.agent.analyze(detection, image_path)
        print(f"      Engine     : {reverse.engine_used}")
        print(f"      Matches    : {reverse.total_found}")
        for h in reverse.hits[:3]:
            print(f"        • {h.source} – {h.title[:60]}")

        # Step 3 – Generate evidence report PDF
        print("\n[3/3] Generating evidence report …")
        report = self.reporter.generate(
            image_path     = image_path,
            detection      = detection,
            reverse_search = reverse,
            output_dir     = self.output_dir,
        )

        # Summary
        print(f"\n{'='*60}")
        print(f"  Analysis Complete")
        print(f"{'='*60}")
        print(f"  Report ID  : {report.report_id}")
        print(f"  Verdict    : {detection.prediction} ({detection.confidence:.2%})")
        print(f"  PDF Report : {report.pdf_path}")
        print(f"\n  Cybercrime Portals:")
        for p in CYBERCRIME_PORTALS:
            print(f"    [{p['country']}] {p['name']}")
            print(f"           {p['url']}")
        print(f"{'='*60}\n")

        return report


# ────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    image_path = sys.argv[1] if len(sys.argv) > 1 else "tayler.jpg"

    platform = DeepTrust(
        # serpapi_key = "your_serpapi_key_here",   # or set SERPAPI_KEY env var
        # openai_key  = "your_openai_key_here",    # or set OPENAI_API_KEY env var
        # mcp_command = ["npx", "@anthropic-ai/mcp-server-web-search"],
        output_dir  = "reports",
    )

    report = asyncio.run(platform.analyze(image_path))
    print(f"Done. Open the report: {report.pdf_path}")
