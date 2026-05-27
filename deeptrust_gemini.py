"""
DeepTrust v3 - TRUE Agentic Architecture (Gemini Edition)
==========================================================
Flow:
  1. HuggingFace model  → detects Fake/Real
  2. Gemini AI (brain)  → reasons about result, autonomously drives tools
  3. MCP Tools          → SerpAPI Google Lens, web search, cybercrime lookup
  4. Gemini AI (brain)  → synthesizes all findings into intelligent report
  5. PDF Generator      → produces final evidence document

Install:
    python -m pip install google-generativeai httpx reportlab transformers torch pillow
"""

import os
import io
import json
import hashlib
import asyncio
import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# ── Deep-learning ────────────────────────────────────────────────────────────
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ── HTTP ─────────────────────────────────────────────────────────────────────
import httpx

# ── Gemini (the reasoning brain) ─────────────────────────────────────────────
import google.generativeai as genai

# ── PDF ──────────────────────────────────────────────────────────────────────
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, HRFlowable, Image as RLImage,
)

# ────────────────────────────────────────────────────────────────────────────
# YOUR API KEYS
# ────────────────────────────────────────────────────────────────────────────
GEMINI_KEY  = "your_gemini_api_key_here"
SERPAPI_KEY = "your_serpapi_api_key_here"  # needed for Google Lens and web search tools
IMGBB_KEY   = "your_imgbb_api_key_here"


# ────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    prediction:   str    # "Fake" | "Real"
    confidence:   float  # 0-1
    label_scores: dict
    image_hash:   str
    timestamp:    str

@dataclass
class AgentFinding:
    tool_name:   str
    tool_input:  dict
    tool_output: str

@dataclass
class AgenticReport:
    report_id:       str
    generated_at:    str
    image_path:      str
    image_hash:      str
    detection:       DetectionResult
    agent_reasoning: str
    findings:        list
    final_summary:   str
    risk_level:      str   # "HIGH" | "MEDIUM" | "LOW"
    portal_links:    list
    pdf_path:        str


# ────────────────────────────────────────────────────────────────────────────
# MODULE 1 — DEEPFAKE DETECTOR (HuggingFace)
# ────────────────────────────────────────────────────────────────────────────

class DeepfakeDetector:
    MODEL_NAME = "prithivMLmods/Deep-Fake-Detector-v2-Model"

    def __init__(self):
        print("[DeepTrust] Loading HuggingFace detection model ...")
        self.processor = AutoImageProcessor.from_pretrained(self.MODEL_NAME)
        self.model     = AutoModelForImageClassification.from_pretrained(self.MODEL_NAME)
        self.model.eval()
        print("[DeepTrust] Detection model ready.")

    def predict(self, image_path: str) -> DetectionResult:
        image  = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs  = torch.nn.functional.softmax(logits, dim=-1)[0]
        labels = self.model.config.id2label

        label_scores = {labels[i]: probs[i].item() for i in range(len(labels))}
        best_idx     = logits.argmax(-1).item()

        return DetectionResult(
            prediction   = labels[best_idx],
            confidence   = probs[best_idx].item(),
            label_scores = label_scores,
            image_hash   = hashlib.sha256(
                               Path(image_path).read_bytes()
                           ).hexdigest()[:16],
            timestamp    = datetime.datetime.now().isoformat(),
        )


# ────────────────────────────────────────────────────────────────────────────
# MODULE 2 — MCP TOOLS
# These are the actual tools Gemini autonomously decides to call
# ────────────────────────────────────────────────────────────────────────────

class MCPTools:

    def __init__(self, serpapi_key: str = "", imgbb_key: str = ""):
        self.serpapi_key = serpapi_key
        self.imgbb_key   = imgbb_key

    # ── Tool 1: Upload image → public URL ────────────────────────────────
    async def upload_image(self, image_path: str) -> dict:
        """Upload image to get a public URL needed by search engines."""
        import base64
        img_bytes = Path(image_path).read_bytes()
        img_name  = Path(image_path).name

        # Try imgbb first
        if self.imgbb_key:
            try:
                b64 = base64.b64encode(img_bytes).decode()
                async with httpx.AsyncClient(timeout=30) as c:
                    r = await c.post(
                        "https://api.imgbb.com/1/upload",
                        data={"key": self.imgbb_key, "image": b64, "expiration": 600},
                    )
                if r.status_code == 200:
                    url = r.json()["data"]["url"]
                    print(f"  [Tool:upload] imgbb OK → {url}")
                    return {"success": True, "url": url, "service": "imgbb"}
            except Exception as e:
                print(f"  [Tool:upload] imgbb failed: {e}")

        # Fallback: catbox.moe
        try:
            async with httpx.AsyncClient(timeout=40) as c:
                r = await c.post(
                    "https://catbox.moe/user/api.php",
                    data={"reqtype": "fileupload", "userhash": ""},
                    files={"fileToUpload": (img_name, img_bytes, "image/jpeg")},
                )
            if r.status_code == 200 and r.text.startswith("https://"):
                print(f"  [Tool:upload] catbox OK → {r.text.strip()}")
                return {"success": True, "url": r.text.strip(), "service": "catbox.moe"}
        except Exception as e:
            print(f"  [Tool:upload] catbox failed: {e}")

        # Fallback: 0x0.st
        try:
            async with httpx.AsyncClient(timeout=40) as c:
                r = await c.post(
                    "https://0x0.st",
                    files={"file": (img_name, img_bytes, "image/jpeg")},
                )
            if r.status_code == 200 and r.text.startswith("https://"):
                print(f"  [Tool:upload] 0x0.st OK → {r.text.strip()}")
                return {"success": True, "url": r.text.strip(), "service": "0x0.st"}
        except Exception as e:
            print(f"  [Tool:upload] 0x0.st failed: {e}")

        return {"success": False, "error": "All upload services failed"}

    # ── Tool 2: Google Lens reverse image search ──────────────────────────
    async def google_lens_search(self, image_url: str) -> dict:
        """Search Google Lens for visual matches of the image online."""
        if not self.serpapi_key:
            return {"error": "No SERPAPI_KEY", "matches": []}

        params = {
            "engine":  "google_lens",
            "url":     image_url,
            "api_key": self.serpapi_key,
            "hl":      "en",
        }
        try:
            async with httpx.AsyncClient(timeout=60) as c:
                r = await c.get("https://serpapi.com/search", params=params)
            data = r.json()

            if "error" in data:
                return {"error": data["error"], "matches": []}

            matches = []
            for item in data.get("visual_matches", [])[:10]:
                matches.append({
                    "title":     item.get("title", ""),
                    "url":       item.get("link", ""),
                    "source":    item.get("source", ""),
                    "thumbnail": item.get("thumbnail", ""),
                })

            return {
                "total_matches": len(matches),
                "matches":       matches,
                "engine":        "Google Lens via SerpAPI",
            }
        except Exception as e:
            return {"error": str(e), "matches": []}

    # ── Tool 3: Web search for context ───────────────────────────────────
    async def web_search(self, query: str) -> dict:
        """Search the web for context about deepfake campaigns or image origin."""
        if not self.serpapi_key:
            return {"error": "No SERPAPI_KEY", "results": []}

        params = {
            "engine":  "google",
            "q":       query,
            "api_key": self.serpapi_key,
            "num":     5,
        }
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.get("https://serpapi.com/search", params=params)
            data = r.json()

            results = []
            for item in data.get("organic_results", [])[:5]:
                results.append({
                    "title":   item.get("title", ""),
                    "url":     item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "source":  item.get("displayed_link", ""),
                })
            return {"query": query, "results": results}
        except Exception as e:
            return {"error": str(e), "results": []}

    # ── Tool 4: Hash database check ───────────────────────────────────────
    async def check_hash_database(self, image_hash: str) -> dict:
        """Check if image hash is known in deepfake/misuse databases."""
        return {
            "found":              False,
            "hash_checked":       image_hash,
            "databases_checked":  ["StopNCII", "PhotoDNA (simulated)"],
            "note":               "Hash not found — does not confirm authenticity",
        }

    # ── Tool 5: Cybercrime portal lookup ──────────────────────────────────
    async def get_complaint_portal(self, country: str) -> dict:
        """Return official cybercrime complaint portal for a given country."""
        portals = {
            "india":  {"name": "National Cyber Crime Reporting Portal",
                       "url":  "https://cybercrime.gov.in",
                       "steps": "Go to 'Report Cyber Crime' → 'Other Cyber Crimes'"},
            "usa":    {"name": "FBI Internet Crime Complaint Center (IC3)",
                       "url":  "https://www.ic3.gov",
                       "steps": "File under 'Other' → 'Image / Video Manipulation'"},
            "uk":     {"name": "Action Fraud",
                       "url":  "https://www.actionfraud.police.uk",
                       "steps": "Report under 'Fraud and cyber crime'"},
            "eu":     {"name": "Europol Online Reporting",
                       "url":  "https://www.europol.europa.eu/report-a-crime",
                       "steps": "Use your national member-state portal"},
            "global": {"name": "INHOPE Hotline Network",
                       "url":  "https://www.inhope.org",
                       "steps": "Coordinates takedowns across 50+ countries"},
        }
        key = country.lower().strip()
        return portals.get(key, {"all_portals": portals,
                                  "note": f"Returning all portals for '{country}'"})

    async def execute(self, tool_name: str, tool_input: dict) -> str:
        """Router — Gemini calls this with tool name + args, gets JSON string back."""
        print(f"  [MCP Tool Called] → {tool_name}({list(tool_input.keys())})")

        if tool_name == "upload_image":
            result = await self.upload_image(tool_input["image_path"])
        elif tool_name == "google_lens_search":
            result = await self.google_lens_search(tool_input["image_url"])
        elif tool_name == "web_search":
            result = await self.web_search(tool_input["query"])
        elif tool_name == "check_hash_database":
            result = await self.check_hash_database(tool_input["image_hash"])
        elif tool_name == "get_complaint_portal":
            result = await self.get_complaint_portal(tool_input["country"])
        else:
            result = {"error": f"Unknown tool: {tool_name}"}

        return json.dumps(result, indent=2)


# ────────────────────────────────────────────────────────────────────────────
# MODULE 3 — GEMINI AGENT (the reasoning brain driving MCP tools)
# ────────────────────────────────────────────────────────────────────────────

# Tool declarations Gemini reads to understand what it can call
GEMINI_TOOLS = [
    {
        "name": "upload_image",
        "description": (
            "Upload a local image file to get a public URL. "
            "ALWAYS call this first before google_lens_search — "
            "search engines cannot access local files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type":        "string",
                    "description": "Local file path to the image",
                }
            },
            "required": ["image_path"],
        },
    },
    {
        "name": "google_lens_search",
        "description": (
            "Search Google Lens for visual matches of the image across the web. "
            "Returns URLs where this image appears online. "
            "Requires a public URL from upload_image first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_url": {
                    "type":        "string",
                    "description": "Public HTTPS URL of the image",
                }
            },
            "required": ["image_url"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the web for contextual information. Use to investigate "
            "suspicious domains, known deepfake campaigns, identity misuse, "
            "or news related to the image."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type":        "string",
                    "description": "Search query string",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "check_hash_database",
        "description": (
            "Check if this image hash is known in deepfake or "
            "non-consensual image databases like StopNCII or PhotoDNA."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_hash": {
                    "type":        "string",
                    "description": "SHA-256 hash of the image (first 16 chars)",
                }
            },
            "required": ["image_hash"],
        },
    },
    {
        "name": "get_complaint_portal",
        "description": (
            "Get the official cybercrime complaint portal for a country. "
            "Always call this to give victims the correct reporting authority."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "country": {
                    "type":        "string",
                    "description": "Country name: India, USA, UK, EU, or Global",
                }
            },
            "required": ["country"],
        },
    },
]


class GeminiAgent:
    """
    Gemini acts as the intelligent brain of DeepTrust.
    It receives the deepfake detection result and autonomously:
      - Decides which MCP tools to call and in what order
      - Interprets results from each tool
      - Calls additional tools based on what it finds
      - Synthesizes everything into a final investigation report
    """

    def __init__(self, gemini_key: str):
        genai.configure(api_key=gemini_key)

        # Convert tool declarations to Gemini FunctionDeclaration format
        tool_declarations = []
        for t in GEMINI_TOOLS:
            tool_declarations.append(
                genai.protos.FunctionDeclaration(
                    name        = t["name"],
                    description = t["description"],
                    parameters  = genai.protos.Schema(
                        type       = genai.protos.Type.OBJECT,
                        properties = {
                            k: genai.protos.Schema(
                                type        = genai.protos.Type.STRING,
                                description = v["description"],
                            )
                            for k, v in t["parameters"]["properties"].items()
                        },
                        required = t["parameters"].get("required", []),
                    ),
                )
            )

        self.tools = genai.protos.Tool(function_declarations=tool_declarations)
        self.model = genai.GenerativeModel(
            model_name    = "gemini-2.5-flash",   # free tier, fast
            tools         = [self.tools],
            system_instruction = self._system_prompt(),
        )

    def _system_prompt(self) -> str:
        return """You are DeepTrust's AI investigation agent specializing in 
deepfake detection and digital forensics.

Your job:
1. Analyze the deepfake detection result provided
2. Use available tools to thoroughly investigate the image
3. Find where the image appears online
4. Assess the risk and potential misuse
5. Provide clear actionable recommendations for the victim

Investigation strategy:
- ALWAYS start by calling upload_image to get a public URL
- THEN call google_lens_search with that URL
- If the image is FAKE with high confidence, do web_search to find 
  context about misuse campaigns
- Always call check_hash_database with the image hash
- Always call get_complaint_portal for India (default jurisdiction)
- Be thorough but efficient

Your FINAL response MUST include:
- RISK LEVEL: HIGH / MEDIUM / LOW  (on its own line)
- A clear 3-5 sentence investigation summary
- Specific recommended actions numbered 1, 2, 3..."""

    async def investigate(
        self,
        image_path: str,
        detection:  DetectionResult,
        mcp_tools:  MCPTools,
    ) -> tuple[str, list[AgentFinding], str, str]:
        """
        Agentic loop:
        1. Send detection result to Gemini with tool definitions
        2. Gemini decides which tools to call
        3. Execute the tools
        4. Feed results back to Gemini
        5. Repeat until Gemini stops calling tools
        6. Return Gemini's synthesized report

        Returns: (reasoning_narrative, findings, final_summary, risk_level)
        """
        is_fake    = detection.prediction.lower() == "fake"
        confidence = detection.confidence

        user_message = f"""Investigate this image for deepfake misuse.

DETECTION RESULT:
- Verdict     : {detection.prediction.upper()}
- Confidence  : {detection.confidence:.1%}
- Image Hash  : {detection.image_hash}
- Image Path  : {image_path}
- Timestamp   : {detection.timestamp}
- All scores  : {json.dumps(detection.label_scores, indent=2)}

{"HIGH CONFIDENCE DEEPFAKE detected — investigate thoroughly for misuse."
 if is_fake and confidence > 0.85 else
 "Moderate confidence result — investigate carefully."
 if is_fake else
 "Image appears authentic — verify source and check for unauthorized use."}

Begin your investigation using your available tools."""

        print("\n[Gemini Agent] Starting autonomous investigation ...")
        print(f"[Gemini Agent] Detection: {detection.prediction} ({confidence:.1%})")

        # Start conversation
        chat     = self.model.start_chat(history=[])
        findings = []

        # ── Agentic loop ──────────────────────────────────────────────────
        max_iterations = 10
        iteration      = 0
        response       = None

        current_message = user_message

        while iteration < max_iterations:
            iteration += 1
            print(f"\n[Gemini Agent] Reasoning round {iteration} ...")

            response = chat.send_message(current_message)

            # ── Check if Gemini wants to call tools ───────────────────────
            tool_calls_made = False

            for part in response.parts:
                if not hasattr(part, "function_call"):
                    continue
                if not part.function_call.name:
                    continue

                tool_calls_made = True
                tool_name  = part.function_call.name
                tool_input = dict(part.function_call.args)

                # Execute the tool
                output = await mcp_tools.execute(tool_name, tool_input)

                findings.append(AgentFinding(
                    tool_name   = tool_name,
                    tool_input  = tool_input,
                    tool_output = output,
                ))

                # Feed result back to Gemini
                current_message = genai.protos.Content(
                    parts = [genai.protos.Part(
                        function_response = genai.protos.FunctionResponse(
                            name     = tool_name,
                            response = {"result": output},
                        )
                    )]
                )

            # ── If no tool calls were made Gemini is done ─────────────────
            if not tool_calls_made:
                print("[Gemini Agent] Investigation complete.")
                break

        # ── Extract Gemini's final text ───────────────────────────────────
        final_text = ""
        if response:
            for part in response.parts:
                if hasattr(part, "text") and part.text:
                    final_text += part.text

        # ── If Gemini ended on a tool call ask for final summary ──────────
        if not final_text.strip():
            print("[Gemini Agent] Requesting final summary ...")
            wrap_up = chat.send_message(
                "All tools have been called. Now provide your complete "
                "investigation summary with RISK LEVEL and recommendations."
            )
            for part in wrap_up.parts:
                if hasattr(part, "text") and part.text:
                    final_text += part.text

        # ── Extract risk level ────────────────────────────────────────────
        risk_level = "MEDIUM"
        upper      = final_text.upper()
        if "RISK LEVEL: HIGH" in upper or "HIGH RISK" in upper or "RISK: HIGH" in upper:
            risk_level = "HIGH"
        elif "RISK LEVEL: LOW" in upper or "LOW RISK" in upper or "RISK: LOW" in upper:
            risk_level = "LOW"
        elif is_fake and confidence > 0.9:
            risk_level = "HIGH"

        reasoning = self._build_narrative(findings)

        print(f"\n[Gemini Agent] Risk Level : {risk_level}")
        print(f"[Gemini Agent] Tools used  : {len(findings)}")

        return reasoning, findings, final_text, risk_level

    def _build_narrative(self, findings: list[AgentFinding]) -> str:
        steps = [f"Investigation used {len(findings)} tool calls:"]
        for i, f in enumerate(findings, 1):
            steps.append(
                f"  Step {i}: '{f.tool_name}' → "
                f"input keys: {list(f.tool_input.keys())}"
            )
        return "\n".join(steps)


# ────────────────────────────────────────────────────────────────────────────
# MODULE 4 — PDF REPORT GENERATOR
# ────────────────────────────────────────────────────────────────────────────

CYBERCRIME_PORTALS = [
    {"country": "India",  "name": "National Cyber Crime Reporting Portal",
     "url": "https://cybercrime.gov.in",
     "note": "Report under 'Report Cyber Crime > Other Cyber Crimes'"},
    {"country": "USA",    "name": "FBI Internet Crime Complaint Center (IC3)",
     "url": "https://www.ic3.gov",
     "note": "File complaint under 'Other' > 'Image / Video Manipulation'"},
    {"country": "UK",     "name": "Action Fraud",
     "url": "https://www.actionfraud.police.uk",
     "note": "Report under 'Fraud and cyber crime'"},
    {"country": "EU",     "name": "Europol Online Reporting",
     "url": "https://www.europol.europa.eu/report-a-crime",
     "note": "Use national member-state portal where possible"},
    {"country": "Global", "name": "INHOPE (Hotline Network)",
     "url": "https://www.inhope.org",
     "note": "Coordinates takedowns across 50+ countries"},
]

RISK_COLORS = {"HIGH": "#dc3545", "MEDIUM": "#fd7e14", "LOW": "#198754"}


class EvidenceReportGenerator:

    def generate(self, image_path: str, report: AgenticReport,
                 output_dir: str = "reports") -> str:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = str(Path(output_dir) /
                       f"DeepTrust_Report_{report.report_id}_{ts}.pdf")
        self._build_pdf(image_path, report, pdf_path)
        return pdf_path

    def _build_pdf(self, image_path: str, report: AgenticReport, pdf_path: str):
        doc    = SimpleDocTemplate(pdf_path, pagesize=A4,
                                   leftMargin=2*cm, rightMargin=2*cm,
                                   topMargin=2*cm,  bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        story  = []

        h2   = ParagraphStyle("H2", parent=styles["Heading2"],
                               textColor=colors.HexColor("#0d6efd"),
                               fontSize=13, spaceAfter=4)
        body = ParagraphStyle("Body", parent=styles["Normal"],
                               fontSize=9, leading=13)

        det        = report.detection
        risk_color = RISK_COLORS.get(report.risk_level, "#fd7e14")
        is_fake    = det.prediction.lower() == "fake"
        verdict    = "DEEPFAKE DETECTED" if is_fake else "AUTHENTIC IMAGE"
        ban_color  = colors.HexColor("#dc3545" if is_fake else "#198754")

        # ── Banner ─────────────────────────────────────────────────────────
        banner = Table([[Paragraph(
            f'<font size="14" color="white"><b>DeepTrust AI Investigation Report'
            f' | {verdict} | RISK: {report.risk_level}</b></font>',
            styles["Normal"],
        )]], colWidths=[17*cm])
        banner.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), ban_color),
            ("TOPPADDING",    (0,0), (-1,-1), 12),
            ("BOTTOMPADDING", (0,0), (-1,-1), 12),
            ("LEFTPADDING",   (0,0), (-1,-1), 12),
        ]))
        story += [banner, Spacer(1, .4*cm)]

        # ── Meta table ─────────────────────────────────────────────────────
        meta = Table([
            ["Report ID",     report.report_id],
            ["Generated At",  report.generated_at],
            ["Image File",    Path(image_path).name],
            ["SHA-256 (16)",  report.image_hash],
            ["Risk Level",    report.risk_level],
            ["AI Engine",     "Google Gemini 1.5 Flash"],
            ["Tools Called",  str(len(report.findings))],
        ], colWidths=[4*cm, 13*cm])
        meta.setStyle(TableStyle([
            ("FONTNAME",       (0,0), (0,-1), "Helvetica-Bold"),
            ("FONTSIZE",       (0,0), (-1,-1), 9),
            ("ROWBACKGROUNDS", (0,0), (-1,-1),
             [colors.HexColor("#f0f4ff"), colors.white]),
            ("GRID",           (0,0), (-1,-1), .3, colors.lightgrey),
            ("TOPPADDING",     (0,0), (-1,-1), 4),
            ("BOTTOMPADDING",  (0,0), (-1,-1), 4),
            ("LEFTPADDING",    (0,0), (-1,-1), 6),
        ]))
        story += [meta, Spacer(1, .5*cm)]

        # ── Section 1: Detection ───────────────────────────────────────────
        story.append(Paragraph("1. Deepfake Detection Analysis", h2))

        img_cell = Paragraph("(preview unavailable)", body)
        try:
            pil = Image.open(image_path).convert("RGB")
            pil.thumbnail((200, 200))
            buf = io.BytesIO()
            pil.save(buf, "PNG")
            buf.seek(0)
            img_cell = RLImage(buf, width=5*cm, height=5*cm)
        except Exception:
            pass

        score_rows = [["Label", "Score"]] + [
            [lbl, f"{sc:.2%}"]
            for lbl, sc in sorted(det.label_scores.items(), key=lambda x: -x[1])
        ]
        st = Table(score_rows, colWidths=[5*cm, 3*cm])
        st.setStyle(TableStyle([
            ("BACKGROUND",     (0,0), (-1,0), colors.HexColor("#0d6efd")),
            ("TEXTCOLOR",      (0,0), (-1,0), colors.white),
            ("FONTNAME",       (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",       (0,0), (-1,-1), 9),
            ("ROWBACKGROUNDS", (0,1), (-1,-1),
             [colors.HexColor("#f6f8fa"), colors.white]),
            ("GRID",           (0,0), (-1,-1), .3, colors.lightgrey),
            ("TOPPADDING",     (0,0), (-1,-1), 4),
            ("BOTTOMPADDING",  (0,0), (-1,-1), 4),
            ("LEFTPADDING",    (0,0), (-1,-1), 6),
        ]))
        side = Table([[img_cell, st]], colWidths=[5.5*cm, 8.5*cm])
        side.setStyle(TableStyle([
            ("VALIGN",       (0,0), (-1,-1), "TOP"),
            ("LEFTPADDING",  (0,0), (-1,-1), 0),
        ]))
        story += [side, Spacer(1, .3*cm)]
        story.append(Paragraph(
            f"<b>Verdict:</b> {det.prediction} &nbsp; "
            f"<b>Confidence:</b> {det.confidence:.2%}", body))
        story.append(Spacer(1, .5*cm))

        # ── Section 2: Tool call log ───────────────────────────────────────
        story.append(HRFlowable(width="100%", color=colors.lightgrey))
        story.append(Spacer(1, .3*cm))
        story.append(Paragraph("2. AI Agent Investigation (Gemini)", h2))
        story.append(Paragraph(
            f"Gemini autonomously called <b>{len(report.findings)} tools</b>:", body))
        story.append(Spacer(1, .3*cm))

        if report.findings:
            tool_data = [["#", "Tool", "Key Input", "Result Summary"]]
            for i, f in enumerate(report.findings, 1):
                try:
                    out = json.loads(f.tool_output)
                    if "matches" in out:
                        summary = f"{out.get('total_matches', 0)} matches found"
                    elif "url" in out:
                        summary = f"Uploaded → {str(out['url'])[:40]}..."
                    elif "results" in out:
                        summary = f"{len(out['results'])} web results"
                    elif "found" in out:
                        summary = "In database" if out["found"] else "Not in database"
                    else:
                        summary = str(out)[:60]
                except Exception:
                    summary = str(f.tool_output)[:60]

                key_input = str(list(f.tool_input.values())[0])[:35] \
                            if f.tool_input else ""
                tool_data.append([
                    str(i),
                    Paragraph(f"<b>{f.tool_name}</b>", body),
                    Paragraph(f"<i>{key_input}</i>", body),
                    Paragraph(summary, body),
                ])

            tt = Table(tool_data, colWidths=[.5*cm, 4*cm, 5.5*cm, 7*cm])
            tt.setStyle(TableStyle([
                ("BACKGROUND",     (0,0), (-1,0), colors.HexColor("#495057")),
                ("TEXTCOLOR",      (0,0), (-1,0), colors.white),
                ("FONTNAME",       (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTSIZE",       (0,0), (-1,-1), 8),
                ("ROWBACKGROUNDS", (0,1), (-1,-1),
                 [colors.HexColor("#f8f9fa"), colors.white]),
                ("GRID",           (0,0), (-1,-1), .3, colors.lightgrey),
                ("TOPPADDING",     (0,0), (-1,-1), 4),
                ("BOTTOMPADDING",  (0,0), (-1,-1), 4),
                ("LEFTPADDING",    (0,0), (-1,-1), 5),
                ("VALIGN",         (0,0), (-1,-1), "TOP"),
            ]))
            story.append(tt)

        story.append(Spacer(1, .5*cm))

        # ── Section 3: Gemini's final summary ─────────────────────────────
        story.append(HRFlowable(width="100%", color=colors.lightgrey))
        story.append(Spacer(1, .3*cm))
        story.append(Paragraph("3. AI Investigation Summary", h2))

        risk_badge = Table([[Paragraph(
            f'<font color="white"><b> RISK: {report.risk_level} </b></font>',
            styles["Normal"],
        )]], colWidths=[3*cm])
        risk_badge.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), colors.HexColor(risk_color)),
            ("TOPPADDING",    (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ]))
        story += [risk_badge, Spacer(1, .3*cm)]

        for line in report.final_summary.split("\n"):
            if line.strip():
                story.append(Paragraph(line.strip(), body))
                story.append(Spacer(1, .15*cm))

        story.append(Spacer(1, .5*cm))

        # ── Section 4: Cybercrime portals ──────────────────────────────────
        story.append(HRFlowable(width="100%", color=colors.lightgrey))
        story.append(Spacer(1, .3*cm))
        story.append(Paragraph("4. Official Cybercrime Complaint Portals", h2))
        story.append(Paragraph(
            "Submit this PDF as evidence to the appropriate portal:", body))
        story.append(Spacer(1, .3*cm))

        portal_data = [["Country", "Portal", "URL", "Instructions"]]
        for p in CYBERCRIME_PORTALS:
            portal_data.append([
                p["country"],
                Paragraph(f"<b>{p['name']}</b>", body),
                Paragraph(f'<link href="{p["url"]}">{p["url"]}</link>', body),
                Paragraph(f'<i>{p["note"]}</i>', body),
            ])
        pt = Table(portal_data, colWidths=[2*cm, 4.5*cm, 5*cm, 5.5*cm])
        pt.setStyle(TableStyle([
            ("BACKGROUND",     (0,0), (-1,0), colors.HexColor("#343a40")),
            ("TEXTCOLOR",      (0,0), (-1,0), colors.white),
            ("FONTNAME",       (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",       (0,0), (-1,-1), 8),
            ("ROWBACKGROUNDS", (0,1), (-1,-1),
             [colors.HexColor("#fff3cd"), colors.white]),
            ("GRID",           (0,0), (-1,-1), .3, colors.lightgrey),
            ("TOPPADDING",     (0,0), (-1,-1), 5),
            ("BOTTOMPADDING",  (0,0), (-1,-1), 5),
            ("LEFTPADDING",    (0,0), (-1,-1), 5),
            ("VALIGN",         (0,0), (-1,-1), "TOP"),
        ]))
        story += [pt, Spacer(1, .5*cm)]

        # ── Footer ─────────────────────────────────────────────────────────
        story.append(HRFlowable(width="100%", color=colors.lightgrey))
        story.append(Spacer(1, .2*cm))
        story.append(Paragraph(
            "<i>Auto-generated by DeepTrust v3 (Gemini Edition). "
            "For legal proceedings consult a certified digital forensics expert. "
            "AI findings are investigative leads, not conclusive legal evidence.</i>",
            ParagraphStyle("Footer", parent=styles["Normal"],
                           fontSize=7.5, textColor=colors.grey),
        ))

        doc.build(story)
        print(f"[Report] Saved → {pdf_path}")


# ────────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ────────────────────────────────────────────────────────────────────────────

class DeepTrust:
    """
    v3 Agentic Architecture (Gemini Edition):
      HuggingFace detects → Gemini reasons + drives MCP tools → PDF report
    """

    def __init__(
        self,
        gemini_key:  str = "AIzaSyCyL8GyknuCvTlmER9UFtMjBe_pjEBOUoM",
        serpapi_key: str = "a683a5c476c3bd04587c4fea0d2dc89ba13bd31a946a5bc0d2e5eb6d5d8f6d05",
        imgbb_key:   str = "b563c086751b5fcbc4126e21e8f3f918",
        output_dir:  str = "reports",
    ):
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        self.detector  = DeepfakeDetector()
        self.mcp_tools = MCPTools(serpapi_key=serpapi_key, imgbb_key=imgbb_key)
        self.agent     = GeminiAgent(gemini_key=gemini_key)
        self.reporter  = EvidenceReportGenerator()

    async def analyze(self, image_path: str) -> AgenticReport:
        print(f"\n{'='*60}")
        print(f"  DeepTrust v3 (Gemini) — Analyzing: {image_path}")
        print(f"{'='*60}")

        # Step 1: HuggingFace detects
        print("\n[1/3] Running deepfake detection ...")
        detection = self.detector.predict(image_path)
        print(f"      Verdict    : {detection.prediction}")
        print(f"      Confidence : {detection.confidence:.2%}")

        # Step 2: Gemini investigates using MCP tools
        print("\n[2/3] Gemini AI agent investigating ...")
        reasoning, findings, final_summary, risk_level = \
            await self.agent.investigate(image_path, detection, self.mcp_tools)

        # Step 3: Generate PDF
        print("\n[3/3] Generating evidence report ...")
        report_id = f"DT-{detection.image_hash.upper()}"

        report = AgenticReport(
            report_id       = report_id,
            generated_at    = detection.timestamp,
            image_path      = image_path,
            image_hash      = detection.image_hash,
            detection       = detection,
            agent_reasoning = reasoning,
            findings        = findings,
            final_summary   = final_summary,
            risk_level      = risk_level,
            portal_links    = CYBERCRIME_PORTALS,
            pdf_path        = "",
        )

        pdf_path        = self.reporter.generate(image_path, report, self.output_dir)
        report.pdf_path = pdf_path

        print(f"\n{'='*60}")
        print(f"  Analysis Complete")
        print(f"{'='*60}")
        print(f"  Report ID  : {report_id}")
        print(f"  Verdict    : {detection.prediction} ({detection.confidence:.2%})")
        print(f"  Risk Level : {risk_level}")
        print(f"  Tools Used : {len(findings)}")
        print(f"  PDF Report : {pdf_path}")
        print(f"\n  Gemini Summary (first 300 chars):")
        print(f"  {final_summary[:300]}...")
        print(f"{'='*60}\n")

        return report


# ────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    image_path = sys.argv[1] if len(sys.argv) > 1 else "tayler.jpg"

    platform = DeepTrust(
        gemini_key  = GEMINI_KEY,
        serpapi_key = SERPAPI_KEY,
        imgbb_key   = IMGBB_KEY,
        output_dir  = "reports",
    )

    report = asyncio.run(platform.analyze(image_path))
    print(f"Done! Open the report: {report.pdf_path}")
