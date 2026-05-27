# DeepTrust 🔐

> **A User-Centric Platform for Deepfake Detection and Reporting System**

---

# 📌 Overview

DeepTrust is an AI-powered forensic investigation platform designed to combat the growing threat of AI-generated deepfakes in digital identity systems and banking workflows.

The platform focuses on the Indian banking sector’s evolving **e-KYC** and **Video Customer Identification Process (V-CIP)** ecosystem, where identity fraud using synthetic media has become a major cybersecurity concern.

DeepTrust enables users to:

- 📤 Upload suspicious images
- 🧠 Detect whether an image is AI-generated or authentic
- 🔎 Trace image occurrences across the web
- ⚠️ Assess risks and malicious usage
- 📄 Generate forensic evidence reports
- 🌐 Access official cybercrime complaint portals

---

# 🎯 Objectives

- Detect AI-generated and manipulated images accurately.
- Enable reverse image search for origin tracking.
- Analyze harmful or fraudulent digital content.
- Generate structured evidence reports.
- Assist users in cybercrime complaint submission.

---

# 🏗️ System Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│                        DEEPTRUST                            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  📤 Module 1: Image Upload & Preprocessing                   │
│               │                                              │
│               ▼                                              │
│  🧠 Module 2: AI-Generated Image Detection                   │
│               │                                              │
│               ▼                                              │
│  🔎 Module 3: Reverse Image Search                           │
│               │                                              │
│               ▼                                              │
│  🌐 Module 4: Cybercrime Portal Integration                  │
│               │                                              │
│               ▼                                              │
│  📄 Final Evidence Report Generation                         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

# 🧩 Functional Modules

---

# 📤 Module 1 — Image Upload & Preprocessing

This module standardizes uploaded images before AI analysis.

## 🔹 Features

### 🖼️ Image Conversion

```python
.convert("RGB")
```

Handles:
- PNG transparency
- Grayscale images
- Multiple formats

---

### ⚙️ Feature Extraction

Uses:

```python
AutoImageProcessor.from_pretrained()
```

Performs:
- Image resizing
- Normalization
- Resolution correction

---

### 🔄 Tensor Conversion

```python
return_tensors="pt"
```

Converts images into PyTorch-readable tensors.

---

# 🧠 Module 2 — AI-Generated Image Detection

This module determines whether the uploaded image is real or AI-generated.

## 🔹 Model Used

```text
MLmods/Deep-Fake-Detector-v2-Model
```

Built using:
- HuggingFace Transformers
- PyTorch
- Vision Transformer (ViT)

---

## 🔹 Detection Process

The Vision Transformer:
- Splits images into patches
- Detects AI-generated artifacts
- Differentiates synthetic textures from natural camera noise

Inference optimization:

```python
torch.no_grad()
```

Probability scores are generated using Softmax classification.

### ✅ Output
- Fake Probability
- Real Probability
- Risk Level (HIGH / MEDIUM / LOW)

---

# 🔎 Module 3 — Reverse Image Search

This module identifies where the uploaded image appears online.

## 🔹 Workflow

1. Converts uploaded images into public URLs using ImgBB/Catbox.
2. Uses SerpAPI Google Lens for reverse image search.
3. Collects:
   - Website titles
   - Source URLs
   - Visual matches
   - Metadata references

---

## 🔹 AI Reasoning

The Gemini Agent:
- Evaluates detection confidence
- Triggers web investigation
- Analyzes image context and credibility

This helps identify:
- Stolen images
- Viral fake media
- Deepfake misuse
- Disinformation campaigns

---

# 🌐 Module 4 — Cybercrime Portal Integration

This module guides users to official cybercrime complaint portals.

## 🔹 Supported Regions

| Region | Portal |
|---|---|
| 🇮🇳 India | National Cyber Crime Reporting Portal |
| 🇺🇸 USA | FBI IC3 |
| 🇬🇧 UK | Action Fraud |
| 🇪🇺 EU | Europol |
| 🌍 Global | INHOPE Network |

---

## 🔹 Evidence Support

Generated reports include:
- Detection results
- Reverse search findings
- Risk assessment
- Complaint guidance links

---

# 📄 Evidence Report

The platform automatically generates tamper-proof forensic reports.

## ✅ Includes
- AI detection summary
- Reverse image findings
- Risk classification
- Cybercrime reporting guidance

## 📂 Output

```bash
./reports/
```

Example:

```bash
DeepTrust_Report_[REPORT_ID]_[TIMESTAMP].pdf
```

---

# ⚙️ Technology Stack

## 🖥️ Languages
- Python

## 🧠 AI Libraries
- PyTorch
- HuggingFace Transformers
- Vision Transformers (ViT)

## 🌐 APIs
- SerpAPI
- MCP SDK
- Google Lens Engine

## 💻 Tools
- VS Code
- React

---

# 🚀 Workflow

```text
Image Upload
      ↓
Preprocessing
      ↓
Deepfake Detection
      ↓
Reverse Image Search
      ↓
Risk Assessment
      ↓
Evidence Generation
      ↓
Cybercrime Guidance
```

---

# 📌 Conclusion

DeepTrust provides an AI-driven forensic ecosystem for detecting, investigating, and reporting deepfake-based identity fraud.

By combining:
- Vision Transformers
- Agentic AI
- Reverse Image Search
- Cybercrime Reporting Assistance

the platform delivers a complete investigation workflow suitable for banking and digital identity verification systems.

---

# 👩‍💻 Project Information

| Field | Details |
|---|---|
| Project Title | DEEPTRUST: A User-Centric Platform for Deepfake Detection and Reporting System |
| Organization | Accenture |
| Domain | AI & Machine Learning |
| Developer | Thanushri S |
| Course | M.Sc – AI & ML |

---

# ⭐ DeepTrust

> Building trust in digital identity systems through AI-powered forensic intelligence.
