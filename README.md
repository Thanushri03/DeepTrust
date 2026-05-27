# DeepTrust 🔐

> **A User-Centric Platform for Deepfake Detection and Reporting System**

---

# 📌 Overview

DeepTrust is an AI-powered forensic investigation platform designed to combat the growing threat of AI-generated deepfakes in the banking and digital identity ecosystem.

The platform focuses on the Indian banking sector’s evolving **e-KYC** and **Video Customer Identification Process (V-CIP)** workflows, where identity fraud using synthetic media has become a major cybersecurity concern.

DeepTrust allows users to:

* 📤 Upload suspicious images
* 🧠 Detect whether the image is AI-generated or authentic
* 🔎 Trace where the image appears online using reverse image search
* ⚠️ Assess potential risks and malicious usage
* 📄 Generate forensic evidence reports
* 🌐 Navigate directly to official cybercrime complaint portals

---

# 🎯 Objectives

## ✅ Core Goals

* Develop an intelligent image analysis system capable of detecting AI-generated images.
* Enable reverse image search functionality to identify image origin and web presence.
* Detect manipulated, fake, or harmful content.
* Generate organized evidence reports for cybercrime investigation.
* Guide users to official cybercrime reporting portals.

---

# 🏗️ System Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│                        DEEPTRUST                            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  📤 Module 1: Image Upload & Preprocessing                   │
│                                                              │
│               │                                              │
│               ▼                                              │
│  🧠 Module 2: AI-Generated Image Detection                   │
│     └── Vision Transformer (ViT)                             │
│                                                              │
│               │                                              │
│               ▼                                              │
│  🔎 Module 3: Reverse Image Search                           │
│     ├── Gemini Agent                                         │
│     ├── MCP Integration                                      │
│     └── SerpAPI Google Lens                                  │
│                                                              │
│               │                                              │
│               ▼                                              │
│  🌐 Module 4: Cybercrime Portal Integration                  │
│                                                              │
│               │                                              │
│               ▼                                              │
│  📄 Final Evidence Report Generation                         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

# 🧩 Functional Modules

The platform is implemented using interconnected AI-driven modules.

---

# 📤 Module 1 — Image Upload & Preprocessing

This module standardizes uploaded images and prepares them for AI-based analysis.

## 🔹 Key Features

### 🖼️ Image Loading & Format Conversion

* Converts uploaded images into RGB format using:

```python
.convert("RGB")
```

* Handles:

  * PNG images with transparency
  * Grayscale images
  * Different image formats

---

### ⚙️ Automated Feature Extraction

Uses:

```python
AutoImageProcessor.from_pretrained()
```

The processor automatically:

* Resizes images
* Applies normalization
* Adjusts image resolution
* Performs color mean/variance correction

---

### 🔄 Tensor Transformation

Transforms image pixels into PyTorch tensors:

```python
return_tensors="pt"
```

This enables compatibility with deep learning models.

---

# 🧠 Module 2 — AI-Generated Image Detection

This module detects whether an uploaded image is authentic or AI-generated.

## 🔹 Deep Learning Model

DeepTrust uses:

```text
MLmods/Deep-Fake-Detector-v2-Model
```

A fine-tuned version of Google’s Vision Transformer (ViT).

---

## 🔹 Technologies Used

* HuggingFace Transformers
* PyTorch
* Vision Transformers (ViT)
* AutoModelForImageClassification

---

## 🔹 Detection Pipeline

### ✅ Vision Transformer Analysis

The Vision Transformer:

* Splits images into patches
* Studies relationships between image regions
* Detects AI artifacts
* Differentiates synthetic textures from natural camera noise

---

### ✅ Optimized Inference

Uses:

```python
torch.no_grad()
```

Benefits:

* Reduced memory usage
* Faster execution
* Efficient inference processing

---

### ✅ Probability Calculation

The model outputs logits which are converted into probabilities using:

```python
Softmax Function
```

Final output includes:

* Fake Probability (%)
* Real Probability (%)
* Risk Category

Risk Levels:

| Confidence | Risk Level |
| ---------- | ---------- |
| > 85%      | HIGH       |
| 60–85%     | MEDIUM     |
| < 60%      | LOW        |

---

# 🔎 Module 3 — Reverse Image Search

This module identifies where the uploaded image appears online.

## 🔹 Intelligent Agent Workflow

The reverse search pipeline is controlled by the Gemini Agent.

The agent:

* Evaluates detection confidence
* Decides whether additional investigation is required
* Triggers reverse image search automatically

---

## 🔹 Search Workflow

### Step 1 — Image Hosting

Uploaded images are temporarily converted into public URLs using:

* ImgBB
* Catbox

---

### Step 2 — Reverse Search

Uses:

```text
SerpAPI Google Lens Engine
```

Search results include:

* Visual matches
* Source URLs
* Website titles
* Related image occurrences

---

## 🔹 Metadata Collection

The module extracts:

* Source domain names
* Thumbnail previews
* Web references
* Search match confidence

---

## 🔹 Structured AI Reasoning

All findings are returned as structured JSON.

This allows the AI system to determine whether the image is:

* A viral meme
* A stolen identity image
* A manipulated deepfake
* Part of a disinformation campaign

---

# 🌐 Module 4 — Cybercrime Portal Integration

This module guides users toward official cybercrime complaint systems.

## 🔹 Complaint Portal Mapping

DeepTrust contains a lookup system that maps countries to official cybercrime authorities.

Supported Regions:

| Country/Region | Portal                                |
| -------------- | ------------------------------------- |
| 🇮🇳 India     | National Cyber Crime Reporting Portal |
| 🇺🇸 USA       | FBI IC3                               |
| 🇬🇧 UK        | Action Fraud                          |
| 🇪🇺 EU        | Europol                               |
| 🌍 Global      | INHOPE Network                        |

---

## 🔹 Evidence Integration

The generated PDF report includes:

* Detection results
* Reverse image findings
* Risk assessment
* Complaint portal links

This enables users to submit evidence directly to authorities.

---

# 📄 Final Evidence Report

The system automatically generates tamper-proof forensic reports.

## ✅ Report Contents

* Detection probabilities
* AI reasoning summary
* Reverse image search results
* Risk classification
* Complaint guidance
* Investigation evidence

---

## 📂 Output Location

```bash
./reports/
```

Example:

```bash
DeepTrust_Report_[REPORT_ID]_[TIMESTAMP].pdf
```

---

# ⚙️ Technology Stack

## 🖥️ Programming Languages

* Python

---

## 🧠 AI & ML Libraries

* PyTorch
* HuggingFace Transformers
* Vision Transformers (ViT)

---

## 🌐 APIs & Integrations

* SerpAPI
* MCP SDK (Python)
* Google Lens Engine

---

## 💻 Development Tools

* VS Code
* React

---

# 🚀 Project Workflow

```text
1️⃣ User Uploads Image
            ↓
2️⃣ Image Preprocessing
            ↓
3️⃣ AI Deepfake Detection
            ↓
4️⃣ Confidence Evaluation
            ↓
5️⃣ Reverse Image Search Triggered
            ↓
6️⃣ Web Context Analysis
            ↓
7️⃣ Risk Assessment
            ↓
8️⃣ PDF Evidence Generation
            ↓
9️⃣ Cybercrime Portal Guidance
```

---

# 🧪 Comparative Advantage

| Feature           | Traditional Detection Systems | DeepTrust Platform              |
| ----------------- | ----------------------------- | ------------------------------- |
| Primary Goal      | Binary Classification         | Complete Forensic Investigation |
| Architecture      | Static CNN/ViT Models         | Agentic AI Architecture         |
| Evidence          | Confidence Score Only         | Tamper-Proof PDF Reports        |
| Analysis          | Pixel-Level Only              | Contextual + OSINT Analysis     |
| Banking Alignment | General Purpose               | RBI & V-CIP Focused             |

---

# 📈 Results Interpretation

DeepTrust combines:

* 🧠 Specialist Vision Models
* 🤖 Agentic AI Reasoning
* 🔎 OSINT Investigation
* 📄 Automated Evidence Generation

The platform can autonomously:

* Trigger Google Lens searches
* Perform web investigations
* Analyze image context
* Identify likely points of origin
* Generate structured forensic reports

---

# 🔮 Future Scope

## 🚀 Planned Enhancements

* Integration with banking KYC databases
* Advanced liveness detection
* Real-time video deepfake detection
* Multi-modal fraud analysis
* Corporate-scale deployment
* Enterprise security integration

---

# 🏦 Banking Sector Relevance

DeepTrust is specifically designed for:

* e-KYC systems
* Video-CIP verification
* RBI-compliant fraud investigation
* Identity fraud prevention
* Financial cybersecurity workflows

---

# 📌 Conclusion

DeepTrust provides a complete AI-driven forensic ecosystem for identifying, investigating, and reporting deepfake-based identity fraud.

Unlike traditional deepfake detectors that only provide confidence scores, DeepTrust combines:

* Vision Transformers
* Agentic AI
* Reverse Image Search
* Cybercrime Reporting Assistance
* Evidence Generation

This makes the platform highly suitable for financial institutions, cybersecurity teams, and digital identity verification systems.

---

# 👩‍💻 Project Information

| Field           | Details                                                                        |
| --------------- | ------------------------------------------------------------------------------ |
| Project Title   | DEEPTRUST: A User-Centric Platform for Deepfake Detection and Reporting System |
| Organization    | Accenture                                                                      |
| Domain          | AI & Machine Learning                                                          |
| Internship Mode | Online                                                                         |
| Developer       | Thanushri S                                                                    |
| Course          | M.Sc – AI & ML                                                                 |

---

# ⭐ DeepTrust

> Building trust in digital identity systems through AI-powered forensic intelligence.
