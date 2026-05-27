"""
DeepTrust v3 — Flask Server
Run: python server.py
Then open: http://localhost:5000
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import asyncio, os
from deeptrust_gemini import DeepTrust   # fixed import

app = Flask(__name__)
CORS(app)

platform = DeepTrust()

@app.route("/")
def index():
    return send_file("deeptrust_ui.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    # Save to same folder as script (Windows compatible)
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file.filename)
    file.save(save_path)

    report = asyncio.run(platform.analyze(save_path))

    return jsonify({
        "prediction":    report.detection.prediction,
        "confidence":    report.detection.confidence,
        "label_scores":  report.detection.label_scores,
        "image_hash":    report.image_hash,
        "risk_level":    report.risk_level,
        "final_summary": report.final_summary,
        "findings":      [{"tool": f.tool_name, "output": f.tool_output} for f in report.findings],
        "pdf_path":      report.pdf_path,
        "report_id":     report.report_id,
    })

@app.route("/download/<path:filename>")
def download(filename):
    return send_file(filename, as_attachment=True)

if __name__ == "__main__":
    print("DeepTrust running at http://localhost:5000")
    app.run(debug=True, port=5000)
