"""Provenance Guard — Flask API.

Milestone 5 scope:
  * POST /submit  — multi-signal detection with real transparency labels (M5).
  * POST /appeal  — creators contest classifications; updates status to under_review.
  * GET  /log     — audit trail with submission and appeal history.

Features:
  - Rate limiting on /submit (10/minute, 100/day)
  - Three distinct transparency label variants based on confidence
  - Appeals workflow with immutable audit trail
"""

import uuid
import json
import sqlite3
from statistics import mean, stdev

from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

import store
from signals import (
    attribution_from_score,
    combine_signals,
    get_llm_score,
    get_stylometric_score,
    get_linguistic_patterns_score,
    get_transparency_label,
)

app = Flask(__name__)
store.init_db()

# Rate limiter setup (Milestone 5)
# In-memory storage for local dev; Redis or SQL in production.
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)

# Placeholder until Milestone 5 implements the real transparency labels.
_PLACEHOLDER_LABEL = "(placeholder label — transparency labels arrive in Milestone 5)"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/submit", methods=["POST"])
@limiter.limit("10 per minute;100 per day")
def submit():
    body = request.get_json(silent=True) or {}
    text = body.get("text")
    creator_id = body.get("creator_id")

    # Input validation — return 400 rather than 500 on bad requests.
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Field 'text' is required and must be non-empty."}), 400
    if not isinstance(creator_id, str) or not creator_id.strip():
        return jsonify({"error": "Field 'creator_id' is required."}), 400

    # --- Signal 1: LLM-based classification (Groq) ---
    try:
        signal1 = get_llm_score(text)
    except RuntimeError as exc:
        # The detection service is unavailable; tell the caller honestly.
        return jsonify({"error": str(exc)}), 502

    llm_score = signal1["llm_score"]

    # --- Signal 2: Stylometric heuristics (pure Python) ---
    try:
        signal2 = get_stylometric_score(text)
    except Exception as exc:
        return jsonify({"error": f"Stylometric analysis failed: {exc}"}), 500

    stylometric_score = signal2["stylometric_score"]

    # --- Signal 3: Linguistic patterns (Stretch Feature - Ensemble) ---
    try:
        signal3 = get_linguistic_patterns_score(text)
    except Exception as exc:
        return jsonify({"error": f"Linguistic analysis failed: {exc}"}), 500

    linguistic_score = signal3["linguistic_score"]

    # --- Combine all three signals into confidence score (Ensemble approach) ---
    confidence = combine_signals(llm_score, stylometric_score, linguistic_score)
    attribution = attribution_from_score(confidence)
    # M5: Use real transparency label instead of placeholder
    label = get_transparency_label(confidence)

    content_id = str(uuid.uuid4())
    timestamp = store.now_iso()

    record = {
        "content_id": content_id,
        "creator_id": creator_id,
        "text": text,
        "attribution": attribution,
        "confidence": confidence,
        "llm_score": llm_score,
        "stylometric_score": stylometric_score,
        "linguistic_score": linguistic_score,
        "label": label,
        "status": "classified",
        "created_at": timestamp,
    }
    store.save_content(record)

    # Check if creator is verified (Stretch Feature: Provenance Certificate)
    is_verified = store.is_creator_verified(creator_id)

    # Append the immutable audit snapshot for this decision.
    store.add_audit_entry(
        content_id,
        event="submission",
        detail={
            "creator_id": creator_id,
            "creator_verified": is_verified,
            "attribution": attribution,
            "confidence": confidence,
            "llm_score": llm_score,
            "stylometric_score": stylometric_score,
            "linguistic_score": linguistic_score,
            "llm_verdict": signal1["verdict"],
            "llm_reasoning": signal1["reasoning"],
            "label": label,
            "status": "classified",
        },
    )

    return jsonify(
        {
            "content_id": content_id,
            "attribution": attribution,
            "confidence": confidence,
            "label": label,
            "creator_verified": is_verified,  # Stretch Feature badge
            "signals": {
                "llm_score": llm_score,
                "llm_verdict": signal1["verdict"],
                "llm_reasoning": signal1["reasoning"],
                "stylometric_score": stylometric_score,
                "stylometric_metrics": signal2["metrics"],
                "linguistic_score": linguistic_score,
                "linguistic_metrics": signal3["metrics"],
            },
            "status": "classified",
        }
    )


@app.route("/log", methods=["GET"])
def log():
    # No auth here — for documentation/grading visibility only. A real system
    # would gate this behind reviewer authentication.
    return jsonify({"entries": store.get_log()})


@app.route("/appeal", methods=["POST"])
def appeal():
    """Handle appeals/contests of attribution decisions (Milestone 5).

    Request body:
        {
          "content_id": "...",
          "creator_reasoning": "I wrote this myself. Here's why I think the classification was wrong..."
        }

    Response:
        {
          "content_id": "...",
          "status": "under_review",
          "message": "Appeal received. A human reviewer will examine your submission."
        }
    """
    body = request.get_json(silent=True) or {}
    content_id = body.get("content_id")
    creator_reasoning = body.get("creator_reasoning")

    # Validation
    if not isinstance(content_id, str) or not content_id.strip():
        return jsonify({"error": "Field 'content_id' is required."}), 400
    if not isinstance(creator_reasoning, str) or not creator_reasoning.strip():
        return jsonify({"error": "Field 'creator_reasoning' is required."}), 400

    # Look up the original content record
    content = store.get_content(content_id)
    if content is None:
        return jsonify({"error": f"Content with ID {content_id} not found."}), 404

    # Update status to under_review
    success = store.update_status(content_id, "under_review")
    if not success:
        return jsonify({"error": "Failed to update content status."}), 500

    # Log the appeal as an immutable audit entry
    store.add_audit_entry(
        content_id,
        event="appeal",
        detail={
            "creator_id": content["creator_id"],
            "appeal_reasoning": creator_reasoning,
            "original_attribution": content["attribution"],
            "original_confidence": content["confidence"],
            "original_llm_score": content["llm_score"],
            "original_stylometric_score": content["stylometric_score"],
            "original_linguistic_score": content.get("linguistic_score"),
            "original_label": content["label"],
            "status": "under_review",
        },
    )

    return jsonify(
        {
            "content_id": content_id,
            "status": "under_review",
            "message": "Appeal received. A human reviewer will examine your submission.",
        }
    )


@app.route("/verify", methods=["POST"])
def verify_creator():
    """Verify a creator and award Provenance Certificate (Stretch Feature).

    In a real system, this would verify email, identity, or other proof.
    For this implementation, we use a simple admin verification endpoint.

    Request body:
        {
          "creator_id": "...",
          "verification_method": "email" | "manual" (optional, defaults to "email")
        }

    Response:
        {
          "creator_id": "...",
          "verified": true,
          "certificate_text": "This creator has been verified as a genuine human author..."
        }
    """
    body = request.get_json(silent=True) or {}
    creator_id = body.get("creator_id")
    method = body.get("verification_method", "email")

    if not isinstance(creator_id, str) or not creator_id.strip():
        return jsonify({"error": "Field 'creator_id' is required."}), 400

    # Award the certificate
    store.verify_creator(creator_id, method=method)

    certificate_text = (
        f"✓ Verified Creator: {creator_id} has been verified as a genuine human author. "
        "This creator's content appears with the verified badge, indicating confidence in their human authorship."
    )

    return jsonify(
        {
            "creator_id": creator_id,
            "verified": True,
            "certificate_text": certificate_text,
            "message": f"Creator {creator_id} has earned the Provenance Certificate!",
        }
    )


def _compute_analytics():
    """Compute comprehensive analytics from audit_log and content tables."""
    conn = sqlite3.connect(store.DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get all submissions
    cursor.execute("""
        SELECT content_id, llm_score, stylometric_score, linguistic_score, confidence, creator_id
        FROM content
        WHERE 1=1
        ORDER BY created_at DESC
    """)
    submissions = [dict(row) for row in cursor.fetchall()]
    
    # Get appeals and appeals status
    cursor.execute("""
        SELECT content_id, status FROM content WHERE status = 'under_review'
    """)
    appeals_data = [dict(row) for row in cursor.fetchall()]
    
    # Get creator verification data
    cursor.execute("""
        SELECT COUNT(*) as total_creators FROM (
            SELECT DISTINCT creator_id FROM content
        ) t
    """)
    total_creators = cursor.fetchone()["total_creators"]
    
    cursor.execute("""
        SELECT COUNT(*) as verified_creators FROM creator_verification WHERE verified = 1
    """)
    verified_creators = cursor.fetchone()["verified_creators"] or 0
    
    conn.close()
    
    # ===== Compute metrics =====
    
    # 1. Confidence distribution
    conf_bands = {
        "likely_human": 0,      # 0.00-0.39
        "uncertain": 0,         # 0.40-0.69
        "likely_ai": 0,         # 0.70-1.00
    }
    for sub in submissions:
        conf = sub["confidence"]
        if conf < 0.4:
            conf_bands["likely_human"] += 1
        elif conf < 0.7:
            conf_bands["uncertain"] += 1
        else:
            conf_bands["likely_ai"] += 1
    
    # 2. Signal statistics
    llm_scores = [s["llm_score"] for s in submissions if s["llm_score"] is not None]
    stylometric_scores = [s["stylometric_score"] for s in submissions if s["stylometric_score"] is not None]
    linguistic_scores = [s["linguistic_score"] for s in submissions if s["linguistic_score"] is not None]
    
    signal_stats = {}
    for signal_name, scores in [("llm", llm_scores), ("stylometric", stylometric_scores), ("linguistic", linguistic_scores)]:
        if len(scores) > 0:
            signal_stats[signal_name] = {
                "mean": round(mean(scores), 3),
                "min": round(min(scores), 3),
                "max": round(max(scores), 3),
                "stdev": round(stdev(scores), 3) if len(scores) > 1 else 0,
                "count": len(scores),
            }
    
    # 3. Signal agreement: do all 3 signals agree on AI/human? (all 3 > 0.65 or all 3 < 0.35)
    agreement_count = 0
    for sub in submissions:
        llm = sub["llm_score"]
        sty = sub["stylometric_score"]
        ling = sub["linguistic_score"]
        if llm and sty and ling:
            # All agree on AI
            if llm > 0.65 and sty > 0.65 and ling > 0.65:
                agreement_count += 1
            # All agree on human
            elif llm < 0.35 and sty < 0.35 and ling < 0.35:
                agreement_count += 1
    
    signal_agreement_pct = round((agreement_count / len(submissions) * 100) if submissions else 0, 1)
    
    # 4. Appeals data
    total_appeals = len(appeals_data)
    
    # 5. Creator metrics
    verification_rate = round((verified_creators / total_creators * 100) if total_creators > 0 else 0, 1)
    
    return {
        "total_submissions": len(submissions),
        "confidence_distribution": conf_bands,
        "signal_statistics": signal_stats,
        "signal_agreement_percentage": signal_agreement_pct,
        "appeals": {
            "total": total_appeals,
            "pending": len([a for a in appeals_data if a["status"] == "under_review"]),
        },
        "creators": {
            "total": total_creators,
            "verified": verified_creators,
            "verification_rate": verification_rate,
        },
        "submissions_list": submissions[:50],  # Latest 50 for detailed view
    }


@app.route("/api/analytics", methods=["GET"])
def api_analytics():
    """Return comprehensive analytics JSON (for integration with external dashboards)."""
    try:
        analytics = _compute_analytics()
        return jsonify(analytics)
    except Exception as e:
        return jsonify({"error": f"Analytics computation failed: {str(e)}"}), 500


@app.route("/dashboard", methods=["GET"])
def dashboard():
    """Serve interactive HTML dashboard with Chart.js visualizations."""
    try:
        analytics = _compute_analytics()
    except Exception as e:
        return f"<h1>Error loading analytics</h1><p>{str(e)}</p>", 500
    
    # Extract data for charts and convert to JSON for JavaScript
    conf_dist = analytics["confidence_distribution"]
    signal_stats = analytics["signal_statistics"]
    creators = analytics["creators"]
    appeals = analytics["appeals"]
    
    # Prepare signal data with defaults
    llm_stats = signal_stats.get("llm", {})
    sty_stats = signal_stats.get("stylometric", {})
    ling_stats = signal_stats.get("linguistic", {})
    
    llm_mean = llm_stats.get("mean", 0)
    sty_mean = sty_stats.get("mean", 0)
    ling_mean = ling_stats.get("mean", 0)
    
    llm_min = llm_stats.get("min", 0)
    sty_min = sty_stats.get("min", 0)
    ling_min = ling_stats.get("min", 0)
    
    llm_max = llm_stats.get("max", 0)
    sty_max = sty_stats.get("max", 0)
    ling_max = ling_stats.get("max", 0)
    
    # Build HTML with proper escaping
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Provenance Guard — Analytics Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f5f5f5; color: #333; }
            .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
            h1 { margin: 20px 0; text-align: center; color: #222; }
            .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 30px 0; }
            .stat-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .stat-value { font-size: 32px; font-weight: bold; color: #2563eb; margin: 10px 0; }
            .stat-label { font-size: 14px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }
            .chart-container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 20px 0; }
            .chart-wrapper { position: relative; height: 300px; margin: 20px 0; }
            .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
            @media (max-width: 768px) { .grid-2 { grid-template-columns: 1fr; } }
            .footer { text-align: center; color: #999; font-size: 12px; margin-top: 40px; padding: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Provenance Guard — Analytics Dashboard</h1>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Total Submissions</div>
                    <div class="stat-value">""" + str(analytics['total_submissions']) + """</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Total Appeals</div>
                    <div class="stat-value">""" + str(appeals['total']) + """</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Verified Creators</div>
                    <div class="stat-value">""" + str(creators['verified']) + """/""" + str(creators['total']) + """</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Signal Agreement</div>
                    <div class="stat-value">""" + str(analytics['signal_agreement_percentage']) + """%</div>
                </div>
            </div>
            
            <div class="grid-2">
                <div class="chart-container">
                    <h2 style="margin-bottom: 20px;">Confidence Distribution</h2>
                    <div class="chart-wrapper">
                        <canvas id="confChart"></canvas>
                    </div>
                </div>
                
                <div class="chart-container">
                    <h2 style="margin-bottom: 20px;">Signal Means</h2>
                    <div class="chart-wrapper">
                        <canvas id="signalChart"></canvas>
                    </div>
                </div>
            </div>
            
            <div class="chart-container">
                <h2 style="margin-bottom: 20px;">Signal Statistics (Mean, Min, Max)</h2>
                <div style="height: 250px; position: relative;">
                    <canvas id="signalStatsChart"></canvas>
                </div>
            </div>
            
            <div class="chart-container">
                <h2 style="margin-bottom: 20px;">Creator Verification</h2>
                <div style="height: 250px; position: relative;">
                    <canvas id="verificationChart"></canvas>
                </div>
            </div>
            
            <div class="footer">
                <p>Last updated: """ + store.now_iso() + """</p>
                <p><a href="/log" style="color: #2563eb; text-decoration: none;">View Full Audit Log →</a></p>
            </div>
        </div>
        
        <script>
            // Chart data
            const confData = [""" + str(conf_dist['likely_human']) + """, """ + str(conf_dist['uncertain']) + """, """ + str(conf_dist['likely_ai']) + """];
            const signalMeans = [""" + str(llm_mean) + """, """ + str(sty_mean) + """, """ + str(ling_mean) + """];
            const signalMins = [""" + str(llm_min) + """, """ + str(sty_min) + """, """ + str(ling_min) + """];
            const signalMaxs = [""" + str(llm_max) + """, """ + str(sty_max) + """, """ + str(ling_max) + """];
            const verifiedCount = """ + str(creators['verified']) + """;
            const unverifiedCount = """ + str(creators['total'] - creators['verified']) + """;
            
            // 1. Confidence Distribution (Pie Chart)
            const confCtx = document.getElementById('confChart').getContext('2d');
            new Chart(confCtx, {
                type: 'pie',
                data: {
                    labels: ['Likely Human (0.00–0.39)', 'Uncertain (0.40–0.69)', 'Likely AI (0.70–1.00)'],
                    datasets: [{
                        data: confData,
                        backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
                        borderColor: '#fff',
                        borderWidth: 2,
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom' } } }
            });
            
            // 2. Signal Means (Bar Chart)
            const signalCtx = document.getElementById('signalChart').getContext('2d');
            new Chart(signalCtx, {
                type: 'bar',
                data: {
                    labels: ['LLM', 'Stylometric', 'Linguistic'],
                    datasets: [{
                        label: 'Mean Score',
                        data: signalMeans,
                        backgroundColor: ['#2563eb', '#7c3aed', '#db2777'],
                        borderRadius: 4,
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true, max: 1 } } }
            });
            
            // 3. Signal Statistics (Min/Mean/Max)
            const statsCtx = document.getElementById('signalStatsChart').getContext('2d');
            new Chart(statsCtx, {
                type: 'bar',
                data: {
                    labels: ['LLM', 'Stylometric', 'Linguistic'],
                    datasets: [
                        {
                            label: 'Min',
                            data: signalMins,
                            backgroundColor: '#fee2e2',
                        },
                        {
                            label: 'Mean',
                            data: signalMeans,
                            backgroundColor: '#2563eb',
                        },
                        {
                            label: 'Max',
                            data: signalMaxs,
                            backgroundColor: '#fecaca',
                        }
                    ]
                },
                options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true, max: 1 } }, plugins: { legend: { position: 'bottom' } } }
            });
            
            // 4. Creator Verification (Donut)
            const verCtx = document.getElementById('verificationChart').getContext('2d');
            new Chart(verCtx, {
                type: 'doughnut',
                data: {
                    labels: ['Verified Creators', 'Unverified Creators'],
                    datasets: [{
                        data: [verifiedCount, unverifiedCount],
                        backgroundColor: ['#10b981', '#d1d5db'],
                        borderColor: '#fff',
                        borderWidth: 2,
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom' } } }
            });
        </script>
    </body>
    </html>
    """
    return html


if __name__ == "__main__":
    app.run(debug=True, port=5000)
