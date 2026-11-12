from __future__ import annotations

import os

from flask import Blueprint, abort, current_app, jsonify, render_template, request

bp = Blueprint("main", __name__)
PAGE_SIZE = 50


def _service():
    return current_app.extensions["analysis_service"]


@bp.route("/")
def dashboard():
    return render_template("dashboard.html", analysis=_service().get_or_create_analysis())


@bp.route("/customers")
def customers():
    analysis = _service().get_or_create_analysis()
    q = request.args.get("q", "").strip()
    risk = request.args.get("risk", "")
    segment = request.args.get("segment", "")
    rows = analysis["customers"]
    if q:
        rows = [r for r in rows if q in str(r["customer_id"])]
    if risk in {"High", "Medium", "Low"}:
        rows = [r for r in rows if r["risk_category"] == risk]
    if segment:
        rows = [r for r in rows if r["segment_name"] == segment]
    page = max(1, request.args.get("page", 1, type=int))
    pages = max(1, (len(rows) + PAGE_SIZE - 1) // PAGE_SIZE)
    page = min(page, pages)
    return render_template(
        "customers.html", analysis=analysis, rows=rows[(page - 1) * PAGE_SIZE: page * PAGE_SIZE],
        total=len(rows), page=page, pages=pages, q=q, risk=risk, segment=segment,
        segments=sorted({r["segment_name"] for r in analysis["customers"]}),
    )


@bp.route("/customers/<customer_id>")
def customer_profile(customer_id: str):
    analysis = _service().get_or_create_analysis()
    customer = _service().get_customer(customer_id)
    if not customer:
        return render_template("error.html", analysis=analysis,
                               message=f"No customer with ID {customer_id} in the current scoring snapshot."), 404
    return render_template("customer_profile.html", analysis=analysis, customer=customer)


@bp.route("/methodology")
def methodology():
    return render_template("methodology.html", analysis=_service().get_or_create_analysis())


@bp.route("/api/summary")
def api_summary():
    analysis = dict(_service().get_or_create_analysis())
    analysis.pop("customers", None)
    return jsonify(analysis)


@bp.route("/api/customers/<customer_id>")
def api_customer(customer_id: str):
    customer = _service().get_customer(customer_id)
    if not customer:
        abort(404)
    return jsonify(customer)


@bp.route("/api/retrain", methods=["POST"])
def api_retrain():
    # Retraining takes about two minutes; keep it off unless explicitly enabled.
    if os.getenv("ALLOW_RETRAIN") != "1":
        return jsonify({"ok": False, "error": "Retraining is disabled. Set ALLOW_RETRAIN=1 to enable."}), 403
    analysis = _service().refresh_analysis(force_retrain=True)
    return jsonify({"ok": True, "generated_at": analysis["generated_at"], "best_model": analysis["model"]["best_model"]})


@bp.route("/health")
def health():
    return jsonify({"status": "ok"})


@bp.app_errorhandler(404)
def not_found(_):
    return render_template("error.html", analysis=None, message="That page does not exist."), 404
