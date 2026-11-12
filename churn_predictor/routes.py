from __future__ import annotations

from flask import Blueprint, current_app, jsonify, render_template, request


bp = Blueprint("main", __name__)


def _analysis_service():
    return current_app.extensions["analysis_service"]


@bp.route("/")
def dashboard():
    analysis = _analysis_service().get_or_create_analysis()
    return render_template("dashboard.html", analysis=analysis)


@bp.route("/analytics")
def analytics():
    analysis = _analysis_service().get_or_create_analysis()
    return render_template("analytics.html", analysis=analysis)


@bp.route("/customers/<customer_id>")
def customer_profile(customer_id: str):
    customer = _analysis_service().get_customer(customer_id)
    if not customer:
        analysis = _analysis_service().get_or_create_analysis()
        return render_template("error.html", message=f"Customer {customer_id} was not found.", analysis=analysis), 404

    return render_template("customer_profile.html", customer=customer)


@bp.route("/api/summary")
def api_summary():
    analysis = _analysis_service().get_or_create_analysis()
    return jsonify(analysis)


@bp.route("/api/retrain", methods=["POST"])
def api_retrain():
    payload = request.get_json(silent=True) or {}
    force_retrain = bool(payload.get("force_retrain", True))
    analysis = _analysis_service().refresh_analysis(force_retrain=force_retrain)
    return jsonify(
        {
            "ok": True,
            "generated_at": analysis["generated_at"],
            "best_model": analysis["model"]["best_model"],
            "metrics": analysis["model"]["metrics"],
        }
    )


@bp.route("/health")
def health():
    return jsonify({"status": "ok"})
