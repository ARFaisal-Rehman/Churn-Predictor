"""Retrain from scratch and print the numbers quoted in the README.

    python -m churn_predictor.evaluate

Writes artifacts/analysis.json, artifacts/predictions.csv and artifacts/metrics.json.
"""
import json
import time

from . import create_app


def main() -> None:
    app = create_app()
    t0 = time.time()
    analysis = app.extensions["analysis_service"].refresh_analysis(force_retrain=True)
    m, d = analysis["model"], analysis["dataset"]
    print(f"done in {time.time() - t0:.0f}s")
    print(json.dumps({"snapshots": d["snapshots"], "rows": d["rows"], "churn_rate": d["churn_rate"],
                      "best_model": m["best_model"], "results": m["results"],
                      "evaluation": {k: v for k, v in m["evaluation"].items() if k != "reliability"},
                      "segments_k": analysis["segments"]["k"], "silhouette": analysis["segments"]["silhouette"],
                      "segment_counts": analysis["segments"]["counts"], "summary": analysis["summary"]}, indent=1))


if __name__ == "__main__":
    main()
