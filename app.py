from flask import Flask, jsonify, request, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
MODEL = joblib.load(MODEL_PATH)

# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/teams")
def api_teams():
    return jsonify(MODEL["team_info"])


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    home = data.get("home", "").strip()
    away = data.get("away", "").strip()

    ti = MODEL["team_info"]
    if not home or not away or home == away or home not in ti or away not in ti:
        return jsonify({"error": "Equipos inválidos"}), 400

    h, a = ti[home], ti[away]
    ef = MODEL["elo_filter"]
    d1 = ef[ef["country"] == home].iloc[-1]
    d2 = ef[ef["country"] == away].iloc[-1]

    partido = pd.DataFrame([[
        h["elo"], a["elo"], h["elo"] - a["elo"],
        float(d1["rank_avg"]), float(d1["rating_avg"]),
        float(d2["rank_avg"]), float(d2["rating_avg"]),
    ]], columns=MODEL["features"])

    partido_norm = MODEL["scaler"].transform(partido)
    pred = int(MODEL["knn"].predict(partido_norm)[0])

    proba = MODEL["knn_proba"].predict_proba(partido_norm)[0]
    classes = list(MODEL["knn_proba"].classes_)
    prob_map = {int(c): float(p) for c, p in zip(classes, proba)}

    # Vecinos más cercanos → marcador esperado
    _, neighbor_pos = MODEL["knn"].kneighbors(partido_norm)
    orig_idx = MODEL["idx_tr"][neighbor_pos[0]]
    nbrs = MODEL["df_merged"].loc[orig_idx]
    avg_h = float(nbrs["home_score"].mean())
    avg_a = float(nbrs["away_score"].mean())
    sh, sa = round(avg_h), round(avg_a)

    if pred == 2 and sh <= sa:
        sh = sa + 1
    elif pred == 0 and sa <= sh:
        sa = sh + 1
    elif pred == 1:
        v = round((avg_h + avg_a) / 2)
        sh = sa = v

    neighbors_info = [
        {
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_score": int(row["home_score"]),
            "away_score": int(row["away_score"]),
            "date": str(row["date"])[:10],
        }
        for _, row in nbrs.iterrows()
    ]

    return jsonify({
        "prediction": pred,
        "probabilities": {
            "home_win": prob_map.get(2, 0),
            "draw":     prob_map.get(1, 0),
            "away_win": prob_map.get(0, 0),
        },
        "score": {"home": sh, "away": sa},
        "neighbors": neighbors_info,
        "home": {"name": home, **h},
        "away": {"name": away, **a},
        "k_values": MODEL["k_values"],
        "k_scores": MODEL["k_scores"],
        "distribution": MODEL["dist"],
        "accuracy": MODEL["accuracy"],
        "train_size": MODEL["train_size"],
        "conf_matrix": MODEL["conf_matrix"],
        "cm_labels": MODEL["cm_labels"],
    })


@app.route("/api/history")
def api_history():
    home = request.args.get("home", "").strip()
    away = request.args.get("away", "").strip()
    results = MODEL["results"]

    mask = (
        results["home_team"].isin([home, away]) &
        results["away_team"].isin([home, away])
    )
    history = (
        results[mask]
        .sort_values("date", ascending=True)
        [["date", "home_team", "home_score", "away_score", "away_team"]]
        .tail(10)
    )

    return jsonify([
        {
            "date": str(r["date"])[:10],
            "home_team": r["home_team"],
            "home_score": int(r["home_score"]),
            "away_score": int(r["away_score"]),
            "away_team": r["away_team"],
        }
        for _, r in history.iterrows()
    ])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, port=port)
