from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import os

app = Flask(__name__)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ---------------------------------------------------------------------------
# PIPELINE — reproducción exacta del notebook KNN_MUNDIAL.ipynb
# ---------------------------------------------------------------------------
def build_pipeline():
    # 1. results.csv
    results = pd.read_csv(os.path.join(DATA_DIR, "results.csv"))
    results.drop(columns=["tournament", "city", "country", "neutral"], inplace=True)
    results.dropna(subset=["home_score", "away_score"], inplace=True)
    results["home_score"] = results["home_score"].astype(int)
    results["away_score"] = results["away_score"].astype(int)
    results["date"] = pd.to_datetime(results["date"])

    # 2. Filtro 2014 → hoy
    results_filter = results[
        (results["date"] >= "2014-01-01") &
        (results["date"] <= pd.Timestamp.today())
    ].copy()
    results_filter["year"] = results_filter["date"].dt.year

    # 3. elo_ratings_wc2026.csv
    elo = pd.read_csv(os.path.join(DATA_DIR, "elo_ratings_wc2026.csv"))
    elo.drop(columns=[c for c in elo.columns
                      if c not in ("year", "country", "rating", "rank_avg", "rating_avg")],
             inplace=True)
    elo_filter = elo[elo["year"] >= 2013].copy()

    # 4. Merge home
    df_merged = pd.merge(
        results_filter,
        elo_filter.rename(columns={
            "country": "home_team", "rating": "home_elo",
            "rank_avg": "home_rank_avg", "rating_avg": "home_rating_avg",
        }),
        left_on=["year", "home_team"], right_on=["year", "home_team"], how="inner",
    )
    df_merged.drop(columns=["year"], inplace=True)
    df_merged["year"] = df_merged["date"].dt.year

    # 5. Merge away
    df_merged = pd.merge(
        df_merged,
        elo_filter.rename(columns={
            "country": "away_team", "rating": "away_elo",
            "rank_avg": "away_rank_avg", "rating_avg": "away_rating_avg",
        }),
        left_on=["year", "away_team"], right_on=["year", "away_team"], how="inner",
    )
    df_merged.drop(columns=["year"], inplace=True)

    # 6. Target + dif_elo
    df_merged["target"] = np.where(df_merged["home_score"] > df_merged["away_score"], 2, 0)
    df_merged["target"] = np.where(df_merged["home_score"] == df_merged["away_score"], 1, df_merged["target"])
    df_merged["dif_elo"] = df_merged["home_elo"] - df_merged["away_elo"]

    # 7. Features / target
    features = ["home_elo", "away_elo", "dif_elo",
                "home_rank_avg", "home_rating_avg",
                "away_rank_avg", "away_rating_avg"]
    X = df_merged[features]
    y = df_merged["target"]

    # 8. Split — guardamos índices originales para recuperar vecinos
    orig_idx = X.index.to_numpy()
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, orig_idx, test_size=0.2, random_state=0
    )

    # 9. Scaler + KNN (k=3 para predicción, k=15 para probabilidades)
    scaler = MinMaxScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_tr_s, y_tr)
    accuracy = float(knn.score(X_te_s, y_te))
    print(f"KNN (k=3) accuracy: {accuracy:.4f}")

    # KNN con K=15 para probabilidades más granulares
    knn_proba = KNeighborsClassifier(n_neighbors=15)
    knn_proba.fit(X_tr_s, y_tr)

    # Matriz de confusión (orden: Victoria=2, Empate=1, Derrota=0)
    y_pred_te = knn.predict(X_te_s)
    cm = confusion_matrix(y_te, y_pred_te, labels=[2, 1, 0])
    cm_labels = ["Victoria Local", "Empate", "Derrota Local"]
    conf_matrix = cm.tolist()

    # 10. Curva accuracy vs K (1–29)
    k_values = list(range(1, 30))
    k_scores = []
    for k in k_values:
        tmp = KNeighborsClassifier(n_neighbors=k)
        tmp.fit(X_tr_s, y_tr)
        k_scores.append(round(tmp.score(X_te_s, y_te), 4))

    # 11. Distribución de resultados del dataset completo
    rm = {0: "Derrota Local", 1: "Empate", 2: "Victoria Local"}
    dist = {rm[k]: int(v) for k, v in df_merged["target"].value_counts().items()}

    # 12. ELO actual (World.tsv)
    elo_actual = pd.read_csv(os.path.join(DATA_DIR, "World.tsv"), sep="\t", header=None)
    elo_actual = elo_actual[[2, 3]]
    elo_actual.columns = ["country_code", "elo_2026"]

    cc = {
        "AR": "Argentina", "FR": "France", "ES": "Spain", "US": "United States",
        "EN": "England", "BR": "Brazil", "CO": "Colombia", "PT": "Portugal",
        "NL": "Netherlands", "NO": "Norway", "JP": "Japan", "DE": "Germany",
        "CH": "Switzerland", "MX": "Mexico", "EC": "Ecuador", "HR": "Croatia",
        "MA": "Morocco", "BE": "Belgium", "UY": "Uruguay", "AT": "Austria",
        "SN": "Senegal", "PY": "Paraguay", "TR": "Turkey", "AU": "Australia",
        "DZ": "Algeria", "CA": "Canada", "CI": "Ivory Coast", "EG": "Egypt",
        "SE": "Sweden", "KR": "South Korea", "PA": "Panama", "CD": "DR Congo",
        "JO": "Jordan", "CV": "Cape Verde", "BA": "Bosnia and Herzegovina",
        "HT": "Haiti", "CW": "Curaçao", "NZ": "New Zealand", "IR": "Iran",
        "SA": "Saudi Arabia", "QA": "Qatar", "IQ": "Iraq", "UZ": "Uzbekistan",
        "IT": "Italy", "DK": "Denmark",
    }
    elo_actual["country"] = elo_actual["country_code"].map(cc)
    elo_actual.dropna(subset=["country"], inplace=True)
    elo_map = dict(zip(elo_actual["country"], elo_actual["elo_2026"]))

    # 13. team_info (ELO 2026 real + últimos rank/rating del histórico)
    elo_latest = elo_filter.sort_values("year").groupby("country").last().reset_index()
    team_info = {}
    for _, row in elo_latest.iterrows():
        name = row["country"]
        if name in elo_map:
            team_info[name] = {
                "elo": float(elo_map[name]),
                "rank_avg": float(row["rank_avg"]),
                "rating_avg": float(row["rating_avg"]),
            }

    return dict(
        knn=knn, knn_proba=knn_proba, scaler=scaler, features=features,
        team_info=team_info, elo_filter=elo_filter,
        df_merged=df_merged, idx_tr=idx_tr,
        k_values=k_values, k_scores=k_scores,
        dist=dist, accuracy=accuracy,
        results=results,
        train_size=len(X_tr),
        conf_matrix=conf_matrix,
        cm_labels=cm_labels,
    )


MODEL = build_pipeline()


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

    # Probabilidades con K=15 para más granularidad (K=3 solo tiene 4 valores posibles)
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

    # Ajustar marcador para que sea consistente con la predicción
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
        .tail(10)   # últimos 10 enfrentamientos
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
    app.run(debug=True, port=port)
