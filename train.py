"""
Entrena el modelo KNN y serializa todo lo necesario en model.pkl.
Se ejecuta UNA vez durante el build (no en runtime).
"""
import os, joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def build_and_save():
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

    # 8. Split
    orig_idx = X.index.to_numpy()
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, orig_idx, test_size=0.2, random_state=0
    )

    # 9. Scaler + KNN k=3 (predicción) + k=15 (probabilidades)
    scaler = MinMaxScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_tr_s, y_tr)
    accuracy = float(knn.score(X_te_s, y_te))
    print(f"KNN k=3 accuracy: {accuracy:.4f}")

    knn_proba = KNeighborsClassifier(n_neighbors=15)
    knn_proba.fit(X_tr_s, y_tr)

    # 10. Matriz de confusión
    y_pred_te = knn.predict(X_te_s)
    cm = confusion_matrix(y_te, y_pred_te, labels=[2, 1, 0])

    # 11. Curva K (1-29)
    k_scores = []
    for k in range(1, 30):
        tmp = KNeighborsClassifier(n_neighbors=k)
        tmp.fit(X_tr_s, y_tr)
        k_scores.append(round(tmp.score(X_te_s, y_te), 4))

    # 12. Distribución
    rm = {0: "Derrota Local", 1: "Empate", 2: "Victoria Local"}
    dist = {rm[k]: int(v) for k, v in df_merged["target"].value_counts().items()}

    # 13. ELO actual (World.tsv)
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

    # 14. team_info
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

    payload = dict(
        knn=knn,
        knn_proba=knn_proba,
        scaler=scaler,
        features=features,
        team_info=team_info,
        elo_filter=elo_filter,
        df_merged=df_merged[["date", "home_team", "away_team",
                              "home_score", "away_score"]],   # solo lo necesario
        idx_tr=idx_tr,
        k_values=list(range(1, 30)),
        k_scores=k_scores,
        dist=dist,
        accuracy=accuracy,
        train_size=len(X_tr),
        conf_matrix=cm.tolist(),
        cm_labels=["Victoria Local", "Empate", "Derrota Local"],
        # results completo para historial H2H
        results=results[["date", "home_team", "away_team", "home_score", "away_score"]],
    )

    joblib.dump(payload, "model.pkl", compress=3)
    print(f"✅ model.pkl guardado ({os.path.getsize('model.pkl') / 1e6:.1f} MB)")

if __name__ == "__main__":
    build_and_save()
