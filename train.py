"""
Entrena el modelo KNN y serializa todo en model.pkl.
Se ejecuta UNA vez durante el build de Render.
"""
import os, gc, joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def build_and_save():
    print("Cargando datos...")
    results = pd.read_csv(os.path.join(DATA_DIR, "results.csv"),
                          usecols=["date", "home_team", "away_team", "home_score", "away_score"])
    results.dropna(subset=["home_score", "away_score"], inplace=True)
    results["home_score"] = results["home_score"].astype(int)
    results["away_score"] = results["away_score"].astype(int)
    results["date"] = pd.to_datetime(results["date"])

    results_full = results.copy()   # guardamos para historial H2H

    results_filter = results[results["date"] >= "2014-01-01"].copy()
    results_filter["year"] = results_filter["date"].dt.year
    del results
    gc.collect()

    elo = pd.read_csv(os.path.join(DATA_DIR, "elo_ratings_wc2026.csv"),
                      usecols=["year", "country", "rating", "rank_avg", "rating_avg"])
    elo_filter = elo[elo["year"] >= 2013].copy()
    del elo
    gc.collect()

    print("Mergeando datasets...")
    df = pd.merge(
        results_filter,
        elo_filter.rename(columns={"country": "home_team", "rating": "home_elo",
                                   "rank_avg": "home_rank_avg", "rating_avg": "home_rating_avg"}),
        on=["year", "home_team"], how="inner",
    )
    df.drop(columns=["year"], inplace=True)
    df["year"] = df["date"].dt.year

    df = pd.merge(
        df,
        elo_filter.rename(columns={"country": "away_team", "rating": "away_elo",
                                   "rank_avg": "away_rank_avg", "rating_avg": "away_rating_avg"}),
        on=["year", "away_team"], how="inner",
    )
    df.drop(columns=["year"], inplace=True)
    del results_filter
    gc.collect()

    df["target"] = np.where(df["home_score"] > df["away_score"], 2, 0)
    df["target"] = np.where(df["home_score"] == df["away_score"], 1, df["target"])
    df["dif_elo"] = df["home_elo"] - df["away_elo"]

    features = ["home_elo", "away_elo", "dif_elo",
                "home_rank_avg", "home_rating_avg",
                "away_rank_avg", "away_rating_avg"]
    X = df[features].values
    y = df["target"].values
    orig_idx = np.arange(len(X))

    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, orig_idx, test_size=0.2, random_state=0
    )

    print("Entrenando scaler y modelos KNN...")
    scaler = MinMaxScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_tr_s, y_tr)
    accuracy = float(knn.score(X_te_s, y_te))
    print(f"KNN k=3 accuracy: {accuracy:.4f}")

    knn_proba = KNeighborsClassifier(n_neighbors=15)
    knn_proba.fit(X_tr_s, y_tr)

    y_pred_te = knn.predict(X_te_s)
    cm = confusion_matrix(y_te, y_pred_te, labels=[2, 1, 0])
    del X_te, y_te, y_pred_te
    gc.collect()

    # Curva K: solo valores impares 1-29 para reducir cómputo
    print("Calculando curva K...")
    k_values = list(range(1, 30))
    k_scores = []
    X_te_s2 = scaler.transform(X_tr[: len(X_tr) // 5])  # subset para validación rápida
    y_te2 = y_tr[: len(y_tr) // 5]
    X_tr_s2 = X_tr_s[len(X_tr) // 5:]
    y_tr2 = y_tr[len(y_tr) // 5:]
    for k in k_values:
        tmp = KNeighborsClassifier(n_neighbors=k)
        tmp.fit(X_tr_s2, y_tr2)
        k_scores.append(round(tmp.score(X_te_s2, y_te2), 4))
        del tmp
    del X_tr_s2, y_tr2, X_te_s2, y_te2
    gc.collect()

    # Distribución
    rm = {0: "Derrota Local", 1: "Empate", 2: "Victoria Local"}
    unique, counts = np.unique(y, return_counts=True)
    dist = {rm[int(k)]: int(v) for k, v in zip(unique, counts)}

    # ELO actual (World.tsv)
    elo_actual = pd.read_csv(os.path.join(DATA_DIR, "World.tsv"), sep="\t", header=None,
                             usecols=[2, 3])
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

    # df_merged solo con columnas necesarias para kneighbors lookup
    df_lookup = df[["home_team", "away_team", "home_score", "away_score", "date"]].copy()

    payload = dict(
        knn=knn,
        knn_proba=knn_proba,
        scaler=scaler,
        features=features,
        team_info=team_info,
        elo_filter=elo_filter[["year", "country", "rank_avg", "rating_avg"]],
        df_merged=df_lookup,
        idx_tr=idx_tr,
        k_values=k_values,
        k_scores=k_scores,
        dist=dist,
        accuracy=accuracy,
        train_size=len(X_tr),
        conf_matrix=cm.tolist(),
        cm_labels=["Victoria Local", "Empate", "Derrota Local"],
        results=results_full[["date", "home_team", "away_team", "home_score", "away_score"]],
    )

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
    joblib.dump(payload, model_path, compress=3)
    size_mb = os.path.getsize(model_path) / 1e6
    print(f"✅ model.pkl guardado ({size_mb:.1f} MB)")

if __name__ == "__main__":
    build_and_save()
