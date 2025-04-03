from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.neighbors import NearestNeighbors

app = FastAPI(title="Movie Recommendation API")

# Modèles Pydantic pour la validation des données d'entrée
class Film(BaseModel):
    title: str
    genres: str
    rating: float

class Profile(BaseModel):
    films: List[Film]

# Fonction de chargement du dataset
def load_dataset() -> pd.DataFrame:
    try:
        df = pd.read_csv("user_ratings_genres_mov.csv")
        return df
    except Exception as e:
        raise Exception(f"Erreur de chargement : {e}")

# Fonction de calcul de la similarité de Jaccard entre deux chaînes de genres
def jaccard_similarity(g1: str, g2: str) -> float:
    s1 = set(g1.split("|"))
    s2 = set(g2.split("|"))
    return len(s1 & s2) / len(s1 | s2) if s1 and s2 else 0

# Fonction pour nettoyer les valeurs NaN dans l'objet à retourner
def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize(i) for i in obj]
    elif isinstance(obj, float):
        if math.isnan(obj):
            return 0.0  # Remplace NaN par 0.0 (ajuste selon tes besoins)
        return obj
    else:
        return obj

# Point d'entrée de l'API pour obtenir des recommandations
@app.post("/recommend")
def recommend(profile: Profile):
    # Vérifier que le profil contient exactement 3 films
    if len(profile.films) != 3:
        raise HTTPException(status_code=400, detail="Le profil doit contenir exactement 3 films.")

    # Vérifier que les titres des films sont distincts
    titles = [film.title for film in profile.films]
    if len(set(titles)) < 3:
        raise HTTPException(status_code=400, detail="Les films doivent être tous différents.")

    # Charger le dataset
    try:
        df = load_dataset()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    if df.empty:
        raise HTTPException(status_code=500, detail="Le dataset est vide ou introuvable.")

    # Mettre à jour le dataset avec les notes du nouvel utilisateur
    new_user_id = "user_new"
    new_user_entries = []
    for film in profile.films:
        new_user_entries.append({
            "userId": new_user_id,
            "title": film.title,
            "rating": film.rating,
            "genres": film.genres
        })
    nouveau_df = pd.DataFrame(new_user_entries)
    df_updated = pd.concat([df, nouveau_df], ignore_index=True)

    # Création de la matrice de notes
    rating_matrix = df_updated.pivot_table(index="userId", columns="title", values="rating")
    rating_matrix_filled = rating_matrix.fillna(0)

    recommendations = {}

    ## 1. Recommandation basée sur le contenu (Jaccard)
    df_unique = df.drop_duplicates(subset="title")[["title", "genres"]]
    best_film = max(profile.films, key=lambda f: f.rating)
    df_unique["similarity"] = df_unique["genres"].apply(lambda g: jaccard_similarity(g, best_film.genres))
    content_reco = df_unique[df_unique["title"] != best_film.title].nlargest(5, "similarity")[["title", "similarity"]]
    recommendations["content_based"] = content_reco.to_dict(orient="records")

    ## 2. Recommandation collaborative – Approche mémoire (Cosine Similarity)
    user_sim = cosine_similarity(rating_matrix_filled)
    user_sim_df = pd.DataFrame(user_sim, index=rating_matrix_filled.index, columns=rating_matrix_filled.index)
    movies_to_predict = rating_matrix.loc[new_user_id][rating_matrix.loc[new_user_id].isna()].index
    sim_new_user = user_sim_df.loc[new_user_id]
    predictions_memory = {}
    for movie in movies_to_predict:
        ratings = rating_matrix[movie]
        if ratings.notna().sum() == 0:
            continue
        predictions_memory[movie] = np.dot(ratings[ratings.notna()], sim_new_user[ratings.notna()]) / sim_new_user[ratings.notna()].sum()
    if predictions_memory:
        memory_reco = pd.DataFrame(list(predictions_memory.items()), columns=["title", "predicted_rating"]).nlargest(5, "predicted_rating")
        recommendations["collaborative_memory"] = memory_reco.to_dict(orient="records")
    else:
        recommendations["collaborative_memory"] = []

    ## 3. Recommandation collaborative – Approche NMF
    try:
        nmf_model = NMF(n_components=20, init='random', random_state=42, max_iter=300)
        W = nmf_model.fit_transform(rating_matrix_filled)
        H = nmf_model.components_
        pred_nmf_df = pd.DataFrame(np.dot(W, H), index=rating_matrix.index, columns=rating_matrix.columns)
        predictions_nmf = {movie: pred_nmf_df.loc[new_user_id, movie] for movie in movies_to_predict}
        if predictions_nmf:
            nmf_reco = pd.DataFrame(predictions_nmf.items(), columns=["title", "predicted_rating"]).nlargest(5, "predicted_rating")
            recommendations["collaborative_nmf"] = nmf_reco.to_dict(orient="records")
        else:
            recommendations["collaborative_nmf"] = []
    except Exception as e:
        recommendations["collaborative_nmf"] = f"Erreur lors de l'exécution de NMF: {e}"

    ## 4. Recommandation collaborative – Approche SVD
    try:
        svd_model = TruncatedSVD(n_components=20, random_state=42)
        U = svd_model.fit_transform(rating_matrix_filled)
        VT = svd_model.components_
        pred_svd_df = pd.DataFrame(np.dot(U, VT), index=rating_matrix_filled.index, columns=rating_matrix_filled.columns)
        predictions_svd = {movie: pred_svd_df.loc[new_user_id, movie] for movie in movies_to_predict}
        if predictions_svd:
            svd_reco = pd.DataFrame(predictions_svd.items(), columns=["title", "predicted_rating"]).nlargest(5, "predicted_rating")
            recommendations["collaborative_svd"] = svd_reco.to_dict(orient="records")
        else:
            recommendations["collaborative_svd"] = []
    except Exception as e:
        recommendations["collaborative_svd"] = f"Erreur lors de l'exécution de SVD: {e}"

    ## 5. Recommandation collaborative – Approche KNN
    try:
        knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
        knn_model.fit(rating_matrix_filled)
        distances, indices = knn_model.kneighbors(rating_matrix_filled.loc[[new_user_id]], n_neighbors=10)
        neighbors = rating_matrix_filled.iloc[indices[0]]
        similarities = 1 - distances[0]
        predictions_knn = {}
        for movie in movies_to_predict:
            neighbor_ratings = neighbors[movie]
            mask = neighbor_ratings != 0
            if not mask.any():
                continue
            weighted_sum = np.dot(neighbor_ratings[mask], similarities[mask])
            total_similarity = similarities[mask].sum()
            predictions_knn[movie] = weighted_sum / total_similarity if total_similarity else 0
        if predictions_knn:
            knn_reco = pd.DataFrame(predictions_knn.items(), columns=["title", "predicted_rating"]).nlargest(5, "predicted_rating")
            recommendations["collaborative_knn"] = knn_reco.to_dict(orient="records")
        else:
            recommendations["collaborative_knn"] = []
    except Exception as e:
        recommendations["collaborative_knn"] = f"Erreur lors de l'exécution de KNN: {e}"

    # Nettoyer la réponse pour remplacer les NaN par 0.0 (ou une autre valeur par défaut)
    return sanitize(recommendations)

@app.get("/")
def home():
    return {"message": "Bienvenue sur l'API de recommandation de films"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
