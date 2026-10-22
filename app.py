import html
import re
import sqlite3
import os
import hashlib
import hmac
import secrets
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="CineMatch | Movie Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# 2. PROJECT PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent


def _resolve_dataset_file(*candidates):
    for name in candidates:
        path = BASE_DIR / "dataset" / name
        if path.exists():
            return path
    return BASE_DIR / "dataset" / candidates[0]


MOVIES_PATH = _resolve_dataset_file("movies.csv", "movies .csv", "movies (1).csv")
RATINGS_PATH = _resolve_dataset_file("ratings.csv", "ratings .csv", "ratings (1).csv")
TAGS_PATH = _resolve_dataset_file("tags.csv", "tags (1).csv")
MODEL_PATH = BASE_DIR / "svd_model.pkl"

# ============================================================
# 3. CUSTOM CSS
# ============================================================
st.html("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root {
        --bg:#08090d; --panel:#11131b; --panel-2:#171923; --line:#282b39;
        --text:#f7f7fb; --muted:#a2a6b7; --accent:#e50914; --accent-2:#ff4450;
        --violet:#a78bfa; --gold:#fbbf24; --cyan:#67e8f9;
    }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    .stApp { background:radial-gradient(ellipse at 12% 0%,rgba(229,9,20,.10),transparent 28%),radial-gradient(ellipse at 90% 15%,rgba(124,58,237,.10),transparent 26%),var(--bg); color:var(--text); }
    .block-container { max-width:1440px; padding:1.1rem 2.2rem 3rem; }
    header[data-testid="stHeader"] { background:rgba(8,9,13,.72); }
    #MainMenu, footer {visibility:hidden;}
    .brandbar {display:flex;align-items:center;justify-content:space-between;padding:4px 2px 20px;border-bottom:1px solid rgba(255,255,255,.08);margin-bottom:22px;}
    .brand {font-family:'Manrope',sans-serif;font-size:25px;font-weight:800;letter-spacing:-1.2px;color:#fff;}
    .brand span {color:var(--accent);}
    .brand-sub {font-size:11px;color:#9ca3af;letter-spacing:2.2px;text-transform:uppercase;font-weight:700;}
    .top-pill {display:inline-flex;align-items:center;gap:8px;border:1px solid #30313d;border-radius:30px;padding:8px 13px;background:#12131a;color:#d7d8e0;font-size:12px;font-weight:700;}
    .live-dot {display:inline-block;width:7px;height:7px;border-radius:50%;background:#4ade80;box-shadow:0 0 12px #4ade8070;}
    .hero {position:relative;overflow:hidden;background:linear-gradient(100deg,rgba(8,9,13,.98) 0%,rgba(17,13,21,.92) 48%,rgba(55,12,21,.70) 100%);border:1px solid #34242c;border-radius:26px;padding:clamp(26px,4vw,54px);margin-bottom:24px;box-shadow:0 22px 70px rgba(0,0,0,.24);}
    .hero:after {content:'C';position:absolute;right:5%;top:-75px;font-family:'Manrope',sans-serif;font-size:360px;font-weight:800;line-height:1;color:rgba(229,9,20,.075);pointer-events:none;}
    .hero-badge {display:inline-flex;align-items:center;gap:8px;padding:7px 12px;border-radius:30px;background:rgba(229,9,20,.12);color:#ff858b;border:1px solid rgba(229,9,20,.32);font-size:10px;font-weight:800;letter-spacing:1.8px;margin-bottom:18px;text-transform:uppercase;}
    .hero h1 {font-family:'Manrope',sans-serif;font-size:clamp(34px,5.1vw,62px);font-weight:800;letter-spacing:-2.8px;line-height:1.06;margin:0 0 17px;color:#fff;max-width:760px;}
    .hero h1 em {font-style:normal;color:#ff3441;}
    .hero p {color:#b7b8c5;font-size:15px;line-height:1.85;max-width:660px;margin:0;}
    .hero-note {margin-top:23px;color:#d4d4dd;font-size:12px;font-weight:600;letter-spacing:.3px;}
    .section-heading {font-family:'Manrope',sans-serif;font-size:25px;font-weight:800;color:#fff;letter-spacing:-.7px;margin:25px 0 5px;}
    .section-description {color:var(--muted);font-size:13px;line-height:1.7;margin-bottom:19px;}
    [data-testid="stMetric"] {background:linear-gradient(145deg,#151720,#101117);border:1px solid #292b37;border-radius:16px;padding:17px 19px;box-shadow:0 10px 30px #00000012;}
    [data-testid="stMetricLabel"] {color:#9da2b4;font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase;}
    [data-testid="stMetricValue"] {color:#fff;font-family:'Manrope',sans-serif;font-size:25px;font-weight:800;}
    [data-testid="stTextInput"] input,[data-testid="stSelectbox"] div[data-baseweb="select"],[data-testid="stSlider"] {background:#12141c;}
    [data-testid="stTextInput"] input {border:1px solid #303341;border-radius:12px;min-height:48px;color:#fff;}
    [data-testid="stTextInput"] input:focus {border-color:#e50914;box-shadow:0 0 0 1px #e50914;}
    div[data-baseweb="select"] > div {background:#12141c;border-color:#303341;border-radius:12px;}
    div.stButton > button {border-radius:11px;font-weight:800;min-height:44px;border:1px solid #343644;background:#191b25;color:#f5f5f7;transition:all .18s ease;}
    div.stButton > button:hover {border-color:#ff4450;color:white;transform:translateY(-1px);}
    div.stButton > button[kind="primary"] {background:linear-gradient(100deg,#d90712,#f02b37);border:1px solid #f02b37;color:white;box-shadow:0 7px 22px #e5091425;}
    div.stButton > button[kind="primary"]:hover {background:linear-gradient(100deg,#ef1722,#ff4550);box-shadow:0 10px 26px #e5091435;}
    .movie-card {display:flex;flex-direction:column;min-height:650px;height:100%;background:linear-gradient(150deg,#171923,#101117 75%);border:1px solid #2a2d39;border-radius:19px;padding:15px;margin-bottom:17px;overflow-wrap:anywhere;transition:transform .22s ease,border-color .22s ease,box-shadow .22s ease;box-shadow:0 10px 30px #00000015;}
    .movie-card:hover {border-color:#7f2630;transform:translateY(-4px);box-shadow:0 18px 38px #00000040;}
    .movie-number {display:inline-block;color:#ff858b;font-size:10px;font-weight:800;letter-spacing:1.4px;margin:1px 0 12px;}
    .movie-poster {display:block;width:100%;height:290px;object-fit:cover;object-position:center;border-radius:12px;margin-bottom:15px;background:#0b0c11;border:1px solid #282a35;}
    .poster-placeholder {height:290px;display:flex;align-items:center;justify-content:center;flex-direction:column;gap:8px;background:radial-gradient(circle at 50% 35%,#24202a,#0c0d12 68%);border:1px solid #2a2b36;border-radius:12px;margin-bottom:15px;color:#a6a6b4;text-align:center;}
    .movie-title {color:#fff;font-family:'Manrope',sans-serif;font-size:17px;line-height:1.4;font-weight:800;margin-bottom:13px;min-height:48px;}
    .rating-label {color:#9297a9;font-size:9px;font-weight:800;letter-spacing:1.15px;margin-top:5px;text-transform:uppercase;}
    .rating-value {color:var(--gold);font-size:23px;font-weight:800;margin:5px 0 8px;}
    .rating-track {width:100%;height:5px;background:#2b2d39;border-radius:20px;overflow:hidden;margin-bottom:13px;}
    .rating-fill {height:5px;border-radius:20px;background:linear-gradient(90deg,#e50914,#ff737b);}
    .community-value {color:var(--cyan);font-size:20px;font-weight:800;margin:4px 0;}
    .rating-count {color:#858a9b;font-size:10px;margin-bottom:12px;}
    .genre-heading {color:#9297a9;font-size:9px;font-weight:800;text-transform:uppercase;letter-spacing:1.15px;margin:8px 0;}
    .genre-list {display:flex;flex-wrap:wrap;gap:6px;margin-top:auto;}
    .genre-tag {display:inline-block;background:#20222d;color:#d9dbe4;border:1px solid #343644;border-radius:6px;padding:5px 7px;font-size:10px;font-weight:700;}
    .trending-card {height:100%;background:linear-gradient(150deg,#171923,#101117 75%);border:1px solid #2a2d39;border-radius:16px;padding:12px;overflow-wrap:anywhere;transition:transform .2s ease,border-color .2s ease;}
    .trending-card:hover {border-color:#7f2630;transform:translateY(-3px);}
    .trending-poster {display:block;width:100%;height:245px;object-fit:cover;border-radius:10px;background:#0b0c11;margin-bottom:11px;}
    .trending-title {font-family:'Manrope',sans-serif;color:#fff;font-size:15px;font-weight:800;line-height:1.4;min-height:42px;margin-bottom:7px;}
    .trending-meta {color:#a2a6b7;font-size:11px;line-height:1.7;}
    .trending-overview {color:#c3c5d0;font-size:11px;line-height:1.6;margin-top:8px;}
    .trending-source {display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;padding:12px 14px;background:#11131b;border:1px solid #282b39;border-radius:12px;color:#a2a6b7;font-size:11px;margin:10px 0 16px;}
    .empty-state {background:linear-gradient(145deg,#151720,#101117);border:1px dashed #393b4b;border-radius:18px;padding:32px;text-align:center;color:#a2a6b7;line-height:1.8;}
    [data-testid="stAlert"] {border-radius:12px;}
    .rated-row {padding:11px 14px;background:#141620;border:1px solid #292c39;border-radius:12px;margin:5px 0;}
    .footer {text-align:center;color:#73788a;border-top:1px solid #252733;margin-top:45px;padding-top:25px;font-size:11px;line-height:1.9;}
    @media (max-width:800px) { .block-container{padding:1rem 1rem 2rem;} .hero{padding:27px 22px;} .hero h1{letter-spacing:-1.5px;} .hero:after{font-size:220px;right:-8%;top:0;} .movie-card{min-height:0;padding:12px;} .movie-poster,.poster-placeholder{height:220px;} .movie-title{font-size:15px;} }
</style>
""")

# ============================================================
# 4. LOAD DATA AND MODEL
# ============================================================
@st.cache_data
def load_movies():
    if not MOVIES_PATH.exists():
        raise FileNotFoundError(f"Movies dataset not found: {MOVIES_PATH}")
    data = pd.read_csv(MOVIES_PATH)
    required = {"movieId", "title", "genres"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing movie columns: {missing}")
    data["movieId"] = pd.to_numeric(data["movieId"], errors="coerce")
    data = data.dropna(subset=["movieId", "title"]).copy()
    data["movieId"] = data["movieId"].astype(int)
    data["title"] = data["title"].astype(str)
    data["genres"] = data["genres"].fillna("")
    # MovieLens embeds the release year in titles, e.g. "Toy Story (1995)".
    # Keep missing years as NA so the UI can label them as unknown.
    data["release_year"] = pd.to_numeric(
        data["title"].str.extract(r"\(((?:18|19|20)\d{2})\)\s*$", expand=False),
        errors="coerce",
    ).astype("Int64")
    return data.reset_index(drop=True)


@st.cache_data
def load_community_ratings():
    if not RATINGS_PATH.exists():
        raise FileNotFoundError(f"Ratings dataset not found: {RATINGS_PATH}")
    ratings = pd.read_csv(RATINGS_PATH)
    required = {"movieId", "rating"}
    missing = required - set(ratings.columns)
    if missing:
        raise ValueError(f"Missing rating columns: {missing}")
    ratings["movieId"] = pd.to_numeric(ratings["movieId"], errors="coerce")
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings.dropna(subset=["movieId", "rating"]).copy()
    ratings["movieId"] = ratings["movieId"].astype(int)
    result = ratings.groupby("movieId")["rating"].agg(
        community_rating="mean", rating_count="count"
    ).reset_index()
    result["community_rating"] = result["community_rating"].round(2)
    return result


@st.cache_data
def load_movie_tags():
    """Load and aggregate the existing MovieLens tags; tags are optional."""
    if not TAGS_PATH.exists():
        return pd.DataFrame(columns=["movieId", "tag_text"])
    tags = pd.read_csv(TAGS_PATH)
    if not {"movieId", "tag"}.issubset(tags.columns):
        return pd.DataFrame(columns=["movieId", "tag_text"])
    tags["movieId"] = pd.to_numeric(tags["movieId"], errors="coerce")
    tags = tags.dropna(subset=["movieId", "tag"]).copy()
    tags["movieId"] = tags["movieId"].astype(int)
    tags["tag"] = tags["tag"].astype(str).str.strip()
    tags = tags[tags["tag"] != ""]
    if tags.empty:
        return pd.DataFrame(columns=["movieId", "tag_text"])
    return (tags.groupby("movieId")["tag"]
            .apply(lambda values: " ".join(values.tolist()))
            .reset_index(name="tag_text"))


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
    loaded_model = joblib.load(MODEL_PATH)
    if not hasattr(loaded_model, "trainset"):
        raise ValueError("The saved file is not a compatible trained Surprise model.")
    return loaded_model


# ============================================================
# 5. TMDB POSTERS
# ============================================================
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"


def get_tmdb_api_key():
    """Read the TMDB API key from Streamlit secrets or an environment variable."""
    try:
        key = st.secrets.get("TMDB_API_KEY", "")
    except Exception:
        key = ""
    return key or os.getenv("TMDB_API_KEY", "")


@st.cache_data(ttl=21600, show_spinner=False)
def get_trending_movies(time_window="week"):
    """Fetch TMDB's trending movie list; cache it for six hours."""
    api_key = get_tmdb_api_key()
    if not api_key:
        return {"configured": False, "results": [], "fetched_at": None, "error": "TMDB_API_KEY is not configured."}

    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/trending/movie/{time_window}",
            params={"api_key": api_key, "language": "en-US", "page": 1},
            timeout=(3, 8),
        )
        response.raise_for_status()
        payload = response.json()
        results = [movie for movie in payload.get("results", []) if movie.get("title")]
        return {
            "configured": True,
            "results": results[:12],
            "fetched_at": datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC"),
            "error": None,
        }
    except (requests.RequestException, ValueError, TypeError) as error:
        return {"configured": True, "results": [], "fetched_at": None, "error": str(error)}


@st.cache_data(ttl=86400, show_spinner=False)
def get_movie_poster(title):
    """Search TMDB by title/year and return a poster URL, cached for 24 hours."""
    try:
        api_key = get_tmdb_api_key()
        if not api_key:
            return None

        title_match = re.match(r"^(.*?)\s*\((\d{4})\)\s*$", str(title))
        search_title = title_match.group(1).strip() if title_match else str(title).strip()
        params = {"api_key": api_key, "query": search_title, "include_adult": "false"}
        if title_match:
            params["year"] = title_match.group(2)

        response = requests.get(
            "https://api.themoviedb.org/3/search/movie", params=params, timeout=(2, 4)
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        if not results:
            return None

        # If the MovieLens title includes a year, prefer the same release year.
        if title_match:
            target_year = title_match.group(2)
            same_year = [r for r in results if r.get("release_date", "")[:4] == target_year]
            if same_year:
                results = same_year

        poster_path = results[0].get("poster_path")
        return f"{TMDB_IMAGE_BASE}{poster_path}" if poster_path else None
    except (requests.RequestException, ValueError, KeyError, TypeError):
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def get_movie_posters(titles):
    """Fetch a recommendation batch concurrently; individual results are cached."""
    from concurrent.futures import ThreadPoolExecutor

    unique_titles = list(dict.fromkeys(str(title) for title in titles))
    if not unique_titles:
        return {}

    with ThreadPoolExecutor(max_workers=8) as executor:
        poster_urls = list(executor.map(get_movie_poster, unique_titles))

    return dict(zip(unique_titles, poster_urls))


try:
    movies = load_movies()
    community_ratings = load_community_ratings()
    movie_tags = load_movie_tags()
    model = load_model()
    movies = movies.merge(community_ratings, on="movieId", how="left")
    movies = movies.merge(movie_tags, on="movieId", how="left")
    movies["tag_text"] = movies["tag_text"].fillna("")
    # Build tag-only text features. Genre similarity is calculated separately so
    # broad genres are not accidentally counted twice.
    movies["content_text"] = movies["tag_text"].fillna("").astype(str).str.lower().str.strip()
    content_vectorizer = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), min_df=1,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9]+\b",
        sublinear_tf=True,
    )
    try:
        content_matrix = content_vectorizer.fit_transform(movies["content_text"].tolist())
    except ValueError:
        # Tags are optional; if no usable tag vocabulary exists, rely on SVD + genres.
        from scipy.sparse import csr_matrix
        content_matrix = csr_matrix((len(movies), 0), dtype=float)
except Exception as error:
    st.error(f"Unable to load CineMatch: {error}")
    st.info("Check that the movies CSV, ratings CSV, and svd_model.pkl are in the expected locations.")
    st.stop()

# ============================================================
# 6. MODEL HELPER FUNCTIONS
# ============================================================
def get_movie_inner_id(trainset, movie_id):
    try:
        movie_id = int(movie_id)
    except (TypeError, ValueError):
        return None
    for raw_id in (movie_id, str(movie_id)):
        try:
            return trainset.to_inner_iid(raw_id)
        except (ValueError, KeyError):
            continue
    return None


def get_known_movie_ids(model):
    known_ids = set()
    for inner_id in model.trainset.all_items():
        raw_id = model.trainset.to_raw_iid(inner_id)
        try:
            known_ids.add(int(raw_id))
        except (TypeError, ValueError):
            pass
    return known_ids


def estimate_new_user_profile(model, user_ratings, epochs=200, learning_rate=0.005, reg=0.1):
    """Estimate a new user's factors/bias without retraining the SVD model."""
    if not user_ratings:
        raise ValueError("Please rate at least one movie.")

    trainset = model.trainset
    user_factors = np.zeros(model.n_factors)
    user_bias = 0.0
    known_ratings = []

    for movie_id, actual_rating in user_ratings.items():
        inner_id = get_movie_inner_id(trainset, movie_id)
        if inner_id is not None:
            known_ratings.append((inner_id, float(actual_rating)))
    if not known_ratings:
        raise ValueError("None of the rated movies are represented in the trained model.")

    for _ in range(epochs):
        for inner_id, actual_rating in known_ratings:
            movie_factors = model.qi[inner_id]
            movie_bias = model.bi[inner_id]
            prediction = (
                trainset.global_mean + user_bias + movie_bias
                + np.dot(user_factors, movie_factors)
            )
            error = actual_rating - prediction
            user_bias += learning_rate * (error - reg * user_bias)
            user_factors += learning_rate * (error * movie_factors - reg * user_factors)
    return user_factors, user_bias


def predict_new_user_rating(model, user_factors, user_bias, movie_id):
    inner_id = get_movie_inner_id(model.trainset, movie_id)
    if inner_id is None:
        return None
    prediction = (
        model.trainset.global_mean + user_bias + model.bi[inner_id]
        + np.dot(user_factors, model.qi[inner_id])
    )
    return float(np.clip(prediction, 0.5, 5.0))


# ============================================================
# 7. PERSONALIZED RECOMMENDATIONS + GENRE DIVERSITY
# ============================================================
def recommend_for_new_user(model, user_ratings, movies, top_n=10, max_per_genre=3):
    """Hybrid ranking with separate positive/negative genre and tag preferences."""
    if not user_ratings:
        raise ValueError("Please rate at least one movie.")

    user_factors, user_bias = estimate_new_user_profile(model, user_ratings)
    rated_ids = {int(movie_id) for movie_id in user_ratings}
    known_ids = get_known_movie_ids(model)
    candidates = movies[
        movies["movieId"].isin(known_ids) & ~movies["movieId"].isin(rated_ids)
    ].copy()
    if candidates.empty:
        candidates["predicted_rating"] = pd.Series(dtype=float)
        return candidates

    candidates["predicted_rating"] = candidates["movieId"].apply(
        lambda movie_id: predict_new_user_rating(model, user_factors, user_bias, movie_id)
    )
    candidates = candidates.dropna(subset=["predicted_rating"]).copy()
    candidates["svd_score"] = ((candidates["predicted_rating"] - 0.5) / 4.5).clip(0, 1)

    movie_row_by_id = {int(mid): idx for idx, mid in enumerate(movies["movieId"].tolist())}
    candidate_indices = [movie_row_by_id[int(mid)] for mid in candidates["movieId"].tolist()]

    def parse_genres(value):
        value = str(value or "")
        if not value or value == "(no genres listed)":
            return set()
        return {g.strip().casefold() for g in value.split("|") if g.strip()}

    # Separate liked and disliked examples. A 3/5 rating is treated as neutral.
    liked, disliked = [], []
    for movie_id, rating in user_ratings.items():
        idx = movie_row_by_id.get(int(movie_id))
        if idx is None:
            continue
        rating = float(rating)
        if rating > 3.0:
            liked.append((idx, (rating - 3.0) / 2.0))
        elif rating < 3.0:
            disliked.append((idx, (3.0 - rating) / 2.5))

    def weighted_similarity(candidate_idx, examples, similarity_fn):
        if not examples:
            return 0.0
        values, weights = [], []
        for idx, weight in examples:
            values.append(similarity_fn(candidate_idx, idx))
            weights.append(weight)
        return float(np.average(values, weights=weights)) if sum(weights) else 0.0

    def genre_similarity(a_idx, b_idx):
        a = parse_genres(movies.iloc[a_idx].get("genres", ""))
        b = parse_genres(movies.iloc[b_idx].get("genres", ""))
        if not a or not b:
            return 0.0
        return len(a & b) / max(len(a | b), 1)

    # Compare candidates only with movies the user rated. This avoids building a
    # huge all-movies similarity matrix and keeps memory usage modest.
    rated_indices = sorted({idx for idx, _weight in liked + disliked})
    rated_col_by_idx = {idx: col for col, idx in enumerate(rated_indices)}
    candidate_tag_sim = np.zeros((len(candidate_indices), len(rated_indices)), dtype=float)
    if content_matrix.shape[1] > 0 and rated_indices:
        candidate_tag_sim = cosine_similarity(
            content_matrix[candidate_indices], content_matrix[rated_indices]
        )

    genre_scores, tag_scores, reasons = [], [], []
    for row_pos, candidate_idx in enumerate(candidate_indices):
        pos_genre = weighted_similarity(candidate_idx, liked, genre_similarity)
        neg_genre = weighted_similarity(candidate_idx, disliked, genre_similarity)
        genre_score = float(np.clip(pos_genre - 0.75 * neg_genre, 0.0, 1.0))

        def tag_sim(candidate_idx_inner, rated_idx):
            # row_pos maps the candidate; rated_col_by_idx maps its rated example.
            col = rated_col_by_idx.get(rated_idx)
            return float(candidate_tag_sim[row_pos, col]) if col is not None else 0.0

        pos_tag = weighted_similarity(candidate_idx, liked, tag_sim)
        neg_tag = weighted_similarity(candidate_idx, disliked, tag_sim)
        tag_score = float(np.clip(pos_tag - 0.75 * neg_tag, 0.0, 1.0))
        genre_scores.append(genre_score)
        tag_scores.append(tag_score)

        # Give a short, transparent explanation based on the strongest liked overlap.
        candidate_genres = parse_genres(movies.iloc[candidate_idx].get("genres", ""))
        shared = set()
        for liked_idx, _weight in liked:
            shared |= candidate_genres & parse_genres(movies.iloc[liked_idx].get("genres", ""))
        if shared:
            reasons.append("Shared genres: " + ", ".join(sorted(g.title() for g in shared)[:3]))
        elif tag_score > 0.05:
            reasons.append("Related community tags")
        else:
            reasons.append("Recommended from your overall rating pattern")

    candidates["genre_match"] = genre_scores
    candidates["tag_match"] = tag_scores
    # Explicitly combine separate signals; weights are a starting point to evaluate,
    # not a claim that these values are optimal for every user.
    candidates["hybrid_score"] = (
        0.30 * candidates["svd_score"]
        + 0.40 * candidates["genre_match"]
        + 0.30 * candidates["tag_match"]
    )
    candidates["recommendation_reason"] = reasons
    candidates = candidates.sort_values(
        ["hybrid_score", "genre_match", "tag_match", "predicted_rating"],
        ascending=False,
    )

    selected = []
    genre_counts = {}
    for _, movie in candidates.iterrows():
        movie_genres = [] if not str(movie.get("genres", "") or "") or str(movie.get("genres", "")) == "(no genres listed)" else [
            g.strip() for g in str(movie.get("genres", "")).split("|") if g.strip()
        ]
        # Apply a soft diversity cap. If too few candidates remain, fill below.
        if movie_genres and any(genre_counts.get(g, 0) >= max_per_genre for g in movie_genres):
            continue
        selected.append(movie)
        for genre in movie_genres:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
        if len(selected) >= top_n:
            break

    if len(selected) < top_n:
        selected_ids = {int(movie["movieId"]) for movie in selected}
        remaining = candidates[~candidates["movieId"].isin(selected_ids)]
        for _, movie in remaining.iterrows():
            selected.append(movie)
            if len(selected) >= top_n:
                break
    return pd.DataFrame(selected).reset_index(drop=True) if selected else candidates.head(0).copy()

# ============================================================
# 8. SEARCH + RECOMMENDATION CARD DISPLAY
# ============================================================
def search_movies(query, movies):
    query = query.strip().casefold()
    if not query:
        return movies.iloc[0:0].copy()
    return movies[movies["title"].str.casefold().str.contains(query, regex=False, na=False)].copy()


def display_movie_card(movie, rank, poster_url=None):
    title = str(movie["title"])
    safe_title = html.escape(title)
    predicted_rating = float(np.clip(float(movie["predicted_rating"]), 0.0, 5.0))
    community_rating = movie.get("community_rating", np.nan)
    rating_count = movie.get("rating_count", np.nan)
    genres = str(movie.get("genres", "") or "")
    genre_list = ["Genres unavailable"] if not genres or genres == "(no genres listed)" else [
        g.strip() for g in genres.split("|") if g.strip()
    ]
    genre_html = "".join(f'<span class="genre-tag">{html.escape(g)}</span>' for g in genre_list)
    percentage = predicted_rating / 5.0 * 100
    recommendation_reason = html.escape(str(movie.get("recommendation_reason", "Recommended from your ratings")))

    # Poster URLs are pre-fetched concurrently before the cards are rendered.
    if poster_url:
        poster_html = f'<img class="movie-poster" src="{html.escape(poster_url, quote=True)}" alt="{safe_title} poster">'
    else:
        poster_html = '<div class="poster-placeholder"><div style="font-size:42px;">🎬</div><div style="font-size:11px;font-weight:800;letter-spacing:1px;">CINEMATIC PICK</div><div style="font-size:10px;color:#707486;">Poster unavailable</div></div>'

    if pd.notna(community_rating):
        count_text = f"{int(rating_count):,} user ratings" if pd.notna(rating_count) else "Rating count unavailable"
        community_html = f'''<div class="rating-label">COMMUNITY RATING</div>
        <div class="community-value">&#9733; {float(community_rating):.2f}
        <span style="font-size:13px;color:#a3adbd;">/ 5</span></div>
        <div class="rating-count">Based on {count_text}</div>'''
    else:
        community_html = '<div class="rating-label">COMMUNITY RATING</div><div class="rating-count">No community ratings available.</div>'

    st.html(f'''<div class="movie-card">
        <div class="movie-number">RECOMMENDATION #{rank}</div>
        {poster_html}
        <div class="movie-title">{safe_title}</div>
        <div class="rating-label">YOUR PREDICTED RATING</div>
        <div class="rating-value">&#9733; {predicted_rating:.2f}
        <span style="font-size:13px;color:#a3adbd;">/ 5</span></div>
        <div class="rating-track"><div class="rating-fill" style="width:{percentage:.2f}%;"></div></div>
        {community_html}
        <div class="rating-label">WHY THIS MOVIE?</div>
        <div style="color:#c4b5fd;font-size:13px;line-height:1.5;margin:6px 0 14px;">{recommendation_reason}</div>
        <div class="genre-heading">GENRES</div><div class="genre-list">{genre_html}</div>
    </div>''')


# ============================================================
# 9. USER DATABASE + AUTHENTICATION
# ============================================================
DB_PATH = BASE_DIR / "cinematch.db"


def get_db_connection():
    # Create a SQLite connection with foreign-key checks enabled.
    connection = sqlite3.connect(DB_PATH, timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database():
    # Create user profiles and per-user movie ratings tables if needed.
    with get_db_connection() as connection:
        connection.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                email TEXT NOT NULL COLLATE NOCASE UNIQUE,
                password_salt BLOB NOT NULL,
                password_hash BLOB NOT NULL,
                created_at TEXT NOT NULL
            )
        ''')
        connection.execute('''
            CREATE TABLE IF NOT EXISTS user_ratings (
                user_id INTEGER NOT NULL,
                movie_id INTEGER NOT NULL,
                rating REAL NOT NULL CHECK (rating >= 0.5 AND rating <= 5.0),
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, movie_id),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        ''')


def hash_password(password, salt=None):
    # Hash a password with PBKDF2-HMAC-SHA256; never store plain text.
    salt = salt or secrets.token_bytes(16)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, 310_000
    )
    return salt, password_hash


def create_user(username, email, password):
    username = username.strip()
    email = email.strip().lower()
    salt, password_hash = hash_password(password)
    created_at = datetime.now(timezone.utc).isoformat()
    try:
        with get_db_connection() as connection:
            connection.execute(
                "INSERT INTO users (username, email, password_salt, password_hash, created_at) VALUES (?, ?, ?, ?, ?)",
                (username, email, salt, password_hash, created_at),
            )
        return True, "Account created successfully. Please sign in."
    except sqlite3.IntegrityError:
        return False, "That username or email is already registered. Try signing in instead."


def authenticate_user(identifier, password):
    identifier = identifier.strip()
    with get_db_connection() as connection:
        user = connection.execute(
            "SELECT id, username, email, password_salt, password_hash, created_at "
            "FROM users WHERE username = ? COLLATE NOCASE OR email = ? COLLATE NOCASE",
            (identifier, identifier),
        ).fetchone()
    if user is None:
        return None
    _, candidate_hash = hash_password(password, bytes(user["password_salt"]))
    if not hmac.compare_digest(candidate_hash, bytes(user["password_hash"])):
        return None
    return {
        "id": int(user["id"]),
        "username": user["username"],
        "email": user["email"],
        "created_at": user["created_at"],
    }


def load_user_ratings(user_id):
    with get_db_connection() as connection:
        rows = connection.execute(
            "SELECT movie_id, rating FROM user_ratings WHERE user_id = ? ORDER BY updated_at DESC",
            (int(user_id),),
        ).fetchall()
    return {int(row["movie_id"]): float(row["rating"]) for row in rows}


def save_user_rating(user_id, movie_id, rating):
    updated_at = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as connection:
        connection.execute(
            "INSERT INTO user_ratings (user_id, movie_id, rating, updated_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(user_id, movie_id) DO UPDATE SET rating = excluded.rating, updated_at = excluded.updated_at",
            (int(user_id), int(movie_id), float(rating), updated_at),
        )


def delete_user_rating(user_id, movie_id):
    with get_db_connection() as connection:
        connection.execute(
            "DELETE FROM user_ratings WHERE user_id = ? AND movie_id = ?",
            (int(user_id), int(movie_id)),
        )


initialize_database()

if "current_user" not in st.session_state:
    st.session_state.current_user = None

if st.session_state.current_user is None:
    st.html('''
    <div class="brandbar"><div><div class="brand">Cine<span>Match</span></div>
    <div class="brand-sub">Your personal cinema guide</div></div>
    <div class="top-pill">🔐 YOUR PRIVATE TASTE PROFILE</div></div>
    <div class="hero"><div class="hero-badge">✦ YOUR NEXT GREAT WATCH STARTS HERE</div>
    <h1>Movies picked for <em>you.</em></h1>
    <p>Create an account to save your movie ratings and return to your personalized recommendations later.</p></div>
    ''')
    login_tab, signup_tab = st.tabs(["Sign in", "Create account"])
    with login_tab:
        with st.form("login_form"):
            login_identifier = st.text_input("Username or email", key="login_identifier")
            login_password = st.text_input("Password", type="password", key="login_password")
            login_submit = st.form_submit_button("Sign in", type="primary", use_container_width=True)
        if login_submit:
            user = authenticate_user(login_identifier, login_password)
            if user:
                st.session_state.current_user = user
                st.session_state.user_ratings = load_user_ratings(user["id"])
                st.session_state.recommendations = None
                st.session_state.recommendation_signature = None
                st.rerun()
            else:
                st.error("We couldn't sign you in with those details. Check them and try again.")
    with signup_tab:
        with st.form("signup_form"):
            new_username = st.text_input("Choose a username", key="signup_username", max_chars=30)
            new_email = st.text_input("Email address", key="signup_email")
            new_password = st.text_input("Create password (at least 8 characters)", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm password", type="password", key="signup_confirm_password")
            signup_submit = st.form_submit_button("Create account", type="primary", use_container_width=True)
        if signup_submit:
            if len(new_username.strip()) < 3:
                st.error("Your username must contain at least 3 characters.")
            elif "@" not in new_email or "." not in new_email.rsplit("@", 1)[-1]:
                st.error("Enter a valid email address.")
            elif len(new_password) < 8:
                st.error("Use a password with at least 8 characters.")
            elif new_password != confirm_password:
                st.error("The passwords do not match.")
            else:
                success, message = create_user(new_username, new_email, new_password)
                (st.success if success else st.error)(message)
    st.caption("Your account and ratings are stored in the app's SQLite database. Passwords are stored as salted hashes, not plain text.")
    st.stop()

# Restore saved ratings from the database for the signed-in user on every rerun.
current_user = st.session_state.current_user
st.session_state.user_ratings = load_user_ratings(current_user["id"])

# ============================================================
# 10. SESSION STATE
# ============================================================
if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = load_user_ratings(current_user["id"])
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None
if "recommendation_signature" not in st.session_state:
    st.session_state.recommendation_signature = None

# ============================================================
# SIGNED-IN PROFILE CONTROLS
# ============================================================
profile_col, logout_col = st.columns([5, 1], vertical_alignment="center")
with profile_col:
    st.caption(f"Signed in as **{current_user['username']}** · {current_user['email']}")
with logout_col:
    if st.button("Log out", use_container_width=True):
        for key in ("current_user", "user_ratings", "recommendations", "recommendation_signature"):
            st.session_state.pop(key, None)
        st.rerun()

# ============================================================
# 11. HERO + DASHBOARD
# ============================================================
st.html('''<div class="brandbar">
    <div><div class="brand">Cine<span>Match</span></div><div class="brand-sub">Your personal cinema guide</div></div>
    <div class="top-pill"><span class="live-dot"></span> PERSONALIZED FOR YOU</div>
</div>
<div class="hero">
    <div class="hero-badge">✦ YOUR NEXT GREAT WATCH STARTS HERE</div>
    <h1>Less scrolling.<br><em>More watching.</em></h1>
    <p>Tell CineMatch what you love. Your taste profile helps uncover movies you might have missed — powered by collaborative filtering, genre signals, and your existing recommendation model.</p>
    <div class="hero-note">✦ &nbsp; Curated around your taste &nbsp;&nbsp; • &nbsp;&nbsp; Built on MovieLens data</div>
</div>''')

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Movies available", f"{len(movies):,}")
with col2:
    st.metric("Movies rated", len(st.session_state.user_ratings))
with col3:
    st.metric("Recommendation engine", "SVD + Content")

# ============================================================
# 12. TRENDING MOVIES (TMDB, refreshed every six hours)
# ============================================================
st.html('''<div class="section-heading">🔥 Trending now</div>
<div class="section-description">Movies gaining attention globally, sourced from TMDB's trending movies feed. This is separate from your personalized recommendations.</div>''')
trend_header_col, trend_refresh_col = st.columns([5, 1], vertical_alignment="center")
with trend_header_col:
    st.caption("Default feed: trending movies this week · Data cache refreshes every 6 hours.")
with trend_refresh_col:
    if st.button("↻ Refresh", key="refresh_trending", use_container_width=True):
        get_trending_movies.clear()
        st.rerun()

with st.spinner("Loading current trending movies..."):
    trending_payload = get_trending_movies("week")

if not trending_payload["configured"]:
    st.info("Trending movies need a TMDB API key. Add TMDB_API_KEY to Streamlit secrets or your environment, then refresh this section.")
elif trending_payload["error"]:
    st.warning("The trending feed could not be loaded right now. Your saved ratings and recommendations are unaffected.")
    st.caption("Check TMDB API access and your TMDB_API_KEY configuration, then use Refresh to try again.")
else:
    st.html(f'''<div class="trending-source"><span>DATA SOURCE: TMDB · TRENDING MOVIES / WEEK</span><span>Last fetched: {html.escape(trending_payload["fetched_at"] or "unknown")}</span></div>''')
    trend_movies = trending_payload["results"]
    if not trend_movies:
        st.info("TMDB returned no trending movies for this period.")
    else:
        for trend_start in range(0, min(len(trend_movies), 12), 4):
            trend_columns = st.columns(4, gap="medium")
            for trend_offset, trend_column in enumerate(trend_columns):
                trend_index = trend_start + trend_offset
                if trend_index >= min(len(trend_movies), 12):
                    break
                trend_movie = trend_movies[trend_index]
                with trend_column:
                    trend_title = str(trend_movie.get("title") or "Untitled")
                    trend_safe_title = html.escape(trend_title)
                    poster_path = trend_movie.get("poster_path")
                    if poster_path:
                        trend_poster = f'<img class="trending-poster" src="{TMDB_IMAGE_BASE}{html.escape(str(poster_path), quote=True)}" alt="{trend_safe_title} poster">'
                    else:
                        trend_poster = '<div class="poster-placeholder" style="height:245px"><div style="font-size:32px">🎬</div><div style="font-size:10px">POSTER UNAVAILABLE</div></div>'
                    release_date = str(trend_movie.get("release_date") or "")
                    release_year = release_date[:4] if len(release_date) >= 4 else "Year unknown"
                    vote_average = trend_movie.get("vote_average")
                    vote_text = f"★ {float(vote_average):.1f}/10" if isinstance(vote_average, (int, float)) else "Rating unavailable"
                    popularity = trend_movie.get("popularity")
                    popularity_text = f"Popularity score: {float(popularity):.0f}" if isinstance(popularity, (int, float)) else ""
                    overview = str(trend_movie.get("overview") or "No synopsis available.").strip()
                    if len(overview) > 170:
                        overview = overview[:167].rsplit(" ", 1)[0] + "..."
                    st.html(f'''<div class="trending-card">{trend_poster}<div class="trending-title">{trend_safe_title}</div><div class="trending-meta">{html.escape(release_year)} · {html.escape(vote_text)}<br>{html.escape(popularity_text)}</div><div class="trending-overview">{html.escape(overview)}</div></div>''')
    st.caption("Trending data and poster images provided by TMDB. Trending popularity is not the same as your personalized recommendation score.")

# ============================================================
# 13. SEARCH + RATE MOVIES
# ============================================================
st.html('''<div class="section-heading">Build your taste profile</div>
<div class="section-description">Search for movies you know and rate them. CineMatch will use your ratings to personalize recommendations.</div>''')

search_query = st.text_input(
    "Search for a movie",
    placeholder="Try Toy Story, Interstellar, or The Dark Knight...",
    key="movie_search",
)
search_results = search_movies(search_query, movies)

if search_query.strip():
    if search_results.empty:
        st.warning("No movies found. Try another title or a shorter search.")
    else:
        search_results = search_results.head(100)
        title_by_id = dict(zip(search_results["movieId"], search_results["title"]))
        selected_movie_id = st.selectbox(
            "Choose a movie",
            options=search_results["movieId"].tolist(),
            format_func=lambda movie_id: title_by_id[movie_id],
            key="selected_movie",
        )
        selected_movie = movies.loc[movies["movieId"] == selected_movie_id].iloc[0]
        genre_text = str(selected_movie["genres"]).replace("|", " · ")
        st.caption(f"Genres: {genre_text or 'Genres unavailable'}")
        if pd.notna(selected_movie.get("community_rating", np.nan)):
            st.caption(
                f"Community rating: ⭐ {selected_movie['community_rating']:.2f}/5 "
                f"({int(selected_movie['rating_count']):,} ratings)"
            )
        rating_col, button_col = st.columns([3, 1], vertical_alignment="bottom")
        with rating_col:
            selected_rating = st.slider(
                "Your rating", min_value=0.5, max_value=5.0,
                value=4.0, step=0.5, key="selected_rating",
            )
        with button_col:
            if st.button("Add rating", type="primary", use_container_width=True):
                save_user_rating(current_user["id"], int(selected_movie_id), float(selected_rating))
                st.session_state.user_ratings[int(selected_movie_id)] = float(selected_rating)
                st.session_state.recommendations = None
                st.session_state.recommendation_signature = None
                st.success(f"Rating saved for {selected_movie['title']}.")
                st.rerun()

# ============================================================
# 13. USER RATING MANAGEMENT
# ============================================================
st.html('''<div class="section-heading">Your rated movies</div>
<div class="section-description">Review or remove ratings before generating recommendations.</div>''')
user_ratings = st.session_state.user_ratings

if not user_ratings:
    st.html('''<div class="empty-state"><div style="font-size:32px;margin-bottom:10px;">&#127916;</div>
    <strong>No ratings yet</strong><br>Search for a movie above and give it a rating to start building your taste profile.</div>''')
else:
    for movie_id, rating in list(user_ratings.items()):
        movie_match = movies.loc[movies["movieId"] == int(movie_id)]
        if movie_match.empty:
            continue
        movie = movie_match.iloc[0]
        rating_col, remove_col = st.columns([5, 1], vertical_alignment="center")
        with rating_col:
            st.markdown(f"**{movie['title']}**  \nYour rating: ⭐ {rating:.1f}/5")
        with remove_col:
            if st.button("Remove", key=f"remove_{movie_id}", use_container_width=True):
                delete_user_rating(current_user["id"], int(movie_id))
                del st.session_state.user_ratings[movie_id]
                st.session_state.recommendations = None
                st.session_state.recommendation_signature = None
                st.rerun()

# ============================================================
# 14. GENERATE RECOMMENDATIONS
# ============================================================
st.html('''<div class="section-heading">Discover your next favorite</div>
<div class="section-description">Personalized picks, with filters to narrow results by genre, release year, and community rating.</div>''')

# Filters are applied to the ranked recommendation pool, not to the trained model.
all_genres = sorted({
    genre.strip()
    for value in movies["genres"].fillna("").astype(str)
    for genre in value.split("|")
    if genre.strip() and genre.strip() != "(no genres listed)"
})
years = movies["release_year"].dropna().astype(int)
if years.empty:
    min_available_year, max_available_year = 1900, 2026
else:
    min_available_year, max_available_year = int(years.min()), int(years.max())

st.markdown("#### 🎛️ Refine your recommendations")
filter_col1, filter_col2, filter_col3 = st.columns([1.25, 1.25, 1])
with filter_col1:
    selected_genres = st.multiselect(
        "Genres",
        options=all_genres,
        default=[],
        help="Leave empty to include every genre. A movie matching at least one selected genre is included.",
        key="recommendation_genre_filter",
    )
with filter_col2:
    selected_years = st.slider(
        "Release year",
        min_value=min_available_year,
        max_value=max_available_year,
        value=(min_available_year, max_available_year),
        help="Choose the inclusive release-year range.",
        key="recommendation_year_filter",
    )
with filter_col3:
    min_community_rating = st.slider(
        "Minimum community rating",
        min_value=0.0,
        max_value=5.0,
        value=0.0,
        step=0.5,
        help="Uses average MovieLens ratings. Movies without a community rating are excluded when this is above 0.",
        key="recommendation_min_rating_filter",
    )

if not st.session_state.user_ratings:
    st.info("Rate at least one movie before generating recommendations.")
else:
    st.caption("For better personalization, rate several movies that reflect your actual preferences.")
    if st.button("Generate recommendations", type="primary", use_container_width=True):
        try:
            with st.spinner("Finding movies that match your preferences..."):
                recommendations = recommend_for_new_user(
                    model=model,
                    user_ratings=st.session_state.user_ratings,
                    movies=movies,
                    # Keep a larger ranked pool so filters can still return enough results.
                    top_n=500,
                    max_per_genre=10,
                )
                st.session_state.recommendations = recommendations
                st.session_state.recommendation_signature = tuple(sorted(st.session_state.user_ratings.items()))
        except Exception as error:
            st.session_state.recommendations = None
            st.error(f"Unable to generate recommendations: {error}")

# ============================================================
# 15. DISPLAY RECOMMENDATIONS
# ============================================================
recommendations = st.session_state.recommendations
current_signature = tuple(sorted(st.session_state.user_ratings.items()))
saved_signature = st.session_state.recommendation_signature

if recommendations is not None:
    if saved_signature != current_signature:
        st.info("Your ratings have changed. Generate recommendations again to reflect your updated preferences.")
    elif recommendations.empty:
        st.warning("No recommendations are available for these ratings.")
    else:
        filtered_recommendations = recommendations.copy()

        if selected_genres:
            wanted_genres = {genre.casefold() for genre in selected_genres}
            filtered_recommendations = filtered_recommendations[
                filtered_recommendations["genres"].fillna("").apply(
                    lambda value: bool(
                        wanted_genres & {
                            item.strip().casefold()
                            for item in str(value).split("|")
                            if item.strip()
                        }
                    )
                )
            ]

        filtered_recommendations = filtered_recommendations[
            filtered_recommendations["release_year"].notna()
            & filtered_recommendations["release_year"].between(
                selected_years[0], selected_years[1], inclusive="both"
            )
        ]

        if min_community_rating > 0:
            filtered_recommendations = filtered_recommendations[
                pd.to_numeric(filtered_recommendations["community_rating"], errors="coerce")
                .ge(min_community_rating)
            ]

        filtered_recommendations = filtered_recommendations.head(10).reset_index(drop=True)
        st.markdown(f"### ✦ Picks for your taste · {len(filtered_recommendations)} movies")
        st.caption(
            "Filters apply to the ranked recommendation pool. Genre matches any selected genre; release year is inclusive; minimum rating uses MovieLens community averages. "
            "Predicted ratings are estimates, not guarantees. Posters are retrieved from TMDB."
        )

        if filtered_recommendations.empty:
            st.info("No movies match these filters. Try selecting more genres, widening the year range, or lowering the minimum rating.")
        else:
            # Fetch poster URLs only for the filtered results currently being displayed.
            with st.spinner("Loading movie posters..."):
                poster_map = get_movie_posters(
                    filtered_recommendations["title"].astype(str).tolist()
                )

            for start in range(0, len(filtered_recommendations), 3):
                columns = st.columns(3, gap="medium")
                for offset, column in enumerate(columns):
                    index = start + offset
                    if index >= len(filtered_recommendations):
                        break
                    movie = filtered_recommendations.iloc[index]
                    with column:
                        display_movie_card(
                            movie,
                            rank=index + 1,
                            poster_url=poster_map.get(str(movie["title"])),
                        )

# ============================================================
# 16. FOOTER
# ============================================================
st.html('''<div class="footer">
<strong style="color:#c4b5fd;">CineMatch</strong><br>
Personalized movie discovery powered by collaborative filtering.<br>
Built with Python, Streamlit, SQLite, and Surprise SVD.<br>
Movie poster images and movie metadata provided by TMDB.<br>
This product uses the TMDB API but is not endorsed or certified by TMDB.
</div>''')
