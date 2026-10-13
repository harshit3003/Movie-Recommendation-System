import html
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


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
MODEL_PATH = BASE_DIR / "svd_model.pkl"


# ============================================================
# 3. CUSTOM CSS
# ============================================================

st.html("""
<style>
    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    :root {
        --background: #0b0d12;
        --surface: #141821;
        --border: #292f3d;
        --text: #f5f7fb;
        --muted: #a3adbd;
        --accent: #a78bfa;
        --gold: #fbbf24;
    }

    .stApp {
        background: var(--background);
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        background:
            radial-gradient(
                circle at 90% 10%,
                rgba(139, 92, 246, 0.25),
                transparent 35%
            ),
            linear-gradient(135deg, #191d2a, #10131c);

        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 38px;
        margin-bottom: 30px;
    }

    .hero-badge {
        display: inline-block;
        padding: 7px 13px;
        border-radius: 30px;
        background: rgba(167, 139, 250, 0.13);
        color: #c4b5fd;
        border: 1px solid rgba(167, 139, 250, 0.25);
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 15px;
    }

    .hero h1 {
        font-size: clamp(32px, 5vw, 52px);
        font-weight: 800;
        letter-spacing: -1.8px;
        line-height: 1.15;
        margin: 0 0 14px 0;
        color: #ffffff;
    }

    .hero p {
        color: #b9c1d0;
        font-size: 16px;
        line-height: 1.8;
        max-width: 700px;
        margin: 0;
    }

    .section-heading {
        font-size: 25px;
        font-weight: 800;
        color: #ffffff;
        margin: 20px 0 6px 0;
    }

    .section-description {
        color: var(--muted);
        font-size: 14px;
        margin-bottom: 22px;
    }

    .movie-card {
        display: flex;
        flex-direction: column;
        min-height: 360px;
        height: 100%;
        background: linear-gradient(145deg, #171b26, #11141c);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 20px;
        margin-bottom: 18px;
        overflow-wrap: anywhere;
        transition: border-color 0.2s ease,
                    transform 0.2s ease;
    }

    .movie-card:hover {
        border-color: var(--accent);
        transform: translateY(-3px);
    }

    .movie-number {
        color: var(--accent);
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 1.5px;
        margin-bottom: 13px;
    }

    .movie-title {
        color: #ffffff;
        font-size: 19px;
        line-height: 1.5;
        font-weight: 700;
        margin-bottom: 18px;
    }

    .rating-label {
        color: var(--muted);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.7px;
        margin-top: 5px;
    }

    .rating-value {
        color: var(--gold);
        font-size: 25px;
        font-weight: 800;
        margin: 5px 0 10px 0;
    }

    .rating-track {
        width: 100%;
        height: 7px;
        background: #292f3d;
        border-radius: 20px;
        overflow: hidden;
        margin-bottom: 17px;
    }

    .rating-fill {
        height: 7px;
        border-radius: 20px;
        background: linear-gradient(90deg, #8b5cf6, #c4b5fd);
    }

    .community-value {
        color: #67e8f9;
        font-size: 23px;
        font-weight: 800;
        margin: 5px 0;
    }

    .rating-count {
        color: var(--muted);
        font-size: 12px;
        margin-bottom: 17px;
    }

    .genre-heading {
        color: var(--muted);
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }

    .genre-list {
        display: flex;
        flex-wrap: wrap;
        gap: 7px;
        margin-top: auto;
    }

    .genre-tag {
        display: inline-block;
        background: #242a38;
        color: #d5d9e4;
        border: 1px solid #353d4e;
        border-radius: 7px;
        padding: 6px 9px;
        font-size: 11px;
        font-weight: 600;
    }

    .empty-state {
        background: var(--surface);
        border: 1px dashed #394153;
        border-radius: 18px;
        padding: 30px;
        text-align: center;
        color: var(--muted);
        line-height: 1.8;
    }

    div.stButton > button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 42px;
    }

    div.stButton > button[kind="primary"] {
        background: #8b5cf6;
        border: 1px solid #8b5cf6;
        color: white;
    }

    .footer {
        text-align: center;
        color: #7d8798;
        border-top: 1px solid var(--border);
        margin-top: 45px;
        padding-top: 25px;
        font-size: 12px;
        line-height: 1.8;
    }
</style>
""")


# ============================================================
# 4. HERO SECTION
# ============================================================

# st.html("""
# <div class="hero">
#     <div class="hero-badge">PERSONALIZED MOVIE DISCOVERY</div>

#     <h1>Find your next<br>favorite movie.</h1>

#     <p>
#         Rate movies you enjoy and discover recommendations
#         tailored to your preferences.
#         Powered by your existing SVD recommendation model.
#     </p>
# </div>
# """)


# ============================================================
# 5. LOAD MOVIES
# ============================================================

@st.cache_data
def load_movies():

    if not MOVIES_PATH.exists():
        raise FileNotFoundError(
            f"Movies dataset not found: {MOVIES_PATH}"
        )

    data = pd.read_csv(MOVIES_PATH)

    required_columns = {"movieId", "title", "genres"}

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing movie columns: {missing_columns}"
        )

    data["movieId"] = pd.to_numeric(
        data["movieId"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["movieId", "title"]
    ).copy()

    data["movieId"] = data["movieId"].astype(int)
    data["title"] = data["title"].astype(str)
    data["genres"] = data["genres"].fillna("")

    return data.reset_index(drop=True)


# ============================================================
# 6. LOAD COMMUNITY RATINGS
# ============================================================

@st.cache_data
def load_community_ratings():

    if not RATINGS_PATH.exists():
        raise FileNotFoundError(
            f"Ratings dataset not found: {RATINGS_PATH}"
        )

    ratings = pd.read_csv(RATINGS_PATH)

    required_columns = {"movieId", "rating"}

    missing_columns = required_columns - set(ratings.columns)

    if missing_columns:
        raise ValueError(
            f"Missing rating columns: {missing_columns}"
        )

    ratings["movieId"] = pd.to_numeric(
        ratings["movieId"],
        errors="coerce"
    )

    ratings["rating"] = pd.to_numeric(
        ratings["rating"],
        errors="coerce"
    )

    ratings = ratings.dropna(
        subset=["movieId", "rating"]
    ).copy()

    ratings["movieId"] = ratings["movieId"].astype(int)

    community_ratings = (
        ratings.groupby("movieId")["rating"]
        .agg(
            community_rating="mean",
            rating_count="count"
        )
        .reset_index()
    )

    community_ratings["community_rating"] = (
        community_ratings["community_rating"].round(2)
    )

    return community_ratings


# ============================================================
# 7. LOAD EXISTING SVD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "trainset"):
        raise ValueError(
            "The saved file is not a compatible trained Surprise model."
        )

    return model


# Load application resources

try:

    movies = load_movies()

    community_ratings = load_community_ratings()

    model = load_model()

    # Merge community ratings into the movie catalogue.
    movies = movies.merge(
        community_ratings,
        on="movieId",
        how="left"
    )

except Exception as error:

    st.error(f"Unable to load CineMatch: {error}")

    st.info(
        "Check that movies (1).csv, ratings (1).csv, "
        "and svd_model.pkl are present in the expected locations."
    )

    st.stop()


# ============================================================
# 8. MODEL HELPER FUNCTIONS
# ============================================================

def get_movie_inner_id(trainset, movie_id):
    """Map an original movie ID to its internal model ID."""

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
    """Return movie IDs represented in the trained SVD model."""

    trainset = model.trainset

    return {
        int(trainset.to_raw_iid(inner_id))
        for inner_id in trainset.all_items()
    }


def estimate_new_user_profile(
    model,
    user_ratings,
    epochs=200,
    learning_rate=0.005,
    reg=0.1,
):
    """
    Estimate a new user's latent preferences while keeping
    the trained SVD model's item factors fixed.
    """

    if not user_ratings:
        raise ValueError(
            "Please rate at least one movie."
        )

    trainset = model.trainset

    user_factors = np.zeros(model.n_factors)
    user_bias = 0.0

    known_ratings = []

    for movie_id, actual_rating in user_ratings.items():

        inner_id = get_movie_inner_id(
            trainset,
            movie_id
        )

        if inner_id is not None:

            known_ratings.append(
                (inner_id, float(actual_rating))
            )

    if not known_ratings:
        raise ValueError(
            "None of the rated movies are represented in "
            "the trained model."
        )

    for _ in range(epochs):

        for inner_id, actual_rating in known_ratings:

            movie_factors = model.qi[inner_id]
            movie_bias = model.bi[inner_id]

            predicted_rating = (
                trainset.global_mean
                + user_bias
                + movie_bias
                + np.dot(user_factors, movie_factors)
            )

            error = actual_rating - predicted_rating

            user_bias += learning_rate * (
                error - reg * user_bias
            )

            user_factors += learning_rate * (
                error * movie_factors
                - reg * user_factors
            )

    return user_factors, user_bias


def predict_new_user_rating(
    model,
    user_factors,
    user_bias,
    movie_id,
):
    """Predict a new user's rating using fixed SVD item factors."""

    inner_id = get_movie_inner_id(
        model.trainset,
        movie_id
    )

    if inner_id is None:
        return None

    movie_factors = model.qi[inner_id]
    movie_bias = model.bi[inner_id]

    predicted_rating = (
        model.trainset.global_mean
        + user_bias
        + movie_bias
        + np.dot(user_factors, movie_factors)
    )

    return float(
        np.clip(predicted_rating, 0.5, 5.0)
    )


# ============================================================
# 9. PERSONALIZED RECOMMENDATIONS + GENRE DIVERSITY
# ============================================================

def recommend_for_new_user(
    model,
    user_ratings,
    movies,
    top_n=10,
    max_per_genre=2,
):
    """
    Generate personalized recommendations with genre diversity.

    The trained SVD model is not retrained.
    """

    user_factors, user_bias = estimate_new_user_profile(
        model,
        user_ratings
    )

    rated_movie_ids = {
        int(movie_id)
        for movie_id in user_ratings
    }

    known_movie_ids = get_known_movie_ids(model)

    candidates = movies[
        movies["movieId"].isin(known_movie_ids)
        & ~movies["movieId"].isin(rated_movie_ids)
    ].copy()

    if candidates.empty:

        candidates["predicted_rating"] = pd.Series(dtype=float)

        return candidates

    # Predict ratings using the existing model.
    candidates["predicted_rating"] = candidates[
        "movieId"
    ].apply(
        lambda movie_id: predict_new_user_rating(
            model,
            user_factors,
            user_bias,
            movie_id
        )
    )

    candidates = candidates.dropna(
        subset=["predicted_rating"]
    )

    candidates = candidates.sort_values(
        "predicted_rating",
        ascending=False
    )

    selected_movies = []
    genre_counts = {}

    # Select high-scoring movies while limiting genre repetition.
    for _, movie in candidates.iterrows():

        genres = str(movie["genres"])

        if not genres or genres == "(no genres listed)":
            movie_genres = []
        else:
            movie_genres = [
                genre.strip()
                for genre in genres.split("|")
                if genre.strip()
            ]

        if movie_genres and any(
            genre_counts.get(genre, 0) >= max_per_genre
            for genre in movie_genres
        ):
            continue

        selected_movies.append(movie)

        for genre in movie_genres:

            genre_counts[genre] = (
                genre_counts.get(genre, 0) + 1
            )

        if len(selected_movies) >= top_n:
            break

    # Fill remaining positions if necessary.
    if len(selected_movies) < top_n:

        selected_ids = {
            int(movie["movieId"])
            for movie in selected_movies
        }

        remaining = candidates[
            ~candidates["movieId"].isin(selected_ids)
        ]

        for _, movie in remaining.iterrows():

            selected_movies.append(movie)

            if len(selected_movies) >= top_n:
                break

    if not selected_movies:
        return candidates.head(0).copy()

    recommendations = pd.DataFrame(selected_movies)

    return recommendations.reset_index(drop=True)


# ============================================================
# 10. MOVIE SEARCH
# ============================================================

def search_movies(query, movies):
    """Case-insensitive search for partial movie titles."""

    query = query.strip().casefold()

    if not query:
        return movies.iloc[0:0].copy()

    return movies[
        movies["title"]
        .str.casefold()
        .str.contains(
            query,
            regex=False,
            na=False
        )
    ].copy()


# ============================================================
# 11. RECOMMENDATION CARD DISPLAY
# ============================================================

def display_movie_card(movie, rank):
    """
    Display predicted rating, community rating,
    rating count, genres, and a visual rating bar.
    """

    title = html.escape(str(movie["title"]))

    predicted_rating = float(movie["predicted_rating"])

    predicted_rating = max(
        0.0,
        min(5.0, predicted_rating)
    )

    community_rating = movie.get(
        "community_rating",
        np.nan
    )

    rating_count = movie.get(
        "rating_count",
        np.nan
    )

    genres = str(movie.get("genres", "") or "")

    if not genres or genres == "(no genres listed)":

        genre_list = ["Genres unavailable"]

    else:

        genre_list = [
            genre.strip()
            for genre in genres.split("|")
            if genre.strip()
        ]

    genre_html = "".join(
        f'<span class="genre-tag">{html.escape(genre)}</span>'
        for genre in genre_list
    )

    predicted_percentage = (
        predicted_rating / 5.0
    ) * 100

    # Community rating section.
    if pd.notna(community_rating):

        community_html = f"""
        <div class="rating-label">
            COMMUNITY RATING
        </div>

        <div class="community-value">
            &#9733; {float(community_rating):.2f}
            <span style="font-size:13px;color:#a3adbd;">
                / 5
            </span>
        </div>

        <div class="rating-count">
            Based on {int(rating_count):,} user ratings
        </div>
        """

    else:

        community_html = """
        <div class="rating-label">
            COMMUNITY RATING
        </div>

        <div class="rating-count">
            No community ratings available.
        </div>
        """

    # Display card.
    st.html(f"""
    <div class="movie-card">

        <div class="movie-number">
            RECOMMENDATION #{rank}
        </div>

        <div class="movie-title">
            {title}
        </div>

        <div class="rating-label">
            YOUR PREDICTED RATING
        </div>

        <div class="rating-value">
            &#9733; {predicted_rating:.2f}

            <span style="font-size:13px;color:#a3adbd;">
                / 5
            </span>
        </div>

        <div class="rating-track">
            <div class="rating-fill"
                 style="width:{predicted_percentage:.2f}%;">
            </div>
        </div>

        {community_html}

        <div class="genre-heading">
            GENRES
        </div>

        <div class="genre-list">
            {genre_html}
        </div>

    </div>
    """)


# ============================================================
# 12. SESSION STATE
# ============================================================

if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = {}

if "recommendations" not in st.session_state:
    st.session_state.recommendations = None

if "recommendation_signature" not in st.session_state:
    st.session_state.recommendation_signature = None


# ============================================================
# 13. HERO AND DASHBOARD
# ============================================================

st.html("""
<div class="hero">
    <div class="hero-badge">PERSONALIZED MOVIE DISCOVERY</div>

    <h1>Find your next<br>favorite movie.</h1>

    <p>
        Rate movies you enjoy and discover recommendations
        tailored to your preferences.
        Powered by your existing SVD recommendation model.
    </p>
</div>
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Movies available",
        f"{len(movies):,}"
    )

with col2:
    st.metric(
        "Movies rated",
        len(st.session_state.user_ratings)
    )

with col3:
    st.metric(
        "Recommendation engine",
        "SVD + Diversity"
    )


# ============================================================
# 14. SEARCH AND RATE MOVIES
# ============================================================

st.html("""
<div class="section-heading">
    Build your taste profile
</div>

<div class="section-description">
    Search for movies you know and rate them.
    CineMatch will use your ratings to personalize recommendations.
</div>
""")

search_query = st.text_input(
    "Search for a movie",
    placeholder="Try Toy Story, Interstellar, or The Dark Knight...",
    key="movie_search"
)

search_results = search_movies(
    search_query,
    movies
)

if search_query.strip():

    if search_results.empty:

        st.warning(
            "No movies found. Try another title or a shorter search."
        )

    else:

        search_results = search_results.head(100)

        selected_movie_id = st.selectbox(
            "Choose a movie",
            options=search_results["movieId"].tolist(),
            format_func=lambda movie_id: search_results.loc[
                search_results["movieId"] == movie_id,
                "title"
            ].iloc[0],
            key="selected_movie"
        )

        selected_movie = movies.loc[
            movies["movieId"] == selected_movie_id
        ].iloc[0]

        genre_text = selected_movie["genres"].replace(
            "|",
            " · "
        )

        st.caption(
            f"Genres: {genre_text or 'Genres unavailable'}"
        )

        if pd.notna(selected_movie.get("community_rating")):

            st.caption(
                f"Community rating: "
                f"⭐ {selected_movie['community_rating']:.2f}/5 "
                f"({int(selected_movie['rating_count']):,} ratings)"
            )

        rating_col, button_col = st.columns(
            [3, 1],
            vertical_alignment="bottom"
        )

        with rating_col:

            selected_rating = st.slider(
                "Your rating",
                min_value=0.5,
                max_value=5.0,
                value=4.0,
                step=0.5,
                key="selected_rating"
            )

        with button_col:

            if st.button(
                "Add rating",
                type="primary",
                use_container_width=True
            ):

                st.session_state.user_ratings[
                    int(selected_movie_id)
                ] = float(selected_rating)

                st.session_state.recommendations = None

                st.session_state.recommendation_signature = None

                st.success(
                    f"Rating saved for {selected_movie['title']}."
                )

                st.rerun()


# ============================================================
# 15. USER RATING MANAGEMENT
# ============================================================

st.html("""
<div class="section-heading">
    Your rated movies
</div>

<div class="section-description">
    Review or remove ratings before generating recommendations.
</div>
""")

user_ratings = st.session_state.user_ratings

if not user_ratings:

    st.html("""
    <div class="empty-state">
        <div style="font-size:32px;margin-bottom:10px;">
            &#127916;
        </div>

        <strong>No ratings yet</strong><br>

        Search for a movie above and give it a rating
        to start building your taste profile.
    </div>
    """)

else:

    for movie_id, rating in list(user_ratings.items()):

        movie_match = movies.loc[
            movies["movieId"] == int(movie_id)
        ]

        if movie_match.empty:
            continue

        movie = movie_match.iloc[0]

        rating_col, remove_col = st.columns(
            [5, 1],
            vertical_alignment="center"
        )

        with rating_col:

            st.markdown(
                f"**{movie['title']}**  \n"
                f"Your rating: ⭐ {rating:.1f}/5"
            )

        with remove_col:

            if st.button(
                "Remove",
                key=f"remove_{movie_id}",
                use_container_width=True
            ):

                del st.session_state.user_ratings[movie_id]

                st.session_state.recommendations = None

                st.session_state.recommendation_signature = None

                st.rerun()


# ============================================================
# 16. GENERATE RECOMMENDATIONS
# ============================================================

st.html("""
<div class="section-heading">
    Discover your next favorite
</div>

<div class="section-description">
    Explore personalized recommendations with community ratings
    and a wider mix of genres.
</div>
""")

if not st.session_state.user_ratings:

    st.info(
        "Rate at least one movie before generating recommendations."
    )

else:

    st.caption(
        "For better personalization, rate several movies "
        "that reflect your actual preferences."
    )

    if st.button(
        "Generate recommendations",
        type="primary",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "Finding movies that match your preferences..."
            ):

                recommendations = recommend_for_new_user(
                    model=model,
                    user_ratings=st.session_state.user_ratings,
                    movies=movies,
                    top_n=10,
                    max_per_genre=2
                )

                st.session_state.recommendations = recommendations

                st.session_state.recommendation_signature = tuple(
                    sorted(
                        st.session_state.user_ratings.items()
                    )
                )

        except Exception as error:

            st.session_state.recommendations = None

            st.error(
                f"Unable to generate recommendations: {error}"
            )


# ============================================================
# 17. DISPLAY RECOMMENDATIONS
# ============================================================

recommendations = st.session_state.recommendations

current_signature = tuple(
    sorted(
        st.session_state.user_ratings.items()
    )
)

saved_signature = st.session_state.recommendation_signature

if recommendations is not None:

    if saved_signature != current_signature:

        st.info(
            "Your ratings have changed. Generate recommendations "
            "again to reflect your updated preferences."
        )

    elif recommendations.empty:

        st.warning(
            "No recommendations are available for these ratings."
        )

    else:

        st.markdown(
            f"### Your top {len(recommendations)} recommendations"
        )

        st.caption(
            "Predicted ratings estimate your preferences. "
            "Community ratings are averages from your MovieLens dataset."
        )

        for start in range(0, len(recommendations), 2):

            columns = st.columns(2)

            for offset, column in enumerate(columns):

                index = start + offset

                if index >= len(recommendations):
                    break

                movie = recommendations.iloc[index]

                with column:

                    display_movie_card(
                        movie,
                        rank=index + 1
                    )


# ============================================================
# 18. FOOTER
# ============================================================

st.html("""
<div class="footer">
    <strong style="color:#c4b5fd;">CineMatch</strong><br>

    Personalized movie discovery powered by collaborative filtering.<br>

    Built with Python, Streamlit, and Surprise SVD.
</div>
""")