# 🎬 CineMatch — Personalized Movie Recommendation System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![CI](https://github.com/harshit3003/Movie-Recommendation-System/actions/workflows/python-app.yml/badge.svg)](https://github.com/harshit3003/Movie-Recommendation-System/actions/workflows/python-app.yml)
[![Database](https://img.shields.io/badge/Database-SQLite-003B57.svg)](https://www.sqlite.org/)
[![API](https://img.shields.io/badge/Metadata%20%26%20Posters-TMDB-01b4e4.svg)](https://www.themoviedb.org/)

**CineMatch** is an end-to-end movie recommendation web application built with **Python, Streamlit, and SQLite**. It leverages a hybrid recommendation architecture combining a pre-trained **Singular Value Decomposition (SVD)** collaborative filtering model with **TF-IDF tag similarity** and **genre affinity signals** to deliver accurate, personalized, and explainable movie recommendations.

The app features a cinematic dark-themed UI, live weekly trending movies via **The Movie Database (TMDB) API**, concurrent poster retrieval, interactive multi-criteria filters, and full persistent user accounts with secure salted password hashing.

---

## 📑 Table of Contents

- [Project at a Glance](#-project-at-a-glance)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Recommendation Engine & Algorithms](#-recommendation-engine--algorithms)
  - [1. SVD Collaborative Filtering](#1-svd-collaborative-filtering)
  - [2. Content-Based TF-IDF & Tag Matching](#2-content-based-tf-idf--tag-matching)
  - [3. Genre Affinity & Diversity Balancing](#3-genre-affinity--diversity-balancing)
  - [4. Explainable AI ("Why This Movie?")](#4-explainable-ai-why-this-movie)
- [Dataset](#-dataset)
- [Evaluation Results](#-evaluation-results)
- [Project Structure](#-project-structure)
- [Installation & Quickstart](#-installation--quickstart)
  - [1. Prerequisites](#1-prerequisites)
  - [2. Clone Repository](#2-clone-repository)
  - [3. Set Up Virtual Environment](#3-set-up-virtual-environment)
  - [4. Install Dependencies](#4-install-dependencies)
  - [5. Configure TMDB API Key (Optional but Recommended)](#5-configure-tmdb-api-key-optional-but-recommended)
  - [6. Launch Application](#6-launch-application)
- [Database & Authentication](#-database--authentication)
- [Technologies Used](#-technologies-used)
- [Troubleshooting](#-troubleshooting)
- [License & Attributions](#-license--attributions)

---

## ⚡ Project at a Glance

| Item | Details |
| :--- | :--- |
| **Application Name** | CineMatch |
| **Framework** | Streamlit |
| **Language** | Python 3.10+ |
| **Primary Algorithm** | SVD Collaborative Filtering (`svd_model.pkl`) |
| **Content Signals** | TF-IDF Tag Vectorization + Jaccard Genre Similarity |
| **User Data Store** | SQLite (`cinematch.db`) with PBKDF2-HMAC-SHA256 password hashing |
| **External Media** | TMDB API (high-resolution movie posters + weekly trending feeds) |
| **Dataset** | MovieLens 100k (Small Dataset) |
| **Total Movies** | 9,742 catalog records |
| **Total Ratings** | 100,836 historical community ratings |

---

## ✨ Key Features

### 🔐 User Authentication & Persistent Profiles
- **Secure Registration & Login**: User accounts stored locally in SQLite with PBKDF2-HMAC-SHA256 password hashing and cryptographic salts.
- **Persistent Ratings**: Add, inspect, adjust, and delete movie ratings at any time; ratings remain saved across sessions.
- **Session Management**: Dedicated profile dashboard displaying rated film counts, account details, and one-click sign-out.

### 🧠 Hybrid Personalized Recommendation Engine
- **Instant Profiling without Retraining**: Learns latent user factors and bias dynamically via gradient steps over the pre-trained SVD item space.
- **Content & Tag Blending**: Analyzes liked/disliked films using TF-IDF on user tags (`dataset/tags.csv`) and weighted genre similarity.
- **Genre Diversity Enforcement**: Avoids repetitive genres through adaptive diversity quotas.
- **Explainable Recommendations**: Transparent "Why This Movie?" cards explain whether a recommendation is driven by shared genres, community tags, or broader taste patterns.

### 🖼️ TMDB Posters & Live Trending Movies
- **Concurrent Poster Fetching**: Multi-threaded poster retrieval using `concurrent.futures.ThreadPoolExecutor` with in-memory caching.
- **Live Weekly Trending Feed**: Displays global trending movies fetched from TMDB's `/trending/movie/week` endpoint, refreshed on a 6-hour cache.
- **Graceful Fallbacks**: Works smoothly offline or without an API key using built-in stylized placeholder cards.

### 🎛️ Interactive Refinement Filters
- **Genre Filter**: Multiselect filter to target specific genres (Action, Sci-Fi, Comedy, etc.).
- **Release Year Range**: Dual-thumb slider to narrow recommendations to specific decades or eras.
- **Community Rating Threshold**: Filter out lower-rated or niche movies based on MovieLens historical averages.

### 🎨 Cinematic Dark UI
- Modern glassmorphism panels, vibrant glowing accents, Manrope / DM Sans typography, rating progress meters, and mobile-responsive grid layouts.

---

## 🏗️ System Architecture

```text
 ┌────────────────────────────────────────────────────────┐
 │                   MovieLens Dataset                    │
 │    movies.csv  •  ratings.csv  •  tags.csv • links.csv │
 └───────────────────────────┬────────────────────────────┘
                             │
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │           Offline Training (main.ipynb)                │
 │    • Surprise SVD Collaborative Filtering              │
 │    • Train/test split (80/20) & cross-validation       │
 │    • Output artifact: svd_model.pkl                    │
 └───────────────────────────┬────────────────────────────┘
                             │
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │               CineMatch App (app.py)                   │
 │                                                        │
 │   ┌───────────────────────┐   ┌────────────────────┐   │
 │   │ SQLite Database       │   │  TMDB REST API     │   │
 │   │ (cinematch.db)        │   │  • Trending feed   │   │
 │   │ • Users & Auth        │   │  • Poster images   │   │
 │   │ • Persistent ratings  │   └─────────┬──────────┘   │
 │   └───────────┬───────────┘             │              │
 │               │                         │              │
 │               ▼                         ▼              │
 │   ┌────────────────────────────────────────────────┐   │
 │   │          Hybrid Recommendation Engine          │   │
 │   │  1. Latent SVD Factor Estimation               │   │
 │   │  2. TF-IDF Tag Cosine Similarity               │   │
 │   │  3. Weighted Genre Affinity (Liked vs Disliked)│   │
 │   │  4. Soft Diversity Filtering                   │   │
 │   └───────────────────────┬────────────────────────┘   │
 │                           │                            │
 │                           ▼                            │
 │   ┌────────────────────────────────────────────────┐   │
 │   │             Streamlit Web Interface            │   │
 │   │   Dashboard • Search & Rating • Filterable Grid│   │
 │   └────────────────────────────────────────────────┘   │
 └────────────────────────────────────────────────────────┘
```

---

## 🔬 Recommendation Engine & Algorithms

### 1. SVD Collaborative Filtering
The foundation of the recommendation system is **Singular Value Decomposition (SVD)** from the `scikit-surprise` library.
- SVD decomposes the sparse user-item interaction matrix into latent representations:
  $$\hat{r}_{u,i} = \mu + b_u + b_i + q_i^T p_u$$
  where $\mu$ is the global rating average, $b_u$ and $b_i$ are user and movie baseline biases, and $p_u, q_i \in \mathbb{R}^k$ are $k$-dimensional latent feature vectors.
- **Fast Cold-Start User Estimation**: Rather than retraining the full model when a user submits new ratings, CineMatch performs stochastic gradient descent updates over user bias $b_u$ and factors $p_u$ while keeping item matrices frozen:
  $$e_{u,i} = r_{u,i} - \hat{r}_{u,i}$$
  $$b_u \leftarrow b_u + \gamma (e_{u,i} - \lambda b_u)$$
  $$p_u \leftarrow p_u + \gamma (e_{u,i} q_i - \lambda p_u)$$

### 2. Content-Based TF-IDF & Tag Matching
- Community tags from `dataset/tags.csv` are aggregated per movie and vectorized using `sklearn.feature_extraction.text.TfidfVectorizer` (sublinear term frequencies, bi-grams).
- Candidate movies are evaluated for tag cosine similarity against movies the user rated highly.

### 3. Genre Affinity & Diversity Balancing
- User preferences are separated into positive examples (ratings $> 3.0$) and negative examples (ratings $< 3.0$).
- Candidate movies receive a net genre score based on Jaccard genre similarity:
  $$\text{Score}_{\text{genre}} = \text{Sim}_{\text{liked}} - 0.75 \times \text{Sim}_{\text{disliked}}$$
- A **soft diversity quota** caps the number of recommendations that can share dominant genres, ensuring a balanced selection.

### 4. Explainable AI ("Why This Movie?")
Every generated recommendation includes an intelligible rationale:
- **Shared Genres**: Lists the specific genres shared with movies you rated highly.
- **Related Community Tags**: Indicates thematic overlap based on MovieLens tags.
- **Overall Rating Pattern**: Explains recommendations driven by latent collaborative patterns.

---

## 📊 Evaluation Results

Model performance was validated on an 80/20 train/test split of the MovieLens ratings dataset:

### Rating Prediction Accuracy
| Model | RMSE | MAE |
| :--- | :---: | :---: |
| User-based KNN | 0.9823 | 0.7559 |
| Item-based KNN | 0.9800 | 0.7615 |
| Baseline SVD | 0.8807 | 0.6766 |
| **Tuned SVD (Selected)** | **0.8718** | **0.6700** |

*Lower RMSE and MAE values represent superior predictive precision.*

### Top-10 Ranking Metrics
| Model | Precision@10 | Recall@10 | NDCG@10 |
| :--- | :---: | :---: | :---: |
| **SVD** | **0.5639** | **0.6668** | **0.7862** |
| User-based KNN | 0.5608 | 0.6637 | 0.7774 |
| Item-based KNN | 0.4910 | 0.6293 | 0.6842 |

---

## 📁 Dataset

The project includes the **MovieLens 100k (Small Dataset)** in the `dataset/` directory:

| File | Size | Description |
| :--- | :--- | :--- |
| `dataset/movies.csv` | ~494 KB | Movie titles, genres, and IDs (9,742 movies) |
| `dataset/ratings.csv` | ~2.48 MB | Historical user ratings (100,836 ratings, 610 users) |
| `dataset/links.csv` | ~198 KB | External database mapping (IMDb and TMDB IDs) |
| `dataset/tags.csv` | ~118 KB | User-assigned keyword tags for movies |
| `dataset/README.txt` | ~8 KB | Official MovieLens dataset documentation |

---

## 📂 Project Structure

```text
Movie_recommendation_system/
│
├── .github/
│   └── workflows/
│       └── python-app.yml     # Automated CI (flake8 linting & testing)
│
├── .streamlit/
│   └── secrets.toml           # Local secrets (TMDB_API_KEY; git-ignored)
│
├── dataset/                   # MovieLens small dataset files
│   ├── links.csv
│   ├── movies.csv
│   ├── ratings.csv
│   ├── tags.csv
│   └── README.txt
│
├── .gitignore                 # Ignores venv, secrets.toml, *.db, cache files
├── app.py                     # Main Streamlit web application
├── cinematch.db               # SQLite database (auto-generated; git-ignored)
├── main.ipynb                 # EDA, model training & benchmark notebook
├── README.md                  # Comprehensive project documentation
├── requirements.txt           # Python package dependencies
└── svd_model.pkl              # Serialized trained SVD model (Surprise)
```

---

## 🚀 Installation & Quickstart

### 1. Prerequisites
- **Python 3.10** or higher installed on your system.
- Git installed.

### 2. Clone Repository
```bash
git clone https://github.com/harshit3003/Movie-Recommendation-System.git
cd Movie-Recommendation-System
```

### 3. Set Up Virtual Environment

#### On Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(If script execution is disabled on PowerShell, first run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`)*

#### On macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Configure TMDB API Key (Optional but Recommended)
To enable live movie posters and the weekly trending movies feed, obtain a free API key from [The Movie Database (TMDB)](https://www.themoviedb.org/documentation/api):

1. Create a `.streamlit/secrets.toml` file in the project root:
   ```toml
   TMDB_API_KEY = "your_tmdb_api_key_here"
   ```
2. Or set it as an environment variable in your terminal:
   ```powershell
   $env:TMDB_API_KEY="your_tmdb_api_key_here"
   ```
*(Note: If no API key is provided, CineMatch functions normally with stylish placeholder art.)*

### 6. Launch Application
```bash
python -m streamlit run app.py
```

Streamlit will launch and display the local URL (typically `http://localhost:8501`).

---

## 🔒 Database & Authentication

CineMatch includes a production-grade authentication and rating persistence layer:
- **Database Engine**: Built-in SQLite (`cinematch.db`), created automatically on first run.
- **Security**: Passwords are never stored in plain text. They are hashed using **PBKDF2-HMAC-SHA256** with 310,000 iterations and a unique 16-byte cryptographic salt per user.
- **Timing-Attack Resistance**: Verification uses constant-time comparison via `hmac.compare_digest`.
- **Relational Integrity**: Foreign key constraints are enforced (`PRAGMA foreign_keys = ON`), with ratings cascaded upon user deletion.

---

## 🛠️ Technologies Used

| Category | Tools & Libraries |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **User Interface** | Streamlit, Custom HTML5/CSS3 (Glassmorphism, DM Sans, Manrope) |
| **Machine Learning** | Scikit-surprise (SVD), Scikit-learn (TfidfVectorizer, cosine_similarity) |
| **Data Processing** | Pandas, NumPy, Scipy |
| **Storage & Persistence**| SQLite3, Joblib |
| **Networking & APIs** | Requests (TMDB REST API), ThreadPoolExecutor |
| **CI / CD** | GitHub Actions |

---

## ❓ Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `ModuleNotFoundError` | Virtual environment is not activated or dependencies are missing | Run `.\venv\Scripts\Activate.ps1` and `pip install -r requirements.txt`. |
| `TMDB_API_KEY is not configured` | Missing TMDB key for posters/trending | Add `TMDB_API_KEY = "..."` to `.streamlit/secrets.toml`. Recommendations still work without it. |
| `Trained model not found` | `svd_model.pkl` is missing | Ensure `svd_model.pkl` is present in the project root directory alongside `app.py`. |
| `Movies dataset not found` | Dataset CSVs are missing or renamed | Ensure `movies.csv`, `ratings.csv`, and `tags.csv` reside in the `dataset/` directory. |
| Port already in use | Previous Streamlit instance is running | Run `streamlit run app.py --server.port 8502`. |

---

## 📜 License & Attributions

- **Dataset**: MovieLens datasets are collected by GroupLens Research at the University of Minnesota.
- **Metadata & Posters**: Movie metadata, release dates, and posters provided by [TMDB](https://www.themoviedb.org/). This product uses the TMDB API but is not endorsed or certified by TMDB.
- **Author**: [Harshit Gupta](https://github.com/harshit3003)
