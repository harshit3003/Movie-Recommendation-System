# CineMatch --- Personalized Movie Recommendation System

CineMatch is a beginner-friendly movie recommendation web application
built with **Python and Streamlit**. It uses a trained **Singular Value
Decomposition (SVD)** collaborative-filtering model to estimate how much
a user may like movies and display personalized recommendations.

The project uses the **MovieLens small dataset** for movie information
and community ratings. It also includes experiments comparing SVD with
user-based and item-based K-nearest-neighbour (KNN) recommendation
models.

> **Project status:** The core app supports movie search, session-based
> user ratings, personalized recommendations, community-rating
> information, and recommendation cards. Movie poster integration and
> real-time trending data are not implemented.

## Table of Contents

-   [Project at a Glance](#project-at-a-glance)
-   [Features](#features)
-   [How It Works](#how-it-works)
-   [Algorithms](#algorithms)
-   [Dataset](#dataset)
-   [Evaluation Results](#evaluation-results)
-   [Project Structure](#project-structure)
-   [Installation on Windows](#installation-on-windows)
-   [Run the Application](#run-the-application)
-   [How to Use CineMatch](#how-to-use-cinematch)
-   [Important Notes and Limitations](#important-notes-and-limitations)
-   [Troubleshooting](#troubleshooting)
-   [Technologies Used](#technologies-used)
-   [Future Improvements](#future-improvements)

## Project at a Glance

  Item                               Details
  ---------------------------------- ---------------------------------------------------
  Project name                       CineMatch
  Type                               Personalized movie recommendation web application
  Interface                          Streamlit
  Language                           Python
  Main model                         SVD collaborative filtering
  Comparison models                  User-based KNN and item-based KNN
  Dataset                            MovieLens small dataset
  Movie records                      9,742
  Rating records                     100,836
  Users in original rating dataset   610
  Saved model                        `svd_model.pkl`

## Features

### Movie search

Search the movie catalogue by title and choose movies to rate.

### Add and remove ratings

Enter ratings on the dataset's 0.5--5.0 scale. Ratings entered in the
app are maintained in the current Streamlit session and may not persist
after the session ends.

### Personalized recommendations

The app estimates the new user's preferences from their entered ratings,
predicts ratings for candidate movies, and recommends movies the user
has not already rated.

### Genre diversity

The recommendation-selection logic attempts to avoid filling the list
with movies from the same genre. This is a selection rule, not a
separate trained model.

### Predicted and community ratings

Recommendation cards show the model's predicted rating and, where
available, the community average and number of ratings from the local
MovieLens dataset.

### Cinematic interface

A dark theme, custom styling, rating indicators, and recommendation
cards make the app easier to browse.

## How It Works

1.  Load movie metadata and ratings from CSV files.
2.  Explore and validate the data.
3.  Split ratings into training and test sets for evaluation.
4.  Train and compare collaborative-filtering models.
5.  Select SVD as the primary model based on the recorded experiments.
6.  Save the trained model to `svd_model.pkl`.
7.  Load the model and movie data in the Streamlit app.
8.  Collect ratings from a new user during the session.
9.  Estimate a profile for the new user using the model's learned movie
    factors.
10. Predict ratings for candidate movies.
11. Exclude movies already rated and apply genre-diversity selection.
12. Display recommendations and community-rating information.

### Architecture overview

``` text
MovieLens CSV files
        |
        +---------------------+
        |                     |
  Movie catalogue        User ratings
        |                     |
        |               Model experiments
        |                     |
        |                Train / evaluate
        |                     |
        |                Trained SVD model
        |                     |
        +-----------+---------+
                    |
               Streamlit app
                    |
            User searches titles
                    |
             User enters ratings
                    |
        Estimate new-user preferences
                    |
          Predict candidate ratings
                    |
       Exclude rated movies and select
          a diverse recommendation list
                    |
          Recommendation cards
```

## Algorithms

### Singular Value Decomposition (SVD)

SVD is the primary model. Collaborative filtering learns latent
representations from historical user--movie ratings and uses them to
estimate ratings for combinations that were not observed.

The app's new-user flow estimates a profile for a visitor who is not an
existing user in the model's training data. It uses the saved model's
learned movie factors and ratings entered in the app. This is a
simplified personalization method; a few ratings may not represent a
user's preferences reliably.

### User-based KNN

User-based K-nearest neighbours looks for users with similar rating
patterns and uses those similarities to estimate preferences.

### Item-based KNN

Item-based K-nearest neighbours uses relationships between movies based
on rating patterns to estimate a user's likely preferences.

SVD was chosen as the primary model because the recorded experiments
showed lower rating-prediction errors and slightly higher ranking
metrics than the tested KNN baselines. This does not prove it will be
best for every dataset or user.

## Dataset

The project expects the MovieLens small dataset in the `dataset`
directory.

  File                     Purpose
  ------------------------ ----------------------------------------------
  `movies.csv`             Movie IDs, titles, and genres (also supports `movies (1).csv`)
  `ratings.csv`            User IDs, movie IDs, ratings, and timestamps (also supports `ratings (1).csv`)
  `links.csv`              External movie database identifiers (optional)
  `tags.csv`               User-applied tags (optional)
  `README.txt`             Dataset-specific documentation (if supplied)

Recorded data checks found:

-   9,742 movie records and 100,836 rating records.
-   610 users in the original rating data.
-   No missing values in the ratings table.
-   No duplicate user--movie rating pairs.
-   Ratings range from 0.5 to 5.0.
-   No missing or duplicate movie IDs in the movie catalogue.
-   All movie IDs in the ratings table existed in the movie catalogue.

These checks apply to the dataset version used in the project. Repeat
them if the CSV files are replaced.

### Train/test split

The recorded experiment used an 80/20 split:

-   Training ratings: **80,668**
-   Test ratings: **20,168**

The saved model was trained on a training split and is not necessarily
trained on every rating in the dataset.

## Evaluation Results

The project evaluated rating-prediction errors and ranking quality.

### Rating prediction

Lower RMSE and MAE indicate smaller differences between predicted and
observed ratings.

  Model                              RMSE          MAE
  ---------------- ---------------------- ------------
  User-based KNN                   0.9823       0.7559
  Item-based KNN     approximately 0.9800       0.7615
  Baseline SVD                     0.8807       0.6766
  Tuned SVD                    **0.8718**   **0.6700**

-   **RMSE (Root Mean Squared Error):** gives more weight to larger
    prediction errors.
-   **MAE (Mean Absolute Error):** measures the average absolute
    prediction error.

### Ranking metrics at cutoff 10

  Model              Precision@10    Recall@10      NDCG@10
  ---------------- -------------- ------------ ------------
  SVD                  **0.5639**   **0.6668**   **0.7862**
  User-based KNN           0.5608       0.6637       0.7774
  Item-based KNN           0.4910       0.6293       0.6842

-   **Precision@10:** how many of the top 10 recommendations are
    relevant under the evaluation's relevance rule.
-   **Recall@10:** how much of the relevant set appears in the top 10.
-   **NDCG@10:** rewards relevant items that appear higher in the ranked
    list.

**Evaluation note:** These are preliminary results recorded from
held-out test ratings. They should not be treated as definitive
full-catalogue or real-world online performance, and the differences
have not been established as statistically significant. Results depend
on the data split and relevance definition.

## Project Structure

Expected structure:

``` text
Movie_recommendation_system/
│
├── app.py                     # Streamlit web application
├── main.ipynb                 # Data exploration and experiments
├── svd_model.pkl              # Saved trained SVD model
├── requirements.txt           # Python dependencies
│
├── dataset/
│   ├── movies.csv             # Movie metadata
│   ├── ratings.csv            # User ratings
│   └── README.txt             # Dataset documentation (if supplied)
│
└── venv/                      # Local virtual environment; do not commit
```

The `venv` directory is created locally and normally should not be
uploaded to Git. Check repository file-size limits before committing the
saved model.

## Installation on Windows

### 1. Open the project directory

Open the folder in VS Code, then open **Terminal → New Terminal**.

``` powershell
cd path\to\Movie_recommendation_system
# Example:
# cd C:\Users\gupta\Desktop\Movie_recommendation_system
```

Replace `path\to\Movie_recommendation_system` with your actual project location.

### 2. Create a virtual environment

``` powershell
python -m venv venv
```

A virtual environment keeps the project's packages separate from other
Python projects.

### 3. Activate it

In PowerShell:

``` powershell
.\venv\Scripts\Activate.ps1
```

> **Note for PowerShell:** If script execution is disabled on your system (`PSSecurityException`), run:
> ``` powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
> ```
> before running the activation script.

Alternatively, in Command Prompt:

``` bat
venv\Scripts\activate.bat
```

### 4. Install dependencies

``` powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `requirements.txt` is missing or incomplete, the main app
dependencies include:

``` powershell
python -m pip install streamlit pandas numpy joblib scikit-surprise
```

Use a Python version compatible with the installed dependencies. Some
versions of the Surprise package may not install cleanly on very new
Python releases.

## Run the Application

Make sure the virtual environment is active and your terminal is in the
project directory:

``` powershell
python -m streamlit run app.py
```

Streamlit normally opens the app in your browser. If it does not, open
the local address printed in the terminal.

To stop the app, focus the terminal and press `Ctrl + C`.

You do not need to retrain the model every time you start the app, as
long as `svd_model.pkl` exists and is compatible with the installed
libraries.

## How to Use CineMatch

1.  Start the Streamlit app.
2.  Search for a movie by title.
3.  Select it and choose a rating from 0.5 to 5.0.
4.  Add the rating to your list.
5.  Repeat for other movies you have watched.
6.  Review your saved ratings and remove any entered by mistake.
7.  Select **Recommend Movies**.
8.  Review the movie titles, genres, predicted ratings, and
    community-rating information.

The ratings entered in the app are session-based; they are not
guaranteed to persist after the session ends.

## Important Notes and Limitations

### Predicted rating versus community rating

-   **Predicted rating:** the model's estimate of how the current user
    might rate a movie.
-   **Community rating:** the average of ratings for that movie in the
    local MovieLens file.
-   **Rating count:** the number of ratings contributing to that
    community average.

These values answer different questions. A high community average does
not necessarily mean a movie will suit a particular user.

### New-user personalization

A new visitor is not already represented as a trained user in the saved
SVD model. The app estimates a user profile from the visitor's ratings
using learned movie factors. This avoids retraining the entire model for
each visitor, but it is a simplified approach and can be unreliable with
very few ratings.

### Model coverage

Because the saved model was trained on a training split, some movies in
the full catalogue may not have learned item factors. The app should
handle unsupported movies safely.

### Other limitations

-   Users need to rate some movies before the recommendations can
    reflect their preferences.
-   User ratings are session-based and may be lost when the session
    ends.
-   Community ratings are local MovieLens data, not live ratings from
    streaming platforms.
-   Real-time trending information is not implemented.
-   Movie poster integration is not implemented.
-   Offline test metrics do not guarantee that each user will find
    recommendations useful.
-   MovieLens ratings are not a universal measure of movie popularity.

## Troubleshooting

### `ModuleNotFoundError`

Activate the virtual environment and install the dependencies:

``` powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Streamlit is not recognized

Run it through Python:

``` powershell
python -m streamlit run app.py
```

### The app cannot find `svd_model.pkl`

Make sure `svd_model.pkl` is in the same directory as `app.py`:

``` text
Movie_recommendation_system/svd_model.pkl
```

### The app cannot find the CSV files

Check that these files exist in the `dataset` directory (either standard names or downloaded copies with `(1)` are automatically detected):

``` text
Movie_recommendation_system/dataset/movies.csv
Movie_recommendation_system/dataset/ratings.csv
```

### Surprise or model compatibility error

The saved model depends on the library versions used to create it. Use
compatible dependencies and, if necessary, a compatible Python version.
Do not delete the model unless you have the code and environment needed
to regenerate it.

### Changes are not appearing

Save `app.py`, check the terminal for errors, and restart Streamlit if
necessary:

``` powershell
python -m streamlit run app.py
```

## Technologies Used

-   **Python** --- main programming language
-   **Pandas** --- loading and manipulating tabular data
-   **NumPy** --- numerical operations
-   **Surprise** --- collaborative-filtering algorithms, including SVD
    and KNN
-   **Joblib** --- saving and loading the trained model
-   **Streamlit** --- interactive web interface
-   **Jupyter Notebook** --- exploration and experiments
-   **HTML/CSS** --- custom styling within the interface

## Future Improvements

These are possible extensions, not features currently implemented:

1.  Save user profiles and ratings in a database.
2.  Add posters through a supported movie metadata API.
3.  Add genre, release-year, and minimum-rating filters.
4.  Add a popularity-based top-rated page.
5.  Add a clearly defined trending section with a regularly updated data
    source.
6.  Improve evaluation with reproducible splits and documented relevance
    thresholds.
7.  Compare collaborative filtering with content-based recommendations
    using genres or tags.
8.  Add automated tests for data validation and recommendation edge
    cases.
9.  Deploy the app and document the hosting process.

## Summary

CineMatch is an end-to-end learning project combining collaborative
filtering, model evaluation, a saved SVD model, and an interactive
Streamlit interface. It demonstrates how a recommendation model can be
connected to a usable application while distinguishing personalized
predicted ratings from community averages.

For reproducibility, keep the application code, dataset files,
dependency list, and saved model together, and document any changes to
the model or evaluation process.
