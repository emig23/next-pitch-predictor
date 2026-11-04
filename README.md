![CI](https://github.com/emig23/next-pitch-predictor/actions/workflows/ci.yml/badge.svg)

# Next Pitch Predictor

Predicts the next pitch type a starting pitcher will throw, using only what is known before the pitch (count, score, baserunners, batter handedness and what he has already thrown in the game), with an XGBoost model trained on MLB Statcast data and tested once on the 2026 season. The model is served by a FastAPI service backed by Postgres, with a public demo page.

## Demo

**[Try the live demo](https://emig23.github.io/next-pitch-predictor/)**

The page has three tabs:
- **Game replay** replays real 2026 games one pitch at a time. For each pitch it shows the chance the model gave every pitch type before the pitch, then reveals what the pitcher actually threw.
- **Results** shows how the model did over the whole test season.
- **Try it live** lets you set up your own situation (pitcher, count, runners, score, earlier pitches in the at-bat) and see the forecast from the deployed API. The API runs on a free host that sleeps when idle, so the first request can take up to a minute.

### Game replay
Before the pitch, the model shows the chance of each pitch type:

<table>
<tr>
<td><b>Before the pitch</b><br><img src="docs/screenshots/replay-before.png" width="420"></td>
</tr>
</table>

After revealing, the pitch he threw is outlined, with the chance the model gave it and a running score for the game:

<table>
<tr>
<td><b>After the reveal</b><br><img src="docs/screenshots/replay-after.png" width="420"></td>
</tr>
</table>

<table>
<tr>
<td><b></b><br><img src="docs/screenshots/replay-game-summary.png" width="420"></td>
</tr>
</table>

### Results
<table>
<tr>
<td><img src="docs/screenshots/results-overview.png" width="420"></td>
<td><img src="docs/screenshots/results-by-pitcher.png" width="420"></td>
</tr>
</table>

## Architecture

```mermaid
flowchart LR
    S[Statcast via pybaseball] -->|daily ingestion, GitHub Actions| DB[(Postgres on Neon)]
    S -->|2023-2026 history| T[Training and evaluation]
    T -->|saved model file| M[models/global_xgb]
    M -->|baked into the image| API[FastAPI service on Render]
    DB -->|/pitchers, /games| API
    API -->|/predict| UI[Demo page on GitHub Pages]
```

- **Ingestion** (`ingest/`): a daily job pulls new pitches for the pitchers already in the database and loads them. It is safe to run twice.
- **Database** (`db/`): Postgres with keys and constraints, loaded with idempotent inserts.
- **Model** (`src/`): feature building, baselines, XGBoost and evaluation. It was trained once on 2023-2025 and saved to `models/`. New rows in the database do not retrain it.
- **API** (`api/`): loads the saved model once at startup and serves forecasts.
- **Demo** (`docs/`): a static page. The Try it live tab calls the API.

## API

Interactive docs are at `/docs` on any running copy of the service.

| Endpoint | What it does |
|---|---|
| `GET /health` | Whether the model is loaded and the database is reachable |
| `GET /pitchers` | Pitchers in the database, and whether the model can forecast each one |
| `GET /games/{game_pk}` | Every stored pitch of one game, in order |
| `POST /predict` | Probability of each of the 10 pitch types for the next pitch |

Example request:

```bash
curl -X POST https://next-pitch-predictor.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pitcher_id": 592332, "pitcher_throws": "R",
    "batter_id": 660271, "batter_stands": "L",
    "at_bat_number": 2, "pitch_number": 2,
    "inning": 1, "inning_half": "Bot",
    "balls": 0, "strikes": 1, "outs": 0,
    "pitching_team_score": 0, "batting_team_score": 0,
    "previous_pitches": [
      {"at_bat_number": 2, "pitch_number": 1, "batter_id": 660271,
       "pitch_type": "FF", "result": "called_strike"}
    ]
  }'
```

The response lists all 10 pitch types, most likely first (the numbers here are illustrative):

```json
{
  "pitcher_id": 592332,
  "top_pick": "FF",
  "probabilities": [
    {"pitch_type": "FF", "name": "Four-seam fastball", "probability": 0.41},
    {"pitch_type": "SL", "name": "Slider", "probability": 0.18}
  ],
  "model_trained_on_seasons": [2023, 2024, 2025]
}
```

Impossible input (4 balls, 3 outs, an unknown pitch type, a pitcher the model was not trained on) is rejected with a `422` and a message saying what is wrong.

## Overview
- **Data**: MLB Statcast pitch-by-pitch data via `pybaseball` (2023-2026, regular season)
- **Task**: 10-class pitch type prediction for 13 starting pitchers
- **Models compared**: three count-based baselines, a per-pitcher XGBoost and a single global XGBoost
- **Evaluation**: trained on 2023-2025, tested once on 2026 (28,778 pitches, 313 games)

## Model Comparison
Evaluated on the 2026 test season. Lower log loss is better.

| Model                              | Log Loss | Top-1 | Top-2 |
|------------------------------------|:--------:|:-----:|:-----:|
| A: pitcher's overall pitch mix     |  1.658   | 36.7% | 61.8% |
| B: A plus ball-strike count        |  1.652   | 39.4% | 63.8% |
| C: B plus batter handedness        |  1.626   | 41.0% | 66.6% |
| Per-pitcher XGBoost                |  1.435   | 41.9% | 67.7% |
| Global XGBoost                     |  1.366   | 42.2% | 68.5% |

**Chosen as the final model: Global XGBoost.** It beats baseline B by 0.286 log loss (95% interval 0.233 to 0.344) and the stronger baseline C by 0.260 (95% interval 0.194 to 0.333), using a paired bootstrap over games (2,000 draws). It also beats baseline B for all 13 of 13 pitchers and one model trained on every pitcher beat separate models trained per pitcher.

The gain in top-1 accuracy is small (42.2% vs 41.0% for C). Most of the improvement is in the probabilities themselves: the model is much better at saying how likely each pitch is, even when its single top pick is only right about 42% of the time. Top-3 accuracy is not in the table, but in the live demo it looked solid: the pitch thrown was usually among the model's top three choices.

## How It Works
1. **Data**: regular-season Statcast pitches, stored as Parquet and in Postgres.
2. **Pitchers**: 15 starters were chosen using 2023-2025 data only (at least 1,500 pitches and at least 80% of appearances as a starter). 13 have 2026 pitches and make up the test set.
3. **Target**: 10 pitch types: four-seam, sinker, cutter, slider, sweeper, curveball, knuckle curve, changeup, splitter, slurve.
4. **Features**: only information available before the pitch, checked by leakage tests.
5. **Split**: during development, fit on 2023-24 and evaluate on 2025. For the final run, fit on 2023-25 and evaluate on 2026.
6. **Metrics**: log loss, top-k accuracy and calibration.

## Design Decisions
- **One feature function for training and serving.** The API builds its features by calling the same `build_features` the model was trained with. A test checks that the API's output equals the offline model's output exactly, so the two cannot quietly drift apart.
- **Idempotent loading.** Rows are keyed by (game, at-bat, pitch number) and inserted with `ON CONFLICT DO NOTHING`, so re-running ingestion never creates duplicates and a failed run can simply be run again. The trade-off: if Statcast later corrects a row that is already stored, the correction is not picked up.
- **Postgres with constraints.** Foreign keys and `CHECK` constraints reject impossible rows (for example 4 balls) at the database, not only in code.
- **The API does not need the database to predict.** `/predict` builds features from the request alone, so forecasts keep working if the database is down. Only `/pitchers` and `/games` use it.
- **The model is fixed, on purpose.** The 2026 season is the held-out test and was scored once. Retraining on new data would end that clean test, so ingestion adds data but never retrains.
- **Database tests use throwaway schemas.** Each test gets its own Postgres schema, so tests cannot interfere with each other or with real data.

## Project Structure
```
next-pitch-predictor/
├── api/                  # FastAPI service: endpoints, schemas, prediction
├── db/                   # Postgres schema, connection, loading, example queries
├── ingest/               # daily ingestion job
├── src/
│   ├── prep/             # data pulling, pitcher selection, cleaning
│   ├── dataset/          # features, encoding, train/test splits
│   ├── modeling/         # baselines and XGBoost models
│   ├── evaluation/       # metrics, calibration, final evaluation
│   └── experiments/      # run_baselines, run_models, run_final, save_model
├── models/               # the saved model the API serves
├── tests/                # unit, leakage, database and API tests
├── results/              # saved settings and final results
├── app/                  # demo data export
├── docs/                 # demo page (index.html, css/, js/), served by GitHub Pages
├── .github/workflows/    # CI and the daily ingestion job
├── Dockerfile            # the API image
└── docker-compose.yml    # the API and Postgres together
```

## Setup
```bash
git clone https://github.com/emig23/next-pitch-predictor
cd next-pitch-predictor

python -m venv .venv
.venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -q
```
Requires Python 3.11 or newer. The tests that need Postgres only run when it is available: start it first with `docker compose up -d db` (see below). CI runs the full suite against a throwaway Postgres on every push, and also builds the Docker image and calls it.

## Run the API and database
Create a `.env` file in the project root (it is git-ignored) and choose your own password:
```
POSTGRES_USER=pitch
POSTGRES_PASSWORD=choose-a-password
POSTGRES_DB=pitch
DATABASE_URL=postgresql+psycopg://pitch:choose-a-password@localhost:5432/pitch
```
Then:
```bash
docker compose up --build
```
The API is at http://localhost:8000 and its docs at http://localhost:8000/docs. The database starts empty, and `/predict` works without it.

## Usage

The data is not in the repo. **Rebuild it** (the pulls take a while, `pybaseball` caches them):
```bash
python -m src.prep.pull_selection_data
python -m src.prep.select_pitchers
python -m src.prep.pull_final_data
python -m src.prep.clean_data
```

**Run the baselines and models:**
```bash
python -m src.experiments.run_baselines
python -m src.experiments.run_models
```

**Final evaluation on the 2026 season:**
```bash
python -m src.experiments.run_final --confirm-final
```
`results/final_results.json` is already committed, so this refuses to run again unless that file is deleted. That is on purpose, because the test season is meant to be used once.

**Save the model for the API and load the database:**
```bash
python -m src.experiments.save_model
python -m db.load_initial
```

**Rebuild the demo data:**
```bash
python -m app.export_demo
```

## Technical Notes
- **Test season used once**: the final run needs `--confirm-final` and refuses to run if results already exist. No setting or model was changed after seeing 2026 results.
- **Leakage tests**: tests check that features use only information from before the pitch.
- **Honest baselines**: the model is compared against count-based and handedness-based baselines, not just a most-common-pitch guess, so the gain reflects more than knowing the count.
- **Rare pitch folded in**: slow curveballs (CS, about 0.03% of pitches) are folded into the curveball class.

## Limitations
- Only 13 pitchers and one test season, so the results should not be read as covering all of MLB.
- It predicts pitch type only, not location or speed.
- Two pitchers, Yusei Kikuchi and Jake Irvin, distort the average gain over baseline C. Baseline C scored 4.09 and 2.53 on them, worse than guessing all ten pitch types equally (2.30). Their 2026 pitch mix likely differs from their history, which a lookup table cannot adjust to, while the model also sees what the pitcher has thrown so far in the game. This explanation is likely but has not been tested. Without those two the gain is smaller but still positive for every pitcher. The demo's Results tab has a checkbox to leave them out.
- The top-1 gain over the strongest baseline is small.
- The model is fixed after training. Daily ingestion keeps the database current, but the model is not retrained from it.
- The Try it live tab only knows the pitches of the at-bat you enter, not the whole game, so its forecasts can differ from the Game replay tab.
- The hosted API is on a free tier that sleeps when idle, so the first request after a quiet period is slow.

## Ideas for extending this project
- Add more pitchers, including relievers and more seasons.
- Add features such as batter-versus-pitcher history or recent fatigue.
- Compare other model types against XGBoost.

## Acknowledgments
- [Claude](https://claude.ai) by Anthropic, used as an AI coding assistant to write and test the code under my direction
- [MLB Statcast](https://baseballsavant.mlb.com/statcast_search) data from Baseball Savant
- [pybaseball](https://github.com/jldbc/pybaseball) for data access