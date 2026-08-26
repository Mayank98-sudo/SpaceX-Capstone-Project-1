# SpaceX Falcon 9 First-Stage Landing Prediction

**Author:** Mayank Raj  
**Date:** 26 August 2026

## Project Overview

This project analyzes SpaceX Falcon 9 launch records to understand the factors associated with successful first-stage landings and to build a classification model that predicts landing success.

The project follows a complete data-science workflow:

1. Collect launch data from the SpaceX REST API.
2. Clean and prepare the data.
3. Explore launch patterns using visualization and SQL.
4. Build geographic analysis with Folium.
5. Create an interactive dashboard with Plotly Dash.
6. Train, tune, and evaluate classification models.

## Business Question

Can historical launch information be used to predict whether the Falcon 9 first stage will land successfully?

This question is useful because booster recovery and reuse are important factors in SpaceX launch economics. Predicting landing success can support launch-risk analysis and operational planning.

## Data Sources

The analysis uses the course-backed SpaceX datasets referenced by the supplied notebooks:

- `dataset_part_1.csv`
- `dataset_part_2.csv`
- `dataset_part_3.csv`
- `Spacex.csv`
- `spacex_launch_geo.csv`

The main modelling dataset contains **90 Falcon 9 missions** from **2010 through 2020**.

## Included Source Files

- `jupyter-labs-spacex-data-collection-api_1787723530695.ipynb` — SpaceX API data collection
- `labs-jupyter-spacex-Data_wrangling_1787723530692.ipynb` — cleaning, transformation, and landing-class creation
- `edadataviz_1787723530694.ipynb` — exploratory data visualization
- `jupyter-labs-eda-sql-coursera_sqllite_1787723530694.ipynb` — SQL analysis
- `lab_jupyter_launch_site_location_1787723530695.ipynb` — Folium map and proximity analysis
- `spacex_dash_app_1787723530693.py` — Plotly Dash dashboard
- `SpaceX_Machine_Learning_Prediction_Part_5_1787723530693.ipynb` — classification modelling

## Methodology

### Data Collection

The API workflow uses the SpaceX past-launches endpoint and helper requests for:

- Rocket and booster details
- Launch-site names and coordinates
- Payload mass and orbit
- Core reuse and landing outcomes

Nested JSON responses are normalized into tabular data for analysis.

No separate web-scraping notebook was provided or used. The project uses structured API data and course-backed extracts instead of inventing web-scraping results.

### Data Wrangling

The data preparation process includes:

- Filtering for Falcon 9 launches
- Inspecting missing values
- Preserving relevant numerical and categorical features
- Creating the target variable:
  - `Class = 1`: successful landing
  - `Class = 0`: unsuccessful landing
- Extracting launch years
- Creating one-hot encoded modelling features

### Exploratory Data Analysis

The visual analysis covers:

- Flight number versus launch site
- Payload mass versus launch site
- Success rate by orbit type
- Flight number versus orbit type
- Payload mass versus orbit type
- Yearly landing-success trends

### SQL Analysis

The SQL section examines:

- Unique launch sites
- Launch sites beginning with `CCA`
- NASA CRS payload totals
- Average payload for F9 v1.1
- First successful ground landing
- Successful drone-ship landings within a payload range
- Mission-outcome counts
- Maximum payload records
- Failed drone-ship landings in 2015
- Ranked landing outcomes between 2010-06-04 and 2017-03-20

### Interactive Analytics

The Folium analysis includes:

- Launch-site markers
- Clustered launch records
- Color-coded landing outcomes
- Distance lines and labels for nearby coastline and railway points

The Plotly Dash dashboard includes:

- Launch-site dropdown selection
- Success pie charts
- Payload-range slider
- Payload-versus-outcome scatter plot

### Predictive Modelling

Four classification models were tuned with 10-fold cross-validation:

- Logistic Regression
- Support Vector Machine
- Decision Tree
- K-Nearest Neighbors

The data was standardized and split into training and test sets using an 80/20 split with `random_state=2`.

## Key Results

- Total missions analyzed: **90**
- Successful landings: **60**
- Overall landing success rate: **66.7%**
- Highest launch-site success rate: **KSC LC 39A**
- KSC LC 39A success rate: **77.3%** — 17 successful landings out of 22
- Largest high-volume orbit group: **GTO**, with 27 launches
- GTO success rate: **51.9%**
- First successful ground landing in the SQL dataset: **2015-12-22**
- Total NASA CRS payload in the SQL dataset: **48,213 kg**
- Average F9 v1.1 payload: **2,928.4 kg**

## Model Results

The supplied prediction notebook reports the following held-out test accuracy:

| Model | Test Accuracy |
|---|---:|
| Logistic Regression | 83.3% |
| SVM | 83.3% |
| Decision Tree | 83.3% |
| KNN | 83.3% |

The notebook identifies **Logistic Regression** as the selected model because it appears first among the tied test results.

Its confusion matrix contains:

- 3 correctly identified non-landings
- 3 non-landings predicted as landings
- 12 correctly identified successful landings
- 0 successful landings missed in the supplied test split

Because the test set contains only 18 records, these results should be treated as a benchmark rather than a production launch-decision system.

## Final Deliverable

The completed presentation is:

`Mayank_Raj_SpaceX_Capstone_Completed.pptx`

The presentation contains 47 slides covering the full project workflow, analysis results, interactive-analysis visuals, predictive modelling, conclusions, and appendix.

## Conclusion

The analysis indicates that landing success improved substantially as SpaceX gained operational experience. Launch site, orbit, payload, hardware, and time-related features all provide useful context. The classification models achieve a promising benchmark accuracy, but additional historical data and stronger validation would be required before using the model for operational decisions.