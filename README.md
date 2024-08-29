# Cricket Match Outcome Predictor

This project is a Cricket Match Outcome Predictor that uses machine learning techniques to predict the likelihood of a team winning a match based on historical data. The project focuses on analyzing and processing cricket match data, training a predictive model, and visualizing the match progression.

## Table of Contents
- [Dataset](#dataset)
- [Installation](#installation)
- [Workflow](#workflow)
- [Model](#model)
- [Visualization](#visualization)
- [Usage](#usage)
- [License](#license)

## Dataset
The project uses two main datasets:
1. `matches.csv`: Contains information about various cricket matches.
2. `deliveries.csv`: Contains ball-by-ball data of cricket matches.

## Installation

### Prerequisites
- Python 3.x
- Required Python libraries (install via `requirements.txt` if provided):
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `pickle`

### Instructions
1. Clone this repository.
2. Ensure that the necessary libraries are installed.
3. Place the datasets `matches.csv` and `deliveries.csv` in the project directory.

## Workflow

### Data Preparation
- **Load Data**: The datasets are loaded using pandas.
- **Data Cleaning and Transformation**: 
  - The total score for each match and inning is calculated.
  - Teams are standardized (e.g., Delhi Daredevils renamed to Delhi Capitals).
  - Only matches between specific teams and without the Duckworth-Lewis method applied are considered.

### Feature Engineering
- **Current Score**: The cumulative score at each delivery.
- **Runs Left and Balls Left**: The runs and balls left for the batting team.
- **Wickets**: The number of wickets remaining.
- **Current Run Rate (CRR) and Required Run Rate (RRR)**: These metrics are calculated to provide context for the model.

### Model Training
- A logistic regression model is trained to predict the match outcome based on the following features:
  - Batting Team
  - Bowling Team
  - City
  - Runs Left
  - Balls Left
  - Wickets Remaining
  - Total Runs
  - CRR
  - RRR

### Model Evaluation
- The model's accuracy is evaluated on a test set using the `accuracy_score` metric.

### Model Serialization
- The trained model pipeline is saved using `pickle` for later use.

## Visualization
The project includes a function to visualize match progression:
- **Match Progression**: Plots the win/loss probabilities and the number of runs scored after each over.

## Usage
- **Prediction**: Use the trained model to predict the outcome of a match based on the input features.
- **Visualization**: The `match_progression` function provides a visual representation of a match's progression.

### Example
```python
import pickle
import pandas as pd

# Load the model
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Example prediction
X_test = pd.DataFrame([...])  # Replace with actual data
y_pred = pipe.predict(X_test)
