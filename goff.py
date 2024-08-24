import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load Jared Goff's historical data
goff_history_path = 'C:/Users/Chuks/Documents/Jared Goff Predictions/jared goff history.csv'
goff_history = pd.read_csv(goff_history_path)
print("Goff history data loaded successfully.")

# Load the team statistics dataset
team_stats_path = 'C:/Users/Chuks/Documents/Jared Goff Predictions/nfl team statistics.txt'
team_stats = pd.read_csv(team_stats_path)
print("Team stats data loaded successfully.")

# Load the schedule dataset and ensure the "Team" row is set as the column name
schedule_path = 'C:/Users/Chuks/Documents/Jared Goff Predictions/schedule.csv'
schedule = pd.read_csv(schedule_path)
if 'Team' in schedule.columns:
    print("Schedule data loaded successfully with 'Team' as a column name.")
else:
    raise ValueError("The 'Team' column is missing from the schedule dataset.")
print("Schedule data loaded successfully.")

# Display column names to identify the correct columns for the strength of schedule calculation
print("Team stats columns:", team_stats.columns)

# Use these columns for calculating the strength of schedule
relevant_defense_columns = ['Defense Grade', 'Coverage Grade', 'Pass Rush Grade', 'Tackling Grade']

# Ensure these columns are numeric
team_stats[relevant_defense_columns] = team_stats[relevant_defense_columns].apply(pd.to_numeric, errors='coerce')
print("Relevant defense columns converted to numeric.")
print(team_stats)

# Create a mapping from schedule team names to team_stats team names
team_name_mapping = {
    'LA Rams': 'Los Angeles Rams',
    'Tampa Bay': 'Tampa Bay Buccaneers',
    'Arizona': 'Arizona Cardinals',
    'Seattle': 'Seattle Seahawks',
    'Dallas': 'Dallas Cowboys',
    'Minnesota': 'Minnesota Vikings',
    'Tennessee': 'Tennessee Titans',
    'Green Bay': 'Green Bay Packers',
    'Houston': 'Houston Texans',
    'Jacksonville': 'Jacksonville Jaguars',
    'Indianapolis': 'Indianapolis Colts',
    'Chicago': 'Chicago Bears',
    'Buffalo': 'Buffalo Bills',
    'San Francisco': 'San Francisco 49ers'
}

# Rename the teams in the schedule dataset using the mapping
schedule['Team'] = schedule['Team'].map(team_name_mapping)
print("Teams in schedule dataset renamed for alignment.")

# Inspect the unique team names after renaming
print("Unique team names in renamed schedule dataset:", schedule['Team'].unique())

# Merge schedule with team stats to get opponents' pass defense stats
merged_schedule = schedule.merge(team_stats, left_on='Team', right_on='Team', how='left')
print("Merged schedule:\n", merged_schedule.head())

# Check for NaN values in merged_schedule
print("NaN values in merged schedule:\n", merged_schedule.isna().sum())

# Calculate Strength of Schedule as a mean of relevant columns
merged_schedule['StrengthOfSchedule'] = merged_schedule[relevant_defense_columns].mean(axis=1)
print("Strength of Schedule calculated:\n", merged_schedule[['Team', 'StrengthOfSchedule']].head())

# Verify if StrengthOfSchedule has NaN values
print("NaN values in StrengthOfSchedule column:\n", merged_schedule['StrengthOfSchedule'].isna().sum())

# Add the strength of schedule to the schedule data
schedule['StrengthOfSchedule'] = merged_schedule['StrengthOfSchedule']

# Filter relevant columns and ensure they are numeric
relevant_columns = ['GP', 'CMP', 'ATT', 'PCT', 'YDS', 'AVG', 'YDS/G', 'TD', 'INT', 'RATE', 'TD%', 'INT%', 'SCK', 'SCKY']
goff_history_filtered = goff_history[relevant_columns].apply(pd.to_numeric, errors='coerce')
print("Filtered Goff history:\n", goff_history_filtered.head())

# Drop rows with NaN values in both features and target datasets
X = goff_history_filtered.dropna()
y = goff_history[relevant_columns].dropna()
print("X shape after dropna:", X.shape)
print("y shape after dropna:", y.shape)

# Align the indices of X_clean and y_clean
common_indices = X.index.intersection(y.index)
X_clean = X.loc[common_indices]
y_clean = y.loc[common_indices]
print("X_clean shape:", X_clean.shape)
print("y_clean shape:", y_clean.shape)

# Standardize the features
scaler = StandardScaler()
X_clean_scaled = scaler.fit_transform(X_clean)

# Split the data into training and testing sets
if X_clean.shape[0] > 0 and y_clean.shape[0] > 0:
    X_train, X_test, y_train, y_test = train_test_split(X_clean_scaled, y_clean, test_size=0.2, random_state=42)
else:
    raise ValueError("The resulting dataset is empty. Please check the data processing steps.")

# List of models to evaluate
models = {
    'Linear Regression': MultiOutputRegressor(LinearRegression()),
    'Decision Tree Regressor': MultiOutputRegressor(DecisionTreeRegressor(random_state=42)),
    'Random Forest Regressor': MultiOutputRegressor(RandomForestRegressor(random_state=42)),
    'Gradient Boosting Regressor': MultiOutputRegressor(GradientBoostingRegressor(random_state=42))
}

# Evaluate each model using cross-validation and store the results
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    if len(X_test) > 1:
        r2 = r2_score(y_test, y_pred)
    else:
        r2 = float('nan')
    
    results[name] = {'MSE': mse, 'MAE': mae, 'R2': r2}

# Display the results
for name, metrics in results.items():
    print(f"Model: {name}")
    print(f"Mean Squared Error: {metrics['MSE']}")
    print(f"Mean Absolute Error: {metrics['MAE']}")
    print(f"R^2 Score: {metrics['R2']}\n")

# Choose the best model based on R2 Score
best_model_name = max(results, key=lambda k: (results[k]['R2'], -results[k]['MSE']))
best_model = models[best_model_name]
print(f"Best Model: {best_model_name}")

# Use the strength of schedule for prediction
strength_of_schedule = schedule['StrengthOfSchedule'].mean()

# Predict Jared Goff's statistics for this year using strength of schedule
goff_stats = goff_history[relevant_columns].apply(pd.to_numeric, errors='coerce').mean().to_frame().T
goff_stats['StrengthOfSchedule'] = strength_of_schedule
goff_stats_clean = scaler.transform(goff_stats[X_clean.columns])
goff_predicted_stats = best_model.predict(goff_stats_clean)

# Define the predicted stats columns
predicted_stats_columns = [
    'GamesPlayed', 'Completions', 'Attempts', 'CompletionPercentage', 
    'Yards', 'AverageYards', 'YardsPerGame', 'Touchdowns', 
    'Interceptions', 'PasserRating', 'TouchdownPercentage', 
    'InterceptionPercentage', 'Sacks', 'SackYards'
]

# Convert predictions to DataFrame
goff_predicted_stats_df = pd.DataFrame(goff_predicted_stats, columns=predicted_stats_columns)

# Correct any negative values except for Interceptions
goff_predicted_stats_df[['GamesPlayed', 'Completions', 'Attempts', 'CompletionPercentage', 
                         'Yards', 'AverageYards', 'YardsPerGame', 'Touchdowns', 
                         'PasserRating', 'TouchdownPercentage', 
                         'InterceptionPercentage', 'Sacks', 'SackYards']] = goff_predicted_stats_df[[
                         'GamesPlayed', 'Completions', 'Attempts', 'CompletionPercentage', 
                         'Yards', 'AverageYards', 'YardsPerGame', 'Touchdowns', 
                         'PasserRating', 'TouchdownPercentage', 
                         'InterceptionPercentage', 'Sacks', 'SackYards']].clip(lower=0)

# Round the values for GamesPlayed, Touchdowns, Interceptions, and Sacks
for col in ['GamesPlayed', 'Touchdowns', 'Interceptions', 'Sacks']:
    goff_predicted_stats_df[col] = goff_predicted_stats_df[col].round().astype(int)

# Cap the maximum value for GamesPlayed at 17
goff_predicted_stats_df['GamesPlayed'] = goff_predicted_stats_df['GamesPlayed'].clip(upper=17)

# Save the predicted stats to a CSV file
goff_predicted_stats_df.to_csv('jared_goff_predicted_stats.csv', index=False)

# Read the file and display its contents for confirmation
with open('jared_goff_predicted_stats.csv', 'r') as file:
    print(file.read())
