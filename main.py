import pandas
import numpy
import matplotlib.pyplot
import seaborn
import sklearn

data = pandas.read_csv("tsl_dataset.csv")

data["Date"] = pandas.to_datetime(data["Date"])

print("Total matches count:", len(data))

print("\n--- MATCH RESULTS ---")
print(data["result"].value_counts())

# To calculate pre-match team performances
data = data.sort_values("Date").reset_index(drop=True)

# Track the number of matches previously played by each team
team_matches = {}

home_form = []
away_form = []

for index, row in data.iterrows():
 
    home_team = row["home"]
    away_team = row["visitor"]

    # Previous matches count for home team
    home_previous_matches = team_matches.get(home_team, 0)

    # Previous matches count for away team
    away_previous_matches = team_matches.get(away_team, 0)

    home_form.append(home_previous_matches)
    away_form.append(away_previous_matches)

    # Increment match count for both teams after this match is played
    team_matches[home_team] = home_previous_matches + 1
    team_matches[away_team] = away_previous_matches + 1

data["home_previous_matches"] = home_form
data["away_previous_matches"] = away_form


team_results = {}
team_goals = {}

home_wins = []
away_wins = []

home_points = []
away_points = []

home_goals_for = []
home_goals_against = []

away_goals_for = []
away_goals_against = []

for index, row in data.iterrows():

    home_team = row["home"]
    away_team = row["visitor"]
    result = row["result"]

    # Get previous match results for teams
    home_recent_results = team_results.get(home_team, [])[-5:]
    away_recent_results = team_results.get(away_team, [])[-5:]

    home_recent_goals = team_goals.get(home_team, [])[-5:]
    away_recent_goals = team_goals.get(away_team, [])[-5:]

    # Number of wins in the last 5 matches
    home_win_count = 0
    away_win_count = 0

    home_point_count = 0
    away_point_count = 0

    for previous_result in home_recent_results:
        if previous_result == "W":
            home_win_count += 1

    for previous_result in away_recent_results:
        if previous_result == "W":
            away_win_count += 1

    for previous_result in home_recent_results:

        if previous_result == "W":
            home_point_count += 3
        
        elif previous_result == "D":
            home_point_count += 1

    for previous_result in away_recent_results:

        if previous_result == "W":
            away_point_count += 3

        elif previous_result == "D":
            away_point_count += 1

    home_wins.append(home_win_count)
    away_wins.append(away_win_count)

    home_points.append(home_point_count)
    away_points.append(away_point_count)

    home_goals_for_count = 0
    home_goals_against_count = 0

    away_goals_for_count = 0
    away_goals_against_count = 0

    for goals_for, goals_against in home_recent_goals:
        home_goals_for_count += goals_for
        home_goals_against_count += goals_against

    for goals_for, goals_against in away_recent_goals:
        away_goals_for_count += goals_for
        away_goals_against_count += goals_against

    home_goals_for.append(home_goals_for_count)
    home_goals_against.append(home_goals_against_count)

    away_goals_for.append(away_goals_for_count)
    away_goals_against.append(away_goals_against_count)

    # Determine current match result from team perspective 
    if result == "H":
        home_team_result = "W"
        away_team_result = "L"
    elif result == "A":
        home_team_result = "L"
        away_team_result = "W"
    else:
        home_team_result = "D"
        away_team_result = "D"

    # Add current match to history
    team_results.setdefault(home_team, []).append(home_team_result)
    team_results.setdefault(away_team, []).append(away_team_result)

    team_goals.setdefault(home_team, []).append(
        (row["home_goal"], row["visitor_goal"])
    )

    team_goals.setdefault(away_team, []).append(
        (row["visitor_goal"], row["home_goal"])
    )

data["home_last5_wins"] = home_wins
data["away_last5_wins"] = away_wins
data["home_last5_points"] = home_points
data["away_last5_points"] = away_points

data["home_last5_goals_for"] = home_goals_for
data["home_last5_goals_against"] = home_goals_against
data["away_last5_goals_for"] = away_goals_for
data["away_last5_goals_against"] = away_goals_against

print("\n--- NEW FEATURES ---")
print(data[[
    "home_previous_matches",
    "away_previous_matches",
    "home_last5_wins",
    "away_last5_wins",
    "home_last5_points",
    "away_last5_points",
    "home_last5_goals_for",
    "home_last5_goals_against",
    "away_last5_goals_for",
    "away_last5_goals_against"
]].head(10))

from sklearn.model_selection import train_test_split

X = data.drop([
    "Date",
    "result",
    "home_goal",
    "visitor_goal",
    "total_goal",
    "goal_difference",
    "FT",
    "HT",
    "home_goal_half",
    "visitor_goal_half",
    "total_goal_half",
    "goal_difference_half",
    "result_half",
    "home_red_card",
    "visitor_red_card"
], axis=1)

y = data["result"]

print("\n--- X COLUMNS ---")
print(X.columns)

print("\n--- NEW FEATURES ---")
print(X[[
    "home_previous_matches",
    "away_previous_matches",
    "home_last5_wins",
    "away_last5_wins",
    "home_last5_points",
    "away_last5_points"
]].head(10))

print("\n--- DATA TYPES ---")
print(X.dtypes)

print("\n--- TEAM COUNTS ---")
print("Number of unique home teams:", data["home"].nunique())
print("Number of unique away teams:", data["visitor"].nunique())

# Convert team names to numerical values
X = pandas.get_dummies(
    X,
    columns=["home", "visitor"],
    dtype=int
)

print("\n--- AFTER ENCODING ---")
print("Total feature count:", len(X.columns))

print("\n--- FIRST 5 ROWS ---")
print(X.head())

# Split train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n--- TRAIN / TEST ---")
print("Training data:", len(X_train))
print("Test data:", len(X_test))

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

# Train the model
model.fit(
    X_train,
    y_train
)

# Calculate feature importance for the model
feature_importance = pandas.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    "importance",
    ascending=False
)

print("\n--- FEATURE IMPORTANCE ---")
print(feature_importance.head(20))

y_pred = model.predict(
    X_test
)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n--- MODEL PERFORMANCE ---")

print("Accuracy:", accuracy)

from sklearn.metrics import classification_report

print("\n--- CLASSIFICATION REPORT ---")

print(classification_report(y_test, y_pred))


from sklearn.metrics import confusion_matrix

matrix = confusion_matrix(
    y_test,
    y_pred
)

print("\n--- CONFUSION MATRIX ---")

print(matrix)

print("\nRow = Actual result, Column = Prediction")

print("A = Away win")

print("D = Draw")

print("\n--- ORDER OF CLASSES ---")
print(sorted(y_test.unique()))

new_match_date = pandas.Timestamp("2019-05-01")

home_team = "Galatasaray"
away_team = "Fenerbahçe"

home_previous_games = data[
    (
        (data["home"] == home_team) |
        (data["visitor"] == home_team)
    ) &
    (data["Date"] < new_match_date)
]

away_previous_games = data[
    (
        (data["home"] == away_team) |
        (data["visitor"] == away_team)
    ) &
    (data["Date"] < new_match_date)
]

print("\n--- GALATASARAY PAST MATCHES COUNT ---")
print(len(home_previous_games))

print("\n--- FENERBAHÇE PAST MATCHES COUNT ---")
print(len(away_previous_games))

home_last_five = home_previous_games.tail(5)
away_last_five = away_previous_games.tail(5)

home_last5_wins = 0
away_last5_wins = 0

home_last5_points = 0
away_last5_points = 0

home_last5_goals_for = 0
home_last5_goals_against = 0

away_last5_goals_for = 0
away_last5_goals_against = 0

for index, row in home_last_five.iterrows():

    if row["home"] == home_team:

        home_last5_goals_for += row["home_goal"]
        home_last5_goals_against += row["visitor_goal"]

        if row["result"] == "H":
            home_last5_wins += 1
            home_last5_points += 3

        elif row["result"] == "D":
            home_last5_points += 1

    elif row["visitor"] == home_team:

        home_last5_goals_for += row["visitor_goal"]
        home_last5_goals_against += row["home_goal"]

        if row["result"] == "A":
            home_last5_wins += 1
            home_last5_points += 3

        elif row["result"] == "D":
            home_last5_points += 1


for index, row in away_last_five.iterrows():

    if row["home"] == away_team:

        away_last5_goals_for += row["home_goal"]
        away_last5_goals_against += row["visitor_goal"]

        if row["result"] == "H":
            away_last5_wins += 1
            away_last5_points += 3

        elif row["result"] == "D":
            away_last5_points += 1

    elif row["visitor"] == away_team:

        away_last5_goals_for += row["visitor_goal"]
        away_last5_goals_against += row["home_goal"]

        if row["result"] == "A":
            away_last5_wins += 1
            away_last5_points += 3

        elif row["result"] == "D":
            away_last5_points += 1

print("\n--- LAST 5 MATCH FORM ---")
print("Galatasaray wins:", home_last5_wins)
print("Galatasaray points:", home_last5_points)

print("Fenerbahçe wins:", away_last5_wins)
print("Fenerbahçe points:", away_last5_points)

print("\n--- LAST 5 MATCH GOALS ---")
print("Galatasaray goals scored:", home_last5_goals_for)
print("Galatasaray goals conceded:", home_last5_goals_against)

print("Fenerbahçe goals scored:", away_last5_goals_for)
print("Fenerbahçe goals conceded:", away_last5_goals_against)

print("\n--- GALATASARAY LAST 5 MATCHES ---")
print(home_last_five[["Date", "home", "visitor", "result"]])

print("\n--- FENERBAHÇE LAST 5 MATCHES ---")
print(away_last_five[["Date", "home", "visitor", "result"]])

# Prediction for a new match
new_match = pandas.DataFrame({
    "Year": [2019],
    "Month": [5],
    "Season": [2019],
    "Week": [10],
    "home": ["Galatasaray"],
    "visitor": ["Fenerbahçe"],
    "fans_allowed": [1],
    "neutral": [0],
    "home_last5_wins": [home_last5_wins],
    "away_last5_wins": [away_last5_wins],
    "home_last5_points": [home_last5_points],
    "away_last5_points": [away_last5_points],
    "home_last5_goals_for": [home_last5_goals_for],
    "home_last5_goals_against": [home_last5_goals_against],
    "away_last5_goals_for": [away_last5_goals_for],
    "away_last5_goals_against": [away_last5_goals_against]
})

# get_dummies() -> Convert text/category values into 0 and 1 columns.
new_match = pandas.get_dummies(
    new_match,
    columns=["home", "visitor"],
    dtype=int
)

new_match = new_match.reindex(
    columns=X.columns,
    fill_value=0
)

prediction = model.predict(new_match)

print("\n--- NEW MATCH PREDICTION ---")
print("Galatasaray - Fenerbahçe:", prediction[0])

if prediction[0] == "H":
    print("Prediction: Home win")
elif prediction[0] == "D":
    print("Prediction: Draw")
else:
    print("Prediction: Away win")

