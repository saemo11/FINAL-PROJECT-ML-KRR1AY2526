import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(BASE_DIR, "dataset", "weather.csv")
model_dir = os.path.join(BASE_DIR, "model")

# Load dataset
data = pd.read_csv(csv_path)

X = data[["Humidity", "WindSpeed", "Rainfall"]]
y = data["MaxTemp"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train models
lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=100, random_state=42)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Evaluate
lr_mae = mean_absolute_error(y_test, lr.predict(X_test))
rf_mae = mean_absolute_error(y_test, rf.predict(X_test))

print("Linear Regression MAE:", lr_mae)
print("Random Forest MAE:", rf_mae)

# Save models
with open(os.path.join(model_dir, "lr_model.pkl"), "wb") as f:
    pickle.dump(lr, f)

with open(os.path.join(model_dir, "rf_model.pkl"), "wb") as f:
    pickle.dump(rf, f)

print("✅ Models saved successfully")
