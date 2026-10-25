import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# 1. Load the dataset
data = pd.read_csv(
    "dataset/Cars Datasets 2025.csv",
    encoding="latin1"
)

# 2. Clean column names and price values
data.columns = data.columns.str.strip()
target = "Cars Prices"

if target not in data.columns:
    raise ValueError(
        f"Price column not found. Available columns: {data.columns.tolist()}"
    )

data[target] = (
    data[target].astype(str)
    .str.replace(",", "", regex=False)
    .str.replace("₹", "", regex=False)
    .str.replace("$", "", regex=False)
    .str.replace("€", "", regex=False)
    .str.replace("£", "", regex=False)
    .str.strip()
)
data[target] = pd.to_numeric(data[target], errors="coerce")

# 3. Remove rows without a valid price
data = data.dropna(subset=[target])

# Exclude columns that are entirely empty
X = data.drop(columns=[target]).dropna(axis=1, how="all")
y = data[target]

# Remove rows with missing values in the target
X = X.loc[y.index]

if len(data) < 10:
    raise ValueError("There are too few valid car-price rows to train.")

# 4. Identify text and numeric columns
categorical_cols = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numeric_cols = X.select_dtypes(
    include=["number"]
).columns.tolist()

# 5. Prepare missing values and text columns
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numbers", numeric_pipeline, numeric_cols),
    ("categories", categorical_pipeline, categorical_cols)
])

# 6. Build the machine-learning model
model = Pipeline([
    ("preprocessing", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

# 7. Split the data and train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

# 8. Evaluate the model
predictions = model.predict(X_test)

print("CAR PRICE PREDICTION RESULTS")
print("----------------------------")
print("Total valid rows:", len(data))
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Mean Absolute Error:", round(mean_absolute_error(y_test, predictions), 2))
print("R2 Score:", round(r2_score(y_test, predictions), 3))

# 9. Save the trained model
joblib.dump(model, "car_price_model.pkl")
print("\nModel saved as car_price_model.pkl")