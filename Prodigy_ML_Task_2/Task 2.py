# 1) Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 2) Load dataset
train = pd.read_csv("train.csv")

# 3) Select only required features
data = train[["GrLivArea", "BedroomAbvGr", "FullBath", "SalePrice"]]

# 4) Check and fill missing values
data = data.dropna()  # simple approach; removes rows with missing values

# 5) Split features & target
X = data[["GrLivArea", "BedroomAbvGr", "FullBath"]]
y = data["SalePrice"]

# 6) Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7) Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 8) Predictions
y_pred = model.predict(X_test)

# 9) Evaluate
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# 10) Print results
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("R^2 Score:", r2)
print("Mean Squared Error:", mse)

# 11) Show some predictions vs actual
result_df = pd.DataFrame({"Actual": y_test.values, "Predicted": np.round(y_pred, 2)})
print(result_df.head())
