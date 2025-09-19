
# ------------------------------
# 1. Import Libraries
# ------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Set styles
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# ------------------------------
# 2. Load and Initial Data Exploration
# ------------------------------
# Load the dataset
# Note: Ensure the file is in the same directory as this script, or provide the correct path.
file_path = "/Users/payalsharma/Downloads/archive-3/Walmart_sales_analysis.csv"
df = pd.read_csv(file_path)

print("\n--- Dataset Info ---")
print(df.info())
print("\n--- First 5 Rows ---")
print(df.head())
print("\n--- Missing Values ---")
print(df.isnull().sum())

# ------------------------------
# 3. Data Cleaning and Preparation
# ------------------------------
# Clean column names by stripping whitespace
df.columns = df.columns.str.strip()

# Convert Date to datetime and sort
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by="Date")

# Clean 'Weekly_Sales' column by removing commas and converting to a numeric type
df['Weekly_Sales'] = df['Weekly_Sales'].str.replace(',', '', regex=False)
df['Weekly_Sales'] = pd.to_numeric(df['Weekly_Sales'])

# Check duplicates and drop
print(f"\nDuplicates in data: {df.duplicated().sum()}")
df = df.drop_duplicates()

# ------------------------------
# 4. Descriptive Statistics & EDA
# ------------------------------
print("\n--- Summary Statistics ---")
print(df.describe())

# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df[['Weekly_Sales','Temperature','Fuel_Price','CPI','Unemployment']].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Sales trends
df.groupby("Date")["Weekly_Sales"].sum().plot()
plt.title("Total Weekly Sales Over Time")
plt.ylabel("Weekly Sales")
plt.xlabel("Date")
plt.show()

# Store-wise average sales
avg_sales_store = df.groupby("Store_Number")["Weekly_Sales"].mean().sort_values(ascending=False)
avg_sales_store.plot(kind="bar", color="teal")
plt.title("Average Weekly Sales per Store")
plt.ylabel("Avg Sales")
plt.show()

# Holiday vs Non-Holiday Sales
sns.boxplot(x="Holiday_Flag", y="Weekly_Sales", data=df, palette="Set2")
plt.title("Holiday vs Non-Holiday Sales")
plt.xticks([0,1], ["Non-Holiday", "Holiday"])
plt.show()

# ------------------------------
# 5. Feature Engineering for Modeling
# ------------------------------
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Week'] = df['Date'].dt.isocalendar().week.astype(int)

# ------------------------------
# 6. Predictive Modeling
# ------------------------------
print("\n--- Model Training ---")
# Define features (X) and target (y) for regression
features = ['Store_Number', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Year', 'Month', 'Week']
target = 'Weekly_Sales'

X = df[features]
y = df[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("✅ Model trained successfully.")

# Make predictions on the test set
y_pred = model.predict(X_test)

# ------------------------------
# 7. Model Assessment
# ------------------------------
print("\n--- Model Evaluation ---")
# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Calculate R-squared (R²)
r2 = r2_score(y_test, y_pred)
print(f"R-squared (R²): {r2:.2f}")

# Optional: Visualize predictions vs. actuals
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, label='Actual vs. Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Predictions')
plt.xlabel("Actual Weekly Sales")
plt.ylabel("Predicted Weekly Sales")
plt.title("Actual vs. Predicted Sales")
plt.legend()
plt.show()

print("\n✅ Walmart Sales Analysis and Forecasting Project Completed Successfully!")