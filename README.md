**Walmart Sales Analysis**

1. 📌 Project Overview

This project analyzes Walmart’s historical sales dataset to uncover patterns, trends, and insights. The analysis focuses on identifying factors affecting weekly sales, seasonality, store performance, and prediction of future sales.

Through **Exploratory Data Analysis (EDA)** and **Machine Learning models**, we aim to provide actionable recommendations that can help improve Walmart’s sales forecasting and decision-making.



2. 📂 Dataset Description

The dataset contains weekly sales information from different Walmart stores and includes:

* **Store** – Store ID
* **Date** – Week of sales
* **Weekly\_Sales** – Sales for the given store on the given week
* **Holiday\_Flag** – Whether the week included a holiday (1 = Yes, 0 = No)
* **Temperature** – Average temperature in the region
* **Fuel\_Price** – Fuel price in the region
* **CPI** – Consumer Price Index
* **Unemployment** – Unemployment rate



3. 🎯 Problem Statement

Walmart needs to:

1. Identify **sales trends** across time, stores, and holidays.
2. Understand how **economic factors** (fuel prices, CPI, unemployment) impact sales.
3. Predict **future weekly sales** with higher accuracy to optimize inventory and promotions.



4.✅ Solution Approach

1. **Data Cleaning & Preprocessing** – Handled missing values, formatted dates, and normalized features.
2. **Exploratory Data Analysis (EDA)** – Identified top-performing stores, seasonal effects, and holiday impacts.
3. **Sales Forecasting Models** – Built and compared regression models for predicting weekly sales.
4. **Evaluation** – Compared predicted vs. actual sales to validate model accuracy.
5. **Insights & Recommendations** – Suggested improvements for marketing, inventory, and resource allocation.



5. 📊 Key Insights

* Holiday weeks show **significant spikes** in sales, requiring better inventory planning.
* Sales have a **seasonal trend**, with peaks during festivals and year-end.
* Economic indicators (fuel price, unemployment, CPI) influence sales but are **less significant** compared to seasonality and holidays.
* Predictive models achieved **high accuracy**, making them reliable for short-term forecasting.



6. 📌 Files in Repository

* `Walmart_Sales_EDA.py` → Python script with full data analysis and visualization.
* `Walmart_Sales_Report.docx` → Detailed project report with findings and recommendations.
* `README.md` → Project overview.


7.  🚀 Future Work

* Implement **time-series forecasting models** (ARIMA, LSTM) for long-term prediction.
* Develop an **interactive dashboard** for real-time sales monitoring.
* Add **external factors** (weather, promotions, regional events) for deeper insights.
