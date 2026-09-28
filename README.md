# Hotel-Operation-Analysis

A data analytics and machine learning project exploring guest booking behavior, cancellations, and revenue patterns for **Elite Hotels International**.

---

## 📌 Project Overview

Elite Hotels International operates City and Resort hotels across multiple locations. Managing high cancellation rates, fluctuating booking windows, and seasonal demand swings makes staffing and revenue forecasting difficult.

This project analyzes 32 operational features across 119k+ bookings to clean data, identify booking trends, predict cancellations, and provide actionable business strategies.

---

## ⚙️ Project Workflow & Methodology

### 1. Data Cleaning & Preparation
* **Handling Missing Values:** Missing entries in `children`, `agent`, and `company` were filled with `0` (missing agent/company IDs represent direct customer bookings on the hotel portal).
* **Removing Duplicates:** Dropped duplicate records across all 32 columns to prevent artificially inflated booking counts and revenue.
* **Date Parsing:** Combined split date fields (`arrival_date_year`, `arrival_date_month`, `arrival_date_day_of_month`) into a single pandas `datetime` column for time-series analysis.

### 2. Exploratory Data Analysis (EDA)
* **Lead Time Tracking:** Analyzed the distribution of advance booking times and their impact on cancellations.
* **Distribution Benchmarking:** Visualized booking proportions across hotel types (City vs. Resort), peak arrival months, and market segments (e.g., Online Travel Agents vs. Direct).
* **ADR Trends:** Tracked the Average Daily Rate (ADR) across arrival years and calculated month-by-month revenue.

### 3. Advanced Statistical & Predictive Modeling
* **Binary Logistic Regression:** Modeled cancellation risk (`is_canceled`) based on key drivers such as `lead_time`, `booking_changes`, and `total_of_special_requests`.
* **Multivariate OLS Regression:** Measured how guest group sizes (`adults`, `children`, `babies`) directly affect room pricing (ADR).
* **Segment Distribution Analysis:** Profiled lead-time dispersion and variance across different market segments using boxplots.

---

## 📊 Key Findings

* **Long Lead Times Drive Cancellations:** Bookings made months in advance show a strong positive correlation with cancellations compared to short-window bookings.
* **Special Requests Lower Cancellation Risk:** Guests with one or more special requests (e.g., twin beds, high floors) are significantly more likely to honor their reservation.
* **Clear Seasonal Demand Spikes:** Arrivals and ADR peak during specific holiday and summer months, highlighting clear seasonal patterns.
* **Demographics Impact Daily Rates:** Regression analysis shows that each additional adult and child creates a measurable, statistically significant increase in the Average Daily Rate.
* **Channel Variation:** Online Travel Agents (OTAs) drive the highest volume, while corporate channels have much shorter lead times and different cancellation rates.

---

## 💡 Practical Recommendations

* **Stricter Deposits on Early Bookings:** Require non-refundable deposits or tiered cancellation fees for bookings made far in advance to cut down long-window cancellations.
* **Data-Driven Dynamic Overbooking:** Calibrate room overbooking thresholds using model cancellation probabilities rather than a flat percentage.
* **Fast-Track Special Requests:** Respond quickly to guests who make special requests to secure commitment and increase guest satisfaction.
* **Targeted Family Room Bundles:** Package room upgrades and amenities specifically for family bookings to capture higher ADR based on group composition.
* **Seasonal Resource Allocation:** Use monthly arrival forecasts to plan front-desk staffing, housekeeping shifts, and parking capacity ahead of peak months.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Data Manipulation:** pandas, numpy
* **Visualization:** matplotlib, seaborn
* **Machine Learning & Statistics:** scikit-learn (`LogisticRegression`), statsmodels (`OLS`)

---
