#  Hotel Operations & Revenue Management: EDA Analytics Case Study

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Hotel%20Booking%20Demand-red.svg)](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)


> An industry-grade data science and revenue management case study analyzing **87,396 validated hotel bookings** across City and Resort properties. This project diagnoses booking cancellation volatility, investigates distribution channel economics, evaluates price elasticity using econometrics, and formulates strategic operational policies for hospitality executives.

---

##  Repository Documentation Index

This repository contains full production-grade documentation tailored for data analysts, data scientists, revenue managers, and academic researchers:

| Document | Primary Focus | Key Contents |
|:---|:---|:---|
| [**CASE_STUDY.md**](CASE_STUDY.md) | **Problem Statement & Curriculum** | Business context, core dilemmas, exhaustive **34-variable Data Dictionary**, and **24 curated analytical questions** structured across Basic, Medium, and Advanced tiers. |
|  [**SOLUTION_GUIDE.md**](SOLUTION_GUIDE.md) | **Verified Solutions & Code** | Complete numerical solutions, step-by-step mathematical derivations, production Python snippets, statistical evaluations, and executive business recommendations for all 24 questions. |
|  [**ADDITIONAL_RESOURCES.md**](ADDITIONAL_RESOURCES.md) | **Further Exploration** | Academic journal citations, hospitality formula handbook (RevPAR, GOPPAR, Critical Ratio Overbooking), recommended Python ML stack, and 4 advanced capstone extensions. |
|  [**Kaggle Notebook**](Kaggle%20Worksheet/eda02-hotel-operations-analysis%20.ipynb) | **Interactive Analysis** | Executable notebook containing end-to-end data cleaning, EDA, visualizations, and econometric models. |

---

##  Key Analytical Insights At-a-Glance

```
+----------------------------------------------------------------------------------------------------+
|                                      EXECUTIVE KPI SUMMARY                                         |
+--------------------------+-----------------------+-------------------------------------------------+
| Metric                   | Value                 | Operational Significance                        |
+--------------------------+-----------------------+-------------------------------------------------+
| Total Analyzed Cohort    | 87,396 records        | 31,994 duplicate rows removed (119,390 raw)     |
| Portfolio Cancellation   | 27.49% (24,025 rooms) | City Hotel: 30.04% vs. Resort Hotel: 23.48%     |
| Average Daily Rate (ADR) | €106.33 overall       | City Hotel (€110.99) commands +12.1% over Resort|
| Peak Arrival Month       | August (11,257 rooms) | August also exhibits peak cancellations (32.2%) |
| Average Lead Time        | 79.89 days (Mean)     | Median: 49 days; positive correlation with churn|
| Top Distribution Channel | TA/TO (79.11% volume) | Extreme intermediary reliance (86.05% via agents|
| Highest Net-Margin Ch.   | Direct (€116.58 ADR)  | Zero OTA commission vs. €96.90 net OTA yield    |
| Length of Stay (LOS)     | 3.64 nights overall   | Weekdays: 2.63 nights | Weekends: 1.01 nights   |
| Repeat Guest Stay Length | 1.93 nights           | Repeat guests stay ~50% shorter than new guests |
| Special Requests Impact  | -32.0% Cancel Odds    | Submitting requests signals strong travel intent|
+--------------------------+-----------------------+-------------------------------------------------+
```

---

##  Visual Analytics Gallery

All visualizations are generated at high resolution ($300\text{ DPI}$) and stored in the [`H-Ops Images/`](H-Ops%20Images/) directory:

| Visual Representation | File Reference | Core Operational Insight |
|:---:|:---:|:---|
| **Property Demand Distribution** | [`Hotel vs Booking.png`](H-Ops%20Images/Hotel%20vs%20Booking.png) | City Hotels account for **61.1%** of reservations (53,428 bookings) vs. **38.9%** for Resort Hotels (33,968 bookings). |
| **Arrival Seasonality** | [`Booking over Arrival Month.png`](H-Ops%20Images/Booking%20over%20Arrival%20Month.png) | European summer holiday peak in July & August; winter trough in November & January. |
| **Market Segment Volume** | [`Booking via Market Segment.png`](H-Ops%20Images/Booking%20via%20Market%20Segment.png) | Online TAs dominate overall transaction volume with **51,618 bookings** (59.1% of market). |
| **Segment ADR Hierarchy** | [`Average Daily Rate (ADR) per Market Segment.png`](H-Ops%20Images/Average%20Daily%20Rate%20(ADR)%20per%20Market%20Segment.png) | Online TA (€118.17) and Direct (€116.58) generate the highest ADRs; Groups (€74.86) and Corporate (€68.15) reflect contract discounts. |
| **Lead Time vs. Cancellation** | [`Scatter Plot _ Lead Time vs Cancellation Rate.png`](H-Ops%20Images/Scatter%20Plot%20_%20Lead%20Time%20vs%20Cancellation%20Rate.png) | Positive correlation ($r = 0.185$); long lead-time reservations ($> 180$ days) experience cancellation rates exceeding $42\%$. |
| **Lead Time by Market Segment** | [`Booking Lead Time Distribution by Market Segment.png`](H-Ops%20Images/Booking%20Lead%20Time%20Distribution%20by%20Market%20Segment.png) | Offline TA/TO and Groups hold the longest advance booking windows (median $> 75-100$ days). |
| **Longitudinal ADR Growth** | [`Trend of Average Daily Rate (ADR) Over the Years.png`](H-Ops%20Images/Trend%20of%20Average%20Daily%20Rate%20(ADR)%20Over%20the%20Years.png) | Consistent compounding room yields: €92.16 (2015) $\rightarrow$ €101.54 (2016) $\rightarrow$ €118.71 (2017). |

---

##  Econometric & Statistical Modeling Highlights

### 1. Multivariate Cancellation Risk (Logistic Regression)
$$\ln\left(\frac{P(\text{Canceled})}{1 - P(\text{Canceled})}\right) = \beta_0 + 0.0050 \cdot (\text{lead\_time}) - 0.4943 \cdot (\text{booking\_changes}) - 0.3852 \cdot (\text{special\_requests})$$
* **Lead Time ($e^{\beta} = 1.0050$):** Every 100 days of lead time increases cancellation odds by **$+64.8\%$**.
* **Booking Modifications ($e^{\beta} = 0.6100$):** Each modification decreases cancellation odds by **$-39.0\%$**.
* **Special Requests ($e^{\beta} = 0.6803$):** Each amenity request decreases cancellation odds by **$-32.0\%$**.

### 2. Party Demographic Pricing Elasticity (OLS Regression)
$$\text{ADR} = €61.18 + €21.18 \cdot (\text{adults}) + €38.66 \cdot (\text{children}) + €6.71 \cdot (\text{babies})$$
* $R^2 = 0.165, \quad F(3, 87392) = 5,752, \quad p < 0.0001$
* **Children add nearly double the rate premium of an adult (+€38.66 vs. +€21.18)** due to mandatory allocation into family suites and premium multi-bed room layouts.

---

---

##  Quickstart & Installation

To run the analysis locally or replicate the econometric models:

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/hotel-operations-analysis.git
cd hotel-operations-analysis
```

### 2. Set Up a Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate on Windows:
.\venv\Scripts\activate

# Activate on macOS/Linux:
source venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn jupyter
```

### 4. Launch Jupyter Notebook
```bash
jupyter notebook "Kaggle Worksheet/eda02-hotel-operations-analysis .ipynb"
```

---

##  Authors & Acknowledgments
* **Dataset Creators:** Nuno Antonio, Ana de Almeida, and Luis Nunes (*Data in Brief*, 2019).
* **Analysis & Implementation:** Hospitality Data Science & Revenue Analytics Pair Programming Project.

---
