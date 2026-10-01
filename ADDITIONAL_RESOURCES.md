# 📚 Hotel Operations & Revenue Analytics: Additional Resources & Exploration Guide

[![Documentation](https://img.shields.io/badge/Documentation-Case%20Study-blue.svg)](CASE_STUDY.md)
[![Solutions](https://img.shields.io/badge/Solutions-Verified%20Guide-success.svg)](SOLUTION_GUIDE.md)
[![Reading Time](https://img.shields.io/badge/reading%20time-10%20min-lightgrey.svg)]()

This curated resource guide provides foundational academic literature, standard hospitality metrics and formulas, recommended Python data science tooling, and advanced capstone extension projects for researchers, data scientists, and hospitality revenue managers.

---

## 📑 Table of Contents
1. [Primary Dataset Citation & Academic Literature](#1-primary-dataset-citation--academic-literature)
2. [Hospitality Industry Metrics & Mathematical Formulas](#2-hospitality-industry-metrics--mathematical-formulas)
3. [Recommended Python Data Science & ML Stack](#3-recommended-python-data-science--ml-stack)
4. [Advanced Project Extensions & Capstone Ideas](#4-advanced-project-extensions--capstone-ideas)
5. [Industry Benchmark Databases & Professional Bodies](#5-industry-benchmark-databases--professional-bodies)

---

## 🔬 1. Primary Dataset Citation & Academic Literature

The data utilized in this study was collected directly from the Property Management Systems (PMS) of two real-world hotels in Portugal (City Hotel in Lisbon and Resort Hotel in the Algarve) and published in peer-reviewed scientific journals.

### Key Academic Papers

1. **Original Dataset Publication:**
   > **Antonio, N., de Almeida, A., and Nunes, L. (2019).** *"Hotel booking demand datasets."*  
   > **Journal:** *Data in Brief*, 22, 41–49.  
   > **DOI:** [10.1016/j.dib.2018.11.126](https://doi.org/10.1016/j.dib.2018.11.126)  
   > *Summary:* Details the extraction architecture, variable definitions, and privacy-preserving de-identification procedures used to generate the benchmark dataset.

2. **Predictive Cancellation Research:**
   > **Antonio, N., de Almeida, A., and Nunes, L. (2017).** *"Predicting hotel booking cancellations to decrease uncertainty and increase revenue."*  
   > **Journal:** *Tourism & Management Studies*, 13(2), 25–39.  
   > **DOI:** [10.18089/tms.2017.13203](https://doi.org/10.18089/tms.2017.13203)  
   > *Summary:* Implements machine learning models (SVMs, Naive Bayes, Decision Trees) to forecast cancellation probability, demonstrating a potential 12% revenue gain through automated overbooking policies.

3. **Revenue Management Foundational Text:**
   > **Talluri, K. T., and van Ryzin, G. J. (2004).**  
   > *"The Theory and Practice of Revenue Management."*  
   > **Publisher:** Springer Science & Business Media.  
   > *Summary:* The definitive mathematical treatise on dynamic pricing, capacity control, network revenue management, and consumer choice behavior.

4. **Hospitality Economics & Lead Time Dynamics:**
   > **Weatherford, L. R., and Kimes, S. E. (2003).** *"A comparison of forecasting methods for hotel revenue management."*  
   > **Journal:** *International Journal of Forecasting*, 19(3), 401–415.  
   > *Summary:* Evaluates additive vs. multiplicative pickup forecasting models for transient hotel demand.

---

## 📐 2. Hospitality Industry Metrics & Mathematical Formulas

In professional hospitality management, operational performance is benchmarked through standardized financial and inventory metrics:

```
+----------------------------------------------------------------------------------------------------+
|                               STANDARD HOSPITALITY FORMULA MATRIX                                  |
+--------------------------+-----------------------------------------------------+-------------------+
| Metric                   | Mathematical Formula                                | Unit / Benchmark  |
+--------------------------+-----------------------------------------------------+-------------------+
| Average Daily Rate (ADR) | Total Room Revenue / Total Paid Rooms Sold          | Currency (€ / $)  |
+--------------------------+-----------------------------------------------------+-------------------+
| Occupancy Rate (OCC)     | (Total Rooms Sold / Total Available Rooms) * 100    | Percentage (%)    |
+--------------------------+-----------------------------------------------------+-------------------+
| Revenue Per Available    | ADR * Occupancy Rate   OR                           | Currency (€ / $)  |
| Room (RevPAR)            | Total Room Revenue / Total Available Rooms          |                   |
+--------------------------+-----------------------------------------------------+-------------------+
| Gross Operating Profit   | Gross Operating Profit (GOP) / Total Available Room | Currency (€ / $)  |
| Per Available Room       | (Captures F&B, Spa, and operational costs)          |                   |
| (GOPPAR)                 |                                                     |                   |
+--------------------------+-----------------------------------------------------+-------------------+
| Net RevPAR (NRevPAR)     | (Room Revenue - Distribution Commissions) / Total   | Currency (€ / $)  |
|                          | Available Rooms                                     |                   |
+--------------------------+-----------------------------------------------------+-------------------+
| Average Length of Stay   | Total Room Nights Sold / Total Distinct Bookings    | Days / Nights     |
| (ALOS)                   |                                                     |                   |
+--------------------------+-----------------------------------------------------+-------------------+
| Cancellation Rate (CR)   | (Total Canceled Bookings / Total Confirmed Bookings)| Percentage (%)    |
|                          | * 100                                               |                   |
+--------------------------+-----------------------------------------------------+-------------------+
```

### Overbooking Optimization Model (Spoilage vs. Spillage)
To maximize expected revenue $E[R]$, revenue managers solve the critical ratio problem:
$$\text{Critical Ratio (CR)} = \frac{C_u}{C_u + C_o}$$
Where:
* $C_u$ (**Cost of Underage / Spoilage**): Opportunity cost of leaving an unsold room empty ($= \text{ADR} - \text{Marginal Cleaning Cost}$).
* $C_o$ (**Cost of Overage / Spillage**): The cost of "walking" an overbooked guest to a competitor hotel ($= \text{Alternative Room Rate} + \text{Transportation} + \text{Goodwill Penalty}$).

The optimal target booking capacity $Q^*$ satisfies:
$$P(\text{Cancellations} + \text{No-Shows} \le Q^* - \text{Capacity}) = \frac{C_u}{C_u + C_o}$$

---

## 🛠️ 3. Recommended Python Data Science & ML Stack

To extend the exploratory analysis in this repository into enterprise-grade production services, the following open-source ecosystem is recommended:

| Layer | Recommended Packages | Typical Use Case in Hospitality Analytics |
|:---|:---|:---|
| **Data Processing** | `pandas`, `polars`, `numpy` | High-performance feature engineering, booking window slicing, aggregation. |
| **Econometrics** | `statsmodels` | Ordinary Least Squares (OLS), Logistic regression, elasticity modeling, hypothesis testing. |
| **Machine Learning** | `scikit-learn`, `xgboost`, `lightgbm`, `catboost` | High-accuracy gradient-boosted trees for cancellation probability prediction. |
| **Model Explainability** | `shap`, `lime` | Generating local and global SHAP waterfall charts to explain individual booking risk to front-desk staff. |
| **Time Series / Seasonality** | `prophet`, `statsmodels.tsa`, `sktime` | Forecasting daily room demand pickup curves and holiday spikes. |
| **Interactive UI & Dashboards** | `streamlit`, `dash`, `plotly` | Deploying web apps for General Managers to explore live risk scores and KPIs. |
| **Production API Serving** | `fastapi`, `uvicorn`, `pydantic` | Serving low-latency microservices for PMS integration. |

---

## 🚀 4. Advanced Project Extensions & Capstone Ideas

The data and solutions in this repository provide an ideal foundation for portfolio-grade advanced projects:

---

### Extension 1: Real-Time Cancellation Early Warning System
* **Architecture:** Train an **XGBoost Classifier** on pre-arrival features (`lead_time`, `market_segment`, `deposit_type`, `previous_cancellations`, `total_of_special_requests`).
* **Output:** Output calibrated probability $p \in [0, 1]$ for every incoming reservation.
* **PMS Integration:**
  * If $p > 0.65$: Flag reservation in PMS; trigger automated email requesting card pre-authorization or deposit.
  * If $p < 0.20$: Whitelist reservation for VIP room allocation and pre-arrival upgrade marketing.

---

### Extension 2: Interactive Streamlit Revenue Dashboard
* **Functionality:**
  1. **Executive KPI Ribbon:** Live RevPAR, ADR, Occupancy, and Cancellation Rate.
  2. **Interactive Filters:** Sliders for Year (2015–2017), Hotel Typology, and Customer Market Segment.
  3. **What-If Scenario Simulator:** Allows revenue managers to simulate the revenue gain of reducing OTA share by 10% in favor of Direct web bookings.

---

### Extension 3: Dynamic Room Rate Elasticity Engine
* **Methodology:** Use two-stage least squares (2SLS) or instrumental variable regression to measure price elasticity of demand ($\epsilon_p$):
  $$\epsilon_p = \frac{\% \Delta \text{ Room Demand}}{\% \Delta \text{ ADR}}$$
* **Application:** Identify off-peak price points where lowering ADR by 10% stimulates sufficient occupancy to maximize total RevPAR.

---

### Extension 4: Monte Carlo Dynamic Overbooking Simulator
* **Simulation:** Model stochastic arrival and cancellation curves across a 30-day forecast horizon.
* **Optimization:** Simulate 10,000 runs to compute the exact overbooking limit (e.g., authorizing 108 bookings on a 100-room property) that minimizes total expected cost without exceeding a 0.5% walking guest threshold.

---

## 🌐 5. Industry Benchmark Databases & Professional Bodies

For real-world comparative studies and market trend intelligence, consult:

* **STR (Smith Travel Research):** Global authority for hotel benchmarking, providing STAR reports and historical market occupancy/RevPAR data ([str.com](https://str.com)).
* **Cornell Center for Hospitality Research (CHR):** Publishes cutting-edge academic studies on pricing algorithms and hotel operations ([sha.cornell.edu](https://sha.cornell.edu/faculty-research/centers-institutes/chr/)).
* **HSMAI (Hospitality Sales & Marketing Association International):** Industry standards and certification for Certified Revenue Management Executives (CRME) ([hsmai.org](https://hsmai.org)).
* **UN Tourism (formerly UNWTO):** Global tourism statistics, regional travel recovery reports, and border arrivals data ([unwto.org](https://www.unwto.org)).

---
*Maintained as part of the Hotel Operations & Revenue Analytics Case Study repository.*
