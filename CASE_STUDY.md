#  Hotel Operations & Revenue Analytics: Comprehensive Case Study


[![Python Version](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/dataset-Hotel%20Booking%20Demand-orange.svg)](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)


---

##  1. Executive Summary & Industry Context

The global hospitality industry operates within a volatile environment characterized by perishable inventory, dynamic customer demand, seasonal fluctuations, and high intermediary commission structures. In hotel operations, a room that sits vacant on a given night represents revenue lost forever. Conversely, high cancellation rates, unpredictable booking lead times, and sub-optimal pricing directly undermine profitability, labor allocation, and supply chain readiness.

This case study investigates operational and revenue dynamics across two contrasting property typologies:
1. **City Hotel (H1):** An urban hotel serving commercial, corporate, short-stay transient, and leisure travelers with steady weekday demand patterns.
2. **Resort Hotel (H2):** A destination resort property situated in the Algarve region of Portugal, driven by seasonal vacationers, longer stays, family travel, and advance tour-operator commitments.

By analyzing **119,390 historical booking records** spanning from **July 1, 2015 to August 31, 2017**, this case study empowers data scientists, revenue managers, and hospitality directors to evaluate operational friction, diagnose cancellation drivers, optimize Average Daily Rate (ADR), and build resilient forecasting frameworks.

---

##  2. Formal Business Problem Statement

Hospitality executives face four core operational and financial dilemmas:

```
+----------------------------------------------------------------------------------------------------+
|                                    CORE BUSINESS DILEMMAS                                          |
+------------------------------------+---------------------------------------------------------------+
| 1. High Cancellation Volatility   | ~27.5% of confirmed bookings cancel prior to arrival,          |
|    & Revenue Leakage               | generating artificial occupancy dips and last-minute vacancies.|
+------------------------------------+---------------------------------------------------------------+
| 2. Intermediary Channel Reliance   | Over 86% of bookings arrive via third-party agents and OTAs,   |
|    & Margin Erosion                | creating 15%-25% commission friction and customer disintermediation.|
+------------------------------------+---------------------------------------------------------------+
| 3. Dynamic Pricing Inefficiency    | Static pricing models fail to capture seasonal demand spikes, |
|    & ADR Misalignment              | guest composition differences (adults vs. children), or lead-time.|
+------------------------------------+---------------------------------------------------------------+
| 4. Operational Capacity Mismatches | Inaccurate staffing, parking availability, and room turnover   |
|    & Customer Dissatisfaction      | schedules driven by volatile stay durations and group spikes. |
+------------------------------------+---------------------------------------------------------------+
```

### Core Business Objectives:
1. **Quantify Operational Baselines:** Measure foundational KPIs including lead time, cancellation volume, seasonal arrivals, parking requirements, and length of stay.
2. **Dissect Channel & Segment Performance:** Compare cancellation risks, ADR generation, and distribution channel efficiencies between Online Travel Agencies (OTAs), Direct bookings, and Corporate groups.
3. **Model Price Sensitivity & Demand Drivers:** Statistically isolate the marginal revenue impact of guest party configurations (adults, children, infants) using econometrics and multivariate regression.
4. **Formulate Predictive Cancellation Mitigations:** Uncover behavioral leading indicators (lead time, historical booking modifications, special request counts) to enable automated risk scoring and dynamic overbooking policies.

---

##  3. Comprehensive Data Dictionary

The underlying dataset comprises **119,390 raw observations** across **32 attributes**, capturing customer demographics, booking channel attributes, arrival dates, stay length, rate economics, and post-booking actions.

### 3.1 Dataset Metadata
* **Original Records:** 119,390 rows
* **Deduplicated Records:** 87,396 rows (31,994 exact duplicate rows removed during hygiene pipeline)
* **Time Span:** July 1, 2015 – August 31, 2017 (26 months)
* **Geographical Origin:** Portugal (PRT) and 177 international source markets

### 3.2 Feature-by-Feature Specification Table

| # | Attribute Name | Data Type | Permissible Range / Values | Missing Values | Business & Operational Description |
|---|----------------|-----------|----------------------------|----------------|-------------------------------------|
| 1 | `hotel` | Categorical (String) | `'City Hotel'`, `'Resort Hotel'` | 0 | Classification of property typology (Urban business vs. Coastal resort). |
| 2 | `is_canceled` | Binary Indicator (Int64) | `0` = Not Canceled, `1` = Canceled | 0 | **Target variable**. Confirmed whether the booking was voided before check-in. |
| 3 | `lead_time` | Discrete Numeric (Int64) | $0 \le x \le 737$ (days) | 0 | Number of elapsed days between booking creation date and arrival date. |
| 4 | `arrival_date_year` | Discrete Numeric (Int64) | `2015`, `2016`, `2017` | 0 | Year of scheduled check-in. |
| 5 | `arrival_date_month` | Categorical (String) | `January` through `December` | 0 | Month of scheduled check-in. |
| 6 | `arrival_date_week_number` | Discrete Numeric (Int64) | $1 \le x \le 53$ | 0 | Calendar week number of arrival. |
| 7 | `arrival_date_day_of_month` | Discrete Numeric (Int64) | $1 \le x \le 31$ | 0 | Day of the month of scheduled arrival. |
| 8 | `stays_in_weekend_nights` | Discrete Numeric (Int64) | $0 \le x \le 19$ (nights) | 0 | Number of weekend nights (Saturday, Sunday) reserved or occupied. |
| 9 | `stays_in_week_nights` | Discrete Numeric (Int64) | $0 \le x \le 50$ (nights) | 0 | Number of weekday nights (Monday through Friday) reserved or occupied. |
| 10 | `adults` | Discrete Numeric (Int64) | $0 \le x \le 55$ | 0 | Number of declared adult occupants ($18+$ years). |
| 11 | `children` | Discrete Numeric (Float64) | $0.0 \le x \le 10.0$ | 4 (Imputed with `0.0`) | Number of children ($2-17$ years). |
| 12 | `babies` | Discrete Numeric (Int64) | $0 \le x \le 10$ | 0 | Number of infants under $2$ years. |
| 13 | `meal` | Categorical (String) | `'BB'`, `'HB'`, `'FB'`, `'SC'`, `'Undefined'` | 0 | Meal package contracted: Bed & Breakfast (`BB`), Half Board (`HB`), Full Board (`FB`), Self Catering (`SC`). |
| 14 | `country` | Categorical (String) | ISO 3166-1 alpha-3 code | 488 (Imputed as `'Unknown'`) | Country of origin of the primary guest. |
| 15 | `market_segment` | Categorical (String) | `'Direct'`, `'Corporate'`, `'Online TA'`, `'Offline TA/TO'`, `'Groups'`, `'Aviation'`, `'Complementary'`, `'Undefined'` | 0 | Designation of market designation and commercial contracting vehicle. |
| 16 | `distribution_channel` | Categorical (String) | `'Direct'`, `'Corporate'`, `'TA/TO'`, `'GDS'`, `'Undefined'` | 0 | Technical infrastructure utilized to distribute and capture the booking. |
| 17 | `is_repeated_guest` | Binary Indicator (Int64) | `0` = First-Time, `1` = Repeat | 0 | Indicates whether guest has stayed at the hotel under previous reservations. |
| 18 | `previous_cancellations` | Discrete Numeric (Int64) | $0 \le x \le 26$ | 0 | Number of previous bookings canceled by customer prior to this reservation. |
| 19 | `previous_bookings_not_canceled` | Discrete Numeric (Int64) | $0 \le x \le 72$ | 0 | Number of prior completed reservations honored by the customer. |
| 20 | `reserved_room_type` | Categorical (Code) | `'A'`, `'B'`, `'C'`, `'D'`, `'E'`, `'F'`, `'G'`, `'H'`, `'L'`, `'P'` | 0 | Room inventory code designated during initial contract reservation. |
| 21 | `assigned_room_type` | Categorical (Code) | `'A'`, `'B'`, `'C'`, `'D'`, `'E'`, `'F'`, `'G'`, `'H'`, `'I'`, `'K'`, `'L'`, `'P'` | 0 | Actual room inventory code allocated at property check-in (may differ due to upgrades or operational overbooking). |
| 22 | `booking_changes` | Discrete Numeric (Int64) | $0 \le x \le 21$ | 0 | Count of modifications made to booking parameters (dates, guests, room type). |
| 23 | `deposit_type` | Categorical (String) | `'No Deposit'`, `'Non Refund'`, `'Refundable'` | 0 | Financial payment security status guaranteed at time of booking. |
| 24 | `agent` | Identifier (Float64) | ID number or `0.0` (None) | 16,340 (Imputed with `0.0`) | ID of travel agency contracted for booking fulfillment. |
| 25 | `company` | Identifier (Float64) | ID number or `0.0` (None) | 112,593 (Imputed with `0.0`) | ID of corporate entity underwriting or contracting the reservation. |
| 26 | `days_in_waiting_list` | Discrete Numeric (Int64) | $0 \le x \le 391$ (days) | 0 | Days the reservation resided on waiting list before customer confirmation. |
| 27 | `customer_type` | Categorical (String) | `'Contract'`, `'Group'`, `'Transient'`, `'Transient-Party'` | 0 | Customer profile grouping according to length, group structure, and billing. |
| 28 | `adr` | Continuous Numeric (Float64) | $-\$6.38 \le x \le \$5,400.00$ | 0 | **Average Daily Rate**: Total lodging revenue divided by total paid room-nights. |
| 29 | `required_car_parking_spaces` | Discrete Numeric (Int64) | $0 \le x \le 8$ | 0 | Number of on-site parking stalls explicitly reserved by guest party. |
| 30 | `total_of_special_requests` | Discrete Numeric (Int64) | $0 \le x \le 5$ | 0 | Count of specific guest requirements (high floor, twin beds, quiet room, late check-in). |
| 31 | `reservation_status` | Categorical (String) | `'Check-Out'`, `'Canceled'`, `'No-Show'` | 0 | Final audit status recorded on the property PMS. |
| 32 | `reservation_status_date` | Date (ISO 8601) | `YYYY-MM-DD` | 0 | Timestamp when the terminal reservation status was executed. |

### 3.3 Engineered Domain Features

| # | Feature Name | Formula / Derivation | Data Type | Business Significance |
|---|--------------|----------------------|-----------|------------------------|
| 33 | `arrival_date` | `pd.to_datetime(year, month, day)` | Datetime | Provides precise continuous time-series index for trend and seasonality analysis. |
| 34 | `total_stay` | `stays_in_weekend_nights + stays_in_week_nights` | Discrete Numeric | Aggregates full length of stay (LOS) per reservation for duration modeling. |

---

##  4. Analytical Case Questions (By Tier)

The analytical investigation is structured into three progressive difficulty tiers:

###  Tier 1: Basic Operational & Foundational Metrics
* **Q1.1 (Lead Time Average):** What is the overall average booking lead time across all properties, and what does it reveal about booking horizons?
* **Q1.2 (Cancellation Magnitude):** What is the total volume and aggregate proportion of canceled bookings versus completed check-outs?
* **Q1.3 (Peak Arrival Month):** Which calendar month attracts the highest arrival traffic, and how does seasonality govern guest intake?
* **Q1.4 (Special Requests Baseline):** What is the average number of special requests submitted per reservation, and how frequently do guests engage with amenity customization?
* **Q1.5 (Primary Source Market):** Which country represents the highest volume of hotel bookings, and what is the domestic vs. international ratio?
* **Q1.6 (ADR by Property Type):** How does the Average Daily Rate (ADR) compare between City Hotels and Resort Hotels?
* **Q1.7 (Parking Demand):** What percentage of arriving guests require dedicated car parking infrastructure?
* **Q1.8 (Stay Distribution):** What is the average duration of stay during weekday nights versus weekend nights?
* **Q1.9 (Agency Dominance):** What proportion of total bookings are brokered through travel agency intermediaries (`agent > 0`)?

---

###  Tier 2: Medium Business Dynamics & Revenue Interactions
* **Q2.1 (Cancellation Rate by Hotel):** How does the cancellation rate differ between City Hotel and Resort Hotel, and why does one suffer greater volatility?
* **Q2.2 (Market Segment ADR Hierarchy):** Which market segments generate the highest Average Daily Rates, and how should sales teams prioritize inventory?
* **Q2.3 (Lead Time vs. Cancellation Risk):** What is the empirical correlation and behavioral relationship between booking lead time and cancellation likelihood?
* **Q2.4 (Leading Distribution Channel):** Which distribution channel processes the largest volume of gross bookings?
* **Q2.5 (Historical Cancellation Behavior):** What is the average prior cancellation rate by hotel property type, and does guest track record predict future behavior?
* **Q2.6 (Annual ADR Growth Trend):** What was the longitudinal trajectory of Average Daily Rate from 2015 through 2017?
* **Q2.7 (Top Monthly Yield):** Which single calendar month achieves the highest average ADR, and how does this align with arrival peaks?
* **Q2.8 (Amenity Request Pricing Elasticity):** What is the statistical correlation between special request frequency and ADR?
* **Q2.9 (Repeat Guest Retention & Stay Length):** How does the average stay duration of repeat guests compare against first-time visitors, and what does this reveal about guest loyalty?
* **Q2.10 (Inventory Room Demand Mode):** Which room category experiences the highest reservation volume across properties?

---

###  Tier 3: Advanced Econometrics, Statistical Modeling & Strategy
* **Q3.1 (Multivariate Cancellation Risk Modeling):** Using Logistic Regression, what are the direction, magnitude, and odds ratios of key behavioral drivers (`lead_time`, `booking_changes`, `total_of_special_requests`) on cancellation probability?
* **Q3.2 (Party Composition Econometrics):** How do additional adults, children, and infants marginally impact room ADR? What does an Ordinary Least Squares (OLS) multivariate regression reveal regarding marginal rate premiums?
* **Q3.3 (Booking Modification vs. Satisfaction):** Does high friction in reservation modification (`booking_changes`) correlate with guest dissatisfaction as captured through special requests?
* **Q3.4 (Monthly Seasonality & Cancellation Waves):** How does cancellation rate vary across the 12 calendar months, and why do peak summer months paradoxically observe peak cancellation ratios?
* **Q3.5 (Lead Time Dispersion Across Segments):** How does lead time distribution vary across market segments, and how can revenue managers construct differentiated release windows?



