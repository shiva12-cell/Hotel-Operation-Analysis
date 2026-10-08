# Hotel Operations & Revenue Management Analysis

- **Domain:** Hospitality Analytics, Yield Management & Revenue Econometrics
- **Primary Tech Stack:** Python 3.10+ (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `statsmodels`)

- **Core Scope:** End-to-end data science and revenue management case study analyzing 87,396 validated hotel reservations across City and Resort properties to evaluate cancellation risk, channel profitability, pricing elasticity, and operational efficiency.

---

## Executive Summary
In the hospitality industry, severe cancellation volatility and over-reliance on third-party Online Travel Agencies (OTAs) erode room yields, create operational friction, and compress net operating margins. 

This case study performs an in-depth exploratory, statistical, and econometric analysis on 87,396 deduped hotel bookings across City and Resort properties. The study isolates structural drivers of reservation churn, models multi-year Average Daily Rate (ADR) dynamics, quantifies channel commission leakages, and estimates price elasticity. The findings demonstrate that booking lead times directly amplify cancellation probability, special amenity requests serve as potent commitments against churn, and direct bookings generate superior net margins compared to intermediary-driven channels.

---

## Key Metrics
- **Analyzed Cohort:** 87,396 validated, unique bookings (31,994 duplicates cleaned from 119,390 raw records).
- **Portfolio Cancellation Rate:** 27.49% overall (City Hotel: 30.04% | Resort Hotel: 23.48%).
- **Average Daily Rate (ADR):** €106.33 portfolio average (City Hotel at €110.99 outpaces Resort Hotel at €99.03 by +12.1%).
- **Lead Time Exposure:** 79.89 days mean lead time (median: 49 days); reservations booked >180 days in advance exhibit cancellation rates exceeding 42%.
- **Distribution Channel Concentration:** Travel Agents/Tour Operators (TA/TO) control 79.11% of booking volume; Online TAs account for 59.1% (51,618 bookings).
- **Net Channel Yield:** Direct bookings achieve €116.58 ADR with zero commission drag versus €96.90 net yield for OTAs after standard intermediary fees.
- **Commitment Multipliers:** Each submitted special request reduces cancellation odds by 32.0% ($e^{\beta} = 0.6803$), and each reservation modification reduces cancellation odds by 39.0% ($e^{\beta} = 0.6100$).
- **Family Demographic Premium:** Each child adds +€38.66 to room ADR versus +€21.18 for an adult in OLS econometric pricing models.

---

## Repository Structure
```text
Hotel-Operation-Analysis/
│
├── Raw_Data/                               # Raw and preprocessed hotel booking demand datasets
├── H-Ops Images/                           # Analytical visualizations and chart exhibits
├── Kaggle Worksheet/                       # Executable Jupyter/Kaggle analysis notebook
│   └── eda02-hotel-operations-analysis .ipynb
├── CASE_STUDY.md                           # Business problem statement, data dictionary & 24 question tiers
├── SOLUTION_GUIDE.md                       # Comprehensive numerical solutions, derivations & code
├── ADDITIONAL_RESOURCES.md                 # Hospitality formula handbook (RevPAR, GOPPAR) & academic citations
└── README.md                               # Primary project documentation
```

---

## Strategic Recommendations
1. **Dynamic Tiered Cancellation & Deposit Policies:**
   - Enforce non-refundable rates or progressive non-refundable deposits on reservations with lead times extending beyond 60–90 days, targeting high-risk cohorts where churn exceeds 40%.
2. **Shift Share Toward High-Yield Direct Channels:**
   - Offer guaranteed room upgrades, complimentary breakfast, or loyalty incentives on direct booking engines to divert demand from high-commission OTAs (15–20% fee drag) toward zero-commission direct bookings (€116.58 gross ADR).
3. **Capitalize on Peak Seasonal Rate Inelasticity:**
   - Implement assertive yield management during July and August peaks by raising minimum length of stay (LOS) restrictions and narrowing discount allocations to maximize RevPAR.
4. **Leverage Pre-Arrival Engagement to Reduce Churn:**
   - Prompt guests via automated pre-stay messaging to submit room preferences and special requests. Engaging guests with personalization significantly lowers cancellation probability (-32% odds per request).
5. **Tailored Family Packaging & Suite Upselling:**
   - Design dedicated family packages with multi-bed arrangements and child-friendly amenities, capitalizing on the high empirical willingness-to-pay (+€38.66 ADR premium per child).
