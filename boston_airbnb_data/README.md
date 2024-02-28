# Boston Airbnb Data Analysis

Guide
- [About This Project](#about-this-project)
- [Dataset Content](#dataset-content)
- [Project Goals](#project-goals)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Project Rationale](#project-rationale)
- [Analysis](#analysis)
- [Licensing](#licensing)

# About This Project
This Jupyter Notebook project evalutates the dataset of Airbnb activity in Boston, Massachusetts. This dataset is from Airbnb and was downloaded from [Kaggle](https://www.kaggle.com/datasets/airbnb/boston). It contains information about homestay listings in the city since 2008.

#### Dataset Content:

- Listings with full descriptions and average review scores
- Reviews, including unique IDs for each reviewer and detailed comments
- Calendar, featuring listing IDs, prices, and availability for each day

#### Project Goals:

1. <ins>Monthly Price Spikes</ins>: Identify the months with the most significant spikes in rental prices to help travelers plan cost-effective visits.


2. <ins>Renter Satisfaction</ins>: Evaluate overall renter satisfaction with accommodations in Boston based on reviews and ratings.

3. <ins>Popular Neighborhoods</ins>: Determine the most popular neighborhoods in Boston based on Airbnb listing activity.

4. <ins>Average Price by Neighborhood</ins>: Analyze and compare the average rental prices in different neighborhoods to guide potential visitors.

5. <ins>Booking Trends</ins>: Examine whether the volume of short-term rental bookings is increasing or declining over time.

## Installation
Python:

Install Python by following the instructions on [python.org]("python.org").

Install Jupyter Notebook:
```
pip install notebook
```

#### Dependencies

Ensure the following Python libraries are installed:

Pandas:
```
pip install pandas
```
NumPy: 
```
pip install numpy
```
Matplotlib: 
```
pip install matplotlib
```
Seaborn: 
```
pip install seaborn
```

# Project Rationale
This analysis aims to provide Boston's short-term rental market insights, benefiting city planners, property owners, travelers, and policymakers. Stakeholders can leverage findings for informed decision-making, property management, budget-conscious trip planning, and housing market assessments. 

# Analysis
Monthly Price Spikes:
- Rental prices exhibit significant spikes above the average during spring and late summer, while experiencing a noticeable decline in the winter months (November to April).
- The analysis hints at higher prices and demand for short-term rentals in late 2016 compared to 2017.

Renter Satisfaction:
- Reviewers express high satisfaction with their rental experiences across all review scores.

Popular Neighborhoods:
- Jamaica Plain and Dorchester are the most frequently reviewed neighborhoods, with Jamaica Plain leading by a substantial margin.
- Allston-Brighton and South End also receive considerable attention.

Average Prices by Neighborhood:
- The Financial District commands the highest average price, just under $300, while Mattapan boasts the lowest average price, slightly over $50, indicating a significant price range.

Additional analysis compares average prices, review scores, and review frequency to identify neighborhoods offering the most value to renters.
Potential renters may find the best value in Jamaica Plain, followed by Dorchester and South End.

Short-Term Rental Bookings Trend
- Examining the number of first-time reviews aggregated by month reveals a drastic increase from 2013 to 2016.
- Recent data suggests a decline in first-time reviews in 2017, particularly during winter months, with a typical mid-year increase during summer.

# Licensing
This information is licensed under Public Domain. More information regarding this can be found on [Kaggle](https://www.kaggle.com/datasets/airbnb/boston).