
# Projects
A repository of projects for Data Science, ranging from sentiment analysis, machine learning, time series forecasting, and more.


Data Cleaning Process

Data Mining Techniques

Data Analysis

Database Connection--- MySql, PostgresSql, Elasticsearch

Dashboard Development

Feature Engineering & Feature Selection Technique

Exploratory Data Analysis

Data Visualization--- MatplotLib, Seaborn

Web Scraping - Scrapy, data collection

Apache Spark

ETL Process

Database connection

ML algorithms - Regression & Classification

Web Scraping



<!--
# Data Science Portfolio


## Overview
This repository contains a collection of projects in various domains of data science, machine learning, and deep learning. The projects are categorized based on the technologies and techniques used.
 

---

## R

 
1. **Data Visualization: Corruption and Human Development**  
   - This project explores the relationship between corruption and human development using UN Human Development Report data.
   - It visualizes the correlation between the **Human Development Index (HDI)** and the **Corruption Perceptions Index (CPI)** through scatter plots.
 

2. **Visualizing Inequalities in Life Expectancy**  
   - This project analyzes life expectancy trends worldwide.
   - It examines gender-based longevity differences, historical trends, and extreme cases using **ggplot2**.
   - Data from the United Nations life expectancy dataset.

3. **Rise and Fall of Programming Languages**  
   - Analyzes programming language trends using **Stack Overflow data**.
   - Identifies which languages are growing or shrinking in popularity.
   - Uses **Stack Exchange Data Explorer** to track historical trends.

4. **Degrees That Pay You Back**  
   - Evaluates financial outcomes of different university degrees.
   - Uses **k-means clustering** to categorize degrees based on earning potential.
   - Data cleaning and visualization to assess long-term benefits.

5. **A Visual History of Nobel Prize Winners**  
   - Analyzes **over 100 years of Nobel Prize data**.
   - Examines country-based distributions and double-award recipients.
   - Identifies potential biases in awards over time.

---

## Python

1. **Telecom Customer Churn**  
   - Predicts customer churn in the telecom industry.
   - Uses **Logistic Regression** to identify key churn factors.
   - Data from [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn).

2. **Directing Customers to Subscription Through App Behavior Analysis**  
   - Identifies users likely to convert to paid subscriptions.
   - Uses behavioral data analysis to optimize marketing strategies.

3. **Minimizing Churn Rate Through Analysis of Financial Habits**  
   - Uses **Random Forest** and feature selection to predict subscription cancellations.
   - Identifies financial behavior patterns leading to disengagement.

4. **Car Price Prediction**  
   - Uses **Ridge & Lasso Regression** to predict car prices.
   - Dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/automobile).

5. **Customer Segmentation Using RFM Analysis**  
   - Segments customers based on **Recency, Frequency, and Monetary (RFM) analysis**.
   - Python implementation of customer segmentation.

6. **Movie Recommendations Using Recommender Systems**  
   - Builds a **collaborative filtering** recommender system for movie suggestions.
   - Uses similarity-based user reviews for recommendations.

7. **Decline in Viewership of a Digital Media Company**  
   - Investigates factors behind the **drop in viewership** of an online streaming platform.
   - Uses **multiple regression modeling** to predict future engagement.

8. **911 Calls Data Analysis**  
   - Performs **exploratory data analysis (EDA)** on 911 emergency call data.
   - Extracts insights on call frequency, categories, and trends.

9. **Predicting the Likelihood of E-Signing a Loan**  
   - Assesses lead quality for online loan applications.
   - Predicts approval likelihood based on financial history.

---

## Deep Learning

1. **Fashion-Class Classification using MNIST Dataset**  
   - Trains deep learning models on the **Fashion MNIST** dataset.
   - Uses **Convolutional Neural Networks (CNNs)** for image classification.

2. **MNIST Using PCA**  
   - Applies **Principal Component Analysis (PCA)** for dimensionality reduction.
   - Evaluates its impact on MNIST digit classification.

3. **Deep Learning for Time Series**  
   - Uses **Recurrent Neural Networks (RNNs) and LSTMs** to predict traffic trends.
   - Forecasts future traffic patterns in the U.S.

---

## Natural Language Processing (NLP)

1. **Named Entity Recognition (NER)**  
   - Extracts **people, organizations, and locations** from text.
   - Uses NLP techniques for automatic entity classification.

2. **Part of Speech (POS) Tagging**  
   - Identifies grammatical structure in sentences.
   - Helps in linguistic analysis and text parsing.

3. **Text Classification**  
   - Classifies documents into different categories.
   - Uses machine learning algorithms for text categorization.

4. **Text Generation with Neural Networks**  
   - Generates synthetic text based on trained models.
   - Uses deep learning techniques like **LSTMs and transformers**.

---

## Time Series Analysis

1. **Mauna Loa Atmospheric CO2 Forecasting (SARIMA)**  
   - Models **atmospheric CO2 levels** using **SARIMA**.
   - Predicts seasonal variations in climate data.

2. **Miles Traveled Forecasting (ARIMA)**  
   - Predicts future U.S. traffic trends using **ARIMA** models.

3. **Avocado Price Prediction**  
   - Forecasts avocado price trends using time series analysis.
   - Uses historical price data for predictive modeling.

--> 


<!--
# Data Science Portfolio

## R

1. [Data Visualization: Corruption and Human Development](): The purpose of this project is to perform data visualization to explore the relationship between Corruption and Human Development across various nations based on UN Human Development Report. A scatter plot for the relationship between the 'Human Development Index' and the 'Corruption Perceptions Index' of countries.

2. [Visualizing Inequalities in Life Expectancy](http://rpubs.com/shantanu97/Title): Do women live longer than men? How long? Does it happen everywhere? Is life expectancy increasing? Everywhere? Which is the country with the lowest life expectancy? Which is the one with the highest? In this Project, I will answer all these questions by manipulating and visualizing United Nations life expectancy data using ggplot2. The dataset can be found [here]() and contains the average life expectancies of men and women by country (in years). It covers four periods: 1985-1990, 1990-1995, 1995-2000, and 2000-2005.

3. [Rise and Fall of Programming Languages](http://rpubs.com/shantanu97/Programming_Languages): How can you tell what programming languages and technologies are used by the most people? How about what languages are growing and which are shrinking, so that you can tell which are most worth investing time in? One excellent source of data is [Stack Overflow](), a programming question and answer site with more than 16 million questions on programming topics. By measuring the number of questions about each technology, you can get an approximate sense of how many people are using it. In this project, you'll use open data from the [Stack Exchange Data Explorer]() to examine the relative popularity of languages like R, Python, Java and Javascript have changed over time.

4. [Degrees that Pay You Back](https://github.com/Shantanu9326/Data-Science-Portfolio/blob/master/Degrees%20that%20pay%20you%20back.ipynb): Wondering if that Philosophy major will really help you pay the bills? Think you're set with an Engineering degree? Whether you're in school or navigating the postgrad world, this project will guide you in exploring the short- and long-term financial implications of this major decision. After doing some data clean up, you'll compare the recommendations from three different methods for determining the optimal number of clusters, apply a k-means cluster analysis, and visualize the results.

5. [A Visual History of Nobel Prize Winners](): The Nobel Prize is perhaps the world's most well-known scientific award. Every year it is given to scientists and scholars in chemistry, literature, physics, medicine, economics, and peace. The first Nobel Prize was handed out in 1901, and at that time the prize was Eurocentric and male-focused, but nowadays it's not biased in any way. Surely, right? Well, let's find out! In this project, you get to explore patterns and trends in over 100 years worth of Nobel Prize winners. What characteristics do the prize winners have? Which country gets it most often? And has anybody gotten it twice? It's up to you to figure this out.

## Python

1. [Telecom Customer Churn](https://github.com/Shantanu9326/Telecom-Customer-Churn/blob/master/Telecom_Customer_Churn.ipynb): Customer churn occurs when customers or subscribers stop doing business with a company or service, also known as customer attrition. It is also referred as loss of clients or customers. One industry in which churn rates are particularly useful is the telecommunications industry, because most customers have multiple options from which to choose within a geographic location. - Data Source: The dataset is available on [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn) data source and you can directly read this notebook into Google Colaboratory. By building a model to predict customer churn with Logistic Regression, ideally we can nip the problem of unsatisfied customers in the bud and keep the revenue flowing.

2. [Directing Customers to Subscription Through App Behavior Analysis](https://github.com/Shantanu9326/Data-Science-Portfolio/blob/master/Directing_Customers_to_Subscription_Through_App_Behavior_Analysis.ipynb): In today’s market, many companies have a mobile presence. Often, these companies provide free products/services in their mobile apps in an attempt to transition their customers to a paid membership. Some examples of paid products, which originate from free ones, are Youtube Red, Pandora premium, audible subscription, and you need a budget. Since marketing efforts are never free, these companies need to know exactly who to target with offers and promotions.

3. [Minimizing Churn Rate Through Analysis of Financial Habits](https://github.com/Shantanu9326/Data-Science-Portfolio/blob/master/Minimizing_Churn_Rate_Through_Analysis_of_Financial_Habits.ipynb): Subscription Products often are the main source of revenue for companies across all industries. These products can come in the form of a ‘one size fits all’ overcompassing subscription, or in multi-level memberships. Regardless of how they structure their memberships, or what industry they are in, companies almost always try to minimize customer churn (a.k.a subscription cancellations). To retain their customers, these companies first need to identify the behavioral pattern that acts as a catalyst in disengagement with the product.

## Deep Learning

1. [Fashion-Class Classification using MNIST Dataset](https://github.com/Shantanu9326/Data-Science-Portfolio/blob/master/Fashion_Class_Classification_using_MNIST_dataset.ipynb): Training AI machine learning models on the Fashion MNIST [dataset](https://github.com/zalandoresearch/fashion-mnist). Read the full article at [Image Recognition for Fashion with Machine Learning](http://www.primaryobjects.com/2017/10/23/image-recognition-for-fashion-with-machine-learning/)

2. [MNIST Using PCA](https://github.com/Shantanu9326/Fashion-Class-Classification-using-MNIST-dataset/blob/master/MNIST_USING_PCA.ipynb): The global fashion industry is valued at three trillion dollars and accounts for 2 percent of the world's GDP. The fashion industry is undergoing a dramatic transformation by adopting new computer vision, machine learning, and deep learning techniques.

3. [Deep Learning for Time Series](https://github.com/Shantanu9326/Data-Science-Portfolio/blob/master/Deep_Learning_for_Time_Series.ipynb): Americans are driving more than ever before. Predicted and plotted the future traffic trends using the RNN & LSTM deep learning models.

## Time Series

1. [Mauna Loa Atmospheric CO2 Concentration Forecasting using SARIMA](https://github.com/Shantanu9326/Forecasting/blob/master/Mauna_Loa_Atmospheric_CO2_Concentration_Forecasting_using_SARIMA.ipynb): Trends and seasonal variation in time-series models. Atmospheric CO2 concentrations (measured in parts per million) derived from air samples collected at Mauna Loa Observatory, Hawaii.

2. [Miles Travelled using ARIMA Model](): Americans are driving more than ever before. Predicted and plotted the future traffic trends using the RNN & LSTM deep learning models.

3. [Avocado Price Prediction](): Predict the avocado prices given Kaggle dataset.
-->



