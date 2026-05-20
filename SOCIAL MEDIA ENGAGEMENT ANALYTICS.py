# =========================================================
# SOCIAL MEDIA ENGAGEMENT ANALYTICS USING PYTHON
# =========================================================

# =========================
# Import Libraries
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

#%matplotlib inline

# =========================================================
# TASK 1 — DATA IMPORT & SETUP
# =========================================================

# Load Dataset
df = pd.read_csv(r"C:\Users\vaish\OneDrive\Desktop\vaishu\Assignment\Python\Python 5\social_media_engagement_5000.csv")

# Display first 5 rows
print("First 5 Rows")
print(df.head())

# Display last 5 rows
print("\nLast 5 Rows")
print(df.tail())

# Shape of dataset
print("\nDataset Shape")
print(df.shape)

# Column names
print("\nColumn Names")
print(df.columns)

# Dataset information
print("\nDataset Info")
print(df.info())

# Data types
print("\nData Types")
print(df.dtypes)

# Convert date column into datetime format
df['posted_at'] = pd.to_datetime(df['posted_at'], errors='coerce')

print("\nConverted Date Column")
print(df['posted_at'].head())

# =========================================================
# TASK 2 — DATA CLEANING
# =========================================================

# -------------------------
# Missing Values
# -------------------------

print("\nMissing Values")
print(df.isnull().sum())

# Numerical columns
num_cols = [
    'likes',
    'comments',
    'shares',
    'watch_time_sec',
    'impression_count',
    'follower_count',
    'engagement_rate',
    'age'
]

# Fill numerical missing values using median
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Categorical columns
cat_cols = [
    'gender',
    'country',
    'post_type',
    'post_category',
    'device_type',
    'sentiment'
]

# Fill categorical missing values using mode
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# -------------------------
# Duplicate Handling
# -------------------------

print("\nDuplicate Rows")
print(df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()

# -------------------------
# Standardize Gender Labels
# -------------------------

df['gender'] = df['gender'].str.lower()

df['gender'] = df['gender'].replace({
    'm': 'male',
    'f': 'female'
})

print("\nUnique Gender Values")
print(df['gender'].unique())

# -------------------------
# Handle Unrealistic Values
# -------------------------

metrics = ['likes', 'comments', 'shares']

for col in metrics:
    df.loc[df[col] < 0, col] = df[col].median()

# -------------------------
# Create Hashtag Count
# -------------------------

df['hashtag_count'] = df['hashtags'].astype(str).apply(
    lambda x: len(x.split())
)

print("\nHashtag Count")
print(df[['hashtags', 'hashtag_count']].head())

# -------------------------
# Clean Sentiment Labels
# -------------------------

df['sentiment'] = df['sentiment'].str.lower().str.strip()

print("\nSentiment Values")
print(df['sentiment'].unique())

# =========================================================
# TASK 3 — DATA EXPLORATION USING PANDAS
# =========================================================

# -------------------------
# Summary Statistics
# -------------------------

print("\nSummary Statistics")
print(df.describe())

# -------------------------
# Unique Values
# -------------------------

print("\nUnique Post Types")
print(df['post_type'].unique())

print("\nPost Category Counts")
print(df['post_category'].value_counts())

print("\nNumber of Unique Countries")
print(df['country'].nunique())

# -------------------------
# Correlation Matrix
# -------------------------

numeric_df = df.select_dtypes(include=np.number)

corr_matrix = numeric_df.corr()

print("\nCorrelation Matrix")
print(corr_matrix)

# -------------------------
# GroupBy Analysis
# -------------------------

print("\nAverage Likes by Post Type")
print(
    df.groupby('post_type')['likes']
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage Impressions by Country")
print(
    df.groupby('country')['impression_count']
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage Engagement Rate by Sentiment")
print(
    df.groupby('sentiment')['engagement_rate']
    .mean()
)

# =========================================================
# TASK 4 — DATA WRANGLING
# =========================================================

# -------------------------
# Create Engagement Score
# -------------------------

df['engagement_score'] = (
    df['likes'] +
    df['comments'] +
    df['shares']
)

print("\nEngagement Score")
print(df[['likes', 'comments', 'shares', 'engagement_score']].head())

# -------------------------
# Log Transformation
# -------------------------

df['log_likes'] = np.log1p(df['likes'])

print("\nLog Likes")
print(df[['likes', 'log_likes']].head())

# -------------------------
# GroupBy Summaries
# -------------------------

print("\nSummary by Post Type")
print(
    df.groupby('post_type')[['likes', 'comments', 'shares']]
    .mean()
)

print("\nSummary by Country")
print(
    df.groupby('country')['engagement_rate']
    .mean()
)

print("\nSummary by Sentiment")
print(
    df.groupby('sentiment')['watch_time_sec']
    .mean()
)

# =========================================================
# TASK 5 — STATISTICAL ANALYSIS
# =========================================================

stats_cols = [
    'likes',
    'comments',
    'shares',
    'watch_time_sec',
    'engagement_rate',
    'follower_count'
]

# Mean
print("\nMean")
print(df[stats_cols].mean())

# Median
print("\nMedian")
print(df[stats_cols].median())

# Mode
print("\nMode")
print(df[stats_cols].mode().iloc[0])

# Standard Deviation
print("\nStandard Deviation")
print(df[stats_cols].std())

# Variance
print("\nVariance")
print(df[stats_cols].var())

# Percentiles
print("\nPercentiles")
print(df[stats_cols].quantile([0.25, 0.50, 0.75]))

# Skewness
print("\nSkewness")
print(df[stats_cols].skew())

# Kurtosis
print("\nKurtosis")
print(df[stats_cols].kurtosis())

# =========================================================
# TASK 6 — DATA VISUALIZATION
# =========================================================

# -------------------------
# 1 Scatter Plot
# -------------------------

plt.figure(figsize=(8,5))

plt.scatter(df['likes'], df['impression_count'])

plt.title('Likes vs Impressions')

plt.xlabel('Likes')

plt.ylabel('Impressions')

plt.show()

# -------------------------
# 2 Line Plot
# -------------------------

df['date'] = df['posted_at'].dt.date

trend = df.groupby('date')['engagement_score'].mean()

plt.figure(figsize=(12,5))

plt.plot(trend)

plt.title('Daily Engagement Trend')

plt.xlabel('Date')

plt.ylabel('Average Engagement')

plt.xticks(rotation=45)

plt.show()

# -------------------------
# 3 Bar Plot
# -------------------------

plt.figure(figsize=(8,5))

df['post_category'].value_counts().plot(kind='bar')

plt.title('Posts by Category')

plt.xlabel('Category')

plt.ylabel('Count')

plt.show()

# -------------------------
# 4 Pie Chart
# -------------------------

df['gender'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%'
)

plt.title('Gender Distribution')

plt.ylabel('')

plt.show()

# -------------------------
# 5 Histogram
# -------------------------

plt.figure(figsize=(8,5))

plt.hist(df['age'], bins=20)

plt.title('Age Distribution')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.show()

# -------------------------
# 6 Box Plot
# -------------------------

plt.figure(figsize=(8,5))

plt.boxplot(df['engagement_rate'])

plt.title('Engagement Rate Boxplot')

plt.show()

# -------------------------
# 7 Count Plot
# -------------------------

plt.figure(figsize=(8,5))

sns.countplot(x='post_type', data=df)

plt.title('Count of Post Types')

plt.show()

# -------------------------
# 8 Bar Plot using Seaborn
# -------------------------

plt.figure(figsize=(8,5))

sns.barplot(
    x='post_category',
    y='likes',
    data=df
)

plt.title('Average Likes by Category')

plt.xticks(rotation=45)

plt.show()

# -------------------------
# 9 Violin Plot
# -------------------------

plt.figure(figsize=(8,5))

sns.violinplot(
    x='sentiment',
    y='follower_count',
    data=df
)

plt.title('Followers vs Sentiment')

plt.show()

# -------------------------
# 10 Heatmap
# -------------------------

plt.figure(figsize=(12,8))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm'
)

plt.title('Correlation Matrix')

plt.show()

# -------------------------
# 11 Swarm Plot
# -------------------------

plt.figure(figsize=(10,5))

sns.stripplot(
    x='device_type',
    y='engagement_rate',
    data=df,
    size=2
)

plt.title('Engagement Rate vs Device Type')

plt.show()

# -------------------------
# 12 Plotly Interactive Plot
# -------------------------

fig = px.scatter(
    df,
    x='likes',
    y='impression_count',
    color='post_type',
    size='engagement_rate',
    title='Interactive Likes vs Impressions'
)

fig.show()

# =========================================================
# FINAL INSIGHTS
# =========================================================

# -------------------------
# Content Performance
# -------------------------

print("\nHighest Engagement Post Type")

print(
    df.groupby('post_type')['engagement_score']
    .mean()
    .sort_values(ascending=False)
)

print("\nBest Performing Category")

print(
    df.groupby('post_category')['engagement_score']
    .mean()
    .sort_values(ascending=False)
)

print("\nCountries with Highest Engagement Rate")

print(
    df.groupby('country')['engagement_rate']
    .mean()
    .sort_values(ascending=False)
)

# -------------------------
# User Trends
# -------------------------

print("\nAge vs Engagement")

print(
    df.groupby('age')['engagement_score']
    .mean()
)

print("\nVerified vs Non Verified")

print(
    df.groupby('is_verified')['engagement_score']
    .mean()
)

# -------------------------
# Behavioral Insights
# -------------------------

df['hour'] = df['posted_at'].dt.hour

print("\nBest Posting Hour")

print(
    df.groupby('hour')['impression_count']
    .mean()
    .sort_values(ascending=False)
)

print("\nDevice Type vs Watch Time")

print(
    df.groupby('device_type')['watch_time_sec']
    .mean()
)

# -------------------------
# Sentiment Analysis
# -------------------------

print("\nSentiment Performance")

print(
    df.groupby('sentiment')['engagement_score']
    .mean()
)

print("\nNegative and Neutral Sentiment Behavior")

print(
    df[df['sentiment'].isin(['negative', 'neutral'])]
    .groupby('sentiment')['watch_time_sec']
    .mean()
)

# =========================================================
# FINAL CONCLUSION
# =========================================================

print("""
\nFINAL CONCLUSION
1. Video and reel content receive higher engagement.
2. Positive sentiment posts generate better interaction.
3. Verified accounts perform better than non-verified accounts.
4. Posting time affects impressions and engagement.
5. Mobile users contribute significantly to watch time.
6. Audience behavior varies by age, country, and content category.
This project demonstrates how Python, Pandas, NumPy, Matplotlib, Seaborn, and Plotly can be used for social media analytics and business insights.
""")