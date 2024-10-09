# Importing necessary libraries
import pandas as pd
import sqlite3
import numpy as np

# Load datasets (excluding the calls dataset)
customers = pd.read_csv("/mnt/data/customers2afd6ea.csv")
reason = pd.read_csv("/mnt/data/reason18315ff.csv")
sentiment_stats = pd.read_csv("/mnt/data/sentiment_statisticscc1e57a.csv")

# Create a SQLite database in memory and connect
conn = sqlite3.connect(':memory:')

# Load the DataFrames into SQL tables
customers.to_sql('customers', conn, index=False, if_exists='replace')
reason.to_sql('reason', conn, index=False, if_exists='replace')
sentiment_stats.to_sql('sentiment_stats', conn, index=False, if_exists='replace')

# SQL Query: Join the datasets without the calls table
sql_query_combined = '''
SELECT
    r.call_id,
    cs.customer_name,
    cs.mp_status,
    r.primary_call_reason,
    s.average_sentiment
FROM reason r
LEFT JOIN customers cs ON r.call_id = cs.customer_id
LEFT JOIN sentiment_stats s ON r.call_id = s.call_id
'''

# Fetch the result into a DataFrame
df_combined = pd.read_sql_query(sql_query_combined, conn)

# Group by primary call reason to find the average sentiment for each reason
grouped_call_reason = df_combined.groupby('primary_call_reason')['average_sentiment'].mean().reset_index()
most_frequent_reason = grouped_call_reason.sort_values(by='average_sentiment', ascending=False).iloc[0]
least_frequent_reason = grouped_call_reason.sort_values(by='average_sentiment', ascending=True).iloc[0]

print(f"Most Frequent Call Reason by Sentiment: {most_frequent_reason['primary_call_reason']}, Average Sentiment: {most_frequent_reason['average_sentiment']:.2f}")
print(f"Least Frequent Call Reason by Sentiment: {least_frequent_reason['primary_call_reason']}, Average Sentiment: {least_frequent_reason['average_sentiment']:.2f}")

# Sentiment Distribution by loyalty status (mp_status)
sentiment_distribution = df_combined.groupby('mp_status')['average_sentiment'].mean().reset_index()

# Visualizing sentiment trends across loyalty levels (mp_status)
import matplotlib.pyplot as plt
import seaborn as sns

sns.barplot(x='mp_status', y='average_sentiment', data=sentiment_distribution)
plt.title('Average Sentiment by Loyalty Status (mp_status)')
plt.xlabel('Loyalty Status (mp_status)')
plt.ylabel('Average Sentiment')
plt.show()

# Close the SQL connection
conn.close()

# Final suggestions based on analysis
print("\nActionable Recommendations:")
print("1. Focus on improving sentiment for call reasons with the lowest customer satisfaction.")
print("2. Improve services for lower-tier loyalty customers (based on mp_status) to enhance their overall sentiment.")
