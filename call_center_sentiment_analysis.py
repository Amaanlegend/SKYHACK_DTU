import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Assuming your dataset is in a CSV format
df = pd.read_csv('/path/to/united_airlines_sentiment_data.csv')

# Check the first few rows of the dataset
print(df.head())

# Handling missing values (if any)
df = df.dropna()

# Converting 'agent_tone' and 'customer_tone' into categorical features
df['agent_tone'] = df['agent_tone'].astype('category')
df['customer_tone'] = df['customer_tone'].astype('category')

# Exploratory Data Analysis (EDA) - Tone Distribution

# Countplot for agent_tone distribution
sns.set_style("whitegrid")
plt.figure(figsize=(13, 7))
ax = sns.countplot(x='agent_tone', data=df, palette="pastel")
plt.title('Agent Tone Distribution', fontsize=16)
plt.xlabel('Agent Tone', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.show()

# Countplot for customer_tone distribution
plt.figure(figsize=(13, 7))
ax = sns.countplot(x='customer_tone', data=df, palette="pastel")
plt.title('Customer Tone Distribution', fontsize=16)
plt.xlabel('Customer Tone', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.show()

# Sentiment Distribution based on Average Sentiment Scores
plt.figure(figsize=(13, 7))
ax = sns.histplot(df['average_sentiment'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Average Sentiment', fontsize=16)
plt.xlabel('Average Sentiment', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.show()

# Relationship between Silence Percentage and Average Sentiment
plt.figure(figsize=(13, 7))
ax = sns.scatterplot(x='silence_percent_average', y='average_sentiment', hue='customer_tone', data=df, palette="Set2", s=100)
plt.title('Silence Percentage vs. Average Sentiment', fontsize=16)
plt.xlabel('Silence Percentage', fontsize=14)
plt.ylabel('Average Sentiment', fontsize=14)
plt.legend(title='Customer Tone')
plt.show()

# Cross-tab analysis: agent_tone vs customer_tone
tone_crosstab = pd.crosstab(df['agent_tone'], df['customer_tone'])
print(tone_crosstab)

# Heatmap for agent_tone vs customer_tone
plt.figure(figsize=(10, 6))
sns.heatmap(tone_crosstab, annot=True, cmap='Blues', fmt='d')
plt.title('Agent Tone vs Customer Tone', fontsize=16)
plt.xlabel('Customer Tone', fontsize=14)
plt.ylabel('Agent Tone', fontsize=14)
plt.show()

# Additional Analysis

# Word Cloud for Angry Customer Comments (if available)
negative_tone_df = df[df['customer_tone'] == 'angry']['customer_tone'].str.cat(sep=' ')
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_tone_df)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Common Theme in Angry Customer Comments')
plt.show()

# Average Sentiment by Agent Tone
agent_sentiment = df.groupby('agent_tone')['average_sentiment'].mean().reset_index()
print("Average Sentiment by Agent Tone:")
print(agent_sentiment)

# Visualizing Average Sentiment by Agent Tone
plt.figure(figsize=(13, 7))
ax = sns.barplot(x='agent_tone', y='average_sentiment', data=agent_sentiment, palette="Set3")
plt.title('Average Sentiment by Agent Tone', fontsize=16)
plt.xlabel('Agent Tone', fontsize=14)
plt.ylabel('Average Sentiment', fontsize=14)
plt.show()

# Silence Percentage for Negative Tones ('angry', 'frustrated')
negative_tone_silence = df[df['customer_tone'].isin(['angry', 'frustrated'])]['silence_percent_average'].mean()
print(f'Average Silence Percentage for Negative Tones: {negative_tone_silence:.2f}%')

# Distribution of Silence Percentage for Negative Tones
plt.figure(figsize=(13, 7))
ax = sns.histplot(df[df['customer_tone'].isin(['angry', 'frustrated'])]['silence_percent_average'], bins=20, kde=True, color='salmon')
plt.title('Silence Percentage Distribution for Negative Tones (Angry, Frustrated)', fontsize=16)
plt.xlabel('Silence Percentage', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.show()
