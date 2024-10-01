import json
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from collections import defaultdict
from collections import Counter

# Load the JSON file
with open('result.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Prepare dictionaries to store results
user_sentiments = defaultdict(lambda: {'positive': 0, 'negative': 0, 'neutral': 0})
negative_messages_per_user = defaultdict(list)

# Extract messages and run sentiment analysis
for message in data['messages']:
    if 'text' in message and isinstance(message['text'], str):
        user = message.get('from', 'Unknown')
        sentiment = sia.polarity_scores(message['text'])
        if sentiment['compound'] >= 0.05:
            user_sentiments[user]['positive'] += 1
        elif sentiment['compound'] <= -0.05:
            user_sentiments[user]['negative'] += 1
            negative_messages_per_user[user].append(message['text'])  # Store negative messages
        else:
            user_sentiments[user]['neutral'] += 1

# Count repeated negative messages per user
repeated_negative_counts = {}
for user, messages in negative_messages_per_user.items():
    message_counts = Counter(messages)
    repeated_negative_counts[user] = {msg: count for msg, count in message_counts.items() if count > 1}

# Display the top 10 repeated negative messages per user
for user, repeated_messages in repeated_negative_counts.items():
    print(f"User: {user}")
    # Sort the messages by the count in descending order
    sorted_messages = sorted(repeated_messages.items(), key=lambda x: x[1], reverse=True)
    # Display the top 10 most repeated messages
    for i, (message, count) in enumerate(sorted_messages[:10]):
        print(f"{i + 1}. Message: '{message}' is repeated {count} times.")
    print('-' * 50)

# Convert results to a DataFrame for easier manipulation
df = pd.DataFrame(user_sentiments).T.reset_index()
df.columns = ['user', 'positive', 'negative', 'neutral']

# Save user_sentiments to CSV
csv_filename = 'user_sentiments.csv'
df.to_csv(csv_filename, index=False)
print(f"Sentiment data saved to {csv_filename}")

# Plot the results
plt.figure(figsize=(10, 7))
ax = df.plot(kind='bar', x='user', stacked=True, figsize=(10, 7))
plt.title('Sentiment Analysis by User')
plt.xlabel('User')
plt.ylabel('Number of Sentences')

# Rotate x-axis labels by 45 degrees for readability
plt.xticks(rotation=45, ha='right')

# Add absolute numbers on the bars
for container in ax.containers:
    ax.bar_label(container, label_type='center', fontsize=10)

# Save the plot to an image file
plot_filename = 'sentiment_analysis_plot.png'
plt.savefig(plot_filename, bbox_inches='tight')
print(f"Plot saved to {plot_filename}")