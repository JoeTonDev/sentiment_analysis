import os
import numpy as np
import pandas as pd
import docx
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentintensityAnalyzer
from nltk import word_tokenize
from docx import Document
# Intitalize SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
sia = SentimentintensityAnalyzer()

# Define a function to analyze the sentiment of text
def analyze_sentiment(text):
  return sia.polarity_scores(text)['compound']


# Read all the Word file in folder
path = "/path/to/folder"
files = [f for f in os.listdir(path) if f.endswith('.docs')]

# Initialize an empty list to store the sentiment scores
sentiments = []

# Loop through each file, extract the text and analyze the sentiment
for file in files:
  text = extract_text_from_docx(os.path.join(path, file))
  sentiment = analyze_sentiment(text)
  sentiments.append(sentiment)
  
  # Convert the list of sentiment scores to a numpy array
  sentiment = np.array(sentiments)
  
  # Plat the distribution of sentiment scores
  plt.hist(sentiments, bins=np.arrange(-1, 1.1, 0.1))
  plt.xlabel('Sentiment Score')
  plt.ylabel('Frequency')
  plt.show()
  
  # Rank the files based on their sentiment scores
  file_sentiments = [(f, s) for f, s in zip(files, sentiment)]
  file_sentiments.sort(key=lambda x: x[1], reverse=True)
  for file, sentiment in file_sentiments:
    print(f'{file}: {sentiment:.2f}')



