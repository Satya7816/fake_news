import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import joblib

# Download NLTK resources
import nltk
nltk.download('stopwords')

def preprocess_text(text):
    # Remove stopwords and perform stemming
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    words = text.split()
    words = [ps.stem(word) for word in words if word.lower() not in stop_words]
    return ' '.join(words)

# Read datasets
fake_data = pd.read_csv('path_to_fake_dataset.csv')  # Replace with the actual path
true_data = pd.read_csv('path_to_true_dataset.csv')  # Replace with the actual path

# Combine datasets
fake_data['label'] = 1  # 1 for fake news
true_data['label'] = 0  # 0 for true news
combined_data = pd.concat([fake_data, true_data])

# Preprocess text
combined_data['text'] = combined_data['text'].apply(preprocess_text)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    combined_data['text'], combined_data['label'], test_size=0.2, random_state=42
)

# Build a pipeline that combines a TF-IDF vectorizer and a Naive Bayes classifier
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print classification report
print(metrics.classification_report(y_test, y_pred))

# Save the trained model for future use
joblib.dump(model, 'fake_news_detection_model.pkl')  # Replace with the desired path
