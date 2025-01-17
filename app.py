from flask import Flask, render_template, request
import pickle
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)
ps = PorterStemmer()

# Load model and vectorizer
model = pickle.load(open('model/model2.pkl', 'rb'))
tfidfvect = pickle.load(open('model/tfidfvect2.pkl', 'rb'))

# Build functionalities


@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/a', methods=['GET','POST'])
def webapp():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        text = request.form['text']
        prediction = predict(text)
        return render_template('home.html', prediction=prediction)

def predict(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word)
              for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    review_vect = tfidfvect.transform([review]).toarray()
    prediction = 'FAKE' if model.predict(review_vect) == 0 else 'REAL'
    return prediction


@app.route('/predict', methods=['POST'])
def webapp2():
    text = request.form['text']
    prediction = predict(text)
    return render_template('predict.html', text=text, result=prediction)


if __name__ == "__main__":
    app.run(debug=True)
