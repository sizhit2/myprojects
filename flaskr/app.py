from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import nltk
from nltk.corpus import stopwords
import numpy as np
app = Flask(__name__)
movie = pd.read_csv('data/combined_corpus_for_search_engine.csv')
data_movie = movie['Combined']
s_movie = []
for i in range(len(data_movie)):
    s_movie.append(data_movie.iloc[i])
vector_movie = TfidfVectorizer(stop_words=set(stopwords.words('english')),max_df=0.8,min_df= 0.01,analyzer= 'word')
vector_movie.fit(s_movie)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/test',methods=['POST','GET'])
def test():
    my_prediction = []

    if request.method == 'POST':
        
        with open('model.pkl','rb') as fid:
            model = pickle.load(fid)
		
        message = request.form['message']
        
        data = vector_movie.transform([message])
		
        result = model.kneighbors(data)

        for i in result[1][0]:
            t = movie.iloc[i]['Title']
            if(pd.isna(t)):
                continue
            else:
                l = [t,data_movie.iloc[i]]
                my_prediction.append(l)
    return render_template('test.html',prediction = my_prediction)

if __name__ == "__main__":
    app.run(debug = True)