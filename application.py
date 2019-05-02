from flask import Flask,render_template,url_for,request
import os
from plotly.offline import plot
import plotly.graph_objs as go
from flask import Markup
import flask
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
application = Flask(__name__)
app = application
movie = pd.read_csv('data/movie.csv')
data_movie = movie['Combined']


@app.route('/')
def home():
    return render_template("index.html")



@app.route('/test',methods=['POST','GET'])
def test():
    if request.method == 'POST':
        
        with open('model.pkl','rb') as fid:
            model = pickle.load(fid)
        with open('model_recommend.pkl','rb') as fid:
            vec = pickle.load(fid)
		
        data_movie = movie['Combined']
        r_movie = []
        for i in range(len(data_movie)):
            r_movie.append(data_movie.iloc[i])
        vec.fit(r_movie)
        message = request.form['message']
        
        data = vec.transform([message])
		
        result = model.kneighbors(data)
        info = []
        labels = []
        idx = []
        for i in result[1][0]:
            t = movie.iloc[i]['Title']
            if(pd.isna(t)):
                continue
            else:
                j = np.argwhere(result[1][0] == i)[0][0]
                idx.append(j)
                l = [t,data_movie.iloc[i]]
                info.append([movie['scores'].iloc[i],movie['Genre'].iloc[i],l,movie['url'].iloc[i]])
                labels.append(t)
    values = (result[0][0])[idx]
    if(values[0] == 1):
        return render_template('test.html',prediction = 'bad')
    colors = ['#FEBFB3', '#E1396C', '#96D38C', '#D0F9B1','#A9D9B1']
    trace = go.Pie(labels=labels, values=values,
               hoverinfo='label+percent', textinfo='value', 
               textfont=dict(size=20),
               marker=dict(colors=colors, 
                           line=dict(color='#000000', width=2)))
    my_plot_div = plot([trace], output_type='div')
    return render_template('test.html',prediction = info,mydiv = Markup(my_plot_div))
if __name__ == "__main__":
    app.run(debug=True)
    # jobs for next time: A dynamic graph in text.html based both rating and distances, Provide some general movie recommendation in index.html page