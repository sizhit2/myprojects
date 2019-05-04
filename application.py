from flask import Flask,render_template,url_for,request
import os
# import dash
# import dash_core_components as dcc
# import dash_html_components as html
from plotly import tools
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
# server = dash.Dash(__name__, server = app, url_base_pathname='/Action/' )


@app.route('/test',methods=['POST','GET'])
def test():
    if request.method == 'POST':
        
        with open('model.pkl','rb') as fid:
            model = pickle.load(fid)
        with open('model_recommend.pkl','rb') as fid:
            vec = pickle.load(fid)
		
        data_movie = movie['Combined']
        message = request.form['message']
        
        data = vec.transform([message])
		
        result = model.kneighbors(data)
        info = []
        labels = []
        idx = []
        scores = []
        for i in result[1][0]:
            t = movie.iloc[i]['Title']
            if(pd.isna(t)):
                continue
            else:
                j = np.argwhere(result[1][0] == i)[0][0]
            if(result[0][0][j] != 1):
                l = [t,data_movie.iloc[i]]
                scores.append(movie['scores'].iloc[i])
                info.append([movie['scores'].iloc[i],movie['Genre'].iloc[i],l,movie['url'].iloc[i]])
                idx.append(j)
                labels.append(t)
    values = result[0][0][idx]
    values_query = 10 - values
    values_score = (10 - values) * 0.3 + np.array(scores) * 0.7
    if(values[0] == 1):
        return render_template('test.html',prediction = 'bad')
    colors = ['#FEBFB3', '#E1396C', '#96D38C', '#D0F9B1','#A9D9B1']
    trace = go.Pie(labels=labels, values=values_query,
               hoverinfo='label+percent', textinfo='value', 
               textfont=dict(size=20),
               marker=dict(colors=colors, 
                           line=dict(color='#000000', width=2)),
                           domain=dict(x=[0,0.5]))
    trace_scores = go.Pie(labels=labels, values=values_score,
               hoverinfo='label+percent', textinfo='value', 
               textfont=dict(size=20),
               marker=dict(colors=colors, 
                           line=dict(color='#000000', width=2)),
                           domain=dict(x=[0.5,1]))
    data = [trace,trace_scores]
    ann1 = dict(font=dict(size=20),
            showarrow=False,
            text='Pure search scores',
            # Specify text position (place text in a hole of pie)
            x=0.20,
            y=1.2
            )
    ann2 = dict(font=dict(size=20),
            showarrow=False,
            text='Combined rating and search scores',
            # Specify text position (place text in a hole of pie)
            x=0.95,
            y=1.2
            )
    layout = go.Layout(title ='Still do not know what to decide? check our plots below',
                   annotations=[ann1,ann2],
                   # Hide legend if you want
                   #showlegend=False
                   )
    fig = go.Figure(data=data,layout=layout)
    my_plot = plot(fig, output_type='div')

    return render_template('test.html',prediction = info,mydiv = Markup(my_plot))
if __name__ == "__main__":
    app.run(debug=True)
    # jobs for next time: A dynamic graph in text.html based both rating and distances, Provide some general movie recommendation in index.html page