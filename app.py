from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    # get uploaded file
    file = request.files['file']
    
    # load data from file
    data = pd.read_csv(file)
    
    # get number of clusters from user input
    n_clusters = request.form.get('n_clusters')
    
    if n_clusters is None:
        return "Please input number of clusters."
    
    n_clusters = int(n_clusters)
    
    # perform clustering with KMeans
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    labels = kmeans.predict(data)
    
    # create scatter plot
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels)
    plt.title('KMeans Clustering Result')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.savefig('static/images/plot.png')
    plt.clf()
    
    return render_template('result.html', n_clusters=n_clusters)

if __name__ == '__main__':
    app.run(debug=True)
