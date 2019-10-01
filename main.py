from flask import Flask, request, url_for, render_template, redirect, jsonify
import pandas as pd
from joblib import load
from sklearn.naive_bayes import MultinomialNB

import json

from flask_cors import CORS

from util import *


app = Flask(__name__)
CORS(app)

#  "./sentiment_data_ARA_pos.txt" 	 './sentiment_data_ARA_neg.txt'
#  "./sentiment_data_TUN_neg.txt" 	 './sentiment_data_TUN_pos.txt'
#



##################
# HOME END POINT #
##################
@app.route('/')
def show_home():
	return render_template('home.html')




'''
Checks the language of sentence.
Returns JSON response that looks like this
    {
        "lang": "['ARA']",
        "match": "[[0.99826534 0.00173466]]",
        "message": "برنامج جميل لو أكون في إرسال صوت يكون أفضل برامج"
    }
'''
###########################
# PROCESSING A WHOLE FILE #
###########################
@app.route('/upload/message', methods=['POST'])
def uploadMessage():
	if request.method == 'POST':
		if not request.json['message']:
			return "key 'message' not found in the body of the request"
		data = str(request.json['message'])
		output = detect_laguage(data)
		return jsonify(output)
	


###########################
# PROCESSING A WHOLE FILE #
###########################
@app.route('/upload/file/<string:pathfile>/', methods=['GET'])
def uploadFile(pathfile):
	corpus = read_text_file("./" + pathfile)
	output = []
	for i in range(10):
		output.append(detect_laguage(corpus[i]))
# 	i = 0
# 	for doc in corpus:
# 		output.append(detect_laguage(doc))
# 		i = i + 1
# 		if (i % 10 == 0):
# 			print ("Processed ...  " + str(i) + "  lines of documents.")
	return jsonify(output) 




if __name__ == '__main__':
    app.run(debug=True)