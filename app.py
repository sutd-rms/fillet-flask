from flask import Flask, request, redirect, url_for, flash, jsonify
from helper_functions import *
from core_functions import rms_pricing_model
import numpy as np
import pickle as p
import json
import pandas as pd
import os
from pathlib import Path


app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"
	

@app.route('/train/', methods=['POST'])
def train():

	data, cv_acc, project_id = parse_training_request(request)

	response_outp = {'result':0,
					 'cv_acc':0
					}
					
	data_df = pd.DataFrame().from_dict(data)
	pdm = rms_pricing_model(data_df)
	pdm.train_all_items(retrain=True)
	item_ids = [int(x.split('_')[1]) for x in pdm.price_columns]
	for item_id in item_ids:
		item_model = pdm.models[item_id]
		model_dir = 'projects/{}/models'.format(project_id)
		if not os.path.exists(model_dir):
			os.makedirs(model_dir)
		p.dump(item_model, open(model_dir + '/model_{}.p'.format(item_id),'wb'))
	response_outp['result'] = 'Success'
	
	if cv_acc == True:
		perf_df = pdm.get_all_performance()
		response_outp['cv_acc'] = perf_df.to_json()

	return jsonify(response_outp)


@app.route('/predict/', methods=['POST'])
def predict():
	prices = request.json['prices']
	project_id = request.get_json()['project_id']

	response_outp = {'prediction':0
	}

	for k in list(prices.keys()):
		prices['Price_'+str(k)] = prices.pop(k)

	prices = pd.DataFrame(prices, index=[0])

	models_list = os.listdir('projects/{}/models'.format(project_id))
	items = [int(x.split('.')[0].split('_')[1]) for x in models_list]

	pred_qty = {}

	for item in items:
		item_model = p.load(open('projects/{}/models/model_{}.p'.format(project_id,item),'rb'))
		prices = prices[item_model.get_booster().feature_names]
		pred_q = item_model.predict(prices)[0]
		pred_qty[item]=int(round(pred_q))

	response_outp['pred_q'] = pred_qty
	print(pred_qty)
	return jsonify(response_outp)

@app.route('/api/', methods=['POST'])
def makecalc():
	data = request.get_json()
	prediction = np.array2string(model.predict(data))

	return jsonify(prediction)

if __name__ == '__main__':
	modelfile = 'models/final_prediction.pickle'
	model = p.load(open(modelfile, 'rb'))
	app.run(debug=True, host='0.0.0.0')
