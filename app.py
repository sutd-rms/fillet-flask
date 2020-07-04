from flask import Flask, request, redirect, url_for, flash, jsonify
from helper_functions import optimize_floats, optimize_memory
from core_functions import rms_pricing_model, GA
import numpy as np
import pickle as p
import json
import pandas as pd
import os, requests
from pathlib import Path
from xgboost import XGBRegressor

import logging

app = Flask('fillet-flask')
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s | %(levelname)-8s | %(name)-25s | %(threadName)-16s : %(message).80s'
	)

	
@app.route("/")
def hello():
	return "Hello World!"
	
@app.route('/optimize/', methods=['POST'])
def optimize():
	app.logger.info('OPTIMIZE REQUEST RECEIVED')
	# get input
	project_id = request.get_json()['project_id']
	constraints = request.get_json()['constraints']
	population =  request.get_json()['population']
	max_epoch = request.get_json()['max_epoch']
	# load price information
	PRICE_INFO_PATH = 'projects/{}/price_info.pkl'.format(project_id)
	assert os.path.isfile(PRICE_INFO_PATH), 'No price info file found.'
	price_std, price_mean, price_names = p.load(open(PRICE_INFO_PATH, 'rb'))
	# load model
	MODEL_PATH = 'projects/{}/models'.format(project_id)
	assert os.path.isdir(MODEL_PATH), 'No model directory found.'
	models_list = [x for x in os.listdir(MODEL_PATH) if x.startswith('model') ]
	# import pdb; pdb.set_trace()
	items = [int(x.split('.')[0].split('_')[1]) for x in models_list]

	models = []
	for item in items:
		item_model = XGBRegressor()
		item_model.load_model(MODEL_PATH+f'/model_{item}.json')
		models.append(item_model)

	# run optimization
	Optimizer = GA()
	Optimizer.properties(models, population, max_epoch, price_std, price_mean, price_names)
	best_price = Optimizer.run()

	# Send result as Dict to avoid confusion
	best_price_dict = {}
	for item, price in zip(items, best_price):
		best_price_dict[str(item)] = round(price,2)

	response_outp = {}
	response_outp['result'] = best_price_dict
	return jsonify(response_outp)


@app.route('/train/', methods=['POST'])
def train():
	app.logger.info('TRAIN REQUEST RECEIVED')
	data = json.loads(request.json['data'])
	cv_acc = request.get_json()['cv_acc']
	project_id = request.get_json()['project_id']

	response_outp = {'result':0,
					 'cv_acc':0
					}
	data_df = pd.DataFrame().from_dict(data)
	pdm = rms_pricing_model(data_df)
	# save price info for optimization use
	PRICE_INFO_PATH = 'projects/{}/'.format(project_id)
	if not os.path.isdir(PRICE_INFO_PATH):
		Path(PRICE_INFO_PATH).mkdir(parents=True)
	pdm.get_and_save_price_info(PRICE_INFO_PATH+'price_info.pkl')
	


	# train models
	pdm.train_all_items(proj_id=project_id,retrain=True)

	# save models
	item_ids = [int(x.split('_')[1]) for x in pdm.price_columns]
	# for item_id in item_ids:
	# 	item_model_json = pdm.models[item_id]
		
	# 	MODEL_PATH = f'projects/{project_id}/models/'
	# 	if not os.path.isdir(MODEL_PATH):
	# 		Path(MODEL_PATH).mkdir(parents=True)

	# 	with open(MODEL_PATH+f'model_{item_id}.json','w') as f:
	# 		json.dump(item_model_json,f)

		# MODEL_PATH = 'projects/{}/models/'.format(project_id)
		# if not os.path.isdir(MODEL_PATH):
		# 	Path(MODEL_PATH).mkdir(parents=True)

		# item_model.save_model(MODEL_PATH+f'model_{item_id}.json')


	

	response_outp['result'] = 'Success'
	
	if cv_acc == True:
		app.logger.info('RUNNING OPTIONAL CROSS VALIDATION')
		perf_df = pdm.get_all_performance()
		response_outp['cv_acc'] = perf_df.to_json()

	return jsonify(response_outp)

@app.route('/predict/', methods=['POST'])
def predict():
	app.logger.info('PREDICT REQUEST RECEIVED')
	with open('keys.json') as f:
			HOST_KEY = json.load(f)['host_key']

	prices = request.json['prices']
	project_id = request.get_json()['project_id']

	for k in list(prices.keys()):
		prices['Price_'+str(k)] = prices.pop(k)

	prices = pd.DataFrame(prices, index=[0])
	prices = prices.reindex(sorted(prices.columns), axis=1)

	prices_json = prices.to_json()

	items = [int(x.split('_')[1]) for x in prices.columns]

	models_json_list = []

	for item in items:
		with open(f'projects/{project_id}/models/model_{item}.json') as f:
			model_json = json.load(f)
		models_json_list.append(model_json)

	data = {
		'prices':prices_json,
		'models':models_json_list,
	}

	payload = {
		'code':HOST_KEY,
	}

	url = 'https://sutdcapstone22-filletofish.azurewebsites.net/api/fillet_func_4_predictbatch'
	app.logger.info('SENDING REQUEST TO FILLET SERVERS')
	result = requests.get(url, params=payload, data=json.dumps(data))
	app.logger.info('RESPONSE RECEIVED FROM FILLET')
	pred_quantities = result.json()['qty_estimates']

	pred_quantities_dict = {}
	for item,qty_estimate in zip(items,pred_quantities):
		pred_quantities_dict[f'Qty_{item}'] = int(round(float(qty_estimate),0))

	response_outp = {'qty_estimates':pred_quantities_dict}
	app.logger.info('RETURNING RESPONSE')
	return jsonify(response_outp)

@app.route('/api/', methods=['POST'])
def makecalc():
	data = request.get_json()
	prediction = np.array2string(model.predict(data))

	return jsonify(prediction)

if __name__ == '__main__':
	# modelfile = 'models/final_prediction.pickle'
	# model = p.load(open(modelfile, 'rb'))
	app.run(debug=True, host='0.0.0.0')
