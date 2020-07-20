from flask import Flask, request, redirect, url_for, flash, jsonify
from helper_functions import optimize_floats, optimize_memory
from core_functions import rms_pricing_model
# from core_functions import rms_pricing_model, GA
import numpy as np
import pickle as p

import json
import pandas as pd
import os, requests
from pathlib import Path
import gc
import shutil
import zlib
# from xgboost import XGBRegressor

import logging

app = Flask('fillet-flask')
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s | %(levelname)-8s | %(name)-25s | %(threadName)-16s : %(message).80s'
	)

	
@app.route("/")
def hello():
	return "Hello World!"
	
# @app.route('/optimize/', methods=['POST'])
# def optimize():
# 	app.logger.info('OPTIMIZE REQUEST RECEIVED')
# 	# get input
# 	project_id = request.get_json()['project_id']
# 	constraints = request.get_json()['constraints']
# 	population =  request.get_json()['population']
# 	max_epoch = request.get_json()['max_epoch']
# 	# load price information
# 	PRICE_INFO_PATH = 'projects/{}/price_info.pkl'.format(project_id)
# 	assert os.path.isfile(PRICE_INFO_PATH), 'No price info file found.'
# 	price_std, price_mean, price_names = p.load(open(PRICE_INFO_PATH, 'rb'))
# 	# load model
# 	MODEL_PATH = 'projects/{}/models'.format(project_id)
# 	assert os.path.isdir(MODEL_PATH), 'No model directory found.'
# 	models_list = [x for x in os.listdir(MODEL_PATH) if x.startswith('model') ]
# 	# import pdb; pdb.set_trace()
# 	items = [int(x.split('.')[0].split('_')[1]) for x in models_list]

# 	models = []
# 	for item in items:
# 		item_model = XGBRegressor()
# 		item_model.load_model(MODEL_PATH+f'/model_{item}.json')
# 		models.append(item_model)

# 	Optimizer = GA()
# 	Optimizer.properties(models, population, max_epoch, price_std, price_mean, price_names)
# 	best_price = Optimizer.run()

# 	# Send result as Dict to avoid confusion
# 	best_price_dict = {}
# 	for item, price in zip(items, best_price):
# 		best_price_dict[str(item)] = round(price,2)
# 	# run optimization

# 	response_outp = {}
# 	response_outp['result'] = best_price_dict
# 	return jsonify(response_outp)


@app.route('/train/', methods=['POST'])
def train():
	app.logger.info('TRAIN REQUEST RECEIVED')

	HOME = os.environ['HOME_SITE']
	# HOME = ''

	cv_acc = request.form['cv_acc']
	print('CV_ACC:', cv_acc)
	project_id = request.form['project_id']

	app.logger.info('ATTEMPTING DATA RETREIVAL')

	data_file = request.files['data']
	temp_data_path = f'temp/staging/{project_id}'
	if not os.path.isdir(temp_data_path):
		Path(temp_data_path).mkdir(parents=True)
	data_file.save(temp_data_path+'/data_staging.parquet')
	data_df = pd.read_parquet(temp_data_path+'/data_staging.parquet')

	app.logger.info('DATA SUCCESSFULLY LOADED')

	shutil.rmtree(temp_data_path)

	response_outp = {'result':0,
					 'cv_acc':0
					}

	del data_file
	gc.collect()

	pdm = rms_pricing_model(data_df)

	del data_df
	gc.collect()

	# save price info for optimization use
	PRICE_INFO_PATH = HOME+f'/projects/{project_id}/'
	if not os.path.isdir(PRICE_INFO_PATH):
		Path(PRICE_INFO_PATH).mkdir(parents=True)
	pdm.get_and_save_price_info(PRICE_INFO_PATH+'price_info.pkl')
	app.logger.info('PRICE INFO SAVED')

	item_ids = [int(x.split('_')[1]) for x in pdm.price_columns]
	proj_properties = {
		'num_items':len(item_ids),
		'items':item_ids
	}
	with open(PRICE_INFO_PATH+'proj_properties.json', 'w') as outfile:
		json.dump(proj_properties, outfile)
	
	# train models
	pdm.train_all_items(proj_id=project_id,retrain=True)

	# save models
	



	response_outp['result'] = 'Success'
	
	if cv_acc == 'True':
		app.logger.info('RUNNING OPTIONAL CROSS VALIDATION')
		perf_df = pdm.get_all_performance(proj_id=project_id)
		response_outp['cv_acc'] = perf_df.to_dict()

		with open(PRICE_INFO_PATH+'proj_cv_perf.json', 'w') as outfile:
			json.dump(perf_df.to_json(), outfile)

		app.logger.info(f'CROSS VALIDATION DONE')


	return jsonify(response_outp)

@app.route('/predict/', methods=['POST'])
def predict():
	app.logger.info('PREDICT REQUEST RECEIVED')

	# with open('keys.json') as f:
	# 		HOST_KEY = json.load(f)['host_key']
	HOST_KEY = os.environ['FUNCTIONS_KEY']

	prices = request.json['prices']
	project_id = request.get_json()['project_id']

	try:

		with open(proj_path+'proj_properties.json') as json_file:
			proj_properties = json.load(json_file)
		project_items = set(proj_properties['items'])

	except:
		return jsonify({'error':'project not found'})

	for k in list(prices.keys()):
		prices['Price_'+str(k)] = prices.pop(k)

	prices = pd.DataFrame(prices, index=[0])
	prices = prices.reindex(sorted(prices.columns), axis=1)

	prices_json = prices.to_json()

	items = [int(x.split('_')[1]) for x in prices.columns]

	models_list = []

	HOME = os.environ['HOME_SITE']
	# HOME = ''


	files = {}

	for item in items:
		with open(HOME+f'/projects/{project_id}/models/model_{item}.p','rb') as f:
			model = p.load(f)
		models_list.append(model)

	data_dict = {
		'prices':prices_json,
		'models':models_list,
	}

	data = p.dumps(data_dict)

	payload = {
		'code':HOST_KEY,
	}

	url = 'https://sutdcapstone22-filletofish.azurewebsites.net/api/fillet_func_4_predictbatch'
	# url = 'http://localhost:7071/api/fillet_func_4_predictbatch'
	app.logger.info('SENDING REQUEST TO FILLET SERVERS')
	result = requests.post(url, params=payload, data=data)
	app.logger.info(f'RESPONSE RECEIVED FROM FILLET {result.status_code}')
	pred_quantities = result.json()['qty_estimates']

	pred_quantities_dict = {}
	for item,qty_estimate in zip(items,pred_quantities):
		pred_quantities_dict[f'Qty_{item}'] = int(round(float(qty_estimate),0))

	response_outp = {'qty_estimates':pred_quantities_dict}
	app.logger.info('RETURNING RESPONSE')
	return jsonify(response_outp)



@app.route('/query_progress/', methods=['POST'])
def query_progress():


	HOME = os.environ['HOME_SITE']
	# HOME = ''

	project_id = request.get_json()['project_id']
	app.logger.info(f'TRAINING PROGRESS QUERY FOR PROJ {project_id}')
	proj_path = HOME+f'/projects/{project_id}/'
	
	try:

		with open(proj_path+'proj_properties.json') as json_file:
			proj_properties = json.load(json_file)
		project_items = set(proj_properties['items'])

	except:
		return jsonify({'error':'project not found'})

	try:
		model_filenames = os.listdir(proj_path+'models')
	except:
		return jsonify({'pct_complete':0})

	trained_models = set([int(x.split('_')[1].split('.')[0]) for x in model_filenames])
	remaining_models = project_items-trained_models


	#check if CV is done
	cv_done = 0
	cv_path = HOME+f'/projects/{project_id}/cv'
	try:
		cv_dir = os.listdir(cv_path)
		cv_progress = len(cv_dir)
	except:
		cv_progress = 0
	


	proj_dir = os.listdir(proj_path)
	if 'proj_cv_perf.json' in proj_dir:
		cv_done = 1


	response_outp = {
		'pct_complete':round((len(trained_models)/len(project_items)),3)*100,
		'project_items':list(project_items),
		'train_complete':list(trained_models),
		'cv_done':cv_done,
		'cv_progress':round(cv_progress/len(project_items),3)*100
	}
	return jsonify(response_outp)

@app.route('/get_cv_results/', methods=['POST'])
def get_cv_results():
	HOME = os.environ['HOME_SITE']
	# HOME = ''

	project_id = request.get_json()['project_id']
	proj_path = HOME+f'/projects/{project_id}/'

	try:

		with open(proj_path+'proj_properties.json') as json_file:
			proj_properties = json.load(json_file)
		project_items = set(proj_properties['items'])

	except:
		return jsonify({'error':'project not found'})

	proj_dir = os.listdir(proj_path)
	if 'proj_cv_perf.json' in proj_dir:
		with open(proj_path+'proj_cv_perf.json') as json_file:
			cv_results = json.load(json_file)
		return jsonify(cv_results)

	try:
		num_cv_done = len(os.listdir(proj_path+'cv'))
	except:
		return jsonify({'status':'incomplete'})

	if num_cv_done>0:
		perf_df = pd.DataFrame(columns=[
			'item_id','avg_sales','r2_score',
			'mae_score','mpe_score','rmse_score']
			)
		for cv_json_filename in os.listdir(proj_path+'cv'):
			with open(proj_path+'cv/'+cv_json_filename) as json_file:
				perf_df = perf_df.append(json.load(json_file), ignore_index=True)

		return jsonify(perf_df.to_json())

	else:
		return jsonify({'status':'incomplete'})
	




if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0')
