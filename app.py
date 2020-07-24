from flask import Flask, request, redirect, url_for, flash, jsonify
from helper_functions import optimize_floats, optimize_memory
from core_functions import rms_pricing_model, GA, GeneticAlgorithm
import numpy as np
import pickle as p
import json
import pandas as pd
import os, requests
from os import listdir
from pathlib import Path
from xgboost import XGBRegressor
import itertools
import cvxpy as cp
import logging
from os.path import isfile, join

app = Flask('fillet-flask')
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s | %(levelname)-8s | %(name)-25s | %(threadName)-16s : %(message).80s'
	)

	
@app.route("/")
def hello():
	return "Hello World!!!!"

@app.route('/detect_conflict/', methods=['POST'])
def detect_conflict():
    # get data
    app.logger.info('DETECT CONFLICT REQUEST RECEIVED')
    constraints = request.get_json()['constraints']
    rule_list = constraints[0]
    hard_rule_list = [i for i in rule_list if i['penalty'] == -1]
    price_range = constraints[1]
    price_range_dic = {}
    for item in price_range:
        price_range_dic[item['item_id']] = [item['max'], item['min']]
    product_list = list(set(list(itertools.chain.from_iterable([rule['products'] for rule in hard_rule_list]))))
    num_item = len(product_list)
    # cvxpy library to solve linear programming problem 
    x = cp.Variable((len(product_list), 1))
    objective = cp.Minimize(cp.sum_squares(x)) # any surrogate objective
    constraints = []
    # add equality constraints
    equal_list = [i for i in hard_rule_list if i['equality']==True]
    if len(equal_list) != 0:
        matrix1 = np.zeros((len(equal_list), len(product_list)))
        shifts1 = []
        for k in range(len(equal_list)):
            products = equal_list[k]['products']
            scales = equal_list[k]['scales']
            scales = [float(s) for s in scales]
            shift = equal_list[k]['shift']
            for j, product in enumerate(products):
                matrix1[k, product_list.index(product)] = float(scales[j])
            shifts1.append(shift)
        shifts1 = np.array(shifts1).reshape(-1, 1)
        constraints.append(matrix1@x-shifts1==0)
    # add inequality constraints
    inequal_list = [i for i in hard_rule_list if i['equality']==False]
    if len(inequal_list) != 0:
        matrix2 = np.zeros((len(inequal_list), len(product_list)))
        penalty = []
        shifts2 = []
        for i in range(len(inequal_list)):
            products = inequal_list[i]['products']
            scales = inequal_list[i]['scales']
            scales = [float(s) for s in scales]
            shift = inequal_list[i]['shift']
            for j, product in enumerate(products):
                print(i, j)
                matrix2[i, product_list.index(product)] = float(scales[j])
            shifts2.append(shift)
        shifts2 = np.array(shifts2).reshape(-1, 1)
        constraints.append(matrix2@x-shifts2>=0)
    # add price range constraints
    for i, product in enumerate(product_list):
        if int(product) not in price_range_dic.keys():
            constraints.append(x[i][0] <= 20.)
            constraints.append(x[i][0] >= 0.)
        else:
            constraints.append(x[i][0] >= price_range_dic[int(product)][1])
            constraints.append(x[i][0] <= price_range_dic[int(product)][0])
    prob = cp.Problem(objective, constraints)
    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve()
    # return the result
    if prob.status == 'infeasible':
        return jsonify({'conflict':'Conflict exists'})
    else:
        return jsonify({'conflict':'No conflict'})
    
    
    
@app.route('/optimize/', methods=['POST'])
def optimize():
    app.logger.info('OPTIMIZE REQUEST RECEIVED')
    # get input
    project_id = request.get_json()['project_id']
    constraints = request.get_json()['constraints']
    population =  request.get_json()['population']
    max_epoch = request.get_json()['max_epoch']
    model_path = request.get_json()['model_path']
    price_info_path = request.get_json()['price_info_path']
    # load price information
    assert os.path.isfile(price_info_path), 'No price info file found.'
    price_std, price_mean, price_names = p.load(open(price_info_path, 'rb'))
    product_to_idx = {column.split('_')[1]: i for i, column in enumerate(price_names)}
    # load model
    assert os.path.isdir(model_path), 'No model directory found.'
    onlyfiles = [f for f in listdir(model_path) if isfile(join(model_path, f))]
    regressors = {}
    for file in onlyfiles:
        name = file.strip().split('.')[0]
        regressors[name] = p.load(open(model_path + file, 'rb'))
    # run optimization
    result = GeneticAlgorithm(price_std, price_mean, price_names, constraints, regressors, population, max_epoch)
                        # costs=None, penalty_hard_constant=1000000, penalty_soft_constant=100000, step=0.05, 
                        # random_seed=1
    if result:
        pop, stats, hof = result
    # Send result as Dict to avoid confusion
        response_outp = {}
        response_outp['result'] = np.array(hof[0]).tolist()
        return jsonify(response_outp)
    else:
        # better raise error here
        return None


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
		response_outp['cv_acc'] = perf_df.to_dict()

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
