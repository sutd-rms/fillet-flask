from flask import Flask, request, redirect, url_for, flash, jsonify
from helper_functions import *
from core_functions import rms_pricing_model, GA
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
    
@app.route('/optimize/', methods=['POST'])
def optimize():
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
    models = [p.load(open(MODEL_PATH+'/model_{}.p'.format(item),'rb')) for item in items]
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

    data, cv_acc, project_id = parse_training_request(request)

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
    pdm.train_all_items(retrain=True)
    # save models
    item_ids = [int(x.split('_')[1]) for x in pdm.price_columns]
    for item_id in item_ids:
    	item_model = pdm.models[item_id]
    	MODEL_PATH = 'projects/{}/models/'.format(project_id)
    	if not os.path.isdir(MODEL_PATH):
    		Path(MODEL_PATH).mkdir(parents=True)
    	p.dump(item_model, open(MODEL_PATH+'model_{}.p'.format(item_id),'wb'))
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
