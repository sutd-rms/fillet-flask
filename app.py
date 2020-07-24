from flask import Flask, request, redirect, url_for, flash, jsonify
from helper_functions import optimize_floats, optimize_memory, get_top_features
from core_functions import rms_pricing_model
# from core_functions import rms_pricing_model, GA
import numpy as np
import pickle as p

import json
import pandas as pd
import os
import requests
from pathlib import Path
import gc
import shutil
import zlib
# from xgboost import XGBRegressor
import itertools
import cvxpy as cp

import logging

app = Flask('fillet-flask')
logging.basicConfig(
    level=logging.INFO,
    format=
    '%(asctime)s | %(levelname)-8s | %(name)-25s | %(threadName)-16s : %(message).80s'
)


# Set current working directory
HOME = os.environ['HOME_SITE']
# HOME = ''

# Function Key required to call fillet-functions
# with open('keys.json') as f:
#       HOST_KEY = json.load(f)['host_key']
HOST_KEY = os.environ['FUNCTIONS_KEY']


@app.route("/")
def hello():
    return "fillet-flask is alive."


# @app.route('/optimize/', methods=['POST'])
# def optimize():
#     app.logger.info('OPTIMIZE REQUEST RECEIVED')
#     # get input
#     project_id = request.get_json()['project_id']
#     constraints = request.get_json()['constraints']
#     population = request.get_json()['population']
#     max_epoch = request.get_json()['max_epoch']
#     # load price information
#     price_info_path = 'projects/{}/price_info.pkl'.format(project_id)
#     assert os.path.isfile(price_info_path), 'No price info file found.'
#     price_std, price_mean, price_names = p.load(open(price_info_path, 'rb'))
#     # load model
#     MODEL_PATH = 'projects/{}/models'.format(project_id)
#     assert os.path.isdir(MODEL_PATH), 'No model directory found.'
#     models_list = [x for x in os.listdir(MODEL_PATH) if x.startswith('model')]
#     # import pdb; pdb.set_trace()
#     items = [int(x.split('.')[0].split('_')[1]) for x in models_list]

#     models = []
#     for item in items:
#         item_model = XGBRegressor()
#         item_model.load_model(MODEL_PATH + f'/model_{item}.json')
#         models.append(item_model)

#     Optimizer = GA()
#     Optimizer.properties(models, population, max_epoch, price_std, price_mean,
#                          price_names)
#     best_price = Optimizer.run()

#     # Send result as Dict to avoid confusion
#     best_price_dict = {}
#     for item, price in zip(items, best_price):
#         best_price_dict[str(item)] = round(price, 2)
#     # run optimization

#     response_outp = {}
#     response_outp['result'] = best_price_dict
#     return jsonify(response_outp)


@app.route('/train/', methods=['POST'])
def train():
    '''This function receives a dataset, a new project id,
    and a training algo name. A project is started, and
    a request is made to fillet-functions for a price->demand
    model to be trained for each item_id in the dataset. The 
    model is returned as .pkl and saved to disk in the project
    folder.
    '''

    # Retrieve request details
    app.logger.info('REQUEST DETAILS RETREIVAL')
    cv_acc = request.form['cv_acc']
    project_id = request.form['project_id']
    modeltype = request.form['modeltype']

    # Log Receive Train Request
    app.logger.info(f'TRAIN REQUEST RECEIVED PROJ_ID: {project_id}')

    # Load Parquet in Memory
    app.logger.info('ATTEMPTING DATA RETREIVAL')
    
    try:
        data_file = request.files['data']
    except Exception as e:
        app.logger.info('ERROR IN DATA RETREIVAL')
        app.logger.info(e)

    # Create temp staging folder
    temp_data_path = f'temp/staging/{project_id}'
    if not os.path.isdir(temp_data_path):
        Path(temp_data_path).mkdir(parents=True)
    
    # Save Parquet Binary to temp disk
    data_file.save(temp_data_path + '/data_staging.parquet')

    # Read Parquet into memory as DataFrame
    data_df = pd.read_parquet(temp_data_path + '/data_staging.parquet')

    # Log data load success.
    app.logger.info('DATA SUCCESSFULLY LOADED')

    # Delete content from temp folder
    shutil.rmtree(temp_data_path)

    # Initialize rms_pricing_model Object with data
    pdm = rms_pricing_model(data_df)

    # Clear unused variables to free up memory
    del data_file
    del data_df
    gc.collect()

    # Create project folder
    price_info_path = HOME + f'/projects/{project_id}/'
    if not os.path.isdir(price_info_path):
        Path(price_info_path).mkdir(parents=True)

    # Save price_info as pickle to project folder
    pdm.get_and_save_price_info(price_info_path + 'price_info.pkl')

    # Log successful save of price info
    app.logger.info('PRICE INFO SAVED')

    # Get all item_ids from price columns in dataset
    item_ids = [int(x.split('_')[1]) for x in pdm.price_columns]

    # Save project properties to disk, for future referencing
    proj_properties = {'num_items': len(item_ids), 'items': item_ids}
    with open(price_info_path + 'proj_properties.json', 'w') as outfile:
        json.dump(proj_properties, outfile)

    # Trigger Train models on fillet-functions and save to disk
    pdm.train_all_items(proj_id=project_id, retrain=True, modeltype=modeltype)

    # Prepare output
    response_outp = {'result': 0, 'cv_acc': 0}
    response_outp['result'] = 'Success'

    # User specified CV option
    if cv_acc == 'True':
        # Log CV Request
        app.logger.info('RUNNING OPTIONAL CROSS VALIDATION')
        # Trigger CV requests on fillet-functions and save to disk
        perf_df = pdm.get_all_performance(proj_id=project_id, modeltype=modeltype)
        # If train + cv completes within timeout, return cv results
        response_outp['cv_acc'] = perf_df.to_dict()
        # Save CV results to project folder
        with open(price_info_path + 'proj_cv_perf.json', 'w') as outfile:
            json.dump(perf_df.to_json(), outfile)

        # Log Successful CV
        app.logger.info(f'CROSS VALIDATION DONE')

    return jsonify(response_outp)


@app.route('/predict/', methods=['POST'])
def predict():
    '''This function receives a project id and set of n prices,
    then loads the n models in the existing project id and 
    uploads the models to fillet-function to get sales quantity
    estimates for the given set of prices.
    '''

    with open(HOME + '/fillet_functions_api_endpoints.json') as f:
        fillet_func_urls = json.load(f)

    

    # Retrieve request details
    prices = request.json['prices']
    project_id = request.json['project_id']

    app.logger.info(f'PREDICT REQUEST RECEIVED PROJECT {project_id}')

    # If the project exists, get its list of item_ids
    proj_path = HOME + f'/projects/{project_id}/'
    try:
        with open(proj_path + 'proj_properties.json') as json_file:
            proj_properties = json.load(json_file)
        project_items = set(proj_properties['items'])

    # Otherwise, return error reponse
    except:
        app.logger.info('PROJECT {project_id} NOT FOUND')
        return jsonify({'error': 'project not found'})

    # "Price_" prefix is added to match feature/column names
    for k in list(prices.keys()):
        prices['Price_' + str(k)] = prices.pop(k)

    # Prices input converted to DataFrame, sorted, saved to json
    prices = pd.DataFrame(prices, index=[0])
    prices = prices.reindex(sorted(prices.columns), axis=1)
    prices_json = prices.to_json()

    # items iterator to retrieve saved models one by one
    items = [int(x.split('_')[1]) for x in prices.columns]

    # Open item_models and save them to list
    models_list = []
    for item in items:
        with open(HOME + f'/projects/{project_id}/models/model_{item}.p',
                  'rb') as f:
            model = p.load(f)
        models_list.append(model)

    pred_quantities_dict = {}

    app.logger.info('ATTEMPT LOCAL PREDICT REQUEST')

    for item in items:
        with open(HOME + f'/projects/{project_id}/models/model_{item}.p',
                  'rb') as f:
            model = p.load(f)
        prices = prices[model._Booster.feature_names]
        pred = model.predict(data=prices)
        pred_value = int(pred[0])
        pred_quantities_dict[f'Qty_{item}'] = pred_value

    app.logger.info('LOCAL PREDICT SUCCESS')
    app.logger.info('RETURNING RESPONSE')
    # Send qty_estimate in reponse
    response_outp = {'qty_estimates': pred_quantities_dict}
    return jsonify(response_outp)


@app.route('/query_progress/', methods=['POST'])
def query_progress():
    '''This function takes in a project_id and checks if there is
    an existing project_id. If there is a project previously
    created by a train request, responds with training progress.
    Otherwise returns error to prompt user to resend train request.
    '''

    # Get request details
    project_id = request.get_json()['project_id']
    app.logger.info(f'TRAINING PROGRESS QUERY FOR PROJ {project_id}')
    
    # Attempt to locate and load in project from project_id
    proj_path = HOME + f'/projects/{project_id}/'
    try:
        with open(proj_path + 'proj_properties.json') as json_file:
            proj_properties = json.load(json_file)
        project_items = set(proj_properties['items'])
    except:
        return jsonify({'error': 'project not found'})

    # Project exists, attempt to locate existing trained models
    try:
        model_filenames = os.listdir(proj_path + 'models')
    except:
        return jsonify({'pct_complete': 0})

    # Trained models exist, determine how many are left to train.
    trained_models = set(
        [int(x.split('_')[1].split('.')[0]) for x in model_filenames])
    remaining_models = project_items - trained_models

    # Check if CV is completed
    cv_done = 0

    # Attempt to locate and count the number of cv results completed
    cv_path = HOME + f'/projects/{project_id}/cv'
    try:
        cv_dir = os.listdir(cv_path)
        cv_progress = len(cv_dir)
    except:
        cv_progress = 0

    # If CV is 100% complete, 'proj_cv_perf.json' also exists.
    proj_dir = os.listdir(proj_path)
    if 'proj_cv_perf.json' in proj_dir:
        cv_done = 1

    # Format response
    response_outp = {
        'pct_complete': round(
            (len(trained_models) / len(project_items)), 3) * 100,
        'project_items': list(project_items),
        'train_complete': list(trained_models),
        'cv_done': cv_done,
        'cv_progress': round(cv_progress / len(project_items), 3) * 100
    }
    return jsonify(response_outp)

@app.route('/batch_query_progress/', methods=['POST'])
def batch_query_progress():
    '''This function takes in a list of project_ids and checks if there is
    an existing project_id. If there is a project previously
    created by a train request, responds with training progress.
    Otherwise returns error to prompt user to resend train request.
    '''

    # Get request details
    project_id_ls = request.get_json()['project_id_ls']
    batch_outp = {}
    for project_id in project_id_ls:
        app.logger.info(f'TRAINING PROGRESS QUERY FOR PROJ {project_id}')
        
        # Attempt to locate and load in project from project_id
        proj_path = HOME + f'/projects/{project_id}/'
        try:
            with open(proj_path + 'proj_properties.json') as json_file:
                proj_properties = json.load(json_file)
            project_items = set(proj_properties['items'])
        except:
            return jsonify({'error': 'project not found'})

        # Project exists, attempt to locate existing trained models
        try:
            model_filenames = os.listdir(proj_path + 'models')
        except:
            return jsonify({'pct_complete': 0})

        # Trained models exist, determine how many are left to train.
        trained_models = set(
            [int(x.split('_')[1].split('.')[0]) for x in model_filenames])
        remaining_models = project_items - trained_models

        # Check if CV is completed
        cv_done = 0

        # Attempt to locate and count the number of cv results completed
        cv_path = HOME + f'/projects/{project_id}/cv'
        try:
            cv_dir = os.listdir(cv_path)
            cv_progress = len(cv_dir)
        except:
            cv_progress = 0

        # If CV is 100% complete, 'proj_cv_perf.json' also exists.
        proj_dir = os.listdir(proj_path)
        if 'proj_cv_perf.json' in proj_dir:
            cv_done = 1

        # Format response
        response_outp = {
            'pct_complete': round(
                (len(trained_models) / len(project_items)), 3) * 100,
            'project_items': list(project_items),
            'train_complete': list(trained_models),
            'cv_done': cv_done,
            'cv_progress': round(cv_progress / len(project_items), 3) * 100
        }
        batch_outp[project_id] = response_outp
    return jsonify(batch_outp)


@app.route('/get_cv_results/', methods=['POST'])
def get_cv_results():
    '''This function attempts to locate and return all
    completed cv results for a given project_id.
    '''

    # Get request details
    project_id = request.get_json()['project_id']
    
    # Attempt to locate and load in project from project_id
    proj_path = HOME + f'/projects/{project_id}/'
    try:

        with open(proj_path + 'proj_properties.json') as json_file:
            proj_properties = json.load(json_file)
        project_items = set(proj_properties['items'])

    except:
        return jsonify({'error': 'project not found'})

    # Attempt to locate and load in cv results
    proj_dir = os.listdir(proj_path)
    if 'proj_cv_perf.json' in proj_dir:
        with open(proj_path + 'proj_cv_perf.json') as json_file:
            cv_results = json.load(json_file)
        return jsonify(cv_results)

    # Check if any CV results have been saved so far
    try:
        num_cv_done = len(os.listdir(proj_path + 'cv'))
    except:
        return jsonify({'status': 'incomplete'})

    # If at least one CV results is available so far
    if num_cv_done > 0:
        # Create placeholder DataFrame
        perf_df = pd.DataFrame(columns=[
            'item_id', 'avg_sales', 'r2_score', 'mae_score', 'mpe_score',
            'rmse_score'
        ])
        # Append all found cv results to DataFrame
        for cv_json_filename in os.listdir(proj_path + 'cv'):
            with open(proj_path + 'cv/' + cv_json_filename) as json_file:
                perf_df = perf_df.append(json.load(json_file),
                                         ignore_index=True)
        # Return filled DataFrame of available CV results
        return jsonify(perf_df.to_json())

    else:
        return jsonify({'status': 'incomplete'})


@app.route('/get_feature_importances/', methods=['POST'])
def get_feature_importances():
    '''This function returns feature importances for all 
    item models in a project.
    '''

    # Get request details
    project_id = request.get_json()['project_id']
    
    # Attempt to locate and load in project from project_id
    proj_path = HOME + f'/projects/{project_id}/'

    # If the project exists, get its list of item_ids
    try:
        with open(proj_path + 'proj_properties.json') as json_file:
            proj_properties = json.load(json_file)
        project_items = set(proj_properties['items'])

    # Otherwise, return error reponse
    except:
        return jsonify({'error': 'project not found'})

    models_path = proj_path + 'models/'
    
    response_outp = {}

    for item in project_items:
        model_filename = models_path + f'model_{item}.p'
        with open(model_filename, 'rb') as f:
            item_model = p.load(f)
        imp_df = get_top_features(item_model, n=10)
        response_outp[item] = imp_df.to_dict()

    return response_outp


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


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')