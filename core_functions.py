import pandas as pd
from helper_functions import optimize_memory
import numpy as np
from scipy import stats
import pickle as p

from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut

import sys, json, requests
import operator
import random
import base64
import zlib
import os
from pathlib import Path

import logging
log = logging.getLogger('fillet-flask.sub')

# < ---------------------- Configuration ---------------------->




master_data_input_path = 'data/data_all.csv'

# input: price_std, price_mean, price_names, constraints(in pre-specified json format), regressors (in dictionary), 
# population, generation, costs(optional), pre-set penalty constants, step(for prices), 
# random_seed for replication of results
def GeneticAlgorithm(price_std, price_mean, price_names, constraints, regressors, population, generation, 
                        costs=None, penalty_hard_constant=1000000, penalty_soft_constant=100000, step=0.05, 
                        random_seed=1):
    product_to_idx = {column.split('_')[1]: i for i, column in enumerate(price_names)}
    num_item = len(product_to_idx)
    # load constraints
    matrix, matrix_largerthan = np.zeros((1, num_item)), np.zeros((1, num_item))
    penalty, penalty_largerthan = [], []
    shifts, shifts_largerthan = [], []
    constraints_hardsoft = constraints[0]
    price_range = constraints[1]
    price_range_dic = {}
    for item in price_range:
        price_range_dic[item['item_id']] = [item['max'], item['min']]
    fixed_rules = [constraint for constraint in constraints_hardsoft if constraints['penalty'] == -1]
    constraints = [constraint for constraint in constraints_hardsoft if constraints['penalty'] != -1]
    
    for fixed_rule in fixed_rules:
        # assume all fixed_rules are equality for now
        products = fixed_rule['products']
        if not (set(products).issubset(set(product_to_idx.keys()))):
            continue
            # better raise error here
        scales = fixed_rule['scales']
        shift = fixed_rule['shift']
        array = np.zeros((1, num_item))
        for i, product in enumerate(products):
            array[0, product_to_idx[product]] = float(scales[i])
        matrix = np.vstack((matrix, array))
        penalty.append(penalty_hard_constant)
        shifts.append(shift)
    
    individuals_solved = []
    #Obtain two valid individuals based on hard constraints and price limits first
    x = cp.Variable((num_item, 1))
    objective1 = cp.Minimize(cp.sum_squares(x))
    objective2 = cp.Minimize(cp.sum(x))
    constraints = [matrix@x-shifts==0]+[matrix_largerthan@x-shifts_largerthan>=0]
    # add price range constraints
    for i, product in enumerate(product_to_idx):
        if int(product) not in price_range_dic.keys():
            continue
            # better raise error here
        else:
            constraints.append(x[i][0] >= price_range_dic[int(product)][1])
            constraints.append(x[i][0] <= price_range_dic[int(product)][0])
    prob1 = cp.Problem(objective1, constraints)
    result = prob1.solve()
    if prob1.status == 'optimal':
        individuals_solved.append(x.value)
    else:
        return None
        # better raise error here    
    prob2 = cp.Problem(objective2, constraints)
    result = prob2.solve()
    if prob2.status == 'optimal':
        individuals_solved.append(x.value)
    else:
        return None
        # better raise error here
    
    # Load soft constraints and price limits
    for constraint in constraints:
        products = constraint['products']
        if not (set(products).issubset(set(product_to_idx.keys()))):
            continue
        scales = constraint['scales']
        shift = constraint['shift']
        pnt = constraint['penlaty']
        equality = constraint['equality']
        array = np.zeros((1, num_item))
        for i, product in enumerate(products):
            array[0, product_to_idx[product]] = float(scales[i])
        if equality == 1: # (less than)
            matrix_largerthan = np.vstack((matrix_largerthan, -array))
            penalty_largerthan.append(penalty_soft_constant*pnt)
            shifts_largerthan.append(-shift)
        elif equality == 2: # (less than)
            matrix_largerthan = np.vstack((matrix_largerthan, array))
            penalty_largerthan.append(penalty_soft_constant*pnt)
            shifts_largerthan.append(shift)
        elif equality == 0:
            matrix = np.vstack((matrix, array))
            penalty.append(penalty_soft_constant*pnt)
            shifts.append(shift)
    
    for i, product in enumerate(product_to_idx):
        if int(product) not in price_range_dic.keys():
            continue
            # better raise error here
        else:
            array = np.zeros((1, num_item))
            array[0, product_to_idx[product]] = 1
            
            shift = price_range_dic[int(product)][0]
            matrix_largerthan = np.vstack((matrix_largerthan, -array))
            penalty_largerthan.append(penlaty_hard_constant)
            shifts_largerthan.append(-shift)
            
            shift = price_range_dic[int(product)][1]
            matrix_largerthan = np.vstack((matrix_largerthan, array))
            penalty_largerthan.append(penlaty_hard_constant)
            shifts_largerthan.append(shift)

    matrix = matrix[1:]
    penalty = np.array(penalty).reshape(-1, 1)
    shifts = np.array(shifts).reshape(-1, 1)
    matrix_largerthan = matrix_largerthan[1:]
    penalty_largerthan = np.array(penalty_largerthan).reshape(-1, 1)
    shifts_largerthan = np.array(shifts_largerthan).reshape(-1, 1)
    
    # components for DEAP package
    creator.create("RevenuePenalty", base.Fitness, weights=(1.,))
    creator.create("Individual", np.ndarray, fitness=creator.RevenuePenalty)
    toolbox = base.Toolbox()
    def get_individual(num_item, price_std, price_mean):
        return creator.Individual((np.floor((np.absolute(np.random.standard_normal(num_item))*price_std*2 + price_mean)/step)*step).round(2))
    toolbox.register("individual", get_individual, num_item, np.array(prices_std_list), np.array(prices_mean_list))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evalObjective(individual):
        """
        returns:
        (revenue, penalty_): revenue of this individual and penalty from it violating the constraints
        """
        # Calculating revenue
        quantity = np.zeros((num_item))
        product_to_idx = {column.split('_')[1]: i for i, column in enumerate(price_columns)}
        individual = individual.round(2)
        for code in regressors: # TODO: use multiple workers here to speedup the optimization process
            quantity[product_to_idx[code]] = regressors[code].predict(pd.DataFrame(individual.reshape(1, -1), columns=price_columns))
        output = individual.dot(quantity)
        # Calculating constraint violation penalty
        temp = (matrix.dot(individual.reshape(-1, 1)) - shifts).round(2)
        mask = temp != 0
        penalty_ = mask.T.dot(penalty)
        temp_largerthan = (matrix_largerthan.dot(individual.reshape(-1, 1)) - shifts_largerthan).round(2)
        mask_largerthan = temp_largerthan > 0
        penalty_largerthan_ = mask_largerthan.T.dot(penalty_largerthan)
        return (output - penalty_[0,0] - penalty_largerthan_[0,0],)

    def evalObjectiveProfit(individual):
        """
        returns:
        (revenue, penalty_): revenue of this individual and penalty from it violating the constraints
        """
        # Calculating revenue
        quantity = np.zeros((num_item))
        product_to_idx = {column.split('_')[1]: i for i, column in enumerate(price_columns)}
        individual = individual.round(2)
        for code in regressors: # TODO: use multiple workers here to speedup the optimization process
            quantity[product_to_idx[code]] = regressors[code].predict(pd.DataFrame(individual.reshape(1, -1), columns=price_columns))
        output = (individual-costs).dot(quantity)
        # Calculating constraint violation penalty
        temp = (matrix.dot(individual.reshape(-1, 1)) - shifts).round(2)
        mask = temp != 0
        penalty_ = mask.T.dot(penalty)
        temp_largerthan = (matrix_largerthan.dot(individual.reshape(-1, 1)) - shifts_largerthan).round(2)
        mask_largerthan = temp_largerthan > 0
        penalty_largerthan_ = mask_largerthan.T.dot(penalty_largerthan)
        return (output - penalty_[0,0] - penalty_largerthan_[0,0],)

    def cxTwoPointCopy(ind1, ind2):
        size = len(ind1)
        cxpoint1 = random.randint(1, size)
        cxpoint2 = random.randint(1, size - 1)
        if cxpoint2 >= cxpoint1:
            cxpoint2 += 1
        else: # Swap the two cx points
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1

        ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
            = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
        return ind1, ind2

    toolbox.register("evaluate", evalObjective) if not costs else toolbox.register("evaluate", evalObjectiveProfit)
    toolbox.register("mate", cxTwoPointCopy)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    from scoop import futures
    toolbox.register("map", futures.map)
    
    random.seed(random_seed)
    pop = toolbox.population(n=population)
    for indvd in selected_individuals:
        pop.append(creator.Individual(indvd.round(2).flatten()))
    hof = tools.ParetoFront(similar=np.array_equal)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=generation, stats=stats,
                        halloffame=hof)
    return pop, stats, hof

class GA(object):

	#initialize variables and lists
	def __init__(self): 

		self.models = []
		self.price_std = []
		self.price_mean = []
		self.parents = []
		self.newparents = []
		self.bests = []
		self.best_p = [] 
		self.price_names = []
		self.iterated = 1
		self.population = 0
		self.epoch = 0
		self.best_price = []

		# increase max recursion for long stack
		iMaxStackSize = 15000
		sys.setrecursionlimit(iMaxStackSize)

	# create the initial population 
	def initialize(self):
		self.num_item = self.price_mean.shape[0]
		for i in range(self.population):
			parent = (np.random.standard_normal(self.num_item)*self.price_std*2 + self.price_mean).tolist()
			self.parents.append(parent)

	# set the details of this problem
	def properties(self, models, population, max_epoch, price_std, price_mean, price_names):

		self.models = models
		self.population = population
		self.epoch = max_epoch
		self.price_std = price_std
		self.price_mean = price_mean
		self.price_names = price_names
		self.initialize()

	# calculate the fitness function for X
	def fitness(self, prices):
		
		df = pd.DataFrame(prices, columns = self.price_names) 
		quantities = np.zeros((self.population, len(self.models)))
		for i in range(len(self.models)):
			quantities[:, i] = self.models[i].predict(df)
		fitness = np.zeros((self.population))
		for i in range(len(self.models)):
			fitness += prices[:, i]*quantities[:, i]
		
		return fitness
   
	# run generations of GA
	def evaluation(self):

		# loop through parents and calculate fitness
		best_pop = self.population // 2
		ft = self.fitness(np.asarray(self.parents))
		for i in range(len(self.parents)):
			parent = self.parents[i]
			self.bests.append((ft[i], parent))

		# sort the fitness list by fitness
		self.bests.sort(key=operator.itemgetter(0), reverse=True)
		self.best_p = self.bests[:best_pop]
		highest_fitness = self.best_p[0][0]
		highest_price = self.best_p[0][1]
		self.best_p = [x[1] for x in self.best_p]
		
		return highest_fitness, highest_price

	# mutate children after certain condition
	def mutation(self, ch):

		for i in range(len(ch)):
			if random.uniform(0, 1) > 0.95:
				ch[i] = np.random.standard_normal(1)[0]*self.price_std[i]*2 + self.price_mean[i]
		return ch

	# crossover two parents to produce two children by miixing them under random ration each time
	def crossover(self, ch1, ch2):

		threshold = random.randint(1, len(ch1)-1)
		tmp1 = ch1[threshold:]
		tmp2 = ch2[threshold:]
		ch1 = ch1[:threshold]
		ch2 = ch2[:threshold]
		ch1.extend(tmp2)
		ch2.extend(tmp1)

		return ch1, ch2

	# run the GA algorithm
	def run(self):
		if self.epoch < self.iterated:
			return self.best_price
		# run the evaluation once
		highest_fitness, highest_price = self.evaluation()
		newparents = []
		pop = len(self.best_p)
		
		# print("{}th generation" .format(self.iterated))
		# print("best solution so far: {}".format(highest_fitness))
		self.best_price = highest_price

		# randomly shuffle the best parents
		random.shuffle(self.best_p)
		for i in range(0, pop):
			if i < pop-1:
				r1 = self.best_p[i]
				r2 = self.best_p[i+1]
				nchild1, nchild2 = self.crossover(r1, r2)
				newparents.append(nchild1)
				newparents.append(nchild2)
			else:
				r1 = self.best_p[i]
				r2 = self.best_p[0]
				nchild1, nchild2 = self.crossover(r1, r2)
				newparents.append(nchild1)
				newparents.append(nchild2)

		# mutate the new children and potential parents to ensure global optima found
		for i in range(len(newparents)):
			newparents[i] = self.mutation(newparents[i])	

		self.parents = newparents
		self.bests = []
		self.best_p = []
		self.iterated += 1
		return self.run()

class rms_pricing_model():
	def __init__(self, data):
		
		self.models = {}
		
		# Ingest Correctly Shaped Data
		sales_data = data[['Wk', 'Tier', 'Store', 'Item_ID', 'Qty_', 'Price_']].copy()

		# Optimize Memory Usage by Downcasting etc.
		sales_data = optimize_memory(sales_data)

		# Convert Data to Wide Format
		sales_data_wide = sales_data.set_index(
			['Wk', 'Tier', 'Store',
			 'Item_ID']).unstack(level=-1).reset_index().copy()
		sales_data_wide.columns = [
			''.join(str(i) for i in col).strip()
			for col in sales_data_wide.columns.values
		]
		sales_data_wide = sales_data_wide.sort_values(
			['Tier', 'Store', 'Wk'], ascending=True).reset_index(drop=True)

		# # Remove Store,Weeks with Nan Sales for Some Items ( Not sure if we want to do this or replace with 0 )
		# # Drops about 421 rows, hence seems reasonable to remove
		sales_data_wide_clean = sales_data_wide.dropna(axis=0).copy()
		self.data = sales_data_wide_clean
		self.price_columns = [
		col for col in sales_data_wide_clean.columns if col.startswith('Price')
		]

	def get_and_save_price_info(self, price_info_path):
		prices = self.data[self.price_columns]
		prices_std = prices.std(axis = 0, skipna = True)
		prices_mean = prices.mean(axis = 0, skipna = True)
		p.dump((prices_std, prices_mean, self.price_columns), open(price_info_path,'wb'))


	def get_performance(self, item_id):

		with open('keys.json') as f:
			HOST_KEY = json.load(f)['host_key']

		sales_data_wide_clean = self.data.copy()

		target_colname = 'Qty_' + str(item_id)

		if target_colname not in sales_data_wide_clean.columns:
			print('Item Not Found in Dataset.')
			return None

		Price_columns = [
			col for col in sales_data_wide_clean.columns
			if col.startswith('Price')
		]
		target_column = [target_colname]

		Week = sales_data_wide_clean[['Wk']].copy()
		X = sales_data_wide_clean[Price_columns].copy()
		X = X.reindex(sorted(X.columns), axis=1)
		y = sales_data_wide_clean[target_column].copy()

		payload = {
		'code':HOST_KEY,
		}

		if not os.path.isdir('temp'):
			Path('temp').mkdir(parents=True)

		X.to_parquet('temp/X.parquet')
		y.to_parquet('temp/y.parquet')
		Week.to_parquet('temp/Wk.parquet')

		files = {'X_file': open('temp/X.parquet', 'rb'),
				 'y_file': open('temp/y.parquet', 'rb'),
				 'Wk_file': open('temp/Wk.parquet', 'rb'),
				 }

		# data = {
		# 	'X':X.to_json(),
		# 	'y':y.to_json(),
		# 	'Week':Week.to_json()
		# }

		url = 'https://sutdcapstone22-filletofish.azurewebsites.net/api/fillet_func_2_cv'
		# url = 'http://localhost:7071/api/fillet_func_2_cv'
		result = requests.get(
			url, params=payload,
			files=files
			# data=base64.b64encode(zlib.compress(json.dumps(data).encode('utf-8')))
			)
		outp = result.json()
		outp['item_id'] = int(item_id)

		files['X_file'].close()
		files['y_file'].close()
		files['Wk_file'].close()

		return outp

	def get_all_performance(self):
		item_ids = [int(x.split('_')[1]) for x in self.price_columns]
		perf_df = pd.DataFrame(columns=[
			'item_id','avg_sales','r2_score',
			'mae_score','mpe_score','rmse_score']
			)
		for item_id in item_ids:
			item_perf = self.get_performance(item_id)
			perf_df = perf_df.append(item_perf, ignore_index=True)
		return perf_df

	def get_model(self, item_id, proj_id):

		log.info(f'TRAINING ITEM_ID {item_id} MODEL')

		with open('keys.json') as f:
			HOST_KEY = json.load(f)['host_key']






		sales_data_wide_clean = self.data.copy()

		target_colname = 'Qty_' + str(item_id)

		if target_colname not in sales_data_wide_clean.columns:
			print('Item Not Found in Dataset.')
			return None

		Price_columns = [
			col for col in sales_data_wide_clean.columns
			if col.startswith('Price')
		]
		target_column = [target_colname]

		X = sales_data_wide_clean[Price_columns].copy()
		X = X.reindex(sorted(X.columns), axis=1)
		y = sales_data_wide_clean[target_column].copy()

		payload = {
		'code':HOST_KEY,
		}

		if not os.path.isdir('temp'):
			Path('temp').mkdir(parents=True)

		X.to_parquet('temp/X.parquet')
		y.to_parquet('temp/y.parquet')

		files = {'X_file': open('temp/X.parquet', 'rb'),
				 'y_file': open('temp/y.parquet', 'rb')}

		# data = {
		# 		'X':X.to_json(),
		# 		'y':y.to_json()
		# 	}

		url = 'https://sutdcapstone22-filletofish.azurewebsites.net/api/fillet_func_1_train'
		# url = 'http://localhost:7071/api/fillet_func_1_train'
		
		result = requests.get(url, params=payload,
			files=files
			)
		# result = requests.get(url, params=payload,
		# 	data=zlib.compress(json.dumps(data).encode('utf-8'))
		# 	)
		model_json = result.json()['model_json']  







		# model = XGBRegressor()
		# model.fit(X, y)
		
		self.models[item_id] = model_json

		MODEL_PATH = f'projects/{proj_id}/models/'
		if not os.path.isdir(MODEL_PATH):
			Path(MODEL_PATH).mkdir(parents=True)

		with open(MODEL_PATH+f'model_{item_id}.json','w') as f:
			json.dump(model_json,f)

		files['X_file'].close()
		files['y_file'].close()

		return model_json
	
	def train_all_items(self, proj_id, retrain=True):
		item_ids = [int(x.split('_')[1]) for x in self.price_columns]
		for item_id in item_ids:
			if retrain==False:
				if item_id not in self.models.keys():
					self.get_model(item_id,proj_id)
			else:
				self.get_model(item_id,proj_id)